import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from collections import deque
import os
import talib
from math import inf
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# --- 1. Define Custom Feature Extractor ---
class CustomLSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.window_size = observation_space.shape[0]
        self.input_dim = observation_space.shape[1]
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=features_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(observations)
        last_step_out=lstm_out[:, -1, :]
        last_step_out=self.dropout(last_step_out)
        return last_step_out

# --- 2. Data Preparation ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
try:
    tech_daily = pd.read_csv(os.path.join("data", "科技股票.csv"))
    tech_daily.set_index('date', inplace=True)
    tech_daily.columns=['AAPL','GOOG','MSFT']

    debt = pd.read_csv(os.path.join("data", "无风险.csv"))
    debt.set_index('date', inplace=True)
    debt.columns=['US_debt']

    tmp = pd.read_csv(os.path.join("data", "指数和贵金属.csv"))
    tmp.columns=['date','SP500','Gold']
    tmp.set_index('date', inplace=True)

    df = pd.merge(tech_daily, debt, how='left', on='date')
    df = pd.merge(df, tmp, how='left', on='date')
    df['date'] = pd.to_datetime(df.index)
    df.set_index('date', inplace=True)
    df['Cash'] = 1
    df.interpolate(method='ffill', inplace=True) 
    df = df.sort_index()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print("Data range:", df.index.min(), "to", df.index.max())

# --- 3. Define Environment ---
class PortfolioOptimizationEnv(gym.Env):
    def __init__(
        self, tickers, window_size, start_date, end_date, initial_balance, lambda_softmax, lambda_turnover, seed=None
    ):
        super().__init__()
        self.tickers = tickers
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.lambda_softmax=lambda_softmax
        self.lambda_turnover=lambda_turnover

        self.raw_data, self.feature_data = self.get_data(tickers, start_date, end_date)
        self.n_features = self.feature_data.shape[1]

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(len(tickers),))
        self.observation_space = gym.spaces.Box(
            low=-inf, high=inf, shape=(window_size, self.n_features)
        )

        self.return_window = deque(maxlen=window_size)
        self.last_action = np.ones(len(tickers)) / len(tickers)

        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def get_data(self, tickers, start_date, end_date):
        data = df.copy().dropna()
        data = data.loc[start_date:end_date, tickers]
        raw_data = data.copy()

        returns = data.pct_change()
        feature_list=[returns]

        volatility=data.pct_change().rolling(window=self.window_size).std()
        volatility.columns=[f'{col}_vol_{self.window_size}' for col in volatility.columns]
        feature_list.append(volatility)

        rsi_df=pd.DataFrame(index=data.index)
        for col in data.columns:
            values=data[col].values.astype(float)
            if np.all(values==values[0]):
                rsi_df[f'{col}_rsi']=50
            else:
                try:
                    rsi_values=talib.RSI(values, timeperiod=14)
                    rsi_df[f'{col}_rsi']=rsi_values
                except Exception as e:
                    print(f'RSI calculation failed: {e}')
        feature_list.append(rsi_df)
        
        macd_df=pd.DataFrame(index=data.index)
        for col in data.columns:
            values=data[col].values.astype(float)
            if np.all(values==values[0]):
                macd_df[f'{col}_macd']=0
            else:
                try:
                    macd_values=talib.MACD(values, fastperiod=12, slowperiod=26, signalperiod=9)[0]
                    macd_df[f'{col}_macd']=macd_values
                except Exception as e:
                    print(f'MACD calculation failed: {e}')
        feature_list.append(macd_df)

        feature_data=pd.concat(feature_list, axis=1)
        feature_data=feature_data.dropna()
        raw_data=raw_data.reindex(feature_data.index).dropna()

        return raw_data, feature_data

    def reset(self, seed=None):
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.return_window.clear()
        self.last_action = np.ones(len(self.tickers)) / len(self.tickers)
        obs = self.feature_data.iloc[
            self.current_step - self.window_size : self.current_step
        ].values
        info = {"balance": self.balance}
        return obs, info

    def step(self, action):
        action = np.asarray(action).ravel()
        action = np.exp((action - np.max(action)) / self.lambda_softmax)
        action = action / np.sum(action)

        if self.current_step >= len(self.raw_data):
            done = True
            obs = self.feature_data.iloc[
                self.current_step - self.window_size : self.current_step
            ].values
            return obs, 0, True, False, {"balance": self.balance}

        current_price = self.raw_data.iloc[self.current_step].values[: len(self.tickers)]
        prev_price = self.raw_data.iloc[self.current_step - 1].values[: len(self.tickers)]
        asset_returns = current_price / prev_price - 1
        self.return_window.append(asset_returns)

        portfolio_return = np.sum(asset_returns * action)
        self.balance = self.balance * (1 + portfolio_return)
        
        volatility = np.std(self.return_window) if len(self.return_window) > 10 else 0.01
        turnover = np.sum(np.abs(action - self.last_action))
        self.last_action = action

        reward = portfolio_return / (volatility + 1e-6) - self.lambda_turnover * turnover

        self.current_step += 1
        done = self.current_step >= len(self.raw_data) - 1
        obs = self.feature_data.iloc[
            self.current_step - self.window_size : self.current_step
        ].values

        return obs, reward, bool(done), False, {"balance": self.balance, "weights": action}

# --- 4. Training Setup (Fixed Split) ---
tickers = df.columns.tolist()
window_size = 20
initial_balance = 10000
seed = 8

# Fixed Train/Test Split
train_start_str = "2015-11-09"
train_end_str = "2023-12-31"
test_start_str = "2024-01-01"
test_end_str = "2025-11-28"

print(f"Train Period: {train_start_str} to {train_end_str}")
print(f"Test Period:  {test_start_str} to {test_end_str}")

log_dir = "./sb3_logs_fixed"
os.makedirs(log_dir, exist_ok=True)

def make_env_for_period(start_date_str, end_date_str, monitor_file=None):
    def _init():
        env_ = PortfolioOptimizationEnv(
            tickers=tickers,
            window_size=window_size,
            start_date=start_date_str,
            end_date=end_date_str,
            initial_balance=initial_balance,
            lambda_turnover=0.001,
            lambda_softmax=0.25,
            seed=seed,
        )
        if monitor_file is not None:
            env_ = Monitor(env_, filename=monitor_file)
        return env_
    return _init

policy_kwargs = dict(
    features_extractor_class=CustomLSTMExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[128, 64]
)

# --- 5. Train ---
monitor_path = os.path.join(log_dir, "monitor_train.csv")
vec_train_env = DummyVecEnv([make_env_for_period(train_start_str, train_end_str, monitor_file=monitor_path)])
vec_train_env = VecNormalize(vec_train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

model = SAC(
    "MlpPolicy",
    vec_train_env,
    verbose=1,
    device=device,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    batch_size=256,
    buffer_size=200_000,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",
    train_freq=(256, "step"),
    gradient_steps=-1,
)

print("Starting training...")
model.learn(total_timesteps=20000)
print("Training finished.")

model_path = os.path.join(log_dir, "sac_fixed_train.zip")
vecnorm_path = os.path.join(log_dir, "sac_vecnorm_fixed.pkl")
model.save(model_path)
vec_train_env.save(vecnorm_path)
print(f"Saved model to {model_path}")

# --- 6. Backtest ---
print("Starting backtest...")
test_env_raw = DummyVecEnv([make_env_for_period(test_start_str, test_end_str, monitor_file=None)])
test_env = VecNormalize.load(vecnorm_path, test_env_raw)
test_env.training = False
test_env.norm_reward = False

obs = test_env.reset()
dones = [False]
balances = []
dates = []

base_env = test_env.envs[0]
while hasattr(base_env, "env"):
    base_env = base_env.env

while not dones[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, infos = test_env.step(action)
    balances.append(float(infos[0]["balance"]))
    
    cur_idx = base_env.current_step - 1
    if cur_idx < len(base_env.raw_data):
        dates.append(base_env.raw_data.index[cur_idx])

# Analysis
balances_arr = np.array(balances)
if len(balances_arr) > 0:
    total_growth = balances_arr[-1] / balances_arr[0]
    print(f"Final Balance: {balances_arr[-1]:.2f} (Initial: {balances_arr[0]:.2f})")
    print(f"Total Growth: {total_growth:.4f}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(dates), balances_arr, label="Strategy")
    plt.title("Fixed Train/Test Split Backtest (2024-2025)")
    plt.xlabel("Date")
    plt.ylabel("Balance")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "backtest_result.png"))
    print(f"Plot saved to {os.path.join(log_dir, 'backtest_result.png')}")
