import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from collections import deque
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Custom LSTM Feature Extractor ---
class CustomLSTMExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor that uses an LSTM to process the time-series observation.
    
    :param observation_space: (gym.Space) The observation space of the environment.
    :param features_dim: (int) The dimension of the output features from the LSTM.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        # Observation space shape is (window_size, n_features)
        # We assume the input is 2D: [window_size, n_features]
        self.window_size = observation_space.shape[0]
        self.input_dim = observation_space.shape[1]
        
        # LSTM Layer
        # input_size: number of features per time step
        # hidden_size: dimension of the LSTM hidden state (and output features)
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=features_dim, batch_first=True)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch_size, window_size, n_features)
        # LSTM output shape: (batch_size, window_size, hidden_size)
        # We take the output of the last time step to capture the temporal context
        lstm_out, _ = self.lstm(observations)
        return lstm_out[:, -1, :]

# --- 2. Data Loading and Preprocessing ---
def load_data():
    # Load Tech Stocks
    tech_daily = pd.read_csv(r"data/科技股票.csv")
    tech_daily['date'] = pd.to_datetime(tech_daily['date'])
    tech_daily.set_index('date', inplace=True)
    tech_daily.columns = ['AAPL', 'GOOG', 'MSFT']

    # Load US Debt (Risk Free)
    us_debt = pd.read_csv(r"data/无风险.csv")
    us_debt['date'] = pd.to_datetime(us_debt['date'])
    us_debt.set_index('date', inplace=True)
    us_debt.columns = ['US_debt']

    # Load Indices and Precious Metals
    indices = pd.read_csv(r"data/指数和贵金属.csv")
    indices['date'] = pd.to_datetime(indices['date'])
    indices.set_index('date', inplace=True)
    indices.columns = ['SP500', 'Gold']

    # Merge DataFrames
    df = tech_daily.join(us_debt, how='outer').join(indices, how='outer')
    
    # Interpolate missing values
    df.interpolate(method='time', inplace=True)
    df = df.sort_index()
    
    # Fill any remaining NaNs
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    return df

# --- 3. Environment Definition ---
class PortfolioOptimizationEnv(gym.Env):
    def __init__(
        self, df, tickers, window_size, start_date, end_date, initial_balance, seed=None
    ):
        super().__init__()
        self.df = df
        self.tickers = tickers
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Get data
        self.raw_data, self.feature_data = self.get_data(start_date, end_date)
        self.n_features = self.feature_data.shape[1]

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(len(tickers),), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, self.n_features), dtype=np.float32
        )

        self.return_window = deque(maxlen=window_size)
        self.last_action = np.ones(len(tickers)) / len(tickers)

        if seed is not None:
            pass # Seeding handled externally or by gym wrappers in newer versions

    def get_data(self, start_date, end_date):
        data = self.df.copy()
        data = data.loc[start_date:end_date]
        
        raw_data = data[self.tickers].copy()

        # Calculate features
        returns = data.pct_change()

        mom_frames = []
        for window in [5, 20]:
            mom = data / data.shift(window) - 1
            mom.columns = [f"{col}_mom_{window}" for col in data.columns]
            mom_frames.append(mom)

        vol = returns.rolling(window=20, min_periods=1).std()
        vol.columns = [f"{col}_vol_20" for col in data.columns]

        ma = data.rolling(window=20, min_periods=1).mean()
        ma_dev = data / ma - 1
        ma_dev.columns = [f"{col}_ma_dev_20" for col in data.columns]

        returns.columns = [f"{col}_ret" for col in data.columns]

        # Feature data
        feature_data = pd.concat([returns, vol, ma_dev] + mom_frames, axis=1)
        
        feature_data = feature_data.dropna()
        raw_data = raw_data.reindex(feature_data.index)
        
        return raw_data, feature_data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = self.window_size

        self.return_window.clear()
        self.last_action = np.ones(len(self.tickers)) / len(self.tickers)

        obs = self.feature_data.iloc[
            self.current_step - self.window_size : self.current_step
        ].values
        info = {"balance": self.balance}
        return obs.astype(np.float32), info

    def step(self, action):
        action = np.asarray(action).ravel()
        action = np.clip(action, 0, 1)
        action = action / np.sum(action + 1e-8)

        prev_balance = self.balance

        current_price = self.raw_data.iloc[self.current_step].values
        prev_price = self.raw_data.iloc[self.current_step - 1].values
        
        asset_returns = current_price / prev_price - 1

        self.return_window.append(asset_returns)

        portfolio_return = np.sum(asset_returns * action)
        self.balance = self.balance * (1 + portfolio_return)
        base_reward = np.log(self.balance / prev_balance)

        risk_penalty = 0
        if len(self.return_window) >= 5:
            R = np.vstack(self.return_window)
            cov_matrix = np.cov(R.T)
            sigma_p2 = action.T @ cov_matrix @ action
            risk_penalty = sigma_p2

        turnover = np.sum(np.abs(action - self.last_action))
        cost = turnover
        self.last_action = action

        lambda_risk = 1
        lambda_turnover = 0.05
        reward = base_reward - lambda_risk * risk_penalty - lambda_turnover * cost

        self.current_step += 1
        done = self.current_step >= len(self.raw_data) - 1

        obs_end = min(len(self.feature_data), self.current_step + self.window_size)
        obs_start = max(0, obs_end - self.window_size)
        obs = self.feature_data.iloc[obs_start:obs_end].values

        terminated = bool(done)
        truncated = False
        info = {"balance": self.balance}

        return obs.astype(np.float32), reward, terminated, truncated, info

# --- 4. Main Execution ---
if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Parameters
    tickers = df.columns.tolist() # Using all columns as tickers/assets for simplicity based on notebook
    window_size = 30
    initial_balance = 10000
    
    # Define training period
    train_start = '2015-11-09'
    train_end = '2024-11-08'
    
    # Create environment factory
    def make_env():
        return PortfolioOptimizationEnv(df, tickers, window_size, train_start, train_end, initial_balance)
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # --- KEY CHANGE: Define Policy Keyword Arguments to use LSTM ---
    policy_kwargs = dict(
        features_extractor_class=CustomLSTMExtractor,
        features_extractor_kwargs=dict(features_dim=128), # Output dimension of LSTM
        net_arch=[128, 128] # MLP layers after LSTM
    )

    # Initialize SAC with custom policy
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs, # Pass the LSTM policy here
        verbose=1,
        device=device,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=100000,
        tau=0.005,
        ent_coef="auto",
    )

    print("Starting training with LSTM feature extractor...")
    model.learn(total_timesteps=10000)
    print("Training finished.")

    # Save model
    model.save("sac_lstm_portfolio")
    print("Model saved to sac_lstm_portfolio.zip")
