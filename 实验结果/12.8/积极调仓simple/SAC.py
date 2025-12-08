# =========================================
# imports & basic setup
# =========================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque
from math import inf

import talib
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3 import SAC

import matplotlib.pyplot as plt
import seaborn as sns

import optuna
import tempfile
import warnings

warnings.filterwarnings("ignore")

# =========================================
# LSTM feature extractor
# =========================================
class CustomLSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.window_size = observation_space.shape[0]
        self.input_dim = observation_space.shape[1]
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=features_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(observations)
        last_step_out = lstm_out[:, -1, :]
        last_step_out = self.dropout(last_step_out)
        return last_step_out


# =========================================
# data loading（顶层加载，子进程也能用）
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tech_daily = pd.read_csv("data/科技股票.csv")
tech_daily.set_index("date", inplace=True)
tech_daily.columns = ["AAPL", "GOOG", "MSFT"]

debt = pd.read_csv("data/无风险.csv")
debt.set_index("date", inplace=True)
debt.columns = ["US_debt"]

tmp = pd.read_csv("data/指数和贵金属.csv")
tmp.columns = ["date", "SP500", "Gold"]
tmp.set_index("date", inplace=True)

df = pd.merge(tech_daily, debt, how="left", on="date")
df = pd.merge(df, tmp, how="left", on="date")
df["date"] = pd.to_datetime(df.index)
df.set_index("date", inplace=True)

df["Cash"] = 1.0

# 插值 & 填补缺失
df.interpolate(method="time", inplace=True)
df.dropna(inplace=True)

# =========================================
# Pre-calc Data
# =========================================
TECH_COLS = ["AAPL", "GOOG", "MSFT", "SP500", "Gold"]


def preprocess_data(df_, tech_cols):
    """Pre-calculate features on the entire dataset to avoid warm-up loss."""
    raw_data = df_.copy().astype(np.float64)
    price_data = raw_data[tech_cols].copy()

    # Features
    returns = price_data.pct_change()
    volatility = returns.rolling(20).std() * np.sqrt(252)
    volatility.columns = [c + "_vol" for c in volatility.columns]

    rsi_df = pd.DataFrame(index=price_data.index)
    for col in tech_cols:
        arr = price_data[col].values.astype(np.float64)
        rsi = talib.RSI(arr, timeperiod=14)
        rsi_df[col + "_rsi"] = rsi / 100.0

    macd_df = pd.DataFrame(index=price_data.index)
    for col in tech_cols:
        arr = price_data[col].values.astype(np.float64)
        macd, _, _ = talib.MACD(
            arr, fastperiod=12, slowperiod=26, signalperiod=9
        )
        macd_df[col + "_macd"] = np.tanh(np.nan_to_num(macd, nan=0.0))

    corr_df = pd.DataFrame(index=price_data.index)
    sp = price_data["SP500"]
    for col in tech_cols:
        corr = price_data[col].rolling(30).corr(sp)
        corr_df[col + "_corr"] = corr

    features = pd.concat(
        [returns, volatility, rsi_df, macd_df, corr_df],
        axis=1,
    )
    features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Clean raw data
    raw_data = (
        raw_data.replace([np.inf, -np.inf], 0.0)
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    return raw_data, features


full_raw, full_features = preprocess_data(df, TECH_COLS)


# =========================================
# Env with tunable reward parameters
# =========================================
class PortfolioOptimizationEnv(gym.Env):
    """
    带 Risk Budgeting + 防 NaN/inf，并支持 reward 参数化的环境
    """

    def __init__(
        self,
        tickers,
        window_size,
        start_date,
        end_date,
        raw_df_all,
        feature_df_all,
        initial_balance=10000.0,
        # ---- reward parameters to tune ----
        w_turn=0.01,
        reward_scale=50.0,
        seed=None,
    ):
        super().__init__()

        self.tickers = tickers  # all assets including US_debt & Cash
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.w_turn = w_turn
        self.reward_scale = reward_scale

        # Slice data for this environment instance
        self.raw_data = raw_df_all.loc[start_date:end_date]
        self.feature_data = feature_df_all.loc[start_date:end_date]

        # Validation
        if len(self.feature_data) < self.window_size + 1:
            raise ValueError(
                f"Feature window too short: {len(self.feature_data)} rows, "
                f"window_size={self.window_size}"
            )

        self.n_features = self.feature_data.shape[1]

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.tickers),)
        )
        self.observation_space = gym.spaces.Box(
            low=-inf, high=inf, shape=(window_size, self.n_features)
        )

        self.last_action = np.ones(len(self.tickers)) / len(self.tickers)

        if seed is not None:
            np.random.seed(seed)

    # ----------------------- reset -----------------------
    def reset(self, seed=None, options=None):
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.last_action = np.ones(len(self.tickers)) / len(self.tickers)

        start = max(0, self.current_step - self.window_size)
        end = min(len(self.feature_data), self.current_step)
        obs = self.feature_data.iloc[start:end].values

        if obs.shape[0] < self.window_size:
            pad = np.zeros((self.window_size - obs.shape[0], obs.shape[1]))
            obs = np.vstack([pad, obs])

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, {"balance": self.balance}

    # ----------------------- step -----------------------
    def step(self, raw_action):
        raw_action = np.asarray(raw_action).ravel()
        raw_action = np.nan_to_num(raw_action, nan=0.0, posinf=0.0, neginf=0.0)
        raw_action = np.clip(raw_action, -1.0, 1.0)

        # Linear Normalization to allow zero weights
        weights = (raw_action + 1.0) / 2.0
        weight_sum = np.sum(weights)
        if weight_sum > 1e-8:
            action = weights / weight_sum
        else:
            action = np.ones_like(weights) / len(weights)

        # end check
        if self.current_step >= len(self.raw_data):
            start = self.current_step - self.window_size
            end = self.current_step
            obs = self.feature_data.iloc[start:end].values
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            return obs, 0.0, True, False, {
                "balance": self.balance,
                "weights": action,
            }

        cur_price = self.raw_data.iloc[self.current_step].values
        prev_price = self.raw_data.iloc[self.current_step - 1].values
        asset_ret = np.nan_to_num(
            cur_price / prev_price - 1.0,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        port_ret = float(np.sum(action * asset_ret))

        # transaction cost
        turnover = float(np.sum(np.abs(action - self.last_action)))
        self.balance *= (1.0 + port_ret)

        # reward: pure return minus turnover cost
        reward = port_ret - self.w_turn * turnover
        
        reward = float(
            np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        )
        reward *= self.reward_scale

        self.last_action = action
        self.current_step += 1
        done = self.current_step >= len(self.raw_data) - 1

        # next obs
        start = max(0, self.current_step - self.window_size)
        end = min(len(self.feature_data), self.current_step)
        obs = self.feature_data.iloc[start:end].values
        if obs.shape[0] < self.window_size:
            pad = np.zeros((self.window_size - obs.shape[0], obs.shape[1]))
            obs = np.vstack([pad, obs])

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        info = {
            "balance": float(self.balance),
            "weights": action,
            "turnover": float(turnover),
        }
        return obs, reward, done, False, info


# =========================================
# Train / test split
# =========================================
tickers = df.columns.tolist()
window_size = 30
initial_balance = 10000.0
seed = 8

full_range = df.index
split_idx = int(len(full_range) * 0.6)
train_start_date = full_range[0]
train_end_date = full_range[split_idx]
test_start_date = full_range[split_idx + 1]
test_end_date = full_range[-1]

# Policy kwargs（LSTM + MLP）
policy_kwargs = dict(
    features_extractor_class=CustomLSTMExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[128, 64],
)


# =========================================
# helper: backtest & compute Sharpe
# =========================================
def evaluate_model(model, vecnorm_path, start_date, end_date, env_kwargs=None):
    """Backtest model on given period and return Sharpe ratio."""
    if env_kwargs is None:
        env_kwargs = {}

    test_env_raw = DummyVecEnv(
        [
            lambda: PortfolioOptimizationEnv(
                tickers=tickers,
                window_size=window_size,
                start_date=start_date,
                end_date=end_date,
                raw_df_all=full_raw,
                feature_df_all=full_features,
                initial_balance=initial_balance,
                **env_kwargs,
            )
        ]
    )
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

    balances.append(initial_balance)
    start_idx = base_env.raw_data.index.get_loc(
        base_env.raw_data.index[base_env.window_size]
    )
    dates.append(base_env.raw_data.index[start_idx])

    while not dones[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = test_env.step(action)
        if not dones[0]:
            balances.append(float(infos[0]["balance"]))
            cur_idx = base_env.current_step
            if cur_idx < len(base_env.raw_data):
                dates.append(base_env.raw_data.index[cur_idx])

    balances_arr = np.array(balances)
    if len(balances_arr) <= 1:
        return 0.0

    balance_series = pd.Series(balances_arr, index=dates)
    daily_returns = balance_series.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0

    ann_ret = daily_returns.mean() * 252
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = float(ann_ret / ann_vol) if ann_vol != 0 else 0.0
    return sharpe


# =========================================
# Optuna objective
# =========================================
N_TRAIN_STEPS = 20000  # 每个 trial 总步数（会除以 N_ENVS）


def objective(trial: optuna.Trial) -> float:
    # ---- sample reward parameters ----
    w_turn = trial.suggest_float("w_turn", 0.0, 0.05)
    reward_scale = trial.suggest_float("reward_scale", 10.0, 100.0)

    # ---- sample SAC hyperparams ----
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 5e-4, log=True
    )
    batch_size = trial.suggest_categorical(
        "batch_size", [256, 512, 1024]
    )
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    tau = trial.suggest_float("tau", 0.001, 0.02, log=True)
    train_freq = trial.suggest_categorical("train_freq", [32, 64, 128])
    gradient_steps = trial.suggest_categorical(
        "gradient_steps", [32, 64]  # 128 很慢，先关掉
    )
    ent_coef = trial.suggest_categorical(
        "ent_coef", ["auto", "auto_0.1", "auto_0.3"]
    )

    # ---- make parallel train env (SubprocVecEnv) ----
    N_ENVS = 8  # CPU 允许的话可以调大

    def make_train_env():
        def _init():
            return PortfolioOptimizationEnv(
                tickers=tickers,
                window_size=window_size,
                start_date=train_start_date,
                end_date=train_end_date,
                raw_df_all=full_raw,
                feature_df_all=full_features,
                initial_balance=initial_balance,
                w_turn=w_turn,
                reward_scale=reward_scale,
                seed=seed,
            )
        return _init

    vec_train_env = SubprocVecEnv([make_train_env() for _ in range(N_ENVS)])

    vec_train_env = VecNormalize(
        vec_train_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )

    model = SAC(
        "MlpPolicy",
        vec_train_env,
        device=device,
        verbose=0,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=300_000,
        ent_coef=ent_coef,
        gamma=gamma,
        tau=tau,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
    )

    # 临时保存 VecNormalize 统计，用于评估
    with tempfile.TemporaryDirectory() as tmpdir:
        vecnorm_path = os.path.join(tmpdir, "vecnorm.pkl")
        vec_train_env.save(vecnorm_path)

        # 训练：注意除以 N_ENVS
        model.learn(total_timesteps=N_TRAIN_STEPS // N_ENVS)

        # 训练后再保存一次（包含最新统计）
        vec_train_env.save(vecnorm_path)

        # 评估（Sharpe）
        env_params = {
            # 注意：这里不再传 raw_df_all / feature_df_all，避免重复参数冲突
            "w_turn": w_turn,
            "reward_scale": reward_scale,
        }
        sharpe = evaluate_model(
            model, vecnorm_path, test_start_date, test_end_date, env_params
        )

    # 释放 env
    vec_train_env.close()
    return sharpe


# =========================================
# main: Windows 上必须这样写，才能安全使用 SubprocVecEnv + Optuna
# =========================================
N_TRIALS = 10  # 可以先小一点调试，OK 再改大


def main():
    print("Using device:", device)
    print("df shape:", df.shape)
    print(df.head())
    print("Train:", train_start_date.date(), "->", train_end_date.date())
    print("Test :", test_start_date.date(), "->", test_end_date.date())

    # ---- Run Optuna ----
    study = optuna.create_study(
        direction="maximize",
        study_name="sac_portfolio_tuning",
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n===== Optuna best trial =====")
    best_trial = study.best_trial
    print("Best Sharpe:", best_trial.value)
    for k, v in best_trial.params.items():
        print(f"{k}: {v}")

    best_params = best_trial.params

    # =========================================
    # 用最佳参数重新训练 + 正式回测 & 画图
    # =========================================
    def make_best_env():
        def _init():
            env_ = PortfolioOptimizationEnv(
                tickers=tickers,
                window_size=window_size,
                start_date=train_start_date,
                end_date=train_end_date,
                raw_df_all=full_raw,
                feature_df_all=full_features,
                initial_balance=initial_balance,
                w_turn=best_params["w_turn"],
                reward_scale=best_params["reward_scale"],
                seed=seed,
            )
            return env_
        return _init

    log_dir = "./sb3_logs_optuna"
    os.makedirs(log_dir, exist_ok=True)

    vec_train_env = DummyVecEnv([make_best_env()])
    vec_train_env = VecNormalize(
        vec_train_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )

    model = SAC(
        "MlpPolicy",
        vec_train_env,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        buffer_size=300_000,
        ent_coef=best_params["ent_coef"],
        gamma=best_params["gamma"],
        tau=best_params["tau"],
        train_freq=best_params["train_freq"],
        gradient_steps=best_params["gradient_steps"],
    )

    print("Starting final training with best params...")
    model.learn(total_timesteps=50000)
    print("Training finished.")

    model_path = os.path.join(log_dir, "sac_optuna_best.zip")
    vecnorm_path = os.path.join(log_dir, "sac_vecnorm_best.pkl")
    model.save(model_path)
    vec_train_env.save(vecnorm_path)
    print("Saved model to", model_path)

    # ---------- 正式回测 & 画图 ----------
    print("Starting backtest with best params...")
    test_env_raw = DummyVecEnv(
        [
            lambda: PortfolioOptimizationEnv(
                tickers=tickers,
                window_size=window_size,
                start_date=test_start_date,
                end_date=test_end_date,
                raw_df_all=full_raw,
                feature_df_all=full_features,
                initial_balance=initial_balance,
                w_turn=best_params["w_turn"],
                reward_scale=best_params["reward_scale"],
                seed=seed,
            )
        ]
    )
    test_env = VecNormalize.load(vecnorm_path, test_env_raw)
    test_env.training = False
    test_env.norm_reward = False

    obs = test_env.reset()
    dones = [False]
    balances = []
    dates = []
    weights_history = []

    base_env = test_env.envs[0]
    while hasattr(base_env, "env"):
        base_env = base_env.env

    balances.append(initial_balance)
    start_idx = base_env.raw_data.index.get_loc(
        base_env.raw_data.index[base_env.window_size]
    )
    dates.append(base_env.raw_data.index[start_idx])

    while not dones[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = test_env.step(action)
        
        weights = infos[0]["weights"]
        weights_history.append(weights)
        
        if not dones[0]:
            balances.append(float(infos[0]["balance"]))
            cur_idx = base_env.current_step
            if cur_idx < len(base_env.raw_data):
                dates.append(base_env.raw_data.index[cur_idx])

    balances_arr = np.array(balances)
    if len(balances_arr) > 0:
        total_growth = balances_arr[-1] / balances_arr[0]
        start_date_ = pd.to_datetime(dates[0])
        end_date_ = pd.to_datetime(dates[-1])
        duration_days = (end_date_ - start_date_).days
        years = duration_days / 365.25 if duration_days > 0 else 0

        balance_series = pd.Series(balances_arr, index=dates)
        daily_returns = balance_series.pct_change().dropna()

        if years > 0:
            cagr = total_growth ** (1 / years) - 1
        else:
            cagr = 0.0

        ann_vol = daily_returns.std() * np.sqrt(252)
        sharpe = cagr / ann_vol if ann_vol != 0 else 0.0
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = cagr / downside_vol if downside_vol != 0 else 0.0
        cumulative_max = np.maximum.accumulate(balances_arr)
        drawdowns = (balances_arr - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdowns)
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

        print("-" * 40)
        print(f"Performance Metrics ({start_date_.date()} to {end_date_.date()})")
        print("-" * 40)
        print(
            f"Final Balance:    {balances_arr[-1]:.2f} (Initial: {balances_arr[0]:.2f})"
        )
        print(f"Total Growth:     {total_growth:.4f}x")
        print(f"CAGR (Ann. Ret):  {cagr:.2%}")
        print(f"Ann. Volatility:  {ann_vol:.2%}")
        print(f"Sharpe Ratio:     {sharpe:.4f}")
        print(f"Sortino Ratio:    {sortino:.4f}")
        print(f"Max Drawdown:     {max_drawdown:.2%}")
        print(f"Calmar Ratio:     {calmar:.4f}")
        print("-" * 40)

        # 画收益曲线
        asset_cols = tickers
        prices = df.loc[dates, asset_cols].copy()
        norm_prices = prices / prices.iloc[0]
        norm_strategy = balances_arr / balances_arr[0]

        strategy_df = pd.DataFrame(
            {"date": pd.to_datetime(dates), "value": norm_strategy}
        )
        strategy_df = strategy_df.sort_values("date").drop_duplicates("date")

        plt.figure(figsize=(12, 6))
        for col in asset_cols:
            plt.plot(
                norm_prices.index,
                norm_prices[col],
                label=col,
                alpha=0.3,
                linestyle="--",
            )
        plt.plot(
            strategy_df["date"],
            strategy_df["value"],
            label="RL Strategy",
            linewidth=2,
            color="black",
        )
        plt.title(
            f"Backtest: {start_date_.date()} to {end_date_.date()} (Dates & Costs)"
        )
        plt.xlabel("Date")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("backtest.png")
        plt.show()

        # 权重堆叠图
        if len(weights_history) > 0:
            sns.set_theme(style="whitegrid")
            df_weights = pd.DataFrame(weights_history, columns=tickers)
            df_weights.index = dates[: len(df_weights)]

            plt.figure(figsize=(15, 8))
            plt.stackplot(df_weights.index, df_weights.T, labels=df_weights.columns)
            plt.title(
                "Asset Allocation Evolution (Position Weights)",
                fontsize=16,
                pad=20,
            )
            plt.xlabel("Date")
            plt.ylabel("Weight (0.0 - 1.0)")
            plt.ylim(0, 1.0)
            plt.margins(x=0)
            plt.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                fontsize=10,
                title="Assets",
            )
            plt.tight_layout()
            plt.savefig("weights.png")
            plt.show()

            print("\n=== Position Statistics ===")
            print(df_weights.mean().sort_values(ascending=False))
            print(
                f"\nMax Single Position: {df_weights.max().max():.2%} ({df_weights.max().idxmax()})"
            )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
