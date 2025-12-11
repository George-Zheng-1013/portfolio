# =========================================
# imports & basic setup
# =========================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from math import inf

import talib
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from sb3_contrib import RecurrentPPO

import matplotlib.pyplot as plt
import seaborn as sns

import optuna
import tempfile
import warnings

warnings.filterwarnings("ignore")




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

# df["Cash"] = 10000.0

# 插值 & 填补缺失
df.interpolate(method="time", inplace=True)
df.dropna(inplace=True)

# =========================================
# Pre-calc Data
# =========================================
TECH_COLS = ["AAPL", "GOOG", "MSFT", "SP500", "Gold","US_debt"]


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
    bias_df = pd.DataFrame(index=price_data.index)
    quantile_df = pd.DataFrame(index=price_data.index)
    
    for col in tech_cols:
        arr = price_data[col].values.astype(np.float64)
        macd, _, _ = talib.MACD(
            arr, fastperiod=12, slowperiod=26, signalperiod=9
        )
        macd_df[col + "_macd"] = np.tanh(np.nan_to_num(macd, nan=0.0))

        # 1. 乖离率 (Bias) = (Price - MA60) / MA60
        # 暗示超跌 (负值大) 或 超买 (正值大)
        ma60 = talib.SMA(arr, timeperiod=60)
        bias = (arr - ma60) / ma60
        bias_df[col + "_bias60"] = np.nan_to_num(bias, nan=0.0)

        # 2. 价格分位数 (Price Quantile) - 过去 1 年 (252天)
        # 用 rolling rank 并不是很快，这里用 rolling min/max 归一化近似
        # Position = (Price - Min252) / (Max252 - Min252)
        # 0.0 = 最低价(极其便宜), 1.0 = 最高价(贵)
        rolling_min = price_data[col].rolling(252).min()
        rolling_max = price_data[col].rolling(252).max()
        quantile = (price_data[col] - rolling_min) / (rolling_max - rolling_min)
        quantile_df[col + "_qtl252"] = np.nan_to_num(quantile, nan=0.5) # 默认填0.5中位数

    corr_df = pd.DataFrame(index=price_data.index)
    sp = price_data["SP500"]
    for col in tech_cols:
        corr = price_data[col].rolling(30).corr(sp)
        corr_df[col + "_corr"] = corr

    features = pd.concat(
        [returns, volatility, rsi_df, macd_df, bias_df, quantile_df, corr_df],
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
        reward_scale=50.0,
        temperature=0.05,
        seed=None,
    ):
        super().__init__()

        self.tickers = tickers  # all assets including US_debt & Cash
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.reward_scale = reward_scale
        self.temperature = temperature

        # Slice data for this environment instance
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
        raw_action = np.nan_to_num(raw_action, nan=0.0, posinf=1.0, neginf=-1.0)
        # raw_action = np.clip(raw_action, -1.0, 1.0)

        # 温度系数 (temperature) 越低，分布越尖锐（越重仓）
        # 动态使用 env 初始化传入的 temperature 参数
        
        # 1. 原始动作是 [-1, 1], 先放大再过 softmax
        exp_values = np.exp(raw_action / self.temperature)
        weights = exp_values / np.sum(exp_values)
        
        # 2. 强行截断微小权重，避免浮点数拖尾
        weights[weights < 0.001] = 0.0
        action = weights / np.sum(weights)

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

        # =========================================
        # 修改处：使用对数收益 (Log Return) + “左侧交易”重奖
        # =========================================
        
        # 1. 基础收益 (Log Return)
        log_ret = np.log1p(port_ret)

        # 2. 左侧交易奖励 (Left-Side Trading Bonus)
        # 逻辑：如果持仓资产处于过去 20 天低点附近，给予额外奖励
        lookback = 20
        # 获取最近窗口价格用于计算 min
        s_idx = max(0, self.current_step - lookback)
        recent_window = self.raw_data.iloc[s_idx : self.current_step + 1].values
        window_min = np.min(recent_window, axis=0)

        # 判断是否在低点 (1% 容差)
        is_dip = cur_price <= (window_min * 1.01)

        # 排除现金 (Cash) 和 美债 (US_debt) 以免模型偷懒不该抄底的地方
        for i, t in enumerate(self.tickers):
            if t in ["Cash", "US_debt"]:
                is_dip[i] = False

        # 计算奖励：(持仓权重 * 是否底部) * 奖励系数
        # 系数 0.05 非常大 (相当于 5% 的日收益)，符合“重奖”的要求，鼓励接飞刀/抄底
        dip_bonus = float(np.sum(weights * is_dip)) * 0.05
        
        reward = log_ret + dip_bonus
        
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

# =========================================
# helper: backtest & compute cagr
# =========================================
def evaluate_model(model, vecnorm_path, start_date, end_date, env_params):
    # Re-create env just for testing
    test_env_raw = DummyVecEnv([
        lambda: PortfolioOptimizationEnv(
            tickers=tickers,
            window_size=window_size,
            start_date=start_date,
            end_date=end_date,
            raw_df_all=full_raw,
            feature_df_all=full_features,
            initial_balance=initial_balance,
            **env_params
        )
    ])
    
    test_env = VecNormalize.load(vecnorm_path, test_env_raw)
    test_env.training = False
    test_env.norm_reward = False

    obs = test_env.reset()
    done = False
    balances = []

    # RecurrentPPO needs state handling
    lstm_states = None
    episode_starts = np.ones((test_env.num_envs,), dtype=bool)

    while not done:
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=True
        )
        obs, reward, dones, info = test_env.step(action)
        done = dones[0]
        episode_starts = dones
        balances.append(info[0]["balance"])

    balances = np.array(balances)
    final_balance = balances[-1]
    initial_balance_val = balances[0]

    # Calculate CAGR
    num_days = len(balances)
    num_years = num_days / 252
    cagr = (final_balance / initial_balance_val) ** (1 / num_years) - 1

    return cagr

# =========================================
# Optuna objective
# =========================================
N_TRAIN_STEPS = 20000  # 每个 trial 总步数（会除以 N_ENVS）


def objective(trial: optuna.Trial) -> float:
    # ---- sample reward & env parameters ----
    reward_scale = trial.suggest_float("reward_scale", 10.0, 100.0)
    temperature = trial.suggest_float("temperature", 0.01, 1.0, log=True) # 0.01=极度重仓, 1.0=分散

    # ---- sample PPO hyperparams ----
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    
    # ---- sample LSTM hyperparams ----
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])
    n_lstm_layers = trial.suggest_categorical("n_lstm_layers", [1, 2])

    policy_kwargs = dict(
        lstm_hidden_size=lstm_hidden_size,
        n_lstm_layers=n_lstm_layers,
        net_arch=[dict(pi=[64, 64], vf=[64, 64])], #稍微加深一点 MLP
    )
    
    # ---- make parallel train env (SubprocVecEnv) ----
    N_ENVS = 8

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
                reward_scale=reward_scale,
                temperature=temperature,
                seed=seed,
            )
        return _init

    vec_train_env = SubprocVecEnv([make_train_env() for _ in range(N_ENVS)])

    vec_train_env = VecNormalize(
        vec_train_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_train_env,
        device=device,
        verbose=0,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
    )

    # 临时保存 VecNormalize 统计，用于评估
    with tempfile.TemporaryDirectory() as tmpdir:
        vecnorm_path = os.path.join(tmpdir, "vecnorm.pkl")
        vec_train_env.save(vecnorm_path)

        # 训练
        model.learn(total_timesteps=N_TRAIN_STEPS)

        # 训练后再保存一次（包含最新统计）
        vec_train_env.save(vecnorm_path)

        # 评估（cagr）
        env_params = {
            "reward_scale": reward_scale,
            "temperature": temperature,
        }
        cagr = evaluate_model(
            model, vecnorm_path, test_start_date, test_end_date, env_params
        )

    # 释放 env
    vec_train_env.close()
    return cagr


# =========================================
# main: Windows 上必须这样写，才能安全使用 SubprocVecEnv + Optuna
# =========================================
N_TRIALS = 5  # 可以先小一点调试，OK 再改大


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
    print("Best cagr:", best_trial.value)
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
                reward_scale=best_params["reward_scale"],
                temperature=best_params["temperature"],
                seed=seed,
            )
            return env_
        return _init

    log_dir = "./sb3_logs_optuna"
    os.makedirs(log_dir, exist_ok=True)

    n_envs = 8
    vec_train_env = SubprocVecEnv([make_best_env() for _ in range(n_envs)])
    vec_train_env = VecNormalize(
        vec_train_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )

    policy_kwargs = dict(
        lstm_hidden_size=best_params["lstm_hidden_size"],
        n_lstm_layers=best_params["n_lstm_layers"],
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_train_env,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        n_steps=best_params["n_steps"],
        gamma=best_params["gamma"],
        gae_lambda=best_params["gae_lambda"],
        clip_range=best_params["clip_range"],
        ent_coef=best_params["ent_coef"],
        max_grad_norm=best_params["max_grad_norm"],
    )

    print("Starting final training with best params...")
    model.learn(total_timesteps=100000)
    print("Training finished.")

    model_path = os.path.join(log_dir, "ppo_recurrent_best.zip")
    vecnorm_path = os.path.join(log_dir, "ppo_vecnorm_best.pkl")
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
                reward_scale=best_params["reward_scale"],
                temperature=best_params["temperature"],
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

    # RecurrentPPO state
    lstm_states = None
    episode_starts = np.ones((test_env.num_envs,), dtype=bool)

    while not dones[0]:
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts,
            deterministic=True
        )
        obs, reward, dones, infos = test_env.step(action)
        episode_starts = dones
        
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
