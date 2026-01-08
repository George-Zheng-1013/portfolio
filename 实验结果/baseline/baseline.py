import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 数据准备
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

# 现金资产价格恒为 1
df["Cash"] = 1.0

# 基本清洗
df.interpolate(method="time", inplace=True)
df.dropna(inplace=True)

# 资产列表
tickers = ["AAPL", "GOOG", "MSFT", "SP500", "Gold", "US_debt", "Cash"]

# 训练/测试切分
full_range = df.index
split_idx = int(len(full_range) * 0.6)
train_start_date = full_range[0]
train_end_date = full_range[split_idx]
test_start_date = full_range[split_idx + 1]
test_end_date = full_range[-1]

# 基本参数
window_size = 30
initial_balance = 10000.0
seed = 8

print("Using same data as SAC+CAAN baseline comparison:")
print("df shape:", df.shape)
print(df.head())
print("Train period:", train_start_date.date(), "->", train_end_date.date())
print("Test  period:", test_start_date.date(), "->", test_end_date.date())


# 环境：动作为权重(非负、归一)、观测为过去窗口价格
class PortfolioOptimizationEnv(gym.Env):
    def __init__(
        self, tickers, window_size, start_date, end_date, initial_balance, seed=None
    ):
        super().__init__()

        self.tickers = tickers
        self.window_size = window_size
        self.initial_balance = initial_balance

        # 使用和 SAC 相同的 df 数据，而不是 yfinance
        self.data = self.get_data(tickers, start_date, end_date)

        # 动作为各资产权重
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(len(tickers),))
        # 观测为过去 window_size 天的价格
        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(window_size, len(tickers))
        )

        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)

        self.balance = None
        self.current_step = None

    def get_data(self, tickers, start_date, end_date):
        # 从全局 df 切片并清洗
        data = df.loc[start_date:end_date, tickers].copy()
        data = data.astype(np.float64)
        # 去掉有缺失的行
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) <= self.window_size + 1:
            raise ValueError(
                f"Not enough data for window_size={self.window_size}, got {len(data)} rows."
            )
        return data

    def reset(self, seed=None, options=None):
        # Handle seed if provided (for Gymnasium API compatibility)
        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)

        # 重置组合
        self.balance = self.initial_balance
        self.current_step = self.window_size

        # 初始观测：过去 window_size 天价格
        obs = self.data.iloc[
            self.current_step - self.window_size : self.current_step
        ].values
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs.reshape(self.observation_space.shape), {}

    def step(self, action):
        # 归一化权重，计算组合收益与奖励，推进时间并返回观测
        action = np.asarray(action).ravel()
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action_sum = np.sum(action)
        if action_sum > 1e-8:
            weights = action / action_sum
        else:
            weights = np.ones_like(action) / len(action)

        prev_balance = self.balance

        # 当前 & 前一日价格
        asset_prices = self.data.iloc[self.current_step].values
        prev_prices = self.data.iloc[self.current_step - 1].values
        asset_returns = np.nan_to_num(
            asset_prices / prev_prices - 1.0, nan=0.0, posinf=0.0, neginf=0.0
        )

        # 更新组合净值
        port_ret = float(np.sum(asset_returns * weights))
        self.balance = self.balance * (1.0 + port_ret)

        # 奖励为对数收益
        reward = float(np.log(self.balance / (prev_balance + 1e-8)))

        # 时间推进
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False  # We don't use episode truncation

        # 新观测
        obs = self.data.iloc[
            self.current_step - self.window_size : self.current_step
        ].values
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        info = {
            "balance": float(self.balance),
            "weights": weights,  # 方便后续画权重堆叠图，与 SAC 一致
        }

        return (
            obs.reshape(self.observation_space.shape),
            reward,
            terminated,
            truncated,
            info,
        )


# 日志目录与 Monitor
log_dir = "./sb3_logs_baseline_ppo"
os.makedirs(log_dir, exist_ok=True)
monitor_path = os.path.join(log_dir, "monitor.csv")


def make_env():
    env_ = PortfolioOptimizationEnv(
        tickers=tickers,
        window_size=window_size,
        start_date=train_start_date,
        end_date=train_end_date,
        initial_balance=initial_balance,
        seed=seed,
    )
    env_ = Monitor(env_, filename=monitor_path)
    return env_


vec_env = DummyVecEnv([make_env])

# ------------ PPO 训练（保持原函数形式）------------
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=20000)  # 可按需调整
model.save("ppo_portfolio_optimization_baseline")

# ------------ 训练过程 reward 曲线（原 baseline 功能保留）------------
train_df = pd.read_csv(monitor_path, comment="#")
if len(train_df) > 0:
    train_df["ep"] = np.arange(1, len(train_df) + 1)
    train_df["reward_smooth"] = (
        train_df["r"].rolling(window=max(5, len(train_df) // 50), min_periods=1).mean()
    )

    plt.figure(figsize=(8, 4))
    plt.plot(train_df["ep"], train_df["r"], alpha=0.3, label="Episode reward")
    plt.plot(train_df["ep"], train_df["reward_smooth"], label="Smoothed")
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title("Training profile (episode rewards) - PPO Baseline")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No episodes logged in monitor file; training may have terminated too early.")


# 回测与指标
# ---------- 构建测试环境并回测 ----------
test_env = DummyVecEnv(
    [
        lambda: PortfolioOptimizationEnv(
            tickers=tickers,
            window_size=window_size,
            start_date=test_start_date,
            end_date=test_end_date,
            initial_balance=initial_balance,
            seed=seed,
        )
    ]
)

obs = test_env.reset()
dones = [False]
balances = []
dates = []
weights_history = []

# 取出底层 env（DummyVecEnv 里的第一个 env）
base_env = test_env.envs[0]
while hasattr(base_env, "env"):
    base_env = base_env.env

# 初始时刻
balances.append(initial_balance)
start_idx = base_env.data.index.get_loc(base_env.data.index[base_env.window_size])
dates.append(base_env.data.index[start_idx])

# 回测循环（与 SAC 逻辑相同，只是算法为 PPO）
while not dones[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, infos = test_env.step(action)

    weights = infos[0].get("weights", None)
    if weights is not None:
        weights_history.append(weights)

    if not dones[0]:
        balances.append(float(infos[0]["balance"]))
        cur_idx = base_env.current_step
        if cur_idx < len(base_env.data):
            dates.append(base_env.data.index[cur_idx])

balances_arr = np.array(balances)

if len(balances_arr) > 0:
    total_growth = balances_arr[-1] / balances_arr[0]
    start_date_ = pd.to_datetime(dates[0])
    end_date_ = pd.to_datetime(dates[-1])
    duration_days = (end_date_ - start_date_).days
    years = duration_days / 365.25 if duration_days > 0 else 0.0

    balance_series = pd.Series(balances_arr, index=dates)
    daily_returns = balance_series.pct_change().dropna()

    if years > 0:
        cagr = total_growth ** (1 / years) - 1.0
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
    print(
        f"Performance Metrics (PPO Baseline) "
        f"{start_date_.date()} to {end_date_.date()}"
    )
    print("-" * 40)
    print(
        f"Final Balance:    {balances_arr[-1]:.2f} " f"(Initial: {balances_arr[0]:.2f})"
    )
    print(f"Total Growth:     {total_growth:.4f}x")
    print(f"CAGR (Ann. Ret):  {cagr:.2%}")
    print(f"Ann. Volatility:  {ann_vol:.2%}")
    print(f"Sharpe Ratio:     {sharpe:.4f}")
    print(f"Sortino Ratio:    {sortino:.4f}")
    print(f"Max Drawdown:     {max_drawdown:.2%}")
    print(f"Calmar Ratio:     {calmar:.4f}")
    print("-" * 40)

    # 保存回测数据与指标
    result_df = pd.DataFrame(
        {
            "date": dates,
            "portfolio_value": balances_arr,
            "daily_return": [0.0] + list(daily_returns.values),
        }
    )
    result_df.to_csv("ppo_baseline_results.csv", index=False)
    print("\nBacktest results saved to: ppo_baseline_results.csv")

    metrics_summary = pd.DataFrame(
        {
            "Metric": [
                "Initial Balance",
                "Final Balance",
                "Total Growth",
                "CAGR",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Max Drawdown",
                "Calmar Ratio",
                "Start Date",
                "End Date",
                "Duration (Days)",
            ],
            "Value": [
                f"{balances_arr[0]:.2f}",
                f"{balances_arr[-1]:.2f}",
                f"{total_growth:.4f}x",
                f"{cagr:.4f}",
                f"{ann_vol:.4f}",
                f"{sharpe:.4f}",
                f"{sortino:.4f}",
                f"{max_drawdown:.4f}",
                f"{calmar:.4f}",
                str(start_date_.date()),
                str(end_date_.date()),
                str(duration_days),
            ],
        }
    )
    metrics_summary.to_csv("ppo_baseline_metrics.csv", index=False)
    print("Performance metrics saved to: ppo_baseline_metrics.csv")

    # 收益曲线
    asset_cols = tickers
    prices = df.loc[dates, asset_cols].copy()
    norm_prices = prices / prices.iloc[0]
    norm_strategy = balances_arr / balances_arr[0]

    strategy_df = pd.DataFrame({"date": pd.to_datetime(dates), "value": norm_strategy})
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
        label="PPO RL Strategy",
        linewidth=2,
        color="black",
    )
    plt.title(f"PPO Baseline Backtest: {start_date_.date()} to {end_date_.date()}")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_baseline_backtest.png")
    plt.show()

    # 权重堆叠图与统计
    if len(weights_history) > 0:
        sns.set_theme(style="whitegrid")
        df_weights = pd.DataFrame(weights_history, columns=tickers)
        df_weights.index = dates[: len(df_weights)]

        plt.figure(figsize=(15, 8))
        plt.stackplot(df_weights.index, df_weights.T, labels=df_weights.columns)
        plt.title(
            "Asset Allocation Evolution (Position Weights) - PPO Baseline",
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
        plt.savefig("ppo_baseline_weights.png")
        plt.show()

        print("\n=== Position Statistics (PPO Baseline) ===")
        print(df_weights.mean().sort_values(ascending=False))
        print(
            f"\nMax Single Position: "
            f"{df_weights.max().max():.2%} ({df_weights.max().idxmax()})"
        )
else:
    print("Not enough data points in backtest to compute performance metrics.")

print("\n===== PPO Baseline Backtest Completed =====")
