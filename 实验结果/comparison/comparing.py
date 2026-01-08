# =========================================
# 实验对比分析系统
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
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

# =========================================
# 数据加载与预处理
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
df["Cash"] = 10000.0

df.interpolate(method="time", inplace=True)
df.dropna(inplace=True)

TECH_COLS = ["AAPL", "GOOG", "MSFT", "SP500", "Gold", "US_debt"]


# =========================================
# 特征工程函数
# =========================================
def preprocess_data(df_, tech_cols):
    """预计算所有技术特征"""
    raw_data = df_.copy().astype(np.float64)
    price_data = raw_data[tech_cols].copy()

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
        macd, _, _ = talib.MACD(arr, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_df[col + "_macd"] = np.tanh(np.nan_to_num(macd, nan=0.0))

        ma60 = talib.SMA(arr, timeperiod=60)
        bias = (arr - ma60) / ma60
        bias_df[col + "_bias60"] = np.nan_to_num(bias, nan=0.0)

        rolling_min = price_data[col].rolling(252).min()
        rolling_max = price_data[col].rolling(252).max()
        quantile = (price_data[col] - rolling_min) / (rolling_max - rolling_min)
        quantile_df[col + "_qtl252"] = np.nan_to_num(quantile, nan=0.5)

    corr_df = pd.DataFrame(index=price_data.index)
    macro_df = pd.DataFrame(index=price_data.index)
    alpha_df = pd.DataFrame(index=price_data.index)

    sp500_ret = price_data["SP500"].pct_change()
    market_vol = sp500_ret.rolling(20).std() * np.sqrt(252)
    us_debt = price_data["US_debt"]
    delta_yield = us_debt.diff()
    sp = price_data["SP500"]

    for col in tech_cols:
        corr = price_data[col].rolling(30).corr(sp)
        corr_df[col + "_corr"] = corr
        macro_df[col + "_vix"] = market_vol
        macro_df[col + "_dyield"] = delta_yield
        asset_ret = price_data[col].pct_change()
        alpha = asset_ret - sp500_ret
        alpha_df[col + "_alpha"] = np.nan_to_num(alpha, nan=0.0)

    features = pd.concat(
        [
            returns,
            volatility,
            rsi_df,
            macd_df,
            bias_df,
            quantile_df,
            corr_df,
            alpha_df,
            macro_df,
        ],
        axis=1,
    )
    features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    raw_data = (
        raw_data.replace([np.inf, -np.inf], 0.0)
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    return raw_data, features


full_raw, full_features = preprocess_data(df, TECH_COLS)


# =========================================
# 环境类定义
# =========================================
class PortfolioEnv(gym.Env):
    """支持特征开关、月度调仓的统一环境"""

    def __init__(
        self,
        tickers,
        window_size,
        start_date,
        end_date,
        raw_df_all,
        feature_df_all,
        initial_balance=10000.0,
        use_features=True,
        monthly_rebalance=False,
        reward_scale=50.0,
        temperature=0.15,
        # 奖励组件开关
        use_dip_bonus=True,  # 是否使用左侧交易奖励
        use_downside_penalty=True,  # 是否使用下行风险惩罚
        use_holding_bonus=True,  # 是否使用持仓稳定性奖励
        # 奖励系数
        dip_bonus_coef=0.05,
        downside_risk_coef=10.0,
        cost_penalty_coef=0.002,
        holding_bonus_coef=0.0005,
        seed=None,
    ):
        super().__init__()
        self.tickers = tickers
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.use_features = use_features
        self.monthly_rebalance = monthly_rebalance
        self.reward_scale = reward_scale
        self.temperature = temperature

        # 奖励组件开关
        self.use_dip_bonus = use_dip_bonus
        self.use_downside_penalty = use_downside_penalty
        self.use_holding_bonus = use_holding_bonus

        # 奖励系数
        self.dip_bonus_coef = dip_bonus_coef
        self.downside_risk_coef = downside_risk_coef
        self.cost_penalty_coef = cost_penalty_coef
        self.holding_bonus_coef = holding_bonus_coef

        self.raw_data = raw_df_all.loc[start_date:end_date]
        self.feature_data = (
            feature_df_all.loc[start_date:end_date] if use_features else None
        )

        if use_features:
            self.n_features = self.feature_data.shape[1]
            self.observation_space = gym.spaces.Box(
                low=-inf, high=inf, shape=(window_size, self.n_features)
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0.0, high=np.inf, shape=(window_size, len(tickers))
            )

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.tickers),)
        )

        self.last_action = np.ones(len(self.tickers)) / len(self.tickers)
        self.last_rebalance_month = None

    def reset(self, seed=None, options=None):
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.last_action = np.ones(len(self.tickers)) / len(self.tickers)
        self.last_rebalance_month = None

        obs = self._get_observation()
        return obs, {"balance": self.balance}

    def _get_observation(self):
        start = max(0, self.current_step - self.window_size)
        end = min(len(self.raw_data), self.current_step)

        if self.use_features:
            obs = self.feature_data.iloc[start:end].values
        else:
            obs = self.raw_data[self.tickers].iloc[start:end].values

        if obs.shape[0] < self.window_size:
            pad = np.zeros((self.window_size - obs.shape[0], obs.shape[1]))
            obs = np.vstack([pad, obs])

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def step(self, raw_action):
        raw_action = np.asarray(raw_action).ravel()
        raw_action = np.nan_to_num(raw_action, nan=0.0, posinf=1.0, neginf=-1.0)

        exp_values = np.exp(raw_action / self.temperature)
        weights = exp_values / np.sum(exp_values)
        weights[weights < 0.001] = 0.0
        action = weights / np.sum(weights)

        if self.current_step >= len(self.raw_data):
            obs = self._get_observation()
            return obs, 0.0, True, False, {"balance": self.balance, "weights": action}

        # 月度调仓逻辑
        current_date = self.raw_data.index[self.current_step]
        current_month = current_date.to_period("M")

        if self.monthly_rebalance:
            if (
                self.last_rebalance_month is None
                or current_month != self.last_rebalance_month
            ):
                # 允许调仓
                self.last_rebalance_month = current_month
                effective_action = action
            else:
                # 保持上次权重
                effective_action = self.last_action
        else:
            effective_action = action

        cur_price = self.raw_data.iloc[self.current_step].values
        prev_price = self.raw_data.iloc[self.current_step - 1].values
        asset_ret = np.nan_to_num(
            cur_price / prev_price - 1.0, nan=0.0, posinf=0.0, neginf=0.0
        )

        port_ret = float(np.sum(effective_action * asset_ret))
        turnover = float(np.sum(np.abs(effective_action - self.last_action)))

        self.balance *= 1.0 + port_ret

        # =========================================
        # 模块化奖励计算
        # =========================================

        # 1. 基础收益 (Log Return)
        log_ret = np.log1p(port_ret)

        # 2. 左侧交易奖励 (可选)
        dip_bonus = 0.0
        if self.use_dip_bonus:
            lookback = 20
            s_idx = max(0, self.current_step - lookback)
            recent_window = self.raw_data.iloc[s_idx : self.current_step + 1].values
            window_min = np.min(recent_window, axis=0)

            is_dip = cur_price <= (window_min * 1.01)

            for i, t in enumerate(self.tickers):
                if t in ["Cash", "US_debt"]:
                    is_dip[i] = False

            dip_bonus = float(np.sum(effective_action * is_dip)) * self.dip_bonus_coef

        # 3. 下行风险惩罚 (可选)
        downside_risk = 0.0
        if self.use_downside_penalty and port_ret < 0:
            downside_risk = (port_ret**2) * self.downside_risk_coef

        # 4. 交易成本惩罚
        cost_penalty = turnover * self.cost_penalty_coef

        # 5. 持仓稳定性奖励 (可选)
        holding_bonus = 0.0
        if self.use_holding_bonus:
            current_top = np.argmax(effective_action)
            last_top = np.argmax(self.last_action)
            if current_top == last_top:
                holding_bonus = self.holding_bonus_coef

        # 综合奖励
        reward = (
            log_ret + dip_bonus - downside_risk - cost_penalty + holding_bonus
        ) * self.reward_scale

        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))

        self.last_action = effective_action
        self.current_step += 1
        done = self.current_step >= len(self.raw_data) - 1

        obs = self._get_observation()
        info = {
            "balance": float(self.balance),
            "weights": effective_action,
            "turnover": float(turnover),
        }
        return obs, reward, done, False, info


# =========================================
# 回测与指标计算
# =========================================
def backtest_model(
    model, vecnorm_path, start_date, end_date, config: Dict, use_lstm=False
) -> Dict:
    """统一的回测函数"""

    test_env_raw = DummyVecEnv(
        [
            lambda: PortfolioEnv(
                tickers=df.columns.tolist(),
                window_size=30,
                start_date=start_date,
                end_date=end_date,
                raw_df_all=full_raw,
                feature_df_all=full_features,
                initial_balance=10000.0,
                **config,
            )
        ]
    )

    if vecnorm_path and os.path.exists(vecnorm_path):
        test_env = VecNormalize.load(vecnorm_path, test_env_raw)
        test_env.training = False
        test_env.norm_reward = False
    else:
        test_env = test_env_raw

    obs = test_env.reset()
    done = False
    balances = []
    weights_history = []

    base_env = test_env.envs[0]
    while hasattr(base_env, "env"):
        base_env = base_env.env

    # LSTM 状态管理
    lstm_states = None
    episode_starts = np.ones((test_env.num_envs,), dtype=bool)

    while not done:
        if use_lstm:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, dones, info = test_env.step(action)
        done = dones[0]

        if use_lstm:
            episode_starts = dones

        balances.append(info[0]["balance"])
        weights_history.append(info[0]["weights"])

    # 计算指标
    balances_arr = np.array(balances)
    final_balance = balances_arr[-1]
    initial_balance_val = balances_arr[0]

    num_days = len(balances)
    num_years = num_days / 252

    total_growth = final_balance / initial_balance_val
    cagr = (total_growth) ** (1 / num_years) - 1 if num_years > 0 else 0.0

    # 日收益率序列
    balance_series = pd.Series(balances_arr)
    daily_returns = balance_series.pct_change().dropna()

    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = cagr / ann_vol if ann_vol != 0 else 0.0

    downside_returns = daily_returns[daily_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = cagr / downside_vol if downside_vol != 0 else 0.0

    cumulative_max = np.maximum.accumulate(balances_arr)
    drawdowns = (balances_arr - cumulative_max) / cumulative_max
    max_drawdown = np.min(drawdowns)
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    test_env.close()

    return {
        "final_balance": final_balance,
        "total_growth": total_growth,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "balances": balances_arr,
        "weights_history": weights_history,
    }


# =========================================
# 训练函数
# =========================================
def train_model(
    config_name: str,
    config: Dict,
    algo: str = "PPO",
    use_lstm: bool = False,
    total_timesteps: int = 50000,
):
    """训练模型并返回路径"""

    print(f"\n{'='*60}")
    print(f"Training: {config_name}")
    print(f"{'='*60}")

    tickers = df.columns.tolist()

    # 数据切分
    full_range = df.index
    split_idx = int(len(full_range) * 0.6)
    train_start_date = full_range[0]
    train_end_date = full_range[split_idx]

    # 创建训练环境
    def make_env():
        return PortfolioEnv(
            tickers=tickers,
            window_size=30,
            start_date=train_start_date,
            end_date=train_end_date,
            raw_df_all=full_raw,
            feature_df_all=full_features,
            initial_balance=10000.0,
            **config,
        )

    n_envs = 16

    vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 选择算法
    if algo == "PPO":
        if use_lstm:
            policy_kwargs = dict(
                lstm_hidden_size=128,
                n_lstm_layers=1,
                net_arch=[dict(pi=[64, 64], vf=[64, 64])],
            )
            model = RecurrentPPO(
                "MlpLstmPolicy",
                vec_env,
                device=device,
                verbose=1,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,
                batch_size=128,
                n_steps=512,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                max_grad_norm=0.5,
            )
        else:
            model = PPO(
                "MlpPolicy",
                vec_env,
                device=device,
                verbose=1,
                learning_rate=3e-4,
                batch_size=256,
                n_steps=1024,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                max_grad_norm=0.5,
            )
    elif algo == "SAC":
        model = SAC(
            "MlpPolicy",
            vec_env,
            device=device,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=50000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # 训练
    model.learn(total_timesteps=total_timesteps)

    # 保存
    save_dir = f"./comparison_results/{config_name}"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "model.zip")
    vecnorm_path = os.path.join(save_dir, "vecnorm.pkl")

    model.save(model_path)
    vec_env.save(vecnorm_path)

    vec_env.close()

    return model_path, vecnorm_path


# =========================================
# 主实验流程
# =========================================
def main():
    print("=" * 60)
    print("Systematic Experiment Comparison - 32 Configurations")
    print("=" * 60)

    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"GPU Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # 数据切分
    full_range = df.index
    split_idx = int(len(full_range) * 0.6)
    test_start_date = full_range[split_idx + 1]
    test_end_date = full_range[-1]

    # 定义5个维度的配置选项
    algorithms = ["PPO", "SAC"]
    use_lstm_options = [False, True]
    use_features_options = [False, True]
    use_full_reward_options = [False, True]
    monthly_rebalance_options = [False, True]

    # 生成所有32种组合
    experiments = {}
    exp_id = 0

    for algo in algorithms:
        for use_lstm in use_lstm_options:
            # SAC 不支持 LSTM，跳过
            if algo == "SAC" and use_lstm:
                continue

            for use_features in use_features_options:
                for use_full_reward in use_full_reward_options:
                    for monthly_rebalance in monthly_rebalance_options:
                        exp_id += 1

                        # 构建实验名称
                        name_parts = [
                            algo,
                            "LSTM" if use_lstm else "MLP",
                            "WithFeature" if use_features else "NoFeature",
                            "FullReward" if use_full_reward else "BaseReward",
                            "Monthly" if monthly_rebalance else "Free",
                        ]
                        exp_name = "_".join(name_parts)

                        # 构建配置
                        config = {
                            "use_features": use_features,
                            "monthly_rebalance": monthly_rebalance,
                        }

                        # 根据是否使用完整奖励设置奖励组件
                        if use_full_reward:
                            config.update(
                                {
                                    "use_dip_bonus": True,
                                    "use_downside_penalty": True,
                                    "use_holding_bonus": True,
                                }
                            )
                        else:
                            config.update(
                                {
                                    "use_dip_bonus": False,
                                    "use_downside_penalty": False,
                                    "use_holding_bonus": False,
                                }
                            )

                        experiments[exp_name] = {
                            "config": config,
                            "algo": algo,
                            "use_lstm": use_lstm,
                        }

    print(f"\nTotal {len(experiments)} experimental configurations generated")
    print("\nExperiment List Preview (First 10):")
    for i, name in enumerate(list(experiments.keys())[:10]):
        print(f"  {i+1}. {name}")
    print("  ...")

    results = {}

    # 运行所有实验
    for idx, (exp_name, exp_config) in enumerate(experiments.items(), 1):
        print(f"\n{'='*80}")
        print(f"Progress: [{idx}/{len(experiments)}] {exp_name}")
        print(f"{'='*80}")

        try:
            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 训练（调整训练步数）
            total_steps = 50000 if exp_config["use_lstm"] else 100000

            model_path, vecnorm_path = train_model(
                exp_name,
                exp_config["config"],
                exp_config["algo"],
                exp_config["use_lstm"],
                total_timesteps=total_steps,
            )

            # 加载模型
            if exp_config["use_lstm"]:
                model = RecurrentPPO.load(model_path, device=device)
            elif exp_config["algo"] == "PPO":
                model = PPO.load(model_path, device=device)
            else:
                model = SAC.load(model_path, device=device)

            # 回测
            metrics = backtest_model(
                model,
                vecnorm_path,
                test_start_date,
                test_end_date,
                exp_config["config"],
                exp_config["use_lstm"],
            )

            results[exp_name] = metrics

            print(f"\n{exp_name} Results:")
            print(f"  Sharpe: {metrics['sharpe']:.4f}")
            print(f"  CAGR: {metrics['cagr']:.2%}")
            print(f"  Max DD: {metrics['max_drawdown']:.2%}")

            # 手动删除模型释放内存
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Experiment {exp_name} failed: {str(e)}")
            import traceback

            traceback.print_exc()

            # 清理内存后继续
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    # =========================================
    # 结果汇总与可视化
    # =========================================

    # 1. 创建对比表格
    comparison_df = pd.DataFrame(
        {
            name: {
                "Algorithm": name.split("_")[0],
                "Model": name.split("_")[1],
                "Features": "Yes" if "WithFeature" in name else "No",
                "Reward": "Full" if "FullReward" in name else "Base",
                "Rebalance": "Monthly" if "Monthly" in name else "Free",
                "Sharpe Ratio": metrics["sharpe"],
                "CAGR": metrics["cagr"],
                "Annual Volatility": metrics["ann_vol"],
                "Sortino Ratio": metrics["sortino"],
                "Max Drawdown": metrics["max_drawdown"],
                "Calmar Ratio": metrics["calmar"],
                "Final Balance": metrics["final_balance"],
            }
            for name, metrics in results.items()
        }
    ).T

    print("\n" + "=" * 80)
    print("Experiment Results Summary (32 Configurations)")
    print("=" * 80)
    print(comparison_df.to_string())

    # 保存结果
    os.makedirs("comparison_results", exist_ok=True)
    comparison_df.to_csv("comparison_results/summary_32_configs.csv")

    # 2. 按维度分组分析
    print("\n" + "=" * 80)
    print("Dimensional Impact Analysis")
    print("=" * 80)

    # 算法影响
    print("\n[Algorithm Impact]")
    algo_analysis = comparison_df.groupby("Algorithm")[
        ["Sharpe Ratio", "CAGR", "Max Drawdown"]
    ].mean()
    print(algo_analysis)

    # 模型影响（仅PPO）
    print("\n[LSTM Impact (PPO Only)]")
    ppo_df = comparison_df[comparison_df["Algorithm"] == "PPO"]
    lstm_analysis = ppo_df.groupby("Model")[
        ["Sharpe Ratio", "CAGR", "Max Drawdown"]
    ].mean()
    print(lstm_analysis)

    # 特征影响
    print("\n[Feature Engineering Impact]")
    feature_analysis = comparison_df.groupby("Features")[
        ["Sharpe Ratio", "CAGR", "Max Drawdown"]
    ].mean()
    print(feature_analysis)

    # 奖励函数影响
    print("\n[Reward Function Impact]")
    reward_analysis = comparison_df.groupby("Reward")[
        ["Sharpe Ratio", "CAGR", "Max Drawdown"]
    ].mean()
    print(reward_analysis)

    # 调仓频率影响
    print("\n[Rebalancing Frequency Impact]")
    rebalance_analysis = comparison_df.groupby("Rebalance")[
        ["Sharpe Ratio", "CAGR", "Max Drawdown"]
    ].mean()
    print(rebalance_analysis)

    # 3. 可视化:夏普比率对比(Top 15)
    plt.figure(figsize=(16, 8))
    sharpe_sorted = (
        comparison_df["Sharpe Ratio"].astype(float).sort_values(ascending=False)[:15]
    )
    colors = ["#2ecc71" if x > 0 else "#e74c3c" for x in sharpe_sorted]

    plt.bar(range(len(sharpe_sorted)), sharpe_sorted.values, color=colors, alpha=0.7)
    plt.xticks(
        range(len(sharpe_sorted)),
        sharpe_sorted.index,
        rotation=45,
        ha="right",
        fontsize=8,
    )
    plt.ylabel("Sharpe Ratio", fontsize=12)
    plt.title("Top 15 Configurations by Sharpe Ratio (Out of 32)", fontsize=14, pad=20)
    plt.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison_results/sharpe_top15.png", dpi=300)
    plt.show()

    # 4. 热力图:维度组合效果
    plt.figure(figsize=(14, 10))

    # 确保数值列是浮点型
    numeric_cols = ["Sharpe Ratio", "CAGR", "Max Drawdown"]
    for col in numeric_cols:
        comparison_df[col] = pd.to_numeric(comparison_df[col], errors="coerce")

    # 创建简化的热力图:算法 + 模型 vs 其他维度
    try:
        pivot_table = comparison_df.pivot_table(
            values="Sharpe Ratio",
            index=["Algorithm", "Model"],
            columns=["Features", "Reward", "Rebalance"],
            aggfunc="mean",
        )

        # 确保 pivot table 是数值型
        pivot_table = pivot_table.astype(float)

        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0,
            linewidths=0.5,
            cbar_kws={"label": "Sharpe Ratio"},
        )
        plt.title("Configuration Heatmap: Sharpe Ratio", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig("comparison_results/config_heatmap.png", dpi=300)
        plt.show()
    except Exception as e:
        print(f"Heatmap generation failed: {e}")
        print("Skipping heatmap visualization...")

        # 备选:简化的对比图
        plt.figure(figsize=(12, 6))

        # 绘制各维度的平均 Sharpe Ratio
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Dimensional Analysis: Average Sharpe Ratio", fontsize=16, y=1.02)

        # 算法
        algo_analysis.plot(
            kind="bar", y="Sharpe Ratio", ax=axes[0, 0], color="skyblue", legend=False
        )
        axes[0, 0].set_title("Algorithm Impact")
        axes[0, 0].set_ylabel("Sharpe Ratio")
        axes[0, 0].tick_params(axis="x", rotation=0)

        # LSTM (仅PPO)
        lstm_analysis.plot(
            kind="bar",
            y="Sharpe Ratio",
            ax=axes[0, 1],
            color="lightcoral",
            legend=False,
        )
        axes[0, 1].set_title("LSTM Impact (PPO Only)")
        axes[0, 1].set_ylabel("Sharpe Ratio")
        axes[0, 1].tick_params(axis="x", rotation=0)

        # 特征
        feature_analysis.plot(
            kind="bar",
            y="Sharpe Ratio",
            ax=axes[0, 2],
            color="lightgreen",
            legend=False,
        )
        axes[0, 2].set_title("Feature Engineering Impact")
        axes[0, 2].set_ylabel("Sharpe Ratio")
        axes[0, 2].tick_params(axis="x", rotation=0)

        # 奖励
        reward_analysis.plot(
            kind="bar", y="Sharpe Ratio", ax=axes[1, 0], color="gold", legend=False
        )
        axes[1, 0].set_title("Reward Function Impact")
        axes[1, 0].set_ylabel("Sharpe Ratio")
        axes[1, 0].tick_params(axis="x", rotation=0)

        # 调仓
        rebalance_analysis.plot(
            kind="bar", y="Sharpe Ratio", ax=axes[1, 1], color="plum", legend=False
        )
        axes[1, 1].set_title("Rebalancing Frequency Impact")
        axes[1, 1].set_ylabel("Sharpe Ratio")
        axes[1, 1].tick_params(axis="x", rotation=0)

        # 综合对比 (Top 10)
        top10_sharpe = (
            comparison_df["Sharpe Ratio"]
            .astype(float)
            .sort_values(ascending=False)[:10]
        )
        top10_sharpe.plot(kind="barh", ax=axes[1, 2], color="steelblue")
        axes[1, 2].set_title("Top 10 Configurations")
        axes[1, 2].set_xlabel("Sharpe Ratio")
        axes[1, 2].tick_params(axis="y", labelsize=8)

        plt.tight_layout()
        plt.savefig("comparison_results/dimensional_analysis.png", dpi=300)
        plt.show()

    # 5. 净值曲线对比(Top 5)
    plt.figure(figsize=(14, 8))
    top5_names = sharpe_sorted.index[:5]

    for exp_name in top5_names:
        if exp_name in results:
            balances = results[exp_name]["balances"]
            normalized = balances / balances[0]
            plt.plot(normalized, label=exp_name, linewidth=2, alpha=0.8)

    plt.xlabel("Trading Days", fontsize=12)
    plt.ylabel("Normalized Portfolio Value", fontsize=12)
    plt.title("Top 5 Strategies: Equity Curve Comparison", fontsize=14, pad=20)
    plt.legend(fontsize=9, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison_results/equity_curves_top5.png", dpi=300)
    plt.show()

    # 6. 生成最佳配置推荐
    print("\n" + "=" * 80)
    print("Best Configuration Recommendations")
    print("=" * 80)

    best_sharpe = comparison_df["Sharpe Ratio"].astype(float).idxmax()
    best_cagr = comparison_df["CAGR"].astype(float).idxmax()
    best_risk_adjusted = comparison_df["Sortino Ratio"].astype(float).idxmax()

    print(f"\nHighest Sharpe Ratio: {best_sharpe}")
    print(f"  Sharpe: {comparison_df.loc[best_sharpe, 'Sharpe Ratio']:.4f}")
    print(f"  CAGR: {comparison_df.loc[best_sharpe, 'CAGR']:.2%}")

    print(f"\nHighest CAGR: {best_cagr}")
    print(f"  CAGR: {comparison_df.loc[best_cagr, 'CAGR']:.2%}")
    print(f"  Sharpe: {comparison_df.loc[best_cagr, 'Sharpe Ratio']:.4f}")

    print(f"\nHighest Sortino Ratio: {best_risk_adjusted}")
    print(f"  Sortino: {comparison_df.loc[best_risk_adjusted, 'Sortino Ratio']:.4f}")
    print(f"  Sharpe: {comparison_df.loc[best_risk_adjusted, 'Sharpe Ratio']:.4f}")

    print("\nAll results saved to comparison_results/ directory")
    print(f"Completed {len(results)}/{len(experiments)} experiments")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
