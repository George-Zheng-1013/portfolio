import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# 数据加载（与 PPO_lstm 保持一致）
# =========================================
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

# 插值 & 填补缺失
df.interpolate(method="time", inplace=True)
df.dropna(inplace=True)

# =========================================
# 训练/测试集分割（与 PPO_lstm 保持一致）
# =========================================
full_range = df.index
split_idx = int(len(full_range) * 0.6)
train_start_date = full_range[0]
train_end_date = full_range[split_idx]
test_start_date = full_range[split_idx + 1]
test_end_date = full_range[-1]

print("数据时间范围:")
print(f"训练集: {train_start_date.date()} -> {train_end_date.date()}")
print(f"测试集: {test_start_date.date()} -> {test_end_date.date()}")


# =========================================
# 等权重策略回测
# =========================================
def equal_weight_backtest(df, start_date, end_date, initial_balance=10000.0):
    """
    等权重策略：投资科技股（AAPL, GOOG, MSFT）+ 黄金（Gold）+ 标普500（SP500），每只资产分配 1/5 权重
    不包含 US_debt 和 Cash
    """
    # 筛选资产（科技股 + 黄金 + 标普500）
    tech_stocks = ["AAPL", "GOOG", "MSFT", "Gold", "SP500"]

    # 筛选时间范围
    backtest_data = df.loc[start_date:end_date, tech_stocks].copy()

    # 等权重分配
    n_assets = len(tech_stocks)
    weights = np.ones(n_assets) / n_assets

    # 计算每日收益率
    daily_returns = backtest_data.pct_change().fillna(0.0)

    # 计算组合收益率
    portfolio_returns = (daily_returns * weights).sum(axis=1)

    # 计算累计净值
    cumulative_value = (1 + portfolio_returns).cumprod() * initial_balance

    return {
        "dates": backtest_data.index,
        "values": cumulative_value.values,
        "returns": portfolio_returns.values,
        "weights": weights,
        "assets": tech_stocks,
    }


# =========================================
# 性能指标计算
# =========================================
def calculate_metrics(values, returns, start_date, end_date, initial_balance):
    """计算策略性能指标"""
    total_growth = values[-1] / initial_balance
    duration_days = (end_date - start_date).days
    years = duration_days / 365.25 if duration_days > 0 else 0

    # CAGR
    cagr = total_growth ** (1 / years) - 1 if years > 0 else 0.0

    # 波动率
    ann_vol = np.std(returns) * np.sqrt(252)

    # Sharpe Ratio
    sharpe = cagr / ann_vol if ann_vol != 0 else 0.0

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_vol = (
        np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.0
    )
    sortino = cagr / downside_vol if downside_vol != 0 else 0.0

    # Max Drawdown
    cumulative_max = np.maximum.accumulate(values)
    drawdowns = (values - cumulative_max) / cumulative_max
    max_drawdown = np.min(drawdowns)

    # Calmar Ratio
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "final_balance": values[-1],
        "total_growth": total_growth,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


# =========================================
# 执行回测
# =========================================
print("\n" + "=" * 50)
print(
    "Equal Weight Strategy Backtest (Tech Stocks + Gold + SP500: AAPL, GOOG, MSFT, Gold, SP500)"
)
print("=" * 50)

# 测试集回测
initial_balance = 10000.0
result = equal_weight_backtest(df, test_start_date, test_end_date, initial_balance)

# 计算性能指标
metrics = calculate_metrics(
    result["values"],
    result["returns"],
    test_start_date,
    test_end_date,
    initial_balance,
)

# 打印结果
print(f"\nStrategy Configuration:")
print(f"  Assets: {', '.join(result['assets'])}")
print(f"  Weights: {result['weights']}")
print(f"\nPerformance Metrics ({test_start_date.date()} to {test_end_date.date()})")
print("-" * 40)
print(f"Initial Balance:     {initial_balance:.2f}")
print(f"Final Balance:       {metrics['final_balance']:.2f}")
print(f"Total Growth:        {metrics['total_growth']:.4f}x")
print(f"CAGR (Ann. Return):  {metrics['cagr']:.2%}")
print(f"Ann. Volatility:     {metrics['ann_vol']:.2%}")
print(f"Sharpe Ratio:        {metrics['sharpe']:.4f}")
print(f"Sortino Ratio:       {metrics['sortino']:.4f}")
print(f"Max Drawdown:        {metrics['max_drawdown']:.2%}")
print(f"Calmar Ratio:        {metrics['calmar']:.4f}")
print("-" * 40)

# =========================================
# 可视化
# =========================================
sns.set_theme(style="whitegrid")

# 1. 净值曲线对比
plt.figure(figsize=(14, 7))

# 等权重策略
norm_eq_weight = result["values"] / initial_balance
plt.plot(
    result["dates"],
    norm_eq_weight,
    label="Equal Weight (AAPL+GOOG+MSFT+Gold+SP500)",
    linewidth=2.5,
    color="blue",
)

# 单个资产作为参考（除了SP500，因为它已经在组合中）
for asset in ["AAPL", "GOOG", "MSFT", "Gold"]:
    asset_prices = df.loc[test_start_date:test_end_date, asset]
    norm_asset = asset_prices / asset_prices.iloc[0]
    plt.plot(
        asset_prices.index,
        norm_asset,
        label=asset,
        alpha=0.5,
        linestyle="--",
    )

sp500_prices = df.loc[test_start_date:test_end_date, "SP500"]
norm_sp500 = sp500_prices / sp500_prices.iloc[0]
plt.plot(
    sp500_prices.index,
    norm_sp500,
    label="S&P 500",
    alpha=0.6,
    linestyle="-.",
    color="gray",
)

plt.title(
    f"Equal Weight Strategy vs Individual Assets\n{test_start_date.date()} to {test_end_date.date()}",
    fontsize=14,
    pad=15,
)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Normalized Value", fontsize=12)
plt.legend(loc="best", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("equalweight_backtest.png", dpi=300, bbox_inches="tight")
plt.show()

# 2. 权重分布图
plt.figure(figsize=(10, 6))
colors = plt.cm.Set3(np.linspace(0, 1, len(result["assets"])))
plt.pie(
    result["weights"],
    labels=result["assets"],
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
)
plt.title("Equal Weight Strategy - Asset Allocation", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("equalweight_allocation.png", dpi=300, bbox_inches="tight")
plt.show()

# 3. 回撤分析
plt.figure(figsize=(14, 6))
cumulative_max = np.maximum.accumulate(result["values"])
drawdowns = (result["values"] - cumulative_max) / cumulative_max
plt.fill_between(
    result["dates"],
    drawdowns * 100,
    0,
    alpha=0.3,
    color="red",
    label="Drawdown",
)
plt.plot(result["dates"], drawdowns * 100, color="darkred", linewidth=1)
plt.title("Equal Weight Strategy - Drawdown Analysis", fontsize=14, pad=15)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Drawdown (%)", fontsize=12)
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("equalweight_drawdown.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================
# 保存结果到CSV
# =========================================
result_df = pd.DataFrame(
    {
        "date": result["dates"],
        "portfolio_value": result["values"],
        "daily_return": result["returns"],
    }
)
result_df.to_csv("equalweight_results.csv", index=False)
print(f"\n回测结果已保存至: equalweight_results.csv")

print("\nEqual Weight Strategy Backtest Completed!")
