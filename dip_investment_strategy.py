import pandas as pd
import numpy as np
import calendar
import matplotlib.pyplot as plt
import backtrader as bt
import yfinance as yf
import collections


def cal_max_drawdown(nav):
    nav = np.asarray(nav, dtype=np.float64)
    peak = np.maximum.accumulate(nav)

    # 只在 peak 非零时计算 drawdown，否则设为 0
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 0, (peak - nav) / peak, 0.0)

    return np.max(drawdown)


class GridStrategy(bt.Strategy):
    params = (
        ("symbol", "SH512890"),
        ("base_amount", 10000),  # 每个网格的投资金额
        ("grid_size", 0.02),     # 网格大小（价格间距）
        ("grid_number", 10),     # 网格数量
        ("price_range", None),   # 价格区间 (min_price, max_price)
    )

    def __init__(self):
        self.grid_prices = []  # 存储网格价格
        self.grid_positions = {}  # 存储每个网格的持仓
        self.last_price = None  # 上一次的价格
        
        # 如果没有指定价格区间，使用历史数据的最高最低价
        if self.p.price_range is None:
            self.p.price_range = (
                min(self.data0.low.get(size=100)),
                max(self.data0.high.get(size=100))
            )
        
        # 计算网格价格
        min_price, max_price = self.p.price_range
        price_step = (max_price - min_price) / self.p.grid_number
        self.grid_prices = [min_price + i * price_step for i in range(self.p.grid_number + 1)]
        
        # 初始化每个网格的持仓
        for price in self.grid_prices:
            self.grid_positions[price] = 0
            
        print(f"网格价格: {self.grid_prices}")

    def get_grid_index(self, price):
        """获取价格所在的网格索引"""
        for i in range(len(self.grid_prices) - 1):
            if self.grid_prices[i] <= price <= self.grid_prices[i + 1]:
                return i
        return -1

    def next(self):
        current_price = self.data0.close[0]
        current_date = self.data0.datetime.date(0)
        
        if self.last_price is None:
            self.last_price = current_price
            return
            
        # 获取当前价格所在的网格
        current_grid = self.get_grid_index(current_price)
        last_grid = self.get_grid_index(self.last_price)
        
        if current_grid == -1 or last_grid == -1:
            self.last_price = current_price
            return
            
        # 价格下跌，买入
        if current_grid > last_grid:
            print(f"买入信号 - 日期: {current_date}, 价格: {current_price:.2f}, 网格: {current_grid}")
            self.buy(data=self.data0, value=self.p.base_amount)
            self.grid_positions[self.grid_prices[current_grid]] += 1
            
        # 价格上涨，卖出
        elif current_grid < last_grid:
            print(f"卖出信号 - 日期: {current_date}, 价格: {current_price:.2f}, 网格: {current_grid}")
            self.sell(data=self.data0, value=self.p.base_amount)
            self.grid_positions[self.grid_prices[current_grid]] -= 1
            
        self.last_price = current_price


def backtest_strategy(
    symbol="512890.SS",
    start_date="2021-01-01",
    end_date=None,
    base_amount=10000,
    grid_size=0.02,
    grid_number=10,
    price_range=None,
):
    """
    回测策略
    """
    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    # 获取历史数据
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    if data.empty:
        print(f"No data available for {symbol}")
        return None

    print(f"数据范围: {data.index[0]} 到 {data.index[-1]}")
    print(f"数据条数: {len(data)}")
    print(f"数据样例:\n{data.head()}")
    
    # 创建回测
    cerebro = bt.Cerebro()

    # 添加数据
    data_feed = bt.feeds.PandasData(
        dataname=data,
        datetime=None,  # 使用索引作为日期
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        openinterest=-1,
    )
    cerebro.adddata(data_feed)

    # 添加策略
    cerebro.addstrategy(
        GridStrategy,
        symbol=symbol,
        base_amount=base_amount,
        grid_size=grid_size,
        grid_number=grid_number,
        price_range=price_range,
    )

    # 设置初始资金
    initial_cash = 10000000
    cerebro.broker.setcash(initial_cash)

    # 设置手续费
    cerebro.broker.setcommission(commission=0.0003)  # 0.03% 手续费

    # 运行回测
    print("初始资金: %.2f" % cerebro.broker.getvalue())
    cerebro.run()
    print("最终资金: %.2f" % cerebro.broker.getvalue())

    # 绘制结果
    cerebro.plot(style="candlestick")
    plt.savefig("backtest_result.png")
    plt.close()

    # 获取回测结果
    portfolio_value = cerebro.broker.getvalue()
    trades = cerebro.broker.get_orders()

    # 计算最大回撤
    max_drawdown = cal_max_drawdown(data["Close"])

    # 计算年化收益率
    total_return = (portfolio_value - initial_cash) / initial_cash
    years = (data.index[-1] - data.index[0]).days / 365
    annual_return = (1 + total_return) ** (1 / years) - 1

    # 计算卡玛比率
    kama_ratio = annual_return / abs(max_drawdown)

    # 打印回测结果
    print("\n=== 回测结果 ===")
    print(f"回测期间: {start_date} 至 {end_date}")
    print(f"最终资金: {portfolio_value:,.2f}")
    print(f"年化收益率: {annual_return*100:.2f}%")
    print(f"最大回撤: {max_drawdown*100:.2f}%")
    print(f"卡玛比率: {kama_ratio:.2f}")
    print(f"总交易次数: {len(trades)}")

    # 计算benchmark
    benchmark_shares = initial_cash / data["Close"].iloc[0]
    benchmark_value = benchmark_shares * data["Close"].iloc[-1]
    benchmark_return = (benchmark_value - initial_cash) / initial_cash
    benchmark_annual_return = (1 + benchmark_return) ** (1 / years) - 1

    print("\n== benchmark ==")
    print(f"benchmark 持仓数量: {benchmark_shares:.2f}")
    print(f"benchmark 最终资金: {benchmark_value:,.2f}")
    print(f"benchmark 年化收益率: {benchmark_annual_return*100:.2f}%")

    return {
        "portfolio_value": portfolio_value,
        "trades": trades,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "kama_ratio": kama_ratio,
    }


def main():
    # 执行回测
    backtest_strategy(start_date="2021-01-01")


if __name__ == "__main__":
    main()
