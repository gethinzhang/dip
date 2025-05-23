import pandas as pd
import numpy as np
import calendar
import matplotlib.pyplot as plt
import backtrader as bt
import yfinance as yf


def cal_max_drawdown(nav):
    nav = np.asarray(nav, dtype=np.float64)
    peak = np.maximum.accumulate(nav)

    # 只在 peak 非零时计算 drawdown，否则设为 0
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 0, (peak - nav) / peak, 0.0)

    return np.max(drawdown)


class ChannelIndicator(bt.Indicator):
    """
    双轨通道指标
    """
    lines = ('upper', 'middle', 'lower',)
    params = (('window', 20),)

    def __init__(self):
        super(ChannelIndicator, self).__init__()
        # 计算价格序列的斜率
        x = np.arange(self.p.window)
        y = self.data.close.get(size=self.p.window)
        slope, intercept = np.polyfit(x, y, 1)

        # 计算价格到趋势线的距离
        trend_line = slope * x + intercept
        distances = y - trend_line

        # 计算上下轨的距离（取最大距离的80%）
        upper_distance = np.percentile(distances, 80)
        lower_distance = np.percentile(distances, 20)

        # 生成通道线
        self.lines.upper = trend_line + upper_distance
        self.lines.middle = trend_line
        self.lines.lower = trend_line + lower_distance


class DipStrategy(bt.Strategy):
    params = (
        ("symbol", "SH512890"),
        ("base_amount", 10000),
        ("dip_threshold", -0.01),
        ("upper_sell_threshold", 0.95),
    )

    def __init__(self):
        self.monthly_trigger_conditions = []
        self.monthly_investments = 0
        self.channel = ChannelIndicator(self.data0)
        self.data = self.get_data(self.data0)
        
    def get_data(self, data):
        # 将 backtrader 数据转换为 pandas DataFrame
        df = pd.DataFrame(
            {
                "close": data.close.array,
                "open": data.open.array,
                "high": data.high.array,
                "low": data.low.array,
                "volume": data.volume.array,
            },
            index=data.datetime.array,
        )
        return df

    def is_last_day_of_third_week(self, date):
        """
        判断是否是当月第三周的最后一天
        """
        year = date.year
        month = date.month
        cal = calendar.monthcalendar(year, month)
        
        # 获取第三周
        third_week = cal[2]
        
        # 找到第三周最后一个非零日期
        last_day = 0
        for day in reversed(third_week):
            if day != 0:
                last_day = day
                break
        
        is_last = date.day == last_day
        print(f"检查第三周最后一天 - 日期: {date}, 是否最后一天: {is_last}, 最后一天日期: {last_day}")
        return is_last

    def get_monthly_position(self, data, middle_line):
        """
        计算月中位置相对于中轨的情况
        """
        # 将 backtrader 数据转换为 pandas DataFrame
        df = self.get_data(data)
        days_in_month = len(df)
        days_below_middle = sum(df["close"] < middle_line)
        return days_below_middle / days_in_month

    def next(self):
        # 获取当前数据
        current_date = self.data0.datetime.date(0)
        current_price = self.data0.close[0]
        
        # 获取通道值
        current_upper = self.channel.upper[0]
        current_lower = self.channel.lower[0]
        current_middle = self.channel.middle[0]
        
        # 检查是否需要卖出
        if current_price >= current_upper * self.p.upper_sell_threshold:
            print(f"卖出信号 - 日期: {current_date}, 价格: {current_price}, 上轨: {current_upper}")
            # 卖出50%的持仓
            self.order_target_percent(data=self.data0, target=0.5)
            return

        # 计算当日涨跌幅
        if len(self.data0) > 1:
            daily_return = (current_price - self.data0.close[-1]) / self.data0.close[-1]
        else:
            daily_return = 0

        # 判断是否触发投资条件
        if self.is_last_day_of_third_week(current_date):
            print(f"第三周最后一天 - 日期: {current_date}")
            # 重置月度状态
            self.monthly_trigger_conditions = []
            self.monthly_investments = 0

        # 条件1：触发下跌，马上买入
        if (
            daily_return <= self.p.dip_threshold
            and 1 not in self.monthly_trigger_conditions
        ):
            print(f"条件1触发 - 日期: {current_date}, 日收益率: {daily_return:.2%}")
            self.buy(data=self.data0, value=self.p.base_amount)
            self.monthly_trigger_conditions.append(1)
            self.monthly_investments += 1

        # 条件2：第三周最后一天，如果本月没有触发条件1，则买入
        elif (
            self.is_last_day_of_third_week(current_date)
            and 1 not in self.monthly_trigger_conditions
        ):
            print(f"条件2触发 - 日期: {current_date}")
            self.buy(data=self.data0, value=self.p.base_amount)
            self.monthly_trigger_conditions.append(2)
            self.monthly_investments += 1

        # 条件3：月中位置判断，如果本月投资次数小于3次，则增加买入
        elif (
            self.is_last_day_of_third_week(current_date)
            and self.get_monthly_position(self.data0, current_middle) >= 0.5
            and self.monthly_investments < 3
        ):
            monthly_pos = self.get_monthly_position(self.data0, current_middle)
            print(f"条件3触发 - 日期: {current_date}, 月中位置: {monthly_pos:.2%}")
            self.buy(data=self.data0, value=self.p.base_amount)
            self.monthly_trigger_conditions.append(3)
            self.monthly_investments += 1

        # 条件4：触达下轨，如果本月投资次数小于3次，则增加买入
        elif (
            current_price <= current_lower
            and self.monthly_investments < 3
            and 4 not in self.monthly_trigger_conditions
        ):
            print(f"条件4触发 - 日期: {current_date}, 价格: {current_price}, 下轨: {current_lower}")
            self.buy(data=self.data0, value=self.p.base_amount)
            self.monthly_trigger_conditions.append(4)
            self.monthly_investments += 1


def backtest_strategy(
    symbol="512890.SS",
    start_date="2021-01-01",
    end_date=None,
    base_amount=10000,
    dip_threshold=-0.01,
    upper_sell_threshold=0.95,
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
        DipStrategy,
        symbol=symbol,
        base_amount=base_amount,
        dip_threshold=dip_threshold,
        upper_sell_threshold=upper_sell_threshold,
    )

    # 设置初始资金
    initial_cash = 10000000
    cerebro.broker.setcash(initial_cash)

    # 设置手续费
    # cerebro.broker.setcommission(commission=0.0003)  # 0.03% 手续费

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
