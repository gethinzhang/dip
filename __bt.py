from datafeed.dataloader import CSVDataloader
import numpy as np
import pandas as pd
import bt

instruments = [
    "159934.SZ",  # 黄金ETF（黄金）
    "511260.SH",  # 十年国债ETF（债券）
    "512890.SH",  # 红利低波（股票）
    "159915.SZ",  # 创业板（股票）
    "159985.SZ",  # 豆粕（商品）
    "513100.SH",  # 纳指100
]
data = CSVDataloader.get_df(instruments, set_index=True)
data = CSVDataloader.calc_expr(data, ["roc(close,20)"], ["roc_20"])
print(data)


def stra1(ticker):
    strat = bt.Strategy(
        ticker,
        [
            bt.algos.SelectThese(tickers=[ticker]),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
    )

    t = bt.Backtest(strat, df_close, integer_positions=False, progress_bar=False)
    return t


def stra2():
    tickers = [
        "159915.SZ",  # 创业板（股票）
    ]
    df_roc = CSVDataloader.get_col_df(data, "roc_20")
    df_roc = df_roc[tickers]
    signal = np.where(
        df_roc > 0.08, 1, np.nan
    )  # 这里不能置成0因为有买卖信号之间的没有大于0.08也没有小于0
    signal = np.where(df_roc < 0, 0, signal)
    signal = pd.DataFrame(signal, index=df_roc.index, columns=df_roc.columns)
    signal = signal.ffill()  # 这一步很关键在1后面的会前向填充，即持仓不变。
    signal = signal.fillna(0)

    print(signal, type(signal))

    strat = bt.Strategy(
        "创业板动量",
        [
            # bt.algos.SelectThese(tickers=tickers),
            bt.algos.SelectThese(tickers=tickers),
            bt.algos.SelectWhere(signal),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
    )

    t = bt.Backtest(strat, df_close, integer_positions=False, progress_bar=False)
    return t


df_close = CSVDataloader.get_col_df(data)
print(df_close)
tests = [stra2()]
for s in [
    "159934.SZ",  # 黄金ETF（黄金）
    "511260.SH",  # 十年国债ETF（债券）
    "512890.SH",  # 红利低波（股票）
    "159985.SZ",  # 豆粕（商品）
    "513100.SH",  # 纳指100
]:
    tests.append(stra1(s))
    # pass

combined_strategy = bt.Strategy(
    "策略组合",
    algos=[
        bt.algos.RunMonthly(),
        bt.algos.SelectAll(),
        bt.algos.WeighEqually(),
        bt.algos.Rebalance(),
    ],
    children=[x.strategy for x in tests],
)

combined_test = bt.Backtest(
    combined_strategy, df_close, integer_positions=False, progress_bar=False
)

res = bt.run(combined_test, stra1("159915.SZ"))
print(res.stats)

import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams["font.family"] = "SimHei"
res.prices.plot()
plt.show()
