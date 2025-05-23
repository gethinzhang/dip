import yfinance as yf
import pandas as pd

# 下载 ETF 512890 的数据
ticker = '512890.SS'  # '.SS' 表示上海证券交易所
etf = yf.Ticker(ticker)
df = etf.history(start="2010-01-01", end="2025-05-20")

# 添加 instrument 列
df['instrument'] = 'SH512890'

# 重置索引并转换日期格式
df.reset_index(inplace=True)
df['Date'] = pd.to_datetime(df['Date']).dt.date
df.rename(columns={'Date': 'datetime'}, inplace=True)

# 保存为 CSV
df.to_csv('SH512890.csv', index=False)
print(f"Data saved to SH512890.csv")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
