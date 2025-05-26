import backtrader as bt
import datetime
import yfinance as yf  # Add yfinance as a more reliable data source
import os
import pandas as pd
import numpy as np
import numpy_financial as npf

TICKERS = ["IWY", "RSP", "MOAT", "PFF", "VNQ"]
#TICKERS = ["SPY", "QQQ", "BRK-B", "RSP"]
RISK_FREE = 0.04
START_DATE = datetime.datetime(1990, 1, 1)
END_DATE = datetime.datetime.now()

class PandasDataWithDividends(bt.feeds.PandasData):
    params = (
        ('dividends', None),
    )
    lines = ('dividends',)

class DCAETFStrategy(bt.Strategy):
    params = (
        ("tickers", TICKERS),
        ("cash_per_contribution", 10000),
        ("human_extreme_threshold", 0.2),
        ("min_period", 10),  # 添加最小交易周期参数，设置为10个交易日（约两周）
    )

    def __init__(self):
        # Map each data feed by its name
        self.datas_by_name = {d._name: d for d in self.datas}
        self.rsp = self.datas_by_name["RSP"]
        self.current_month = None
        self.current_month_cash_flow = self.p.cash_per_contribution
        self.first_exec = False
        self.second_exec = False
        self.third_exec = False
        self.month_start_price = None
        
        # Track orders and cash
        self.orders = {}  # Track pending orders
        # record the cash flow and portfolio value for each month
        self.monthly_cash_flows = [-self.p.cash_per_contribution]
        self.monthly_portfolio_values = []
        self.monthly_dates = []
        
        # Track dividends and taxes
        self.dividend_history = []  # List to store dividend payments
        self.total_dividends = 0
        self.total_tax_paid = 0
        self.non_resident_tax_rate = 0.30  # 30% non-resident tax rate
        
        self.broker.setcash(self.p.cash_per_contribution)
        self.log(f"Initial cash: {self.broker.getcash()}")
        
        # Add order tracking
        self.order_history = []  # List to store all order details

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            # Record order details
            order_info = {
                'date': self.datas[0].datetime.date(0),
                'ticker': order.data._name,
                'type': 'BUY' if order.isbuy() else 'SELL',
                'price': order.executed.price,
                'size': order.executed.size,
                'value': order.executed.value,
                'commission': order.executed.comm,
                'reason': getattr(order, 'reason', 'N/A')  # Get the reason if available
            }
            self.order_history.append(order_info)
            
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, {order.data._name}, Price: {order.executed.price:.2f}, '
                    f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}, '
                    f'Size: {order.executed.size}'
                )
            else:
                self.log(
                    f'SELL EXECUTED, {order.data._name}, Price: {order.executed.price:.2f}, '
                    f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}, '
                    f'Size: {order.executed.size}'
                )
        elif order.status in [order.Canceled, order.Rejected]:
            self.log(f'Order Canceled/Rejected: {order.data._name}, reason: {str(order.info)}')
        elif order.status in [order.Margin]:
            self.log(f'Order Margin not enough {order.data._name}')

        if order.ref in self.orders:
            del self.orders[order.ref]

    def is_third_week_end(self):
        # Determine the last valid trading day (Mon-Fri) of the 3rd calendar week
        dt = self.datas[0].datetime.date(0)
        try:
            next_dt = self.datas[0].datetime.date(1)
        except IndexError:
            next_dt = None

        # if next_dt is in 4th week and dt is in 3rd week, then it is the last trading day of the 3rd week
        if (next_dt is None or 22 <= next_dt.day <= 28) and 15 <= dt.day <= 21:
            return True
        return False

    def check_human_extreme(self):
        # Placeholder: integrate real 60-day SPX breadth data
        return False

    def buy_etfs(self, reason):
        self.log(f"Executing contribution due to {reason}")
        
        # Calculate allocation per ETF
        alloc = self.p.cash_per_contribution / len(self.p.tickers)
        
        # Place orders for each ETF
        for name, data in self.datas_by_name.items():
            order = self.buy(data, int(alloc / data.close[0]))
            if order:
                order.reason = reason  # Add reason to order for tracking
                self.orders[order.ref] = order

    def rebalance(self):
        self.log("Annual rebalance to equal weights")
        for name, data in self.datas_by_name.items():
            order = self.order_target_percent(data, 1.0 / len(self.p.tickers))
            if order:
                order.reason = "rebalance"
                self.orders[order.ref] = order

    def next(self):
        # 检查是否达到最小交易周期
        if len(self) < self.p.min_period:
            #print(f"Not enough data to trade, current length: {len(self)}")
            return

        dt = self.datas[0].datetime.date(0)
        
        # Track monthly cash flows and portfolio values
        if self.current_month is None or self.current_month != dt.month:
            portfolio_value = self.broker.getvalue()
            self.monthly_portfolio_values.append(portfolio_value)
            self.monthly_dates.append(dt)
            
            # Calculate monthly cash flow (negative for investments)
            self.monthly_cash_flows.append(-self.current_month_cash_flow)
            
            self.current_month = dt.month
            self.first_exec = self.second_exec = self.third_exec = False
            self.month_start_price = self.rsp.close[0]
            self.current_month_cash_flow = 0  # Reset monthly cash flow
            
            self.log(f"Month start - Cash: {self.broker.getcash()}, Portfolio value: {portfolio_value}")

        # 1st contribution: RSP daily drop ≤ -1% or fallback on 3rd week's last trading day
        if not self.first_exec:
            prev_close = self.rsp.close[-1]
            today_close = self.rsp.close[0]
            daily_ret = (today_close / prev_close) - 1
            if daily_ret <= -0.01:
                self.buy_etfs("RSP daily drop <= -1%")
                self.first_exec = True
            elif self.is_third_week_end():
                self.buy_etfs("Fallback on 3rd week end")
                self.first_exec = True

        # 2nd contribution: MTD drop ≤ -5%
        if not self.second_exec:
            mtd_ret = (self.rsp.close[0] / self.month_start_price) - 1
            if mtd_ret <= -0.05:
                self.buy_etfs("RSP MTD drop <= -5%")
                self.second_exec = True

        # 3rd contribution: human extreme event
        if not self.third_exec:
            if self.check_human_extreme():
                self.buy_etfs("Human extreme event")
                self.third_exec = True

        # Annual rebalance on August 1st
        if dt.month == 8 and dt.day == 1:
            self.rebalance()

        # Add cash for this contribution
        if self.broker.getcash() < 2*self.p.cash_per_contribution:
            self.broker.add_cash(2*self.p.cash_per_contribution)
            self.current_month_cash_flow += 2*self.p.cash_per_contribution

    def stop(self):
        # Add final portfolio value
        final_value = self.broker.getvalue()
        final_cash = self.broker.getcash()
        print(f"final_cash: {final_cash} final_value: {final_value} stock_value: {final_value - final_cash}")
        self.monthly_portfolio_values.append(final_value)
        self.monthly_dates.append(self.datas[0].datetime.date(0))
        # Calculate and print key metrics
        self.calculate_metrics()
        
        # Export all results to Excel
        self.export_to_excel()

    def calculate_metrics(self):
        # Calculate total return
        total_invested = -sum(self.monthly_cash_flows)
        final_value = self.broker.getvalue()
        total_return = (final_value - total_invested) / total_invested
        
        # Calculate IRR
        cash_flows = self.monthly_cash_flows.copy()
        cash_flows.append(final_value)  # Add final portfolio value as positive cash flow
        
        try:
            irr = npf.irr(cash_flows)
            annual_irr = (1 + irr) ** 12 - 1  # Convert monthly IRR to annual
        except:
            irr = None
            annual_irr = None
        
        # Calculate maximum drawdown
        for data in self.datas:
            pos = self.broker.getposition(data)
            print(f"data: {data._name} size: {pos.size} price: {pos.price}")
        portfolio_values = np.array(self.monthly_portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Print metrics
        print("\nBacktest Results:")
        print(f"Total invested: {self.monthly_cash_flows[0] + sum(self.monthly_cash_flows[1:])}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annual IRR: {annual_irr:.2%}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        
        # Plot results
        #self.plot_results()

    def plot_results(self):
        import matplotlib.pyplot as plt
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot portfolio value
        ax1.plot(self.monthly_dates, self.monthly_portfolio_values, label='Portfolio Value')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Value ($)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot drawdown
        portfolio_values = np.array(self.monthly_portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        ax2.fill_between(self.monthly_dates, drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def notify_cashvalue(self, cash, value):
        # Track dividends and apply non-resident tax
        current_date = self.datas[0].datetime.date(0)
        
        for data in self.datas:
            if data.dividends[0] is not None and data.dividends[0] > 0:
                print(f"date {current_date} dividend: {data.dividends[0]}")
                pos = self.broker.getposition(data)
                if pos.size > 0:
                    dividend_amount = pos.size * data.dividends[0]
                    tax_amount = dividend_amount * self.non_resident_tax_rate
                    net_dividend = dividend_amount - tax_amount
                    
                    # Add dividend to order history
                    dividend_order = {
                        'date': current_date,
                        'ticker': data._name,
                        'type': 'DIVIDEND',
                        'price': data.dividends[0],
                        'size': pos.size,
                        'value': dividend_amount,
                        'commission': tax_amount,  # Use commission field to store tax
                        'reason': f'Dividend payment (Net: ${net_dividend:.2f})'
                    }
                    self.order_history.append(dividend_order)
                    
                    self.dividend_history.append({
                        'date': current_date,
                        'ticker': data._name,
                        'amount': dividend_amount,
                        'tax': tax_amount,
                        'net': net_dividend
                    })
                    
                    self.total_dividends += dividend_amount
                    self.total_tax_paid += tax_amount
                    
                    # Add net dividend to cash
                    self.broker.add_cash(net_dividend)
                    self.log(f'Dividend received for {data._name}: ${dividend_amount:.2f}, Tax: ${tax_amount:.2f}, Net: ${net_dividend:.2f}')

    def export_to_excel(self):
        # Create a Pandas Excel writer using XlsxWriter as the engine
        fn = f"QUANT_FOR_{'-'.join(TICKERS)}_{START_DATE.strftime('%Y-%m-%d')}_{END_DATE.strftime('%Y-%m-%d')}.xlsx"
        writer = pd.ExcelWriter(fn, engine='xlsxwriter')
        print(f"Results exported to {fn}")
        
        # Export order history
        orders_df = pd.DataFrame(self.order_history)
        orders_df = orders_df.sort_values('date')
        
        # Create formats for different order types
        workbook = writer.book
        buy_format = workbook.add_format({'bg_color': '#C6EFCE'})  # Light green
        sell_format = workbook.add_format({'bg_color': '#FFC7CE'})  # Light red
        dividend_format = workbook.add_format({'bg_color': '#FFEB9C'})  # Light yellow
        
        # Create date format
        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
        
        # First write the DataFrame to Excel
        orders_df.to_excel(writer, sheet_name='Results', startrow=0, index=False)
        
        # Now get the worksheet and apply formatting
        worksheet = writer.sheets['Results']
        
        # Apply formatting based on order type
        for idx, row in orders_df.iterrows():
            # Format date column
            worksheet.write(idx + 1, 0, row['date'], date_format)
            
            if row['type'] == 'BUY':
                for col in range(1, len(orders_df.columns)):
                    worksheet.write(idx + 1, col, row[orders_df.columns[col]], buy_format)
            elif row['type'] == 'SELL':
                for col in range(1, len(orders_df.columns)):
                    worksheet.write(idx + 1, col, row[orders_df.columns[col]], sell_format)
            elif row['type'] == 'DIVIDEND':
                for col in range(1, len(orders_df.columns)):
                    worksheet.write(idx + 1, col, row[orders_df.columns[col]], dividend_format)
        
        annual_irr = self.calculate_annual_irr()
        max_drawdown = self.calculate_max_drawdown()
        sharpe_ratio = (annual_irr - RISK_FREE) / max_drawdown * np.sqrt(12)
        # Export metrics starting from row 7 (O7)
        metrics = {
            'Metric': [
                'Total Invested',
                'Final Portfolio Value',
                'Total Return',
                'Annual IRR',
                'Maximum Drawdown',
                'Sharpe Ratio',
                'Final Cash',
                'Final Stock Value',
                'Total Orders',
                'Total Buy Orders',
                'Total Sell Orders',
                'Average Order Size',
                'Total Commission Paid',
                'Total Dividends',
                'Total Tax Paid',
                'Net Dividends'
            ],
            'Value': [
                -sum(self.monthly_cash_flows),
                self.monthly_portfolio_values[-1],
                (self.monthly_portfolio_values[-1] - (-sum(self.monthly_cash_flows))) / (-sum(self.monthly_cash_flows)),
                annual_irr,
                max_drawdown,
                sharpe_ratio,
                self.broker.getcash(),
                self.monthly_portfolio_values[-1] - self.broker.getcash(),
                len(self.order_history),
                len([o for o in self.order_history if o['type'] == 'BUY']),
                len([o for o in self.order_history if o['type'] == 'SELL']),
                sum(o['size'] for o in self.order_history) / len(self.order_history) if self.order_history else 0,
                sum(o['commission'] for o in self.order_history),
                self.total_dividends,
                self.total_tax_paid,
                self.total_dividends - self.total_tax_paid
            ]
        }
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_excel(writer, sheet_name='Results', startrow=7, startcol=12, index=False)
        
        # Get the workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Results']
        
        # Add some formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Format the headers
        for col_num, value in enumerate(orders_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        for col_num, value in enumerate(metrics_df.columns.values):
            worksheet.write(7, col_num+12, value, header_format)
        
        # Add final positions header and data
        worksheet.write(7, 15, 'Final Positions', header_format)
        worksheet.write(7, 16, 'Shares', header_format)
        worksheet.write(7, 17, 'Invested', header_format)
        worksheet.write(7, 18, 'Final Value', header_format)
        worksheet.write(7, 19, 'Return', header_format)
        worksheet.write(7, 20, 'Dividends', header_format)
        worksheet.write(7, 21, 'Net Dividends', header_format)
        
        # Write final positions data
        row = 8
        total_value = self.broker.getvalue()
        total_shares = 0
        total_stock_value = 0
        total_invested = 0
        total_dividends = 0
        total_net_dividends = 0
        
        for data in self.datas:
            pos = self.broker.getposition(data)
            if pos.size > 0:
                latest_price = data.close[0]
                value = pos.size * latest_price
                invested = pos.size * pos.price
                position_dividends = sum(d['amount'] for d in self.dividend_history if d['ticker'] == data._name)
                position_net_dividends = sum(d['net'] for d in self.dividend_history if d['ticker'] == data._name)
                
                worksheet.write(row, 15, data._name)
                worksheet.write(row, 16, pos.size)
                worksheet.write(row, 17, invested)
                worksheet.write(row, 18, value)
                worksheet.write(row, 19, (value - invested) / invested)
                worksheet.write(row, 20, position_dividends)
                worksheet.write(row, 21, position_net_dividends)
                
                total_shares += pos.size
                total_stock_value += value
                total_invested += invested
                total_dividends += position_dividends
                total_net_dividends += position_net_dividends
                row += 1
        
        # Add sum row
        worksheet.write(row, 15, 'Total', header_format)
        worksheet.write(row, 16, total_shares)
        worksheet.write(row, 17, total_invested)
        worksheet.write(row, 18, total_stock_value)
        worksheet.write(row, 19, (total_stock_value - total_invested) / total_invested)
        worksheet.write(row, 20, total_dividends)
        worksheet.write(row, 21, total_net_dividends)
        
        # Adjust column widths
        worksheet.set_column('A:A', 15)  # Date column
        worksheet.set_column('B:B', 10)  # Ticker column
        worksheet.set_column('C:C', 8)   # Type column
        worksheet.set_column('D:H', 12)  # Numeric columns
        
        # Save the Excel file
        writer.close()
        self.log(f"Results exported to {fn}")

    def calculate_annual_irr(self):
        try:
            cash_flows = self.monthly_cash_flows.copy()
            cash_flows.append(self.monthly_portfolio_values[-1])
            irr = npf.irr(cash_flows)
            return (1 + irr) ** 12 - 1
        except:
            return None

    def calculate_max_drawdown(self):
        portfolio_values = np.array(self.monthly_portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return np.max(drawdown)


def download_multiple_tickers(tickers, start, end, cache_dir="cache"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    data_dict = {}
    for ticker in tickers:
        cache_file = os.path.join(cache_dir, f"{ticker}-{start.strftime('%Y-%m-%d')}-{end.strftime('%Y-%m-%d')}.csv")
        if os.path.exists(cache_file):
            data_dict[ticker] = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            try:
                ticker_obj = yf.Ticker(ticker)
                # Download both price and dividend data
                df = ticker_obj.history(start=start, end=end)
                # Get dividend data
                dividends = ticker_obj.dividends
                if not dividends.empty:
                    # Align dividend data with price data
                    df['Dividends'] = dividends.reindex(df.index).fillna(0)
                else:
                    df['Dividends'] = 0
                df.to_csv(cache_file)
                data_dict[ticker] = df
            except Exception as e:
                print(f"Error downloading {ticker}: {str(e)}")
    return data_dict


if __name__ == "__main__":
    cerebro = bt.Cerebro()

    data_dict = download_multiple_tickers(TICKERS, START_DATE, END_DATE)

    for ticker in TICKERS:
        if ticker in data_dict:
            data = PandasDataWithDividends(
                dataname=data_dict[ticker],
                datetime=None,  # 使用索引作为日期
                open="Open",
                high="High",
                low="Low",
                close="Close",
                volume="Volume",
                dividends="Dividends",
                openinterest=-1,
            )
            cerebro.adddata(data, name=ticker)

    #cerebro.broker.setcash(10000000)
    cerebro.broker.setcommission(commission=0.003)
    cerebro.addstrategy(DCAETFStrategy)

    results = cerebro.run()
