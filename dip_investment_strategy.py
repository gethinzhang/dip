import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime
import backtrader as bt

def process_backtrader_orders(cerebro) -> List[Dict]:
    """
    处理backtrader的交易记录
    
    Args:
        cerebro: backtrader的cerebro对象
        
    Returns:
        处理后的交易记录列表
    """
    trades = []
    
    # 获取所有交易记录
    for strategy in cerebro.runstrats:
        for order in strategy.orders:
            if order.status == order.Completed:
                trade = {
                    'date': order.data.datetime.datetime(0).strftime('%Y-%m-%d'),
                    'price': order.executed.price,
                    'size': order.executed.size,
                    'direction': 'buy' if order.isbuy() else 'sell',
                    'grid': getattr(order, 'grid_level', 0)  # 从order对象获取网格级别
                }
                trades.append(trade)
    
    return trades

class InvestmentMetrics:
    def __init__(self, risk_free_rate: float = 0.03):
        """
        初始化投资指标计算类
        
        Args:
            risk_free_rate: 无风险利率，默认3%
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_metrics(self, 
                         portfolio_values: List[float],
                         cash_flows: List[float],
                         dates: List[str],
                         trades: List[Dict] = None) -> Dict:
        """
        计算投资组合的量化评估指标
        
        Args:
            portfolio_values: 投资组合每日价值列表
            cash_flows: 每日现金流列表（正数表示流入，负数表示流出）
            dates: 对应的日期列表
            trades: 交易记录列表，每个记录包含日期、价格、数量、方向等信息
            
        Returns:
            包含各项指标的字典
        """
        # 转换为numpy数组
        portfolio_values = np.array(portfolio_values)
        cash_flows = np.array(cash_flows)
        
        # 计算基础指标
        total_investment = np.sum(cash_flows[cash_flows > 0])
        final_value = portfolio_values[-1]
        
        # 计算每日收益率
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # 计算累计收益率
        total_return = (final_value - total_investment) / total_investment
        
        # 计算年化收益率
        days = len(portfolio_values)
        annual_return = (1 + total_return) ** (252 / days) - 1
        
        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # 计算风险调整后收益指标
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        
        # 计算定投相关指标
        monthly_irr = self._calculate_monthly_irr(cash_flows)
        annual_irr = (1 + monthly_irr) ** 12 - 1 if monthly_irr is not None else 0
        
        # 计算其他指标
        win_rate = self._calculate_win_rate(daily_returns)
        profit_factor = self._calculate_profit_factor(daily_returns)
        
        # 计算网格策略特定指标
        grid_metrics = self._calculate_grid_metrics(trades) if trades else {}
        
        return {
            "total_investment": total_investment,
            "final_value": final_value,
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "annual_irr": annual_irr,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            **grid_metrics
        }
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """计算最大回撤"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return np.max(drawdown)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """计算夏普比率"""
        excess_returns = returns - self.risk_free_rate/252
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """计算索提诺比率"""
        excess_returns = returns - self.risk_free_rate/252
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)
    
    def _calculate_monthly_irr(self, cash_flows: np.ndarray) -> float:
        """计算月度IRR"""
        try:
            return np.irr(cash_flows)
        except:
            return None
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """计算胜率"""
        winning_trades = np.sum(returns > 0)
        total_trades = len(returns)
        return winning_trades / total_trades if total_trades > 0 else 0
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """计算盈亏比"""
        gains = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))
        return gains / losses if losses != 0 else float('inf')
    
    def _calculate_grid_metrics(self, trades: List[Dict]) -> Dict:
        """计算网格策略特定指标"""
        if not trades:
            return {}
            
        # 计算网格交易统计
        buy_trades = [t for t in trades if t.get('direction') == 'buy']
        sell_trades = [t for t in trades if t.get('direction') == 'sell']
        
        # 计算网格交易收益
        grid_profits = []
        grid_profit_details = []
        
        # 按网格级别统计
        grid_levels = {}
        for trade in trades:
            grid_level = trade.get('grid', 0)
            if grid_level not in grid_levels:
                grid_levels[grid_level] = {'buy': 0, 'sell': 0, 'profit': 0}
            
            if trade.get('direction') == 'buy':
                grid_levels[grid_level]['buy'] += 1
            else:
                grid_levels[grid_level]['sell'] += 1
        
        # 计算每个网格的收益
        for i in range(len(sell_trades)):
            if i < len(buy_trades):
                buy_price = buy_trades[i].get('price', 0)
                sell_price = sell_trades[i].get('price', 0)
                buy_grid = buy_trades[i].get('grid', 0)
                sell_grid = sell_trades[i].get('grid', 0)
                
                profit = (sell_price - buy_price) / buy_price
                grid_profits.append(profit)
                
                grid_profit_details.append({
                    'buy_grid': buy_grid,
                    'sell_grid': sell_grid,
                    'profit': profit,
                    'buy_price': buy_price,
                    'sell_price': sell_price
                })
                
                # 更新网格级别统计
                if buy_grid in grid_levels:
                    grid_levels[buy_grid]['profit'] += profit
        
        # 计算网格策略指标
        grid_win_rate = len([p for p in grid_profits if p > 0]) / len(grid_profits) if grid_profits else 0
        avg_grid_profit = np.mean(grid_profits) if grid_profits else 0
        max_grid_profit = np.max(grid_profits) if grid_profits else 0
        min_grid_profit = np.min(grid_profits) if grid_profits else 0
        
        # 计算网格使用效率
        grid_usage = {level: stats['buy'] + stats['sell'] for level, stats in grid_levels.items()}
        most_used_grid = max(grid_usage.items(), key=lambda x: x[1])[0] if grid_usage else 0
        
        return {
            "grid_trades": len(trades),
            "grid_buy_trades": len(buy_trades),
            "grid_sell_trades": len(sell_trades),
            "grid_win_rate": grid_win_rate,
            "avg_grid_profit": avg_grid_profit,
            "max_grid_profit": max_grid_profit,
            "min_grid_profit": min_grid_profit,
            "grid_levels": grid_levels,
            "most_used_grid": most_used_grid,
            "grid_profit_details": grid_profit_details
        }

def format_metrics(metrics: Dict) -> str:
    """
    格式化指标输出
    
    Args:
        metrics: 指标字典
    
    Returns:
        格式化后的字符串
    """
    base_metrics = f"""
投资组合表现指标:
-----------------
总投资金额: {metrics['total_investment']:,.2f}
最终价值: {metrics['final_value']:,.2f}
总收益率: {metrics['total_return']:.2%}
年化收益率: {metrics['annual_return']:.2%}
最大回撤: {metrics['max_drawdown']:.2%}
夏普比率: {metrics['sharpe_ratio']:.2f}
索提诺比率: {metrics['sortino_ratio']:.2f}
定投年化收益率: {metrics['annual_irr']:.2%}
胜率: {metrics['win_rate']:.2%}
盈亏比: {metrics['profit_factor']:.2f}
"""
    
    # 如果有网格策略指标，添加网格策略部分
    if 'grid_trades' in metrics:
        grid_metrics = f"""
网格策略指标:
-----------------
总交易次数: {metrics['grid_trades']}
买入交易次数: {metrics['grid_buy_trades']}
卖出交易次数: {metrics['grid_sell_trades']}
网格胜率: {metrics['grid_win_rate']:.2%}
平均网格收益: {metrics['avg_grid_profit']:.2%}
最大网格收益: {metrics['max_grid_profit']:.2%}
最小网格收益: {metrics['min_grid_profit']:.2%}
最活跃网格: {metrics['most_used_grid']}

网格级别统计:
"""
        # 添加每个网格级别的统计信息
        for level, stats in sorted(metrics['grid_levels'].items()):
            grid_metrics += f"""
网格 {level}:
  买入次数: {stats['buy']}
  卖出次数: {stats['sell']}
  总收益: {stats['profit']:.2%}
"""
        
        return base_metrics + grid_metrics
    
    return base_metrics

def analyze_dca_performance(portfolio_values: List[float],
                          cash_flows: List[float],
                          dates: List[str],
                          trades: List[Dict] = None,
                          risk_free_rate: float = 0.03) -> str:
    """
    分析定投策略表现
    
    Args:
        portfolio_values: 投资组合每日价值列表
        cash_flows: 每日现金流列表
        dates: 对应的日期列表
        trades: 交易记录列表
        risk_free_rate: 无风险利率
        
    Returns:
        格式化后的分析结果
    """
    metrics_calculator = InvestmentMetrics(risk_free_rate)
    metrics = metrics_calculator.calculate_metrics(portfolio_values, cash_flows, dates, trades)
    return format_metrics(metrics)

# 使用示例
if __name__ == "__main__":
    # 示例数据
    portfolio_values = [10000, 10200, 10100, 10300, 10500]
    cash_flows = [10000, 1000, 1000, 1000, 1000]
    dates = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
    
    # 示例交易记录
    trades = [
        {'date': '2023-01-02', 'price': 1.02, 'direction': 'buy', 'grid': 1},
        {'date': '2023-01-03', 'price': 1.01, 'direction': 'sell', 'grid': 0},
        {'date': '2023-01-04', 'price': 1.03, 'direction': 'buy', 'grid': 2},
        {'date': '2023-01-05', 'price': 1.05, 'direction': 'sell', 'grid': 1}
    ]
    
    # 分析定投表现
    result = analyze_dca_performance(portfolio_values, cash_flows, dates, trades)
    print(result)

# 在网格策略回测后
cerebro = bt.Cerebro()
# ... 设置策略和其他参数 ...
cerebro.run()

# 获取并处理交易记录
trades = process_backtrader_orders(cerebro)

# 分析策略表现
result = analyze_dca_performance(
    portfolio_values=portfolio_values,
    cash_flows=cash_flows,
    dates=dates,
    trades=trades
)
print(result) 