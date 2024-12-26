# backtest.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from technical_analysis.technical_analysis import TechnicalAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BacktestEngine:
    def __init__(self, initial_capital: float = 10000):
        self.logger = logging.getLogger('DecryptoBot.Backtest')
        self.initial_capital = initial_capital
        self.ta = TechnicalAnalyzer()
        
        # Trading parameters (mirror live trading params)
        self.params = {
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 3.0,
            'max_position_size': 1000,  # USDT
            'risk_per_trade': 0.01,  # 1% risk per trade
        }
        
        # Initialize performance tracking
        self.reset_performance_metrics()

    def reset_performance_metrics(self):
        """Reset all performance metrics"""
        self.capital = self.initial_capital
        self.positions = []
        self.trades_history = []
        self.equity_curve = []
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'total_return': 0.0
        }

    def load_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load and prepare historical data for backtesting.
        This method should be implemented to load data from your preferred source
        (e.g., saved CSV files, database, or API)
        """
        try:
            # Example using saved CSV (implement your data loading logic)
            df = pd.read_csv('data/historical_data.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Filter date range
            mask = (df.index >= start_date) & (df.index <= end_date)
            df = df[mask].copy()
            
            # Add technical indicators
            df = self.ta.add_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return None

    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """Run backtest on historical data"""
        try:
            self.reset_performance_metrics()
            
            for i in range(len(data)):
                current_bar = data.iloc[i]
                
                # Update open positions
                self._update_positions(current_bar)
                
                # Generate trading signals
                if i >= 50:  # Ensure enough data for indicators
                    historical_data = data.iloc[max(0, i-100):i+1]
                    signal, strength, details = self.ta.generate_trading_signals(historical_data)
                    
                    # Execute trades based on signals
                    if signal in ['buy', 'sell']:
                        self._execute_trade(signal, current_bar, strength)
                
                # Track equity curve
                self._update_equity_curve(current_bar)
            
            # Calculate final performance metrics
            self._calculate_performance_metrics()
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}")
            return None

    def _update_positions(self, current_bar: pd.Series):
        """Update open positions with current market data"""
        for position in self.positions[:]:  # Copy list for iteration
            # Check stop loss
            if current_bar['low'] <= position['stop_loss']:
                self._close_position(position, current_bar, 'stop_loss')
                continue
                
            # Check take profit
            if current_bar['high'] >= position['take_profit']:
                self._close_position(position, current_bar, 'take_profit')
                continue
                
            # Update trailing stop if enabled
            if position['trailing_stop']:
                new_stop = current_bar['close'] * 0.98  # 2% trailing stop
                if new_stop > position['stop_loss']:
                    position['stop_loss'] = new_stop

    def _execute_trade(self, signal: str, current_bar: pd.Series, strength: float):
        """Execute trade based on signal"""
        # Skip if already in position
        if self.positions:
            return
            
        # Calculate position size
        risk_amount = self.capital * self.params['risk_per_trade']
        stop_price = (current_bar['close'] * 
                     (1 - self.params['stop_loss_percentage'] / 100))
        position_size = risk_amount / (current_bar['close'] - stop_price)
        
        # Ensure position size doesn't exceed max
        position_size = min(
            position_size,
            self.params['max_position_size'] / current_bar['close']
        )
        
        # Open position
        if signal == 'buy':
            position = {
                'entry_price': current_bar['close'],
                'size': position_size,
                'stop_loss': stop_price,
                'take_profit': current_bar['close'] * 
                             (1 + self.params['take_profit_percentage'] / 100),
                'trailing_stop': True,
                'entry_time': current_bar.name,
                'entry_capital': self.capital
            }
            self.positions.append(position)

    def _close_position(self, position: Dict, current_bar: pd.Series, reason: str):
        """Close a position and record the trade"""
        # Calculate profit/loss
        pnl = (current_bar['close'] - position['entry_price']) * position['size']
        
        # Update capital
        self.capital += pnl
        
        # Record trade
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': current_bar.name,
            'entry_price': position['entry_price'],
            'exit_price': current_bar['close'],
            'size': position['size'],
            'pnl': pnl,
            'return': (pnl / position['entry_capital']) * 100,
            'reason': reason
        }
        self.trades_history.append(trade)
        
        # Remove position
        self.positions.remove(position)

    def _update_equity_curve(self, current_bar: pd.Series):
        """Update equity curve with current portfolio value"""
        portfolio_value = self.capital
        for position in self.positions:
            portfolio_value += (current_bar['close'] - position['entry_price']) * position['size']
        self.equity_curve.append({
            'timestamp': current_bar.name,
            'equity': portfolio_value
        })

    def _calculate_performance_metrics(self):
        """Calculate final performance metrics"""
        if not self.trades_history:
            return
            
        # Basic metrics
        self.metrics['total_trades'] = len(self.trades_history)
        self.metrics['winning_trades'] = sum(1 for t in self.trades_history if t['pnl'] > 0)
        self.metrics['losing_trades'] = sum(1 for t in self.trades_history if t['pnl'] <= 0)
        
        # Win rate
        self.metrics['win_rate'] = (self.metrics['winning_trades'] / 
                                  self.metrics['total_trades']) * 100
        
        # Average returns
        winning_returns = [t['return'] for t in self.trades_history if t['pnl'] > 0]
        losing_returns = [t['return'] for t in self.trades_history if t['pnl'] <= 0]
        
        self.metrics['avg_win'] = np.mean(winning_returns) if winning_returns else 0
        self.metrics['avg_loss'] = np.mean(losing_returns) if losing_returns else 0
        
        # Equity curve analysis
        equity_series = pd.DataFrame(self.equity_curve).set_index('timestamp')['equity']
        
        # Calculate drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        self.metrics['max_drawdown'] = abs(drawdown.min())
        
        # Total return
        self.metrics['total_return'] = (
            (self.capital - self.initial_capital) / self.initial_capital * 100
        )
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        returns = equity_series.pct_change().dropna()
        excess_returns = returns - 0.02/252  # Daily risk-free rate
        self.metrics['sharpe_ratio'] = np.sqrt(252) * (
            excess_returns.mean() / excess_returns.std()
        )
        
        # Profit factor
        total_gains = sum(t['pnl'] for t in self.trades_history if t['pnl'] > 0)
        total_losses = sum(abs(t['pnl']) for t in self.trades_history if t['pnl'] <= 0)
        self.metrics['profit_factor'] = (
            total_gains / total_losses if total_losses != 0 else float('inf')
        )

    def plot_results(self) -> go.Figure:
        """Generate interactive plot of backtest results"""
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxis=True,
                           vertical_spacing=0.03,
                           row_heights=[0.7, 0.3])

        # Add equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        fig.add_trace(
            go.Scatter(x=equity_df['timestamp'], 
                      y=equity_df['equity'],
                      name='Portfolio Value',
                      line=dict(color='blue')),
            row=1, col=1
        )

        # Add trade markers
        for trade in self.trades_history:
            color = 'green' if trade['pnl'] > 0 else 'red'
            
            # Entry marker
            fig.add_trace(
                go.Scatter(x=[trade['entry_time']], 
                          y=[trade['entry_price']],
                          mode='markers',
                          marker=dict(color=color, size=10, symbol='triangle-up'),
                          name=f'Entry ({trade["entry_price"]:.2f})'),
                row=1, col=1
            )
            
            # Exit marker
            fig.add_trace(
                go.Scatter(x=[trade['exit_time']], 
                          y=[trade['exit_price']],
                          mode='markers',
                          marker=dict(color=color, size=10, symbol='triangle-down'),
                          name=f'Exit ({trade["exit_price"]:.2f})'),
                row=1, col=1
            )

        # Add drawdown chart
        equity_series = pd.DataFrame(self.equity_curve).set_index('timestamp')['equity']
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        
        fig.add_trace(
            go.Scatter(x=equity_df['timestamp'],
                      y=drawdown,
                      name='Drawdown',
                      fill='tonexty',
                      line=dict(color='red')),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title='Backtest Results',
            yaxis_title='Portfolio Value (USDT)',
            yaxis2_title='Drawdown %',
            showlegend=True,
            height=800
        )

        return fig
    
    def plot_results(self):
        """Generate and save plot of backtest results"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

        # Example plot data
        fig.add_trace(go.Candlestick(
            x=self.data.index,
            open=self.data['open'],
            high=self.data['high'],
            low=self.data['low'],
            close=self.data['close'],
            name='Candlestick'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['equity_curve'],
            mode='lines',
            name='Equity Curve'
        ), row=2, col=1)

        fig.update_layout(title='Backtest Results', xaxis_title='Date', yaxis_title='Price')
        fig.write_html('backtest_results.html')
        fig.show()    

    def generate_report(self) -> str:
        """Generate detailed backtest report"""
        report = [
            "=== Backtest Results ===\n",
            f"Initial Capital: {self.initial_capital:,.2f} USDT",
            f"Final Capital: {self.capital:,.2f} USDT",
            f"Total Return: {self.metrics['total_return']:.2f}%\n",
            "Trading Statistics:",
            f"Total Trades: {self.metrics['total_trades']}",
            f"Winning Trades: {self.metrics['winning_trades']}",
            f"Losing Trades: {self.metrics['losing_trades']}",
            f"Win Rate: {self.metrics['win_rate']:.2f}%\n",
            "Returns:",
            f"Average Win: {self.metrics['avg_win']:.2f}%",
            f"Average Loss: {self.metrics['avg_loss']:.2f}%",
            f"Profit Factor: {self.metrics['profit_factor']:.2f}\n",
            "Risk Metrics:",
            f"Maximum Drawdown: {self.metrics['max_drawdown']:.2f}%",
            f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}\n",
            "=== End Report ===\n"
        ]
        
        return "\n".join(report)
    
    # Example usage
if __name__ == "__main__":
    backtest_engine = BacktestEngine()
    data = backtest_engine.load_data('data/historical_data.csv')
    backtest_engine.run_backtest(data)
    backtest_engine.plot_results()
    print(data.head())