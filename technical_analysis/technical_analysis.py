# technical_analysis.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging

class TechnicalAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger('DecryptoBot.TA')

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the dataframe"""
        df = self._add_moving_averages(df)
        df = self._add_bollinger_bands(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_atr(df)
        df = self._identify_support_resistance(df)
        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages"""
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        return df

    def _add_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df['bb_middle'] = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()
        df['bb_upper'] = df['bb_middle'] + (std * num_std)
        df['bb_lower'] = df['bb_middle'] - (std * num_std)
        
        # Calculate bandwidth and %B
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df

    def _add_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator"""
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df

    def _add_atr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=window).mean()
        return df

    def _identify_support_resistance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Identify potential support and resistance levels"""
        df['rolling_high'] = df['high'].rolling(window=window, center=True).max()
        df['rolling_low'] = df['low'].rolling(window=window, center=True).min()
        
        # Mark potential support and resistance zones
        df['potential_support'] = df.apply(
            lambda x: x['low'] if x['low'] == x['rolling_low'] else np.nan, axis=1)
        df['potential_resistance'] = df.apply(
            lambda x: x['high'] if x['high'] == x['rolling_high'] else np.nan, axis=1)
        return df

    def identify_range(self, df: pd.DataFrame) -> Dict[str, float]:
        """Identify if the market is in a range and potential trading boundaries"""
        latest_data = df.tail(20)  # Look at recent price action
        
        range_metrics = {
            'is_ranging': False,
            'range_top': None,
            'range_bottom': None,
            'range_strength': 0
        }
        
        # Check if price is moving sideways using Bollinger Bandwidth
        avg_bandwidth = latest_data['bb_bandwidth'].mean()
        if avg_bandwidth < 0.1:  # Narrow bands indicate ranging market
            range_metrics['is_ranging'] = True
            range_metrics['range_top'] = latest_data['bb_upper'].mean()
            range_metrics['range_bottom'] = latest_data['bb_lower'].mean()
            range_metrics['range_strength'] = 1 - avg_bandwidth
            
        return range_metrics

    def generate_trading_signals(self, df: pd.DataFrame) -> Tuple[str, float, Dict]:
        """Generate trading signals based on technical analysis"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = {
            'rsi_signal': False,
            'macd_signal': False,
            'bb_signal': False,
            'trend_signal': False
        }
        
        # RSI signals
        signals['rsi_signal'] = (
            'buy' if latest['rsi'] < 30 else
            'sell' if latest['rsi'] > 70 else
            None
        )
        
        # MACD signals
        signals['macd_signal'] = (
            'buy' if (latest['macd'] > latest['macd_signal'] and 
                     prev['macd'] <= prev['macd_signal']) else
            'sell' if (latest['macd'] < latest['macd_signal'] and 
                      prev['macd'] >= prev['macd_signal']) else
            None
        )
        
        # Bollinger Bands signals
        signals['bb_signal'] = (
            'buy' if latest['close'] < latest['bb_lower'] else
            'sell' if latest['close'] > latest['bb_upper'] else
            None
        )
        
        # Trend signals using moving averages
        signals['trend_signal'] = (
            'buy' if (latest['sma_20'] > latest['sma_50'] and 
                     latest['close'] > latest['sma_20']) else
            'sell' if (latest['sma_20'] < latest['sma_50'] and 
                      latest['close'] < latest['sma_20']) else
            None
        )
        
        # Combine signals for final decision
        buy_signals = sum(1 for signal in signals.values() if signal == 'buy')
        sell_signals = sum(1 for signal in signals.values() if signal == 'sell')
        
        signal_strength = abs(buy_signals - sell_signals) / len(signals)
        
        if buy_signals > sell_signals:
            return 'buy', signal_strength, signals
        elif sell_signals > buy_signals:
            return 'sell', signal_strength, signals
        else:
            return 'neutral', 0, signals

    def calculate_risk_reward(self, df: pd.DataFrame, side: str) -> Dict[str, float]:
        """Calculate risk/reward ratio for potential trade"""
        latest = df.iloc[-1]
        atr = latest['atr']
        
        risk_reward = {
            'stop_loss': None,
            'take_profit': None,
            'risk_reward_ratio': None
        }
        
        if side == 'buy':
            risk_reward['stop_loss'] = latest['close'] - (atr * 2)
            risk_reward['take_profit'] = latest['close'] + (atr * 3)
        elif side == 'sell':
            risk_reward['stop_loss'] = latest['close'] + (atr * 2)
            risk_reward['take_profit'] = latest['close'] - (atr * 3)
            
        if risk_reward['stop_loss'] and risk_reward['take_profit']:
            risk = abs(latest['close'] - risk_reward['stop_loss'])
            reward = abs(risk_reward['take_profit'] - latest['close'])
            risk_reward['risk_reward_ratio'] = reward / risk if risk != 0 else 0
            
        return risk_reward