# DecryptoBot
# A conservative crypto trading bot focusing on new listings and range trading

import os
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from dotenv import load_dotenv

class DecryptoBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('decryptobot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DecryptoBot')
        
        # Initialize Binance client
        try:
            self.client = Client(
                os.getenv('BINANCE_API_KEY'),
                os.getenv('BINANCE_API_SECRET'),
                testnet=True  # Set to False for real trading
            )
            self.logger.info("Successfully connected to Binance API")
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {str(e)}")
            raise

        # Trading parameters
        self.trading_params = {
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '100')),  # USDT
            'stop_loss_percentage': float(os.getenv('STOP_LOSS_PERCENTAGE', '2.0')),
            'take_profit_percentage': float(os.getenv('TAKE_PROFIT_PERCENTAGE', '3.0')),
            'max_trades_per_day': int(os.getenv('MAX_TRADES_PER_DAY', '5')),
        }

    def get_account_balance(self):
        """Get current account balance"""
        try:
            account = self.client.get_account()
            balances = {asset['asset']: float(asset['free']) 
                       for asset in account['balances'] 
                       if float(asset['free']) > 0}
            self.logger.info(f"Current balances: {balances}")
            return balances
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get account balance: {str(e)}")
            return None

    def fetch_historical_data(self, symbol: str, interval: str, limit: int = 100):
        """Fetch historical klines/candlestick data"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert price columns to float
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            # Convert timestamps to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        except BinanceAPIException as e:
            self.logger.error(f"Failed to fetch historical data: {str(e)}")
            return None

    def calculate_risk_metrics(self, df: pd.DataFrame):
        """Calculate basic risk metrics from historical data"""
        metrics = {
            'volatility': df['close'].pct_change().std(),
            'avg_daily_volume': df['volume'].mean(),
            'price_range': (df['high'].max() - df['low'].min()) / df['low'].min() * 100
        }
        return metrics

    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = 'MARKET'):
        """Place an order with integrated risk management"""
        try:
            # Check if we're within daily trade limit
            # Implement daily trade counting logic here
            
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity
            )
            
            self.logger.info(f"Order placed: {order}")
            
            # If market buy order successful, place stop loss and take profit orders
            if order['status'] == 'FILLED' and side == 'BUY':
                fill_price = float(order['fills'][0]['price'])
                
                # Place stop loss order
                stop_price = fill_price * (1 - self.trading_params['stop_loss_percentage'] / 100)
                self.place_stop_loss(symbol, quantity, stop_price)
                
                # Place take profit order
                take_profit_price = fill_price * (1 + self.trading_params['take_profit_percentage'] / 100)
                self.place_take_profit(symbol, quantity, take_profit_price)
            
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Failed to place order: {str(e)}")
            return None

    def place_stop_loss(self, symbol: str, quantity: float, stop_price: float):
        """Place a stop-loss order"""
        try:
            order = self.client.create_order(
                symbol=symbol,
                side='SELL',
                type='STOP_LOSS_LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                stopPrice=stop_price,
                price=stop_price * 0.99  # Slightly lower to ensure execution
            )
            self.logger.info(f"Stop loss placed: {order}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Failed to place stop loss: {str(e)}")
            return None

    def place_take_profit(self, symbol: str, quantity: float, take_profit_price: float):
        """Place a take-profit order"""
        try:
            order = self.client.create_order(
                symbol=symbol,
                side='SELL',
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=take_profit_price
            )
            self.logger.info(f"Take profit placed: {order}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Failed to place take profit: {str(e)}")
            return None

if __name__ == "__main__":
    bot = DecryptoBot()
    # Add your trading logic here
