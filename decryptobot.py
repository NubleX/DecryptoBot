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
from technical_analysis.technical_analysis import TechnicalAnalyzer
from typing import List, Dict, Optional, Tuple
from listing_monitor.listing_monitor import ListingMonitor
import asyncio
import time


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

        # Initialize listing monitor

        self.listing_monitor = ListingMonitor(self.client)

        self.listing_params = {
            'max_listing_age': int(os.getenv('MAX_LISTING_AGE_SECONDS', '300')),  # 5 minutes
            'initial_position_size': float(os.getenv('LISTING_POSITION_SIZE', '50')),  # USDT
            'quick_profit_target': float(os.getenv('LISTING_PROFIT_TARGET', '3.0')),  # Percentage
            'stop_loss': float(os.getenv('LISTING_STOP_LOSS', '2.0'))  # Percentage
        }

        # Initialize technical analyzer

        self.ta = TechnicalAnalyzer()
        
        # Add technical analysis parameters
        self.ta_params = {
            'min_signal_strength': float(os.getenv('MIN_SIGNAL_STRENGTH', '0.6')),
            'min_risk_reward_ratio': float(os.getenv('MIN_RISK_REWARD_RATIO', '1.5')),
            'range_trading_enabled': os.getenv('RANGE_TRADING_ENABLED', 'True').lower() == 'true'
        }

    async def run(self):
        """Main bot loop"""
        try:
            # Start listing monitor in the background
            listing_monitor_task = asyncio.create_task(self.listing_monitor.monitor_new_listings())
            
            # Main trading loop
            while True:
                # Regular trading logic here
                await asyncio.sleep(10)
                
        except Exception as e:
            self.logger.error(f"Error in main bot loop: {str(e)}")
            
        finally:
            listing_monitor_task.cancel()

    async def handle_new_listing(self, listing_analysis: Dict):
        """Handle trading decision for new listing"""
        if not listing_analysis['tradeable']:
            self.logger.info(f"Skipping {listing_analysis['symbol']} - does not meet criteria")
            return
            
        try:
            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=listing_analysis['symbol'])
            current_price = float(ticker['price'])
            
            # Calculate position size
            quantity = self.listing_params['initial_position_size'] / current_price
            
            # Place initial buy order
            order = self.place_order(
                symbol=listing_analysis['symbol'],
                side='BUY',
                quantity=quantity
            )
            
            if order and order['status'] == 'FILLED':
                # Place take profit and stop loss orders
                stop_price = current_price * (1 - self.listing_params['stop_loss'] / 100)
                target_price = current_price * (1 + self.listing_params['quick_profit_target'] / 100)
                
                self.place_stop_loss(listing_analysis['symbol'], quantity, stop_price)
                self.place_take_profit(listing_analysis['symbol'], quantity, target_price)
                
                self.logger.info(f"Successfully entered position for new listing {listing_analysis['symbol']}")
                
        except Exception as e:
            self.logger.error(f"Failed to handle new listing {listing_analysis['symbol']}: {str(e)}")

        # Trading parameters
        self.trading_params = {
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '100')),  # USDT
            'stop_loss_percentage': float(os.getenv('STOP_LOSS_PERCENTAGE', '2.0')),
            'take_profit_percentage': float(os.getenv('TAKE_PROFIT_PERCENTAGE', '5.0')),
            'max_trades_per_day': int(os.getenv('MAX_TRADES_PER_DAY', '10')),
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
        
    def analyze_market(self, symbol: str, interval: str = '1h'):
        """Analyze market conditions and generate trading signals"""
        try:
            # Fetch historical data
            df = self.fetch_historical_data(symbol, interval)
            if df is None:
                return None
                
            # Add technical indicators
            df = self.ta.add_indicators(df)
            
            # Generate trading signals
            signal, strength, details = self.ta.generate_trading_signals(df)
            
            # Calculate risk/reward
            risk_reward = self.ta.calculate_risk_reward(df, signal)
            
            # Check if market is ranging
            range_metrics = self.ta.identify_range(df)
            
            analysis_result = {
                'symbol': symbol,
                'signal': signal,
                'signal_strength': strength,
                'signal_details': details,
                'risk_reward': risk_reward,
                'range_metrics': range_metrics
            }
            
            self.logger.info(f"Market analysis completed for {symbol}: {analysis_result}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}")
            return None
    
    def should_trade(self, analysis_result: Dict) -> bool:
        """Determine if we should trade based on analysis results"""
        if not analysis_result:
            return False
            
        # Check signal strength
        if analysis_result['signal_strength'] < self.ta_params['min_signal_strength']:
            self.logger.info(f"Signal strength too low: {analysis_result['signal_strength']}")
            return False
            
        # Check risk/reward ratio
        if (analysis_result['risk_reward']['risk_reward_ratio'] < 
            self.ta_params['min_risk_reward_ratio']):
            self.logger.info("Risk/reward ratio unfavorable")
            return False
            
        # Range trading checks
        if self.ta_params['range_trading_enabled'] and analysis_result['range_metrics']['is_ranging']:
            current_price = float(self.client.get_symbol_ticker(symbol=analysis_result['symbol'])['price'])
            
            if (analysis_result['signal'] == 'buy' and 
                current_price > analysis_result['range_metrics']['range_bottom'] * 1.02):
                self.logger.info("Price too high in range for buy")
                return False
                
            if (analysis_result['signal'] == 'sell' and 
                current_price < analysis_result['range_metrics']['range_top'] * 0.98):
                self.logger.info("Price too low in range for sell")
                return False
                
        return True  
      
    async def start(self):
        """Initialize and start the trading bot"""
        self.logger.info("Starting DecryptoBot...")
        
        # Initialize trading pairs
        self.trading_pairs = self._get_trading_pairs()
        self.logger.info(f"Monitoring {len(self.trading_pairs)} trading pairs")
        
        try:
            # Start main loop
            await self.run_trading_loop()
        except KeyboardInterrupt:
            self.logger.info("Shutting down DecryptoBot...")
        except Exception as e:
            self.logger.error(f"Fatal error in bot execution: {str(e)}")
        finally:
            # Cleanup any open orders
            self._cleanup()

    def _get_trading_pairs(self) -> List[str]:
        """Get list of trading pairs based on configuration"""
        try:
            # Get exchange info
            exchange_info = self.client.get_exchange_info()
            
            # Filter for USDT pairs with sufficient volume
            all_pairs = []
            for symbol in exchange_info['symbols']:
                if (symbol['quoteAsset'] == 'USDT' and 
                    symbol['status'] == 'TRADING'):
                    
                    # Get 24h stats
                    stats = self.client.get_24hr_ticker(symbol=symbol['symbol'])
                    
                    # Check minimum volume requirement (e.g., 100,000 USDT)
                    if float(stats['quoteVolume']) > 100000:
                        all_pairs.append(symbol['symbol'])
            
            return all_pairs
            
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get trading pairs: {str(e)}")
            return []

    async def run_trading_loop(self):
        """Main trading loop"""
        while True:
            try:
                # 1. Check account status
                balance = self.get_account_balance()
                if not balance or balance.get('USDT', 0) < self.trading_params['max_position_size']:
                    self.logger.warning("Insufficient USDT balance")
                    await asyncio.sleep(60)
                    continue
                
                # 2. Monitor existing positions
                await self._manage_positions()
                
                # 3. Look for new trading opportunities
                await self._find_opportunities()
                
                # 4. Brief pause before next iteration
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(30)

    async def _manage_positions(self):
        """Manage existing positions"""
        try:
            positions = self._get_open_positions()
            for position in positions:
                # Update stop loss and take profit based on market conditions
                await self._update_position_orders(position)
                
                # Check if we should close the position
                if await self._should_close_position(position):
                    self._close_position(position)
                    
        except Exception as e:
            self.logger.error(f"Error managing positions: {str(e)}")

    def _get_open_positions(self) -> List[Dict]:
        """Get current open positions"""
        try:
            positions = []
            account = self.client.get_account()
            
            for asset in account['balances']:
                if float(asset['free']) > 0 or float(asset['locked']) > 0:
                    # Skip USDT
                    if asset['asset'] == 'USDT':
                        continue
                        
                    symbol = f"{asset['asset']}USDT"
                    if symbol in self.trading_pairs:
                        ticker = self.client.get_symbol_ticker(symbol=symbol)
                        positions.append({
                            'symbol': symbol,
                            'amount': float(asset['free']) + float(asset['locked']),
                            'current_price': float(ticker['price'])
                        })
                        
            return positions
            
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get open positions: {str(e)}")
            return []

    async def _find_opportunities(self):
        """Find new trading opportunities"""
        for symbol in self.trading_pairs:
            try:
                # Skip if we already have a position
                if self._has_position(symbol):
                    continue
                    
                # Analyze market
                analysis = self.analyze_market(symbol)
                if not analysis:
                    continue
                    
                # Check if we should trade
                if self.should_trade(analysis):
                    # Execute trade
                    await self._execute_trade(analysis)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {str(e)}")

    async def _execute_trade(self, analysis: Dict):
        """Execute trade based on analysis"""
        try:
            symbol = analysis['symbol']
            signal = analysis['signal']
            
            if signal not in ['buy', 'sell']:
                return
                
            # Calculate position size
            position_size = self._calculate_position_size(analysis)
            if not position_size:
                return
                
            # Place orders
            if signal == 'buy':
                # Place buy order
                order = self.place_order(
                    symbol=symbol,
                    side='BUY',
                    quantity=position_size
                )
                
                if order and order['status'] == 'FILLED':
                    # Place stop loss and take profit
                    stop_loss = analysis['risk_reward']['stop_loss']
                    take_profit = analysis['risk_reward']['take_profit']
                    
                    self.place_stop_loss(symbol, position_size, stop_loss)
                    self.place_take_profit(symbol, position_size, take_profit)
                    
                    self.logger.info(f"Successfully entered position for {symbol}")
                    
        except Exception as e:
            self.logger.error(f"Failed to execute trade for {symbol}: {str(e)}")

    def _calculate_position_size(self, analysis: Dict) -> float:
        """Calculate position size based on risk management rules"""
        try:
            balance = self.get_account_balance()
            usdt_balance = balance.get('USDT', 0)
            
            # Risk only 1% of account per trade
            risk_amount = usdt_balance * 0.01
            
            # Calculate position size based on stop loss
            current_price = float(self.client.get_symbol_ticker(
                symbol=analysis['symbol'])['price'])
            stop_loss = analysis['risk_reward']['stop_loss']
            
            risk_per_unit = abs(current_price - stop_loss)
            position_size = risk_amount / risk_per_unit
            
            # Ensure position size doesn't exceed max_position_size
            max_position = self.trading_params['max_position_size'] / current_price
            position_size = min(position_size, max_position)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Failed to calculate position size: {str(e)}")
            return None

    def _cleanup(self):
        """Cleanup function for bot shutdown"""
        try:
            # Cancel all open orders
            open_orders = self.client.get_open_orders()
            for order in open_orders:
                self.client.cancel_order(
                    symbol=order['symbol'],
                    orderId=order['orderId']
                )
                
            self.logger.info("Cleaned up all open orders")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


if __name__ == "__main__":
    # Initialize and start the bot
    bot = DecryptoBot()
    
    # Create and run event loop
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()