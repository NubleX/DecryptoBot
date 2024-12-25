# listing_monitor.py

import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
import requests
from typing import Dict, List, Optional
import json
import asyncio

class ListingMonitor:
    def __init__(self, client: Client):
        self.client = client
        self.logger = logging.getLogger('DecryptoBot.ListingMonitor')
        
        # Cache for existing symbols to compare against
        self.known_symbols = set()
        
        # Parameters for new listing analysis
        self.listing_params = {
            'volume_threshold': 100000,  # Minimum volume in USDT
            'price_change_threshold': 5.0,  # Maximum initial price change percentage
            'min_trades_threshold': 100,  # Minimum number of trades
            'monitoring_period': 180,  # Seconds to monitor after detection
            'order_book_depth': 20  # Depth of order book to analyze
        }
        
        # Initialize symbol cache
        self._initialize_symbol_cache()

    def _initialize_symbol_cache(self):
        """Initialize cache of existing symbols"""
        try:
            exchange_info = self.client.get_exchange_info()
            self.known_symbols = {s['symbol'] for s in exchange_info['symbols'] 
                                if s['status'] == 'TRADING'}
            self.logger.info(f"Initialized symbol cache with {len(self.known_symbols)} symbols")
        except BinanceAPIException as e:
            self.logger.error(f"Failed to initialize symbol cache: {str(e)}")
            raise

    async def monitor_new_listings(self):
        """Continuously monitor for new listings"""
        while True:
            try:
                new_listings = await self._check_new_listings()
                for listing in new_listings:
                    await self._analyze_new_listing(listing)
                    
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in listing monitor: {str(e)}")
                await asyncio.sleep(30)  # Wait longer on error

    async def _check_new_listings(self) -> List[str]:
        """Check for new trading pairs"""
        try:
            exchange_info = self.client.get_exchange_info()
            current_symbols = {s['symbol'] for s in exchange_info['symbols'] 
                             if s['status'] == 'TRADING'}
            
            new_symbols = current_symbols - self.known_symbols
            
            if new_symbols:
                self.logger.info(f"Detected new symbols: {new_symbols}")
                self.known_symbols = current_symbols
                
            return list(new_symbols)
        except BinanceAPIException as e:
            self.logger.error(f"Failed to check new listings: {str(e)}")
            return []

    async def _analyze_new_listing(self, symbol: str) -> Dict:
        """Analyze a newly listed symbol"""
        try:
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'initial_metrics': await self._get_initial_metrics(symbol),
                'market_depth': await self._analyze_market_depth(symbol),
                'trading_metrics': await self._analyze_trading_metrics(symbol)
            }
            
            # Determine if the listing meets our trading criteria
            analysis['tradeable'] = self._evaluate_listing_opportunity(analysis)
            
            self.logger.info(f"New listing analysis completed: {json.dumps(analysis, indent=2)}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze new listing {symbol}: {str(e)}")
            return None

    async def _get_initial_metrics(self, symbol: str) -> Dict:
        """Get initial metrics for new listing"""
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            return {
                'first_price': float(ticker['openPrice']),
                'current_price': float(ticker['lastPrice']),
                'price_change': float(ticker['priceChangePercent']),
                'volume': float(ticker['volume']),
                'quote_volume': float(ticker['quoteVolume']),
                'trade_count': int(ticker['count'])
            }
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get initial metrics for {symbol}: {str(e)}")
            return None

    async def _analyze_market_depth(self, symbol: str) -> Dict:
        """Analyze market depth for new listing"""
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=self.listing_params['order_book_depth'])
            
            bids = pd.DataFrame(depth['bids'], columns=['price', 'quantity'], dtype=float)
            asks = pd.DataFrame(depth['asks'], columns=['price', 'quantity'], dtype=float)
            
            analysis = {
                'bid_ask_spread': float(asks['price'].iloc[0]) - float(bids['price'].iloc[0]),
                'bid_depth': float(bids['quantity'].sum()),
                'ask_depth': float(asks['quantity'].sum()),
                'bid_ask_ratio': float(bids['quantity'].sum()) / float(asks['quantity'].sum()) if asks['quantity'].sum() > 0 else 0,
                'price_impact_buy': self._calculate_price_impact(asks),
                'price_impact_sell': self._calculate_price_impact(bids)
            }
            
            return analysis
            
        except BinanceAPIException as e:
            self.logger.error(f"Failed to analyze market depth for {symbol}: {str(e)}")
            return None

    def _calculate_price_impact(self, orders: pd.DataFrame, target_volume: float = 10000) -> float:
        """Calculate price impact for a given target volume"""
        cumulative_volume = 0
        for _, order in orders.iterrows():
            cumulative_volume += float(order['price']) * float(order['quantity'])
            if cumulative_volume >= target_volume:
                return (float(order['price']) - float(orders['price'].iloc[0])) / float(orders['price'].iloc[0]) * 100
        return None

    async def _analyze_trading_metrics(self, symbol: str) -> Dict:
        """Analyze trading metrics during initial period"""
        start_time = int(time.time() * 1000)
        end_time = start_time + (self.listing_params['monitoring_period'] * 1000)
        
        trades = []
        async for trade in self._get_historical_trades(symbol, start_time, end_time):
            trades.append(trade)
            
        if not trades:
            return None
            
        df = pd.DataFrame(trades)
        
        return {
            'avg_trade_size': float(df['quantity'].mean()),
            'max_trade_size': float(df['quantity'].max()),
            'trade_frequency': len(trades) / self.listing_params['monitoring_period'],
            'buy_sell_ratio': len(df[df['isBuyerMaker']]) / len(df) if len(df) > 0 else 0
        }

    async def _get_historical_trades(self, symbol: str, start_time: int, end_time: int):
        """Generator for historical trades"""
        try:
            trades = self.client.get_historical_trades(symbol=symbol, limit=1000)
            for trade in trades:
                if int(trade['time']) > end_time:
                    break
                yield {
                    'price': float(trade['price']),
                    'quantity': float(trade['qty']),
                    'time': int(trade['time']),
                    'isBuyerMaker': trade['isBuyerMaker']
                }
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get historical trades for {symbol}: {str(e)}")

    def _evaluate_listing_opportunity(self, analysis: Dict) -> bool:
        """Evaluate if the new listing presents a trading opportunity"""
        if not all([analysis['initial_metrics'], analysis['market_depth'], analysis['trading_metrics']]):
            return False
            
        criteria = [
            # Volume threshold
            analysis['initial_metrics']['quote_volume'] >= self.listing_params['volume_threshold'],
            
            # Price stability
            abs(analysis['initial_metrics']['price_change']) <= self.listing_params['price_change_threshold'],
            
            # Sufficient trading activity
            analysis['initial_metrics']['trade_count'] >= self.listing_params['min_trades_threshold'],
            
            # Healthy market depth
            analysis['market_depth']['bid_ask_ratio'] >= 0.5 and analysis['market_depth']['bid_ask_ratio'] <= 2.0,
            
            # Reasonable price impact
            analysis['market_depth']['price_impact_buy'] <= 2.0 if analysis['market_depth']['price_impact_buy'] else False,
            
            # Active trading
            analysis['trading_metrics']['trade_frequency'] >= 0.5  # At least 1 trade every 2 seconds
        ]
        
        return all(criteria)