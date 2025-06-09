"""
Market Data Collector Module - Public Demo Version
Simulates market data collection for demonstration purposes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from collections import deque

from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataCollector:
    """
    Simulates market data collection for demonstration
    
    Production version includes:
    - Real-time WebSocket connections to multiple exchanges
    - Integration with Alpaca, Polygon, Finnhub APIs
    - Level 2 order book data
    - Tick-by-tick trade data
    - Options flow analysis
    """
    
    def __init__(self):
        """Initialize the data collector in demo mode"""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Data storage
        self.order_books = {}
        self.trades = {}
        self.quotes = {}
        
        # Demo data buffers
        self.trade_buffer = deque(maxlen=1000)
        self.quote_buffer = deque(maxlen=1000)
        
        self.logger.info("MarketDataCollector initialized in demo mode")
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Generate simulated historical OHLCV data
        
        Production version connects to:
        - Alpaca Markets API
        - Yahoo Finance
        - Polygon.io
        With automatic failover between sources
        """
        self.logger.info(f"Generating demo data for {symbol} ({days} days)")
        
        # Generate realistic demo data
        df = self.generate_demo_data(symbol, days)
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        return df
    
    def generate_demo_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate realistic demo market data"""
        
        # Base prices for different symbols
        base_prices = {
            'SPY': 450.0,
            'QQQ': 380.0,
            'AAPL': 185.0,
            'MSFT': 370.0,
            'GOOGL': 140.0,
            'TSLA': 250.0,
            'JPM': 150.0,
            'BAC': 35.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Generate hourly data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Generate realistic price movement
        np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol
        
        prices = []
        current_price = base_price
        
        for i, timestamp in enumerate(date_range):
            # Market hours effect
            hour = timestamp.hour
            if 9 <= hour <= 16:  # Market hours
                volatility = 0.002
                volume_multiplier = 1.0
            else:
                volatility = 0.0005
                volume_multiplier = 0.1
            
            # Price movement
            returns = np.random.normal(0, volatility)
            current_price *= (1 + returns)
            
            # OHLC generation
            high = current_price * (1 + abs(np.random.normal(0, volatility)))
            low = current_price * (1 - abs(np.random.normal(0, volatility)))
            open_price = current_price * (1 + np.random.normal(0, volatility * 0.5))
            
            # Volume
            base_volume = 1000000
            volume = int(base_volume * volume_multiplier * np.random.lognormal(0, 0.3))
            
            prices.append({
                'Open': open_price,
                'High': max(high, open_price, current_price),
                'Low': min(low, open_price, current_price),
                'Close': current_price,
                'Volume': volume
            })
        
        return pd.DataFrame(prices, index=date_range)
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators"""
        
        # Returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(100).mean()
        
        # Volume features
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['obv'] = (df['Volume'] * np.sign(df['returns'])).cumsum()
        
        # Fill NaN values
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        
        return df
    
    def get_market_microstructure_data(self, symbol: str) -> Dict:
        """
        Get simulated market microstructure metrics
        
        Production version calculates from real order book data:
        - Real-time order book imbalance
        - Actual trade flow toxicity (VPIN)
        - Live bid-ask spreads
        - Market maker participation
        """
        
        # Simulate realistic microstructure data
        base_spread = 0.0002 if symbol in ['SPY', 'QQQ'] else 0.0005
        
        return {
            'symbol': symbol,
            'order_book_imbalance': np.random.normal(0.05, 0.3),
            'trade_flow_toxicity': min(1.0, max(0.0, np.random.beta(2, 5))),
            'spread': base_spread * (1 + np.random.exponential(0.5)),
            'timestamp': datetime.now().isoformat()
        }
    
    def simulate_real_time_update(self, symbol: str) -> Dict:
        """
        Simulate a market data update
        
        Production version receives:
        - Real-time WebSocket feeds
        - Nanosecond-precision timestamps
        - Full order book updates
        - Trade-by-trade data
        """
        
        if symbol not in self.quotes:
            self.quotes[symbol] = {'price': 100.0}
        
        current_price = self.quotes[symbol]['price']
        
        # Simulate price movement
        returns = np.random.normal(0, 0.0001)
        new_price = current_price * (1 + returns)
        
        # Create quote
        spread = 0.0002
        quote = {
            'symbol': symbol,
            'bid': new_price * (1 - spread/2),
            'ask': new_price * (1 + spread/2),
            'bid_size': int(np.random.lognormal(7, 1)),
            'ask_size': int(np.random.lognormal(7, 1)),
            'timestamp': datetime.now().isoformat()
        }
        
        self.quote_buffer.append(quote)
        self.quotes[symbol] = {'price': new_price}
        
        return quote