import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
    
    # Trading Parameters
    SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM', 'BAC']
    LOOKBACK_DAYS = 30
    UPDATE_FREQUENCY = 60  # seconds
    
    # Model Parameters
    SIGNAL_THRESHOLD = 0.7
    MAX_POSITIONS = 5
    RISK_LIMIT = 0.02  # 2% per trade