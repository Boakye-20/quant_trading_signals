import requests
import json
import random
from datetime import datetime

# Generate fake signals for testing
symbols = ['SPY', 'QQQ', 'AAPL']

for symbol in symbols:
    signal = {
        "symbol": symbol,
        "action": random.choice(['BUY', 'SELL', 'HOLD']),
        "score": random.uniform(-1, 1),
        "strength": random.uniform(0.5, 0.9),
        "confidence": random.uniform(0.6, 0.95)
    }
    
    print(f"Generated {signal['action']} signal for {symbol}")
    
    # Send to API
    response = requests.post(
        "http://localhost:8000/api/signals",
        json={"symbols": [symbol]},
        headers={"Authorization": "Bearer test_token"}
    )
    print(f"Response: {response.status_code}")