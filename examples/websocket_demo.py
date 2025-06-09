"""
WebSocket Demo - Real-time Trading Signals
Shows how to connect and receive live trading signals
"""

import asyncio
import websockets
import json
from datetime import datetime

async def receive_trading_signals():
    """Connect to the trading signal WebSocket and receive real-time signals"""
    
    uri = "ws://localhost:8000/ws"
    
    print("🔌 Connecting to Trading Signal System...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected! Subscribing to symbols...")
            
            # Subscribe to specific symbols
            subscribe_message = {
                "type": "subscribe",
                "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA"]
            }
            
            await websocket.send(json.dumps(subscribe_message))
            print(f"📊 Subscribed to: {', '.join(subscribe_message['symbols'])}")
            print("\n⏳ Waiting for signals...\n")
            
            # Receive and process signals
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Handle different message types
                    if data.get('type') == 'signal':
                        display_signal(data)
                    elif data.get('type') == 'enhanced_signal':
                        display_enhanced_signal(data)
                    elif data.get('type') == 'metrics':
                        display_metrics(data)
                    else:
                        print(f"ℹ️  {data.get('type', 'Unknown')}: {data}")
                        
                except websockets.exceptions.ConnectionClosed:
                    print("❌ Connection closed")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("Make sure the API is running at http://localhost:8000")

def display_signal(data):
    """Display a trading signal in a formatted way"""
    signal = data.get('signal', {})
    symbol = data.get('symbol', 'Unknown')
    timestamp = datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
    
    # Determine emoji based on action
    action = signal.get('action', 'HOLD')
    emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
    
    print(f"\n{emoji} {symbol} - {action} Signal")
    print(f"   Confidence: {signal.get('confidence', 0)*100:.1f}%")
    print(f"   Entry: ${signal.get('current_price', 0):.2f}")
    print(f"   Stop Loss: ${signal.get('stop_loss', 0):.2f}")
    print(f"   Take Profit: ${signal.get('take_profit', 0):.2f}")
    print(f"   Position Size: {signal.get('position_size', 0)*100:.1f}%")
    print(f"   Time: {timestamp.strftime('%H:%M:%S')}")

def display_enhanced_signal(data):
    """Display an enhanced signal with sentiment data"""
    display_signal(data)
    
    sentiment = data.get('sentiment', {})
    if sentiment:
        print(f"   Sentiment Score: {sentiment.get('overall_score', 0):.0f}")
        warnings = data.get('warnings', [])
        if warnings:
            print(f"   ⚠️  Warnings: {', '.join(warnings)}")

def display_metrics(data):
    """Display portfolio metrics"""
    metrics = data.get('data', {})
    print(f"\n📊 Portfolio Update:")
    print(f"   Active Signals: {metrics.get('total_active_signals', 0)}")
    print(f"   Buy Signals: {metrics.get('buy_signals', 0)}")
    print(f"   Sell Signals: {metrics.get('sell_signals', 0)}")
    print(f"   Risk Status: {metrics.get('risk_status', 'NORMAL')}")

async def send_test_commands(websocket):
    """Example of sending commands to the WebSocket"""
    # Example: Request specific symbol update
    request_update = {
        "type": "request_update",
        "symbol": "AAPL"
    }
    await websocket.send(json.dumps(request_update))
    
    # Example: Change subscription
    change_subscription = {
        "type": "subscribe",
        "symbols": ["SPY", "QQQ", "NVDA"]
    }
    await websocket.send(json.dumps(change_subscription))

if __name__ == "__main__":
    print("="*50)
    print("Quantum Trading Signals - WebSocket Demo")
    print("="*50)
    print("\nThis demo will connect to the WebSocket and display")
    print("real-time trading signals as they are generated.\n")
    
    try:
        # Run the WebSocket client
        asyncio.run(receive_trading_signals())
    except KeyboardInterrupt:
        print("\n\n👋 Disconnected. Goodbye!")