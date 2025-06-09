import asyncio
import websockets
import json
import requests

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe to symbols
        await websocket.send(json.dumps({
            "type": "subscribe",
            "symbols": ["AAPL", "MSFT", "SPY"]
        }))
        
        # Listen for messages
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Received: {data}")

def test_rest_api():
    # Test signal endpoint
    response = requests.post(
        "http://localhost:8000/api/signals",
        json={"symbols": ["AAPL", "MSFT"]},
        headers={"Authorization": "Bearer test_token"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    # Test REST API
    print("Testing REST API...")
    test_rest_api()
    
    # Test WebSocket
    print("\nTesting WebSocket...")
    asyncio.run(test_websocket())