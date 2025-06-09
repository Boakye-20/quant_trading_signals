import requests

print("Starting API test...")

try:
    # Test root endpoint
    print("\n1. Testing root endpoint...")
    response = requests.get("http://localhost:8000/", timeout=5)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")

try:
    # Test config endpoint
    print("\n2. Testing config endpoint...")
    response = requests.get("http://localhost:8000/test-config", timeout=5)
    print(f"   Config: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")

print("\nTest complete!")
# Add this to test_simple.py after the existing code

# Test signals endpoint (will fail without auth)
print("\n3. Testing signals endpoint...")
try:
    response = requests.post(
        "http://localhost:8000/api/signals",
        json={"symbols": ["AAPL", "MSFT"], "lookback_days": 30},
        headers={"Authorization": "Bearer test_token"}
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Signals: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")