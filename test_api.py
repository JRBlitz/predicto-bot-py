import asyncio
from src.utils.polymarket_client import PolymarketTradingClient
from src.config.settings import settings

async def test_api():
    client = PolymarketTradingClient(settings.private_key, test_mode=True)
    await client.initialize()
    
    markets = await client.get_markets()
    print(f"Markets type: {type(markets)}")
    print(f"Markets keys: {list(markets.keys()) if markets else 'None'}")
    
    if markets:
        # Get first market ID
        first_market_id = list(markets.keys())[0]
        print(f"First market ID: {first_market_id}")
        print(f"First market data: {markets[first_market_id]}")
        
        # Test orderbook
        orderbook = await client.get_orderbook(first_market_id)
        print(f"Orderbook type: {type(orderbook)}")
        print(f"Orderbook keys: {list(orderbook.keys()) if orderbook else 'None'}")

if __name__ == "__main__":
    asyncio.run(test_api())
