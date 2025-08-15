"""
Live Polymarket API client for real-time market data.
Integrates with Polymarket's API to fetch live market data for crypto predictions.
"""
import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from loguru import logger
import websockets
import ssl


@dataclass
class LiveMarketData:
    """Live market data structure."""
    market_id: str
    timestamp: datetime
    best_bid: float
    best_ask: float
    mid_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume_24h: float
    price_change_24h: float
    active: bool
    metadata: Dict[str, Any]


@dataclass
class OrderbookSnapshot:
    """Orderbook snapshot structure."""
    market_id: str
    timestamp: datetime
    bids: List[Dict[str, float]]  # [{'price': float, 'size': float}]
    asks: List[Dict[str, float]]  # [{'price': float, 'size': float}]
    spread: float
    mid_price: float


class LivePolymarketClient:
    """
    Live Polymarket API client with WebSocket support for real-time data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the live Polymarket client."""
        self.api_key = api_key
        self.base_url = "https://clob.polymarket.com"
        self.ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        
        # Connection management
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        
        # Data storage
        self.market_data: Dict[str, LiveMarketData] = {}
        self.orderbooks: Dict[str, OrderbookSnapshot] = {}
        self.subscribed_markets: set = set()
        
        # Callbacks
        self.data_callbacks: List[Callable[[LiveMarketData], None]] = []
        self.orderbook_callbacks: List[Callable[[OrderbookSnapshot], None]] = []
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self):
        """Establish connection to Polymarket API."""
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Test API connection
            await self._test_connection()
            
            self.is_connected = True
            logger.success("Connected to Polymarket API")
            
        except Exception as e:
            logger.error(f"Failed to connect to Polymarket API: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Polymarket API."""
        try:
            # Close WebSocket connection
            if self.ws_connection:
                await self.ws_connection.close()
                self.ws_connection = None
            
            # Close HTTP session
            if self.session:
                await self.session.close()
                self.session = None
            
            self.is_connected = False
            logger.info("Disconnected from Polymarket API")
            
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
    
    async def _test_connection(self):
        """Test API connection."""
        try:
            async with self.session.get(f"{self.base_url}/markets") as response:
                if response.status != 200:
                    raise Exception(f"API test failed with status {response.status}")
                
                data = await response.json()
                logger.info(f"API test successful, found {len(data.get('data', []))} markets")
                
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            raise
    
    async def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def get_market_info(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Get market information."""
        if not self.is_connected:
            raise Exception("Not connected to Polymarket API")
        
        try:
            await self._rate_limit()
            
            async with self.session.get(f"{self.base_url}/markets/{market_id}") as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    logger.warning(f"Market {market_id} not found")
                    return None
                else:
                    logger.error(f"Failed to get market info: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting market info for {market_id}: {e}")
            return None
    
    async def search_crypto_markets(self) -> List[Dict[str, Any]]:
        """Search for crypto-related markets on Polymarket."""
        if not self.is_connected:
            raise Exception("Not connected to Polymarket API")
        
        crypto_keywords = [
            "bitcoin", "btc", "ethereum", "eth", "xrp", "ripple", 
            "solana", "sol", "crypto", "cryptocurrency"
        ]
        
        all_markets = []
        
        try:
            # Get all markets
            await self._rate_limit()
            async with self.session.get(f"{self.base_url}/markets") as response:
                if response.status == 200:
                    data = await response.json()
                    markets = data.get('data', [])
                    
                    # Filter for crypto-related markets
                    for market in markets:
                        market_text = (market.get('question', '') + ' ' + 
                                     market.get('description', '')).lower()
                        
                        if any(keyword in market_text for keyword in crypto_keywords):
                            all_markets.append({
                                'id': market.get('id'),
                                'question': market.get('question'),
                                'description': market.get('description'),
                                'active': market.get('active', False),
                                'end_date': market.get('end_date'),
                                'volume': market.get('volume', 0)
                            })
                    
                    logger.info(f"Found {len(all_markets)} crypto-related markets")
                    return all_markets
                    
        except Exception as e:
            logger.error(f"Error searching crypto markets: {e}")
            return []
    
    async def get_orderbook(self, market_id: str) -> Optional[OrderbookSnapshot]:
        """Get current orderbook for a market."""
        if not self.is_connected:
            raise Exception("Not connected to Polymarket API")
        
        try:
            await self._rate_limit()
            
            async with self.session.get(f"{self.base_url}/book/{market_id}") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse orderbook data
                    bids = [{'price': float(bid['price']), 'size': float(bid['size'])} 
                           for bid in data.get('bids', [])]
                    asks = [{'price': float(ask['price']), 'size': float(ask['size'])} 
                           for ask in data.get('asks', [])]
                    
                    # Calculate spread and mid price
                    best_bid = bids[0]['price'] if bids else 0
                    best_ask = asks[0]['price'] if asks else 1
                    spread = best_ask - best_bid
                    mid_price = (best_bid + best_ask) / 2
                    
                    return OrderbookSnapshot(
                        market_id=market_id,
                        timestamp=datetime.now(),
                        bids=bids,
                        asks=asks,
                        spread=spread,
                        mid_price=mid_price
                    )
                    
        except Exception as e:
            logger.error(f"Error getting orderbook for {market_id}: {e}")
            return None
    
    async def get_market_data(self, market_id: str) -> Optional[LiveMarketData]:
        """Get live market data."""
        if not self.is_connected:
            raise Exception("Not connected to Polymarket API")
        
        try:
            # Get market info and orderbook
            market_info, orderbook = await asyncio.gather(
                self.get_market_info(market_id),
                self.get_orderbook(market_id),
                return_exceptions=True
            )
            
            if isinstance(market_info, Exception) or isinstance(orderbook, Exception):
                logger.error(f"Error getting data for {market_id}")
                return None
            
            if not market_info or not orderbook:
                return None
            
            # Create market data object
            market_data = LiveMarketData(
                market_id=market_id,
                timestamp=datetime.now(),
                best_bid=orderbook.bids[0]['price'] if orderbook.bids else 0,
                best_ask=orderbook.asks[0]['price'] if orderbook.asks else 1,
                mid_price=orderbook.mid_price,
                bid_size=orderbook.bids[0]['size'] if orderbook.bids else 0,
                ask_size=orderbook.asks[0]['size'] if orderbook.asks else 0,
                last_price=market_info.get('last_price', orderbook.mid_price),
                volume_24h=market_info.get('volume_24h', 0),
                price_change_24h=market_info.get('price_change_24h', 0),
                active=market_info.get('active', True),
                metadata=market_info
            )
            
            # Store and notify callbacks
            self.market_data[market_id] = market_data
            self.orderbooks[market_id] = orderbook
            
            for callback in self.data_callbacks:
                try:
                    callback(market_data)
                except Exception as e:
                    logger.warning(f"Data callback error: {e}")
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data for {market_id}: {e}")
            return None
    
    async def start_websocket_stream(self, market_ids: List[str]):
        """Start WebSocket stream for real-time updates."""
        if not market_ids:
            logger.warning("No market IDs provided for WebSocket stream")
            return
        
        try:
            # Create SSL context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Connect to WebSocket
            self.ws_connection = await websockets.connect(
                self.ws_url,
                ssl=ssl_context,
                ping_interval=30,
                ping_timeout=10
            )
            
            # Subscribe to markets
            for market_id in market_ids:
                subscribe_msg = {
                    "type": "subscribe",
                    "market": market_id,
                    "feed": ["book", "trades"]
                }
                await self.ws_connection.send(json.dumps(subscribe_msg))
                self.subscribed_markets.add(market_id)
            
            logger.success(f"WebSocket connected, subscribed to {len(market_ids)} markets")
            
            # Start message handling loop
            asyncio.create_task(self._handle_websocket_messages())
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.ws_connection = None
    
    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages."""
        try:
            while self.ws_connection and not self.ws_connection.closed:
                try:
                    message = await asyncio.wait_for(
                        self.ws_connection.recv(), 
                        timeout=60
                    )
                    
                    data = json.loads(message)
                    await self._process_websocket_message(data)
                    
                except asyncio.TimeoutError:
                    logger.warning("WebSocket message timeout")
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    break
                    
        except Exception as e:
            logger.error(f"WebSocket message handling error: {e}")
        finally:
            self.ws_connection = None
    
    async def _process_websocket_message(self, data: Dict[str, Any]):
        """Process individual WebSocket message."""
        try:
            message_type = data.get('type')
            market_id = data.get('market')
            
            if message_type == 'book' and market_id:
                # Update orderbook
                await self._update_orderbook_from_ws(market_id, data)
            elif message_type == 'trade' and market_id:
                # Update last price from trade
                await self._update_price_from_trade(market_id, data)
                
        except Exception as e:
            logger.warning(f"Error processing WebSocket message: {e}")
    
    async def _update_orderbook_from_ws(self, market_id: str, data: Dict[str, Any]):
        """Update orderbook from WebSocket data."""
        try:
            bids = [{'price': float(bid[0]), 'size': float(bid[1])} 
                   for bid in data.get('bids', [])]
            asks = [{'price': float(ask[0]), 'size': float(ask[1])} 
                   for ask in data.get('asks', [])]
            
            if bids and asks:
                best_bid = bids[0]['price']
                best_ask = asks[0]['price']
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2
                
                orderbook = OrderbookSnapshot(
                    market_id=market_id,
                    timestamp=datetime.now(),
                    bids=bids,
                    asks=asks,
                    spread=spread,
                    mid_price=mid_price
                )
                
                self.orderbooks[market_id] = orderbook
                
                # Notify callbacks
                for callback in self.orderbook_callbacks:
                    try:
                        callback(orderbook)
                    except Exception as e:
                        logger.warning(f"Orderbook callback error: {e}")
                        
        except Exception as e:
            logger.warning(f"Error updating orderbook from WebSocket: {e}")
    
    async def _update_price_from_trade(self, market_id: str, data: Dict[str, Any]):
        """Update price from trade data."""
        try:
            price = float(data.get('price', 0))
            if price > 0 and market_id in self.market_data:
                self.market_data[market_id].last_price = price
                self.market_data[market_id].timestamp = datetime.now()
                
        except Exception as e:
            logger.warning(f"Error updating price from trade: {e}")
    
    def add_data_callback(self, callback: Callable[[LiveMarketData], None]):
        """Add callback for market data updates."""
        self.data_callbacks.append(callback)
    
    def add_orderbook_callback(self, callback: Callable[[OrderbookSnapshot], None]):
        """Add callback for orderbook updates."""
        self.orderbook_callbacks.append(callback)
    
    async def get_historical_data(self, market_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical data for backtesting (simulated for now)."""
        # Note: This is a placeholder since Polymarket doesn't provide historical OHLC data
        # In practice, you would need to collect and store this data over time
        
        logger.warning("Historical data not available from Polymarket API - using simulated data")
        
        # Generate simulated historical data for testing
        historical_data = []
        base_price = 0.5
        
        for i in range(hours):
            timestamp = datetime.now() - timedelta(hours=hours-i)
            price_change = (hash(f"{market_id}_{i}") % 1000 - 500) / 10000  # Random walk
            price = max(0.01, min(0.99, base_price + price_change))
            
            historical_data.append({
                'timestamp': timestamp,
                'open': base_price,
                'high': max(base_price, price),
                'low': min(base_price, price),
                'close': price,
                'volume': abs(hash(f"{market_id}_{i}") % 1000)
            })
            
            base_price = price
        
        return historical_data


# Convenience functions for easy integration
async def get_crypto_markets() -> List[Dict[str, Any]]:
    """Get all available crypto markets."""
    async with LivePolymarketClient() as client:
        return await client.search_crypto_markets()


async def get_live_data(market_ids: List[str]) -> Dict[str, LiveMarketData]:
    """Get live data for multiple markets."""
    async with LivePolymarketClient() as client:
        results = {}
        
        for market_id in market_ids:
            data = await client.get_market_data(market_id)
            if data:
                results[market_id] = data
        
        return results


if __name__ == "__main__":
    async def main():
        """Test the live client."""
        async with LivePolymarketClient() as client:
            # Search for crypto markets
            crypto_markets = await client.search_crypto_markets()
            print(f"Found {len(crypto_markets)} crypto markets")
            
            for market in crypto_markets[:5]:  # Show first 5
                print(f"  • {market['id']}: {market['question']}")
    
    asyncio.run(main())
