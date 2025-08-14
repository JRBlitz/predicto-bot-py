"""
Polymarket CLOB client wrapper for trading operations.
"""
from typing import Dict, List, Optional, Any
import asyncio
from loguru import logger

# Import Polymarket libraries
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.constants import POLYGON
    from py_order_utils.order_builder.constants import BUY, SELL
    from py_order_utils.order_builder.order_builder import OrderBuilder
    from py_order_utils.order_builder.order_converter import OrderConverter
    from py_order_utils.order_builder.order_parser import OrderParser
except ImportError:
    logger.warning("Polymarket libraries not installed. Install from requirements.txt")


class PolymarketTradingClient:
    """Wrapper for Polymarket CLOB trading operations."""
    
    def __init__(self, private_key: str, host: str = "https://clob.polymarket.com"):
        """Initialize the trading client."""
        self.private_key = private_key
        self.host = host
        self.client = None
        self.order_builder = None
        self.order_converter = None
        self.order_parser = None
        
    async def initialize(self):
        """Initialize the client and order utilities."""
        try:
            # Initialize CLOB client
            self.client = ClobClient(
                host=self.host,
                key=self.private_key,
                chain_id=POLYGON
            )
            
            # Initialize order utilities
            self.order_builder = OrderBuilder()
            self.order_converter = OrderConverter()
            self.order_parser = OrderParser()
            
            logger.info("Polymarket trading client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Polymarket client: {e}")
            raise
    
    async def get_markets(self) -> List[Dict[str, Any]]:
        """Get available markets."""
        try:
            markets = await self.client.get_markets()
            return markets
        except Exception as e:
            logger.error(f"Failed to get markets: {e}")
            return []
    
    async def get_orderbook(self, market_id: str) -> Dict[str, Any]:
        """Get order book for a specific market."""
        try:
            orderbook = await self.client.get_orderbook(market_id)
            return orderbook
        except Exception as e:
            logger.error(f"Failed to get orderbook for {market_id}: {e}")
            return {}
    
    async def place_order(
        self,
        market_id: str,
        side: str,
        size: str,
        price: str,
        post_only: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Place a limit order."""
        try:
            # Build the order
            order = self.order_builder.build_order(
                market_id=market_id,
                side=side,
                size=size,
                price=price,
                post_only=post_only
            )
            
            # Convert to API format
            api_order = self.order_converter.to_api_order(order)
            
            # Place the order
            response = await self.client.post_order(api_order)
            
            logger.info(f"Order placed successfully: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            await self.client.delete_order(order_id)
            logger.info(f"Order {order_id} cancelled successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        try:
            positions = await self.client.get_positions()
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get open orders."""
        try:
            orders = await self.client.get_orders()
            return orders
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
