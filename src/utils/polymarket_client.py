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
except ImportError as e:
    logger.warning(f"Polymarket libraries not installed: {e}")


class PolymarketTradingClient:
    """Wrapper for Polymarket CLOB trading operations."""
    
    def __init__(self, private_key: str, host: str = "https://clob.polymarket.com", test_mode: bool = True):
        """Initialize the trading client."""
        self.private_key = private_key
        self.host = host
        self.client = None
        self.test_mode = test_mode
        
    async def initialize(self):
        """Initialize the client."""
        try:
            # Initialize CLOB client
            self.client = ClobClient(
                host=self.host,
                key=self.private_key,
                chain_id=POLYGON
            )
            
            logger.info(f"Polymarket trading client initialized successfully (Test Mode: {self.test_mode})")
            
        except Exception as e:
            logger.error(f"Failed to initialize Polymarket client: {e}")
            raise
    
    async def get_markets(self) -> List[Dict[str, Any]]:
        """Get available markets."""
        try:
            # Fix: Use the correct async method
            markets = self.client.get_markets()
            if hasattr(markets, '__await__'):
                markets = await markets
            return markets
        except Exception as e:
            logger.error(f"Failed to get markets: {e}")
            return []
    
    async def get_orderbook(self, market_id: str) -> Dict[str, Any]:
        """Get order book for a specific market."""
        try:
            orderbook = self.client.get_orderbook(market_id)
            if hasattr(orderbook, '__await__'):
                orderbook = await orderbook
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
            if self.test_mode:
                logger.info(f"🧪 TEST MODE: Would place {side} order - {size} @ {price} for market {market_id}")
                return {
                    "test_mode": True,
                    "order_id": f"test_{market_id}_{side}_{size}_{price}",
                    "status": "test_order_placed"
                }
            
            # Create order payload
            order_payload = {
                "market_id": market_id,
                "side": side,
                "size": size,
                "price": price,
                "post_only": post_only
            }
            
            # Place the order
            response = self.client.post_order(order_payload)
            if hasattr(response, '__await__'):
                response = await response
            
            logger.info(f"Order placed successfully: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            if self.test_mode:
                logger.info(f"🧪 TEST MODE: Would cancel order {order_id}")
                return True
            
            response = self.client.delete_order(order_id)
            if hasattr(response, '__await__'):
                response = await response
            
            logger.info(f"Order {order_id} cancelled successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        try:
            positions = self.client.get_positions()
            if hasattr(positions, '__await__'):
                positions = await positions
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get open orders."""
        try:
            orders = self.client.get_orders()
            if hasattr(orders, '__await__'):
                orders = await orders
            return orders
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
