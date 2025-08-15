"""
Polymarket CLOB client wrapper for trading operations.
"""
from typing import Dict, List, Optional, Any
import asyncio
from loguru import logger
# Simplified settings for crypto-only mode
import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    crypto_only_mode = True
    allowed_market_keywords = [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
        "blockchain", "defi", "solana", "sol", "xrp", "ripple"
    ]

settings = Settings()

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
        self.crypto_only_mode = settings.crypto_only_mode
        self.allowed_keywords = [kw.lower() for kw in settings.allowed_market_keywords]
        
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
    
    def _is_crypto_market(self, market: Dict[str, Any]) -> bool:
        """Check if a market is crypto-related based on keywords."""
        if not self.crypto_only_mode:
            return True
            
        # Extract market information for keyword matching
        market_text = ""
        
        # Check various fields that might contain market description
        for field in ['question', 'title', 'description', 'market_slug', 'slug']:
            if field in market and market[field]:
                market_text += f" {market[field]}".lower()
        
        # Check if any crypto keywords are present
        return any(keyword in market_text for keyword in self.allowed_keywords)
    
    async def get_markets(self) -> List[Dict[str, Any]]:
        """Get available markets, filtered for crypto if crypto_only_mode is enabled."""
        try:
            # Get markets from Polymarket API
            markets = self.client.get_markets()
            if hasattr(markets, '__await__'):
                markets = await markets
            
            logger.info(f"Raw markets response type: {type(markets)}")
            
            if not markets:
                logger.warning("No markets returned from API")
                return []
            
            # Handle different response formats
            market_list = []
            
            if isinstance(markets, list):
                market_list = markets
            elif isinstance(markets, dict):
                # Convert dict to list of market objects
                if 'data' in markets:
                    market_list = markets['data'] if isinstance(markets['data'], list) else []
                elif 'markets' in markets:
                    market_list = markets['markets'] if isinstance(markets['markets'], list) else []
                else:
                    # Assume dict values are market objects
                    market_list = list(markets.values())
            else:
                logger.warning(f"Unexpected markets format: {type(markets)}")
                return []
            
            logger.info(f"Processing {len(market_list)} markets")
            
            # Apply crypto filtering if enabled
            if self.crypto_only_mode and market_list:
                filtered_markets = []
                for market in market_list:
                    if isinstance(market, dict) and self._is_crypto_market(market):
                        filtered_markets.append(market)
                
                logger.info(f"Filtered {len(market_list)} markets down to {len(filtered_markets)} crypto markets")
                return filtered_markets
            
            return market_list
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
