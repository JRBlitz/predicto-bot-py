"""
Base strategy class for implementing trading strategies.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from loguru import logger
import asyncio


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str, client, config: Dict[str, Any]):
        """Initialize the strategy."""
        self.name = name
        self.client = client
        self.config = config
        self.is_running = False
        self.total_trades = 0
        self.pnl = 0.0
        
    @abstractmethod
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data and return trading signals."""
        pass
    
    @abstractmethod
    async def should_enter_position(self, signals: Dict[str, Any]) -> bool:
        """Determine if we should enter a position."""
        pass
    
    @abstractmethod
    async def should_exit_position(self, position: Dict[str, Any], signals: Dict[str, Any]) -> bool:
        """Determine if we should exit a position."""
        pass
    
    @abstractmethod
    async def calculate_position_size(self, signals: Dict[str, Any]) -> float:
        """Calculate the position size for the trade."""
        pass
    
    async def execute_trade(self, market_id: str, side: str, size: float, price: float):
        """Execute a trade."""
        try:
            logger.info(f"Executing {side} order: {size} @ {price} for {market_id}")
            
            # Convert to string format required by Polymarket
            size_str = str(size)
            price_str = str(price)
            
            result = await self.client.place_order(
                market_id=market_id,
                side=side,
                size=size_str,
                price=price_str
            )
            
            if result:
                logger.success(f"Trade executed successfully: {result}")
                self.total_trades += 1
                
                # Update web server with trade info
                try:
                    from ..ui.web_server import web_server
                    web_server.update_bot_status(
                        total_trades=self.total_trades,
                        pnl=self.pnl
                    )
                except ImportError:
                    pass  # Web server not available
            else:
                logger.error("Trade execution failed")
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def run(self):
        """Main strategy loop."""
        self.is_running = True
        crypto_mode = self.config.get('crypto_optimized', False)
        logger.info(f"Starting strategy: {self.name} (Crypto-optimized: {crypto_mode})")
        
        while self.is_running:
            try:
                # Get available markets
                markets = await self.client.get_markets()
                
                # Handle different market data structures
                if isinstance(markets, list) and len(markets) > 0:
                    # Take first few markets for testing, more for crypto-optimized
                    market_limit = 10 if crypto_mode else 5
                    markets_to_process = markets[:market_limit]
                    logger.info(f"Processing {len(markets_to_process)} markets (crypto-optimized: {crypto_mode})")
                elif isinstance(markets, dict):
                    market_limit = 10 if crypto_mode else 5
                    markets_to_process = [markets[key] for key in list(markets.keys())[:market_limit]]
                    logger.info(f"Processing {len(markets_to_process)} markets (crypto-optimized: {crypto_mode})")
                else:
                    logger.warning("No markets available or unexpected format")
                    await asyncio.sleep(self.config.get('interval', 60))
                    continue
                
                for market in markets_to_process:
                    try:
                        # Handle different market ID formats
                        if isinstance(market, dict):
                            market_id = market.get('id') or market.get('market_id') or market.get('token_id')
                        else:
                            logger.warning(f"Unexpected market format: {type(market)}")
                            continue
                        
                        if not market_id:
                            logger.warning("Market missing ID, skipping")
                            continue
                        
                        # Get market data
                        orderbook = await self.client.get_orderbook(market_id)
                    
                        # Analyze market
                        signals = await self.analyze_market({
                            'market': market,
                            'orderbook': orderbook
                        })
                        
                        # Check for entry signals
                        if await self.should_enter_position(signals):
                            size = await self.calculate_position_size(signals)
                            trade_price = signals.get('price', signals.get('current_price', 0.5))
                            await self.execute_trade(
                                market_id=market_id,
                                side=signals.get('side', 'BUY'),
                                size=size,
                                price=trade_price
                            )
                        
                        # Check for exit signals
                        positions = await self.client.get_positions()
                        for position in positions:
                            if position.get('market_id') == market_id:
                                if await self.should_exit_position(position, signals):
                                    exit_price = signals.get('exit_price', signals.get('current_price', 0.5))
                                    await self.execute_trade(
                                        market_id=market_id,
                                        side='SELL' if position.get('side') == 'BUY' else 'BUY',
                                        size=abs(float(position.get('size', 0))),
                                        price=exit_price
                                    )
                    
                    except Exception as market_error:
                        logger.error(f"Error processing market {market_id if 'market_id' in locals() else 'unknown'}: {market_error}")
                        continue
                
                # Wait before next iteration
                await asyncio.sleep(self.config.get('interval', 60))
                
            except Exception as e:
                logger.error(f"Error in strategy loop: {e}")
                await asyncio.sleep(10)
    
    def stop(self):
        """Stop the strategy."""
        self.is_running = False
        logger.info(f"Stopping strategy: {self.name}")
