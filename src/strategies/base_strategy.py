"""
Base strategy class for implementing trading strategies.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from loguru import logger


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str, client, config: Dict[str, Any]):
        """Initialize the strategy."""
        self.name = name
        self.client = client
        self.config = config
        self.is_running = False
        
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
            else:
                logger.error("Trade execution failed")
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def run(self):
        """Main strategy loop."""
        self.is_running = True
        logger.info(f"Starting strategy: {self.name}")
        
        while self.is_running:
            try:
                # Get available markets
                markets = await self.client.get_markets()
                
                for market in markets:
                    # Get market data
                    orderbook = await self.client.get_orderbook(market['id'])
                    
                    # Analyze market
                    signals = await self.analyze_market({
                        'market': market,
                        'orderbook': orderbook
                    })
                    
                    # Check for entry signals
                    if await self.should_enter_position(signals):
                        size = await self.calculate_position_size(signals)
                        await self.execute_trade(
                            market_id=market['id'],
                            side=signals.get('side', 'BUY'),
                            size=size,
                            price=signals.get('price', 0.5)
                        )
                    
                    # Check for exit signals
                    positions = await self.client.get_positions()
                    for position in positions:
                        if position['market_id'] == market['id']:
                            if await self.should_exit_position(position, signals):
                                await self.execute_trade(
                                    market_id=market['id'],
                                    side='SELL' if position['side'] == 'BUY' else 'BUY',
                                    size=abs(float(position['size'])),
                                    price=signals.get('exit_price', 0.5)
                                )
                
                # Wait before next iteration
                await asyncio.sleep(self.config.get('interval', 60))
                
            except Exception as e:
                logger.error(f"Error in strategy loop: {e}")
                await asyncio.sleep(10)
    
    def stop(self):
        """Stop the strategy."""
        self.is_running = False
        logger.info(f"Stopping strategy: {self.name}")


# Import asyncio for the sleep function
import asyncio
