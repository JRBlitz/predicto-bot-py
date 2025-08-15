"""
Mean reversion trading strategy for Polymarket.
"""
from typing import Dict, List, Any
import numpy as np
from loguru import logger
from .base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy that trades when price deviates from moving average."""
    
    def __init__(self, client, config: Dict[str, Any]):
        """Initialize the mean reversion strategy."""
        super().__init__("Mean Reversion", client, config)
        self.price_history = {}
        self.lookback_period = config.get('lookback_period', 20)
        self.deviation_threshold = config.get('deviation_threshold', 0.05)  # 5%
        self.position_size_pct = config.get('position_size_pct', 0.1)  # 10% of portfolio
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data for mean reversion opportunities."""
        market = market_data['market']
        orderbook = market_data['orderbook']
        
        # Handle different market ID formats
        if isinstance(market, dict):
            market_id = market.get('id') or market.get('market_id') or market.get('token_id')
        else:
            return {'signal': 'HOLD', 'confidence': 0}
        
        if not market_id:
            return {'signal': 'HOLD', 'confidence': 0}
        
        # Calculate current mid price
        if orderbook.get('bids') and orderbook.get('asks'):
            best_bid = float(orderbook['bids'][0]['price'])
            best_ask = float(orderbook['asks'][0]['price'])
            current_price = (best_bid + best_ask) / 2
        else:
            return {'signal': 'HOLD', 'confidence': 0}
        
        # Update price history
        if market_id not in self.price_history:
            self.price_history[market_id] = []
        
        self.price_history[market_id].append(current_price)
        
        # Keep only recent prices
        if len(self.price_history[market_id]) > self.lookback_period:
            self.price_history[market_id] = self.price_history[market_id][-self.lookback_period:]
        
        # Calculate moving average
        if len(self.price_history[market_id]) >= self.lookback_period:
            moving_avg = np.mean(self.price_history[market_id])
            deviation = abs(current_price - moving_avg) / moving_avg
            
            # Determine signal
            if deviation > self.deviation_threshold:
                if current_price < moving_avg:
                    signal = 'BUY'
                    confidence = min(deviation / self.deviation_threshold, 1.0)
                else:
                    signal = 'SELL'
                    confidence = min(deviation / self.deviation_threshold, 1.0)
            else:
                signal = 'HOLD'
                confidence = 0
        else:
            signal = 'HOLD'
            confidence = 0
            moving_avg = current_price
            deviation = 0
        
        return {
            'signal': signal,
            'confidence': confidence,
            'current_price': current_price,
            'moving_avg': moving_avg,
            'deviation': deviation,
            'side': signal if signal != 'HOLD' else None,
            'price': current_price
        }
    
    async def should_enter_position(self, signals: Dict[str, Any]) -> bool:
        """Check if we should enter a position."""
        return (
            signals['signal'] in ['BUY', 'SELL'] and
            signals['confidence'] > 0.7  # 70% confidence threshold
        )
    
    async def should_exit_position(self, position: Dict[str, Any], signals: Dict[str, Any]) -> bool:
        """Check if we should exit a position."""
        # Exit when price returns to moving average
        if signals['deviation'] < self.deviation_threshold * 0.5:
            return True
        
        # Exit on opposite signal
        position_side = position.get('side', 'BUY')
        if position_side == 'BUY' and signals['signal'] == 'SELL':
            return True
        if position_side == 'SELL' and signals['signal'] == 'BUY':
            return True
        
        return False
    
    async def calculate_position_size(self, signals: Dict[str, Any]) -> float:
        """Calculate position size based on confidence and portfolio size."""
        base_size = self.config.get('base_position_size', 100)  # USD
        confidence_multiplier = signals['confidence']
        
        return base_size * confidence_multiplier
