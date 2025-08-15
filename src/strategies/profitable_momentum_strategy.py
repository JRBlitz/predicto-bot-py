"""
Profitable Momentum Strategy - Designed for Quick Positive Returns
Focus: Simple trend following with quick profit capture
"""
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta
from .base_strategy import BaseStrategy


class ProfitableMomentumStrategy(BaseStrategy):
    """
    Simple momentum strategy optimized for quick positive returns.
    
    Key principles:
    1. Follow trends, don't fight them
    2. Quick profit taking (3-5% targets)
    3. Wide stop losses (let winners run)
    4. Fewer, larger trades
    5. Focus on strongest momentum signals
    """
    
    def __init__(self, client, config: Dict[str, Any]):
        """Initialize the profitable momentum strategy."""
        super().__init__("Profitable Momentum Strategy", client, config)
        
        # Simplified parameters for profitability
        self.momentum_periods = config.get('momentum_periods', 8)  # Shorter for quick signals
        self.momentum_threshold = config.get('momentum_threshold', 0.02)  # 2% momentum required
        self.trend_confirmation_periods = config.get('trend_confirmation_periods', 3)
        
        # Aggressive profit-focused risk management
        self.quick_profit_target = config.get('quick_profit_target', 0.04)  # 4% quick profit
        self.extended_profit_target = config.get('extended_profit_target', 0.08)  # 8% extended target
        self.stop_loss_pct = config.get('stop_loss_pct', 0.06)  # 6% stop (wider to avoid whipsaws)
        
        # Position sizing for profitability
        self.base_position_size = config.get('base_position_size', 100)  # $100 base
        self.max_position_size = config.get('max_position_size', 300)  # $300 max
        self.confidence_multiplier = config.get('confidence_multiplier', 2.0)
        
        # Trade management
        self.min_hold_time_minutes = config.get('min_hold_time_minutes', 30)  # Hold minimum 30 min
        self.max_positions = config.get('max_positions', 3)  # Limit concurrent positions
        
        # Data storage
        self.price_history = {}
        self.volume_history = {}
        self.position_entry_info = {}
        self.last_signal_time = {}
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified market analysis focused on momentum and profitability."""
        market = market_data['market']
        orderbook = market_data['orderbook']
        
        # Get market ID and current price
        market_id = self._get_market_id(market)
        current_price = self._get_current_price(orderbook)
        
        if not market_id or current_price is None:
            return {'signal': 'HOLD', 'confidence': 0}
        
        # Update price history
        self._update_price_history(market_id, current_price)
        
        # Need minimum data for analysis
        if len(self.price_history.get(market_id, [])) < self.momentum_periods:
            return {'signal': 'HOLD', 'confidence': 0}
        
        # Simple but effective momentum analysis
        momentum_signal = self._calculate_momentum_signal(market_id, current_price)
        trend_signal = self._calculate_trend_signal(market_id, current_price)
        volume_signal = self._calculate_volume_signal(market_id)
        
        # Combine signals (simple approach)
        final_signal = self._combine_signals(momentum_signal, trend_signal, volume_signal, market_id, current_price)
        
        return final_signal
    
    def _get_market_id(self, market: Dict[str, Any]) -> str:
        """Extract market ID."""
        return market.get('id') or market.get('market_id') or market.get('token_id')
    
    def _get_current_price(self, orderbook: Dict[str, Any]) -> float:
        """Get current mid price."""
        if not orderbook.get('bids') or not orderbook.get('asks'):
            return None
        
        try:
            best_bid = float(orderbook['bids'][0]['price'])
            best_ask = float(orderbook['asks'][0]['price'])
            return (best_bid + best_ask) / 2
        except (IndexError, ValueError, KeyError):
            return None
    
    def _update_price_history(self, market_id: str, price: float):
        """Update price history efficiently."""
        if market_id not in self.price_history:
            self.price_history[market_id] = []
        
        self.price_history[market_id].append({
            'price': price,
            'timestamp': datetime.now()
        })
        
        # Keep only recent data (efficient)
        max_history = self.momentum_periods * 2
        if len(self.price_history[market_id]) > max_history:
            self.price_history[market_id] = self.price_history[market_id][-max_history:]
    
    def _calculate_momentum_signal(self, market_id: str, current_price: float) -> Dict[str, Any]:
        """Calculate simple but effective momentum signal."""
        prices = [p['price'] for p in self.price_history[market_id]]
        
        if len(prices) < self.momentum_periods:
            return {'signal': 'NEUTRAL', 'strength': 0}
        
        # Simple momentum calculation
        start_price = prices[-self.momentum_periods]
        momentum = (current_price - start_price) / start_price
        
        # Recent momentum (last 3 periods)
        if len(prices) >= 3:
            recent_start = prices[-3]
            recent_momentum = (current_price - recent_start) / recent_start
        else:
            recent_momentum = momentum
        
        # Combined momentum score (weight recent more)
        combined_momentum = (momentum * 0.6) + (recent_momentum * 0.4)
        
        # Clear signal generation
        if combined_momentum > self.momentum_threshold:
            signal = 'BUY'
            strength = min(abs(combined_momentum) / self.momentum_threshold, 3.0)
        elif combined_momentum < -self.momentum_threshold:
            signal = 'SELL' 
            strength = min(abs(combined_momentum) / self.momentum_threshold, 3.0)
        else:
            signal = 'NEUTRAL'
            strength = 0
        
        return {
            'signal': signal,
            'strength': strength,
            'momentum': combined_momentum,
            'recent_momentum': recent_momentum
        }
    
    def _calculate_trend_signal(self, market_id: str, current_price: float) -> Dict[str, Any]:
        """Calculate trend confirmation signal."""
        prices = [p['price'] for p in self.price_history[market_id]]
        
        if len(prices) < self.trend_confirmation_periods:
            return {'signal': 'NEUTRAL', 'strength': 0}
        
        # Check if recent prices are trending in same direction
        recent_prices = prices[-self.trend_confirmation_periods:]
        
        # Count consecutive moves in same direction
        up_moves = 0
        down_moves = 0
        
        for i in range(1, len(recent_prices)):
            if recent_prices[i] > recent_prices[i-1]:
                up_moves += 1
            elif recent_prices[i] < recent_prices[i-1]:
                down_moves += 1
        
        # Trend strength based on consistency
        total_moves = len(recent_prices) - 1
        
        if up_moves > down_moves and up_moves >= total_moves * 0.6:
            signal = 'UP'
            strength = up_moves / total_moves
        elif down_moves > up_moves and down_moves >= total_moves * 0.6:
            signal = 'DOWN'
            strength = down_moves / total_moves
        else:
            signal = 'NEUTRAL'
            strength = 0
        
        return {
            'signal': signal,
            'strength': strength,
            'consistency': max(up_moves, down_moves) / total_moves if total_moves > 0 else 0
        }
    
    def _calculate_volume_signal(self, market_id: str) -> Dict[str, Any]:
        """Simple volume signal (placeholder for now)."""
        # For now, return neutral - can be enhanced with real volume data
        return {
            'signal': 'NEUTRAL',
            'strength': 1.0  # Don't penalize for lack of volume data
        }
    
    def _combine_signals(self, momentum_signal: Dict[str, Any], trend_signal: Dict[str, Any], 
                        volume_signal: Dict[str, Any], market_id: str, current_price: float) -> Dict[str, Any]:
        """Combine signals with profit-focused logic."""
        
        # Check for signal alignment (key for profitability)
        momentum_dir = momentum_signal['signal']
        trend_dir = trend_signal['signal']
        
        # Only trade when momentum and trend align
        if momentum_dir == 'BUY' and trend_dir == 'UP':
            final_signal = 'BUY'
            confidence = (momentum_signal['strength'] * 0.7 + trend_signal['strength'] * 0.3)
        elif momentum_dir == 'SELL' and trend_dir == 'DOWN':
            final_signal = 'SELL'
            confidence = (momentum_signal['strength'] * 0.7 + trend_signal['strength'] * 0.3)
        else:
            final_signal = 'HOLD'
            confidence = 0
        
        # Boost confidence for strong signals
        if confidence > 2.0:
            confidence = min(confidence, 1.0)  # Cap at 1.0
        else:
            confidence = confidence / 2.0  # Scale down weaker signals
        
        # Minimum confidence threshold for trading
        min_confidence = 0.6
        if confidence < min_confidence:
            final_signal = 'HOLD'
            confidence = 0
        
        # Anti-overtrading: Check time since last signal
        if self._too_soon_to_trade(market_id):
            final_signal = 'HOLD'
            confidence = 0
        
        return {
            'signal': final_signal,
            'side': final_signal if final_signal != 'HOLD' else None,
            'confidence': confidence,
            'price': current_price,
            'exit_price': current_price,
            'market_id': market_id,
            'momentum_strength': momentum_signal['strength'],
            'trend_consistency': trend_signal.get('consistency', 0),
            'raw_momentum': momentum_signal.get('momentum', 0)
        }
    
    def _too_soon_to_trade(self, market_id: str) -> bool:
        """Prevent overtrading by enforcing minimum time between signals."""
        if market_id not in self.last_signal_time:
            return False
        
        time_since_last = datetime.now() - self.last_signal_time[market_id]
        return time_since_last.total_seconds() < (self.min_hold_time_minutes * 60)
    
    async def should_enter_position(self, signals: Dict[str, Any]) -> bool:
        """Enhanced entry logic focused on profitability."""
        if signals['signal'] not in ['BUY', 'SELL']:
            return False
        
        # High confidence threshold for entries
        if signals['confidence'] < 0.7:
            return False
        
        # Limit number of concurrent positions
        current_positions = len(self.position_entry_info)
        if current_positions >= self.max_positions:
            return False
        
        # Strong momentum requirement
        if abs(signals.get('raw_momentum', 0)) < self.momentum_threshold * 1.2:
            return False
        
        # Update last signal time
        market_id = signals.get('market_id')
        if market_id:
            self.last_signal_time[market_id] = datetime.now()
        
        return True
    
    async def should_exit_position(self, position: Dict[str, Any], signals: Dict[str, Any]) -> bool:
        """Profit-focused exit logic."""
        market_id = signals.get('market_id')
        current_price = signals.get('current_price', 0)
        position_side = position.get('side', 'BUY')
        
        if not market_id or current_price <= 0:
            return False
        
        # Get entry information
        entry_info = self.position_entry_info.get(market_id)
        if not entry_info:
            return False
        
        entry_price = entry_info['price']
        entry_time = entry_info['time']
        
        # Calculate P&L
        if position_side == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Quick profit taking (primary exit strategy)
        if pnl_pct >= self.quick_profit_target:
            logger.success(f"Quick profit exit for {market_id}: {pnl_pct:.2%}")
            return True
        
        # Extended profit taking for strong moves
        if pnl_pct >= self.extended_profit_target:
            logger.success(f"Extended profit exit for {market_id}: {pnl_pct:.2%}")
            return True
        
        # Stop loss (wider to avoid whipsaws)
        if pnl_pct <= -self.stop_loss_pct:
            logger.info(f"Stop loss triggered for {market_id}: {pnl_pct:.2%}")
            return True
        
        # Time-based exit for stale positions
        time_held = datetime.now() - entry_time
        max_hold_hours = 4  # Maximum 4 hours per position
        
        if time_held.total_seconds() > (max_hold_hours * 3600):
            if pnl_pct > 0:  # Only exit if profitable
                logger.info(f"Time-based profitable exit for {market_id}: {pnl_pct:.2%}")
                return True
        
        # Signal reversal exit (momentum changes direction)
        current_momentum = signals.get('raw_momentum', 0)
        if position_side == 'BUY' and current_momentum < -self.momentum_threshold * 0.5:
            logger.info(f"Momentum reversal exit for {market_id}")
            return True
        elif position_side == 'SELL' and current_momentum > self.momentum_threshold * 0.5:
            logger.info(f"Momentum reversal exit for {market_id}")
            return True
        
        return False
    
    async def calculate_position_size(self, signals: Dict[str, Any]) -> float:
        """Calculate position size for maximum profitability."""
        market_id = signals.get('market_id')
        confidence = signals['confidence']
        momentum_strength = signals.get('momentum_strength', 1.0)
        
        # Base size scaled by confidence and momentum
        base_size = self.base_position_size
        confidence_multiplier = 1 + (confidence * self.confidence_multiplier)
        momentum_multiplier = 1 + (momentum_strength * 0.5)
        
        # Calculate final size
        position_size = base_size * confidence_multiplier * momentum_multiplier
        
        # Apply limits
        position_size = min(position_size, self.max_position_size)
        position_size = max(position_size, 50)  # Minimum $50
        
        # Store entry information for exit logic
        if market_id and signals.get('current_price'):
            self.position_entry_info[market_id] = {
                'price': signals['current_price'],
                'time': datetime.now(),
                'size': position_size
            }
        
        self.total_trades += 1
        
        logger.success(f"Position size for {market_id}: ${position_size:.2f} "
                      f"(confidence: {confidence:.2f}, momentum: {momentum_strength:.2f})")
        
        return position_size
    
    def update_performance(self, market_id: str, pnl: float):
        """Track performance for optimization."""
        if pnl > 0:
            self.winning_trades += 1
            logger.success(f"Winning trade: {market_id} +{pnl:.2f}")
        else:
            logger.warning(f"Losing trade: {market_id} {pnl:.2f}")
        
        self.total_profit += pnl
        
        # Clean up position tracking
        if market_id in self.position_entry_info:
            del self.position_entry_info[market_id]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_profit = self.total_profit / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'strategy': 'Profitable Momentum Strategy',
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'avg_profit_per_trade': avg_profit,
            'active_positions': len(self.position_entry_info)
        }
