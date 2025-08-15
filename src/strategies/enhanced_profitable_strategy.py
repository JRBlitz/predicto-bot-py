"""
Enhanced Profitable Strategy - Next Generation
Builds on the successful profitable momentum strategy with advanced improvements.
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta
from .base_strategy import BaseStrategy
import asyncio


class EnhancedProfitableStrategy(BaseStrategy):
    """
    Enhanced version of the profitable momentum strategy with:
    1. Advanced signal filtering and confirmation
    2. Dynamic position sizing based on market conditions
    3. Adaptive profit targets and stop losses
    4. Cross-asset correlation analysis
    5. Real-time performance optimization
    6. Machine learning signal enhancement
    """
    
    def __init__(self, client, config: Dict[str, Any]):
        """Initialize the enhanced profitable strategy."""
        super().__init__("Enhanced Profitable Strategy", client, config)
        
        # Core momentum parameters (proven profitable)
        self.momentum_periods = config.get('momentum_periods', 6)
        self.momentum_threshold = config.get('momentum_threshold', 0.015)
        self.trend_confirmation_periods = config.get('trend_confirmation_periods', 3)
        
        # Enhanced signal filtering
        self.signal_strength_threshold = config.get('signal_strength_threshold', 0.8)
        self.confirmation_indicators = config.get('confirmation_indicators', 3)
        self.noise_filter_periods = config.get('noise_filter_periods', 5)
        
        # Dynamic profit management
        self.base_profit_target = config.get('base_profit_target', 0.03)  # 3%
        self.volatility_profit_multiplier = config.get('volatility_profit_multiplier', 1.5)
        self.momentum_profit_multiplier = config.get('momentum_profit_multiplier', 2.0)
        self.adaptive_stop_loss = config.get('adaptive_stop_loss', True)
        
        # Advanced position sizing
        self.base_position_size = config.get('base_position_size', 100)
        self.volatility_position_adjustment = config.get('volatility_position_adjustment', True)
        self.correlation_position_reduction = config.get('correlation_position_reduction', 0.3)
        self.kelly_criterion_sizing = config.get('kelly_criterion_sizing', True)
        
        # Performance optimization
        self.adaptive_parameters = config.get('adaptive_parameters', True)
        self.performance_window = config.get('performance_window', 20)  # Last 20 trades
        self.optimization_frequency = config.get('optimization_frequency', 10)  # Every 10 trades
        
        # Data storage
        self.price_history = {}
        self.volume_history = {}
        self.volatility_history = {}
        self.correlation_matrix = {}
        self.signal_history = {}
        self.performance_history = []
        self.position_entry_info = {}
        
        # ML-enhanced features
        self.feature_history = {}
        self.signal_weights = {
            'momentum': 0.4,
            'trend': 0.3,
            'volatility': 0.15,
            'correlation': 0.1,
            'volume': 0.05
        }
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        self.total_profit = 0
        
        # Real-time optimization
        self.last_optimization = datetime.now()
        self.current_parameters = self._get_current_parameters()
        
    def _get_current_parameters(self) -> Dict[str, float]:
        """Get current strategy parameters."""
        return {
            'momentum_threshold': self.momentum_threshold,
            'profit_target': self.base_profit_target,
            'signal_threshold': self.signal_strength_threshold,
            'position_size_base': self.base_position_size
        }
    
    def _get_market_id(self, market: Dict[str, Any]) -> str:
        """Extract market ID from market data."""
        return market.get('id') or market.get('market_id') or market.get('token_id')
    
    def _get_current_price(self, orderbook: Dict[str, Any]) -> float:
        """Get current mid price from orderbook."""
        if not orderbook.get('bids') or not orderbook.get('asks'):
            return None
        
        try:
            best_bid = float(orderbook['bids'][0]['price'])
            best_ask = float(orderbook['asks'][0]['price'])
            return (best_bid + best_ask) / 2
        except (IndexError, ValueError, KeyError):
            return None
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced market analysis with multiple confirmation layers."""
        market = market_data['market']
        orderbook = market_data['orderbook']
        
        market_id = self._get_market_id(market)
        current_price = self._get_current_price(orderbook)
        
        if not market_id or current_price is None:
            return {'signal': 'HOLD', 'confidence': 0}
        
        # Update all data histories
        self._update_all_histories(market_id, current_price)
        
        # Need sufficient data
        if len(self.price_history.get(market_id, [])) < max(self.momentum_periods, self.noise_filter_periods):
            return {'signal': 'HOLD', 'confidence': 0}
        
        # Multi-layer signal analysis
        signals = {}
        
        # 1. Enhanced momentum analysis
        signals['momentum'] = self._calculate_enhanced_momentum(market_id, current_price)
        
        # 2. Advanced trend confirmation
        signals['trend'] = self._calculate_advanced_trend(market_id, current_price)
        
        # 3. Volatility regime analysis
        signals['volatility'] = self._calculate_volatility_regime(market_id, current_price)
        
        # 4. Cross-asset correlation analysis
        signals['correlation'] = self._calculate_correlation_signal(market_id, current_price)
        
        # 5. Volume confirmation (if available)
        signals['volume'] = self._calculate_volume_confirmation(market_id)
        
        # 6. Noise filtering
        signals['noise_filter'] = self._calculate_noise_filter(market_id, current_price)
        
        # 7. ML-enhanced signal scoring
        signals['ml_score'] = self._calculate_ml_enhanced_score(market_id, signals)
        
        # Combine all signals with dynamic weighting
        final_signal = self._combine_enhanced_signals(signals, market_id, current_price)
        
        # Store signal for learning
        self._store_signal_history(market_id, final_signal, signals)
        
        return final_signal
    
    def _update_all_histories(self, market_id: str, current_price: float):
        """Update all data histories efficiently."""
        current_time = datetime.now()
        
        # Price history
        if market_id not in self.price_history:
            self.price_history[market_id] = []
        
        self.price_history[market_id].append({
            'price': current_price,
            'timestamp': current_time
        })
        
        # Volatility history
        if len(self.price_history[market_id]) >= 2:
            prices = [p['price'] for p in self.price_history[market_id][-10:]]  # Last 10 prices
            if len(prices) >= 2:
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns)
                
                if market_id not in self.volatility_history:
                    self.volatility_history[market_id] = []
                
                self.volatility_history[market_id].append({
                    'volatility': volatility,
                    'timestamp': current_time
                })
        
        # Keep histories manageable
        max_history = 100
        for history in [self.price_history, self.volatility_history]:
            if market_id in history and len(history[market_id]) > max_history:
                history[market_id] = history[market_id][-max_history:]
    
    def _calculate_enhanced_momentum(self, market_id: str, current_price: float) -> Dict[str, Any]:
        """Calculate enhanced momentum with multiple timeframes."""
        prices = [p['price'] for p in self.price_history[market_id]]
        
        if len(prices) < self.momentum_periods:
            return {'signal': 'NEUTRAL', 'strength': 0, 'quality': 0}
        
        # Multi-timeframe momentum
        short_momentum = self._calculate_momentum_for_period(prices, 3)
        medium_momentum = self._calculate_momentum_for_period(prices, self.momentum_periods)
        long_momentum = self._calculate_momentum_for_period(prices, min(len(prices)-1, self.momentum_periods * 2))
        
        # Momentum acceleration (rate of change of momentum)
        if len(prices) >= self.momentum_periods + 2:
            prev_momentum = self._calculate_momentum_for_period(prices[:-1], self.momentum_periods)
            momentum_acceleration = medium_momentum - prev_momentum
        else:
            momentum_acceleration = 0
        
        # Combined momentum score with acceleration
        combined_momentum = (
            short_momentum * 0.5 + 
            medium_momentum * 0.3 + 
            long_momentum * 0.2 + 
            momentum_acceleration * 0.3
        )
        
        # Enhanced threshold with dynamic adjustment
        current_volatility = self._get_current_volatility(market_id)
        adjusted_threshold = self.momentum_threshold * (1 + current_volatility * 2)
        
        # Signal generation with quality scoring
        if abs(combined_momentum) > adjusted_threshold:
            signal = 'BUY' if combined_momentum > 0 else 'SELL'
            strength = min(abs(combined_momentum) / adjusted_threshold, 3.0)
            
            # Quality score based on momentum alignment
            alignment_score = self._calculate_momentum_alignment(short_momentum, medium_momentum, long_momentum)
            quality = alignment_score * (1 + min(momentum_acceleration, 0.5))
        else:
            signal = 'NEUTRAL'
            strength = 0
            quality = 0
        
        return {
            'signal': signal,
            'strength': strength,
            'quality': quality,
            'combined_momentum': combined_momentum,
            'acceleration': momentum_acceleration,
            'threshold': adjusted_threshold,
            'volatility_adjustment': current_volatility
        }
    
    def _calculate_momentum_for_period(self, prices: List[float], period: int) -> float:
        """Calculate momentum for specific period."""
        if len(prices) < period:
            return 0
        return (prices[-1] - prices[-period]) / prices[-period]
    
    def _calculate_momentum_alignment(self, short: float, medium: float, long: float) -> float:
        """Calculate how well different timeframe momentums align."""
        # All same direction = high alignment
        if (short > 0 and medium > 0 and long > 0) or (short < 0 and medium < 0 and long < 0):
            return 1.0
        
        # Two same direction = medium alignment
        same_direction_count = sum([
            1 if short * medium > 0 else 0,
            1 if medium * long > 0 else 0,
            1 if short * long > 0 else 0
        ])
        
        return same_direction_count / 3.0
    
    def _get_current_volatility(self, market_id: str) -> float:
        """Get current market volatility."""
        if market_id not in self.volatility_history or not self.volatility_history[market_id]:
            return 0.02  # Default volatility
        
        return self.volatility_history[market_id][-1]['volatility']
    
    def _calculate_advanced_trend(self, market_id: str, current_price: float) -> Dict[str, Any]:
        """Calculate advanced trend with multiple confirmations."""
        prices = [p['price'] for p in self.price_history[market_id]]
        
        if len(prices) < self.trend_confirmation_periods + 2:
            return {'signal': 'NEUTRAL', 'strength': 0, 'consistency': 0}
        
        # Multiple trend analysis methods
        
        # 1. Moving average trend
        short_ma = np.mean(prices[-3:])
        long_ma = np.mean(prices[-self.trend_confirmation_periods-2:])
        ma_trend = 'UP' if short_ma > long_ma else 'DOWN'
        
        # 2. Price action trend (higher highs/lower lows)
        recent_prices = prices[-self.trend_confirmation_periods:]
        price_trend = self._analyze_price_action_trend(recent_prices)
        
        # 3. Linear regression trend
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]
        regression_trend = 'UP' if slope > 0 else 'DOWN'
        
        # Trend consensus
        trends = [ma_trend, price_trend, regression_trend]
        up_votes = trends.count('UP')
        down_votes = trends.count('DOWN')
        
        if up_votes > down_votes:
            signal = 'UP'
            strength = up_votes / len(trends)
        elif down_votes > up_votes:
            signal = 'DOWN'
            strength = down_votes / len(trends)
        else:
            signal = 'NEUTRAL'
            strength = 0
        
        # Trend consistency (how smooth is the trend)
        consistency = self._calculate_trend_consistency(recent_prices)
        
        return {
            'signal': signal,
            'strength': strength,
            'consistency': consistency,
            'ma_trend': ma_trend,
            'price_trend': price_trend,
            'regression_trend': regression_trend,
            'slope': slope
        }
    
    def _analyze_price_action_trend(self, prices: List[float]) -> str:
        """Analyze price action for trend direction."""
        if len(prices) < 3:
            return 'NEUTRAL'
        
        # Look for higher highs and higher lows (uptrend) or lower highs and lower lows (downtrend)
        highs = []
        lows = []
        
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                highs.append(prices[i])
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                lows.append(prices[i])
        
        if len(highs) >= 2 and len(lows) >= 2:
            higher_highs = highs[-1] > highs[0]
            higher_lows = lows[-1] > lows[0]
            
            if higher_highs and higher_lows:
                return 'UP'
            elif not higher_highs and not higher_lows:
                return 'DOWN'
        
        # Fallback to simple comparison
        return 'UP' if prices[-1] > prices[0] else 'DOWN'
    
    def _calculate_trend_consistency(self, prices: List[float]) -> float:
        """Calculate how consistent the trend is."""
        if len(prices) < 3:
            return 0
        
        # Calculate correlation with ideal trend line
        x = np.arange(len(prices))
        correlation = abs(np.corrcoef(x, prices)[0, 1])
        
        return correlation if not np.isnan(correlation) else 0
    
    def _calculate_volatility_regime(self, market_id: str, current_price: float) -> Dict[str, Any]:
        """Analyze volatility regime for position sizing adjustments."""
        if market_id not in self.volatility_history or len(self.volatility_history[market_id]) < 5:
            return {'regime': 'normal', 'adjustment': 1.0, 'percentile': 0.5}
        
        volatilities = [v['volatility'] for v in self.volatility_history[market_id]]
        current_vol = volatilities[-1]
        
        # Calculate volatility percentile
        vol_percentile = np.percentile(volatilities, 50)  # Median
        vol_75 = np.percentile(volatilities, 75)
        vol_25 = np.percentile(volatilities, 25)
        
        # Determine regime
        if current_vol > vol_75:
            regime = 'high_volatility'
            adjustment = 0.7  # Reduce position size
        elif current_vol < vol_25:
            regime = 'low_volatility'
            adjustment = 1.3  # Increase position size
        else:
            regime = 'normal'
            adjustment = 1.0
        
        # Calculate exact percentile for fine-tuning
        percentile = (np.sum(np.array(volatilities) <= current_vol) / len(volatilities))
        
        return {
            'regime': regime,
            'adjustment': adjustment,
            'percentile': percentile,
            'current_volatility': current_vol,
            'volatility_trend': 'increasing' if len(volatilities) >= 3 and volatilities[-1] > volatilities[-3] else 'decreasing'
        }
    
    def _calculate_correlation_signal(self, market_id: str, current_price: float) -> Dict[str, Any]:
        """Calculate cross-asset correlation signal."""
        if len(self.price_history) < 2:
            return {'signal': 'NEUTRAL', 'strength': 1.0, 'correlation': 0}
        
        # Get price series for correlation analysis
        correlation_strength = 0
        correlations = {}
        
        current_prices = [p['price'] for p in self.price_history[market_id][-20:]]  # Last 20 prices
        
        for other_market_id, other_history in self.price_history.items():
            if other_market_id == market_id or len(other_history) < 20:
                continue
            
            other_prices = [p['price'] for p in other_history[-20:]]
            
            if len(current_prices) == len(other_prices):
                correlation = np.corrcoef(current_prices, other_prices)[0, 1]
                
                if not np.isnan(correlation):
                    correlations[other_market_id] = correlation
                    correlation_strength += abs(correlation)
        
        # Average correlation strength
        avg_correlation = correlation_strength / len(correlations) if correlations else 0
        
        # Signal based on correlation
        if avg_correlation > 0.7:
            signal = 'CORRELATED'  # High correlation - be cautious about multiple positions
            strength = 0.5  # Reduce signal strength
        elif avg_correlation < 0.3:
            signal = 'INDEPENDENT'  # Low correlation - good for diversification
            strength = 1.2  # Boost signal strength
        else:
            signal = 'NEUTRAL'
            strength = 1.0
        
        return {
            'signal': signal,
            'strength': strength,
            'avg_correlation': avg_correlation,
            'correlations': correlations
        }
    
    def _calculate_volume_confirmation(self, market_id: str) -> Dict[str, Any]:
        """Calculate volume confirmation (placeholder for future enhancement)."""
        # For now, return neutral - can be enhanced with real volume data
        return {
            'signal': 'NEUTRAL',
            'strength': 1.0,
            'volume_trend': 'stable'
        }
    
    def _calculate_noise_filter(self, market_id: str, current_price: float) -> Dict[str, Any]:
        """Filter out market noise using statistical methods."""
        prices = [p['price'] for p in self.price_history[market_id]]
        
        if len(prices) < self.noise_filter_periods:
            return {'signal': 'CLEAN', 'noise_level': 0}
        
        # Calculate price changes
        recent_prices = prices[-self.noise_filter_periods:]
        price_changes = np.diff(recent_prices)
        
        # Noise metrics
        volatility = np.std(price_changes)
        mean_change = np.mean(np.abs(price_changes))
        
        # Noise-to-signal ratio
        if mean_change > 0:
            noise_ratio = volatility / mean_change
        else:
            noise_ratio = 1.0
        
        # Determine if market is too noisy
        if noise_ratio > 2.0:
            signal = 'NOISY'
            filter_strength = 0.3  # Reduce signal strength significantly
        elif noise_ratio > 1.5:
            signal = 'MODERATE_NOISE'
            filter_strength = 0.7  # Reduce signal strength moderately
        else:
            signal = 'CLEAN'
            filter_strength = 1.0  # No reduction
        
        return {
            'signal': signal,
            'noise_level': noise_ratio,
            'filter_strength': filter_strength
        }
    
    def _calculate_ml_enhanced_score(self, market_id: str, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ML-enhanced signal score using feature engineering."""
        # Extract features from signals
        features = self._extract_signal_features(signals)
        
        # Store features for learning
        if market_id not in self.feature_history:
            self.feature_history[market_id] = []
        
        self.feature_history[market_id].append({
            'features': features,
            'timestamp': datetime.now()
        })
        
        # Simple ML scoring (can be enhanced with actual ML models)
        ml_score = self._calculate_simple_ml_score(features)
        
        return {
            'score': ml_score,
            'features': features,
            'confidence': min(ml_score * 2, 1.0)
        }
    
    def _extract_signal_features(self, signals: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from signals."""
        features = {}
        
        # Momentum features
        momentum_signal = signals.get('momentum', {})
        features['momentum_strength'] = momentum_signal.get('strength', 0)
        features['momentum_quality'] = momentum_signal.get('quality', 0)
        features['momentum_acceleration'] = momentum_signal.get('acceleration', 0)
        
        # Trend features
        trend_signal = signals.get('trend', {})
        features['trend_strength'] = trend_signal.get('strength', 0)
        features['trend_consistency'] = trend_signal.get('consistency', 0)
        
        # Volatility features
        vol_signal = signals.get('volatility', {})
        features['volatility_adjustment'] = vol_signal.get('adjustment', 1.0)
        features['volatility_percentile'] = vol_signal.get('percentile', 0.5)
        
        # Correlation features
        corr_signal = signals.get('correlation', {})
        features['correlation_strength'] = corr_signal.get('strength', 1.0)
        features['avg_correlation'] = corr_signal.get('avg_correlation', 0)
        
        # Noise features
        noise_signal = signals.get('noise_filter', {})
        features['noise_filter_strength'] = noise_signal.get('filter_strength', 1.0)
        features['noise_level'] = noise_signal.get('noise_level', 0)
        
        return features
    
    def _calculate_simple_ml_score(self, features: Dict[str, float]) -> float:
        """Calculate simple ML-like score from features."""
        # Weighted combination of features
        score = (
            features.get('momentum_strength', 0) * 0.3 +
            features.get('momentum_quality', 0) * 0.2 +
            features.get('trend_strength', 0) * 0.2 +
            features.get('trend_consistency', 0) * 0.15 +
            features.get('noise_filter_strength', 1.0) * 0.1 +
            features.get('correlation_strength', 1.0) * 0.05
        )
        
        # Apply volatility adjustment
        vol_adjustment = features.get('volatility_adjustment', 1.0)
        adjusted_score = score * vol_adjustment
        
        return min(adjusted_score, 1.0)
    
    def _combine_enhanced_signals(self, signals: Dict[str, Any], market_id: str, current_price: float) -> Dict[str, Any]:
        """Combine all enhanced signals with dynamic weighting."""
        
        # Extract signal components
        momentum_signal = signals['momentum']['signal']
        trend_signal = signals['trend']['signal']
        volatility_regime = signals['volatility']
        correlation_signal = signals['correlation']
        noise_filter = signals['noise_filter']
        ml_score = signals['ml_score']['score']
        
        # Dynamic weight adjustment based on recent performance
        weights = self._get_dynamic_weights()
        
        # Calculate weighted scores
        buy_score = 0
        sell_score = 0
        
        # Momentum contribution
        if momentum_signal == 'BUY':
            buy_score += weights['momentum'] * signals['momentum']['strength'] * signals['momentum']['quality']
        elif momentum_signal == 'SELL':
            sell_score += weights['momentum'] * signals['momentum']['strength'] * signals['momentum']['quality']
        
        # Trend contribution
        if trend_signal == 'UP' and momentum_signal == 'BUY':
            buy_score += weights['trend'] * signals['trend']['strength'] * signals['trend']['consistency']
        elif trend_signal == 'DOWN' and momentum_signal == 'SELL':
            sell_score += weights['trend'] * signals['trend']['strength'] * signals['trend']['consistency']
        
        # Apply filters and adjustments
        volatility_adjustment = volatility_regime['adjustment']
        correlation_adjustment = correlation_signal['strength']
        noise_adjustment = noise_filter['filter_strength']
        ml_adjustment = ml_score
        
        # Combined adjustments
        total_adjustment = (volatility_adjustment * correlation_adjustment * 
                          noise_adjustment * ml_adjustment)
        
        buy_score *= total_adjustment
        sell_score *= total_adjustment
        
        # Determine final signal
        signal_threshold = self.signal_strength_threshold
        
        if buy_score > signal_threshold and buy_score > sell_score:
            final_signal = 'BUY'
            confidence = min(buy_score, 1.0)
        elif sell_score > signal_threshold and sell_score > buy_score:
            final_signal = 'SELL'
            confidence = min(sell_score, 1.0)
        else:
            final_signal = 'HOLD'
            confidence = 0
        
        # Calculate adaptive profit target
        adaptive_profit_target = self._calculate_adaptive_profit_target(
            volatility_regime, signals['momentum'], ml_score
        )
        
        return {
            'signal': final_signal,
            'side': final_signal if final_signal != 'HOLD' else None,
            'confidence': confidence,
            'price': current_price,
            'exit_price': current_price,
            'market_id': market_id,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'adaptive_profit_target': adaptive_profit_target,
            'volatility_regime': volatility_regime['regime'],
            'ml_score': ml_score,
            'total_adjustment': total_adjustment,
            'signal_breakdown': signals
        }
    
    def _get_dynamic_weights(self) -> Dict[str, float]:
        """Get dynamically adjusted weights based on recent performance."""
        if len(self.performance_history) < 5:
            return self.signal_weights.copy()
        
        # Analyze recent performance by signal type
        recent_performance = self.performance_history[-self.performance_window:]
        
        # For now, return base weights (can be enhanced with actual performance analysis)
        return self.signal_weights.copy()
    
    def _calculate_adaptive_profit_target(self, volatility_regime: Dict[str, Any], 
                                        momentum_signal: Dict[str, Any], ml_score: float) -> float:
        """Calculate adaptive profit target based on market conditions."""
        base_target = self.base_profit_target
        
        # Volatility adjustment
        vol_multiplier = 1.0
        if volatility_regime['regime'] == 'high_volatility':
            vol_multiplier = self.volatility_profit_multiplier
        elif volatility_regime['regime'] == 'low_volatility':
            vol_multiplier = 0.8
        
        # Momentum strength adjustment
        momentum_strength = momentum_signal.get('strength', 1.0)
        momentum_multiplier = 1 + (momentum_strength - 1) * self.momentum_profit_multiplier * 0.1
        
        # ML confidence adjustment
        ml_multiplier = 1 + (ml_score - 0.5) * 0.5
        
        # Combined adaptive target
        adaptive_target = base_target * vol_multiplier * momentum_multiplier * ml_multiplier
        
        # Reasonable bounds
        return max(0.02, min(adaptive_target, 0.08))  # Between 2% and 8%
    
    def _store_signal_history(self, market_id: str, final_signal: Dict[str, Any], 
                            component_signals: Dict[str, Any]):
        """Store signal history for learning and optimization."""
        if market_id not in self.signal_history:
            self.signal_history[market_id] = []
        
        self.signal_history[market_id].append({
            'timestamp': datetime.now(),
            'final_signal': final_signal,
            'component_signals': component_signals
        })
        
        # Keep reasonable history
        if len(self.signal_history[market_id]) > 100:
            self.signal_history[market_id] = self.signal_history[market_id][-100:]
    
    async def should_enter_position(self, signals: Dict[str, Any]) -> bool:
        """Enhanced entry logic with multiple confirmations."""
        if signals['signal'] not in ['BUY', 'SELL']:
            return False
        
        # High confidence threshold
        if signals['confidence'] < self.signal_strength_threshold:
            return False
        
        # Check volatility regime
        if signals.get('volatility_regime') == 'high_volatility' and signals['confidence'] < 0.9:
            return False  # Need higher confidence in volatile markets
        
        # Check ML score
        if signals.get('ml_score', 0) < 0.6:
            return False  # Need good ML score
        
        # Position limits
        current_positions = len(self.position_entry_info)
        max_positions = 2  # Conservative
        
        if current_positions >= max_positions:
            return False
        
        # Check correlation (avoid too many correlated positions)
        correlation_signal = signals.get('signal_breakdown', {}).get('correlation', {})
        if correlation_signal.get('signal') == 'CORRELATED' and current_positions > 0:
            return False
        
        return True
    
    async def calculate_position_size(self, signals: Dict[str, Any]) -> float:
        """Enhanced position sizing with multiple factors."""
        market_id = signals.get('market_id')
        confidence = signals['confidence']
        
        # Base size
        base_size = self.base_position_size
        
        # Confidence multiplier
        confidence_multiplier = 1 + confidence
        
        # Volatility adjustment
        volatility_regime = signals.get('volatility_regime', 'normal')
        if volatility_regime == 'high_volatility':
            vol_multiplier = 0.6
        elif volatility_regime == 'low_volatility':
            vol_multiplier = 1.4
        else:
            vol_multiplier = 1.0
        
        # ML score adjustment
        ml_score = signals.get('ml_score', 0.5)
        ml_multiplier = 0.5 + ml_score
        
        # Kelly criterion sizing (simplified)
        if self.kelly_criterion_sizing and len(self.performance_history) >= 10:
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.5
            avg_win = np.mean([p for p in self.performance_history[-20:] if p > 0]) if any(p > 0 for p in self.performance_history[-20:]) else 0.03
            avg_loss = abs(np.mean([p for p in self.performance_history[-20:] if p < 0])) if any(p < 0 for p in self.performance_history[-20:]) else 0.04
            
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_multiplier = max(0.1, min(kelly_fraction, 2.0))  # Bounded Kelly
            else:
                kelly_multiplier = 1.0
        else:
            kelly_multiplier = 1.0
        
        # Calculate final size
        position_size = (base_size * confidence_multiplier * vol_multiplier * 
                        ml_multiplier * kelly_multiplier)
        
        # Apply bounds
        position_size = max(50, min(position_size, 300))  # $50 to $300
        
        # Store entry info
        if market_id and signals.get('current_price'):
            self.position_entry_info[market_id] = {
                'price': signals['current_price'],
                'time': datetime.now(),
                'size': position_size,
                'adaptive_profit_target': signals.get('adaptive_profit_target', self.base_profit_target),
                'volatility_regime': volatility_regime
            }
        
        self.total_trades += 1
        
        logger.success(f"Enhanced position size for {market_id}: ${position_size:.2f} "
                      f"(confidence: {confidence:.2f}, ML: {ml_score:.2f}, "
                      f"volatility: {volatility_regime})")
        
        return position_size
    
    async def should_exit_position(self, position: Dict[str, Any], signals: Dict[str, Any]) -> bool:
        """Enhanced exit logic with adaptive targets."""
        market_id = signals.get('market_id')
        current_price = signals.get('current_price', 0)
        position_side = position.get('side', 'BUY')
        
        if not market_id or current_price <= 0:
            return False
        
        entry_info = self.position_entry_info.get(market_id)
        if not entry_info:
            return False
        
        entry_price = entry_info['price']
        entry_time = entry_info['time']
        adaptive_profit_target = entry_info.get('adaptive_profit_target', self.base_profit_target)
        
        # Calculate P&L
        if position_side == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Adaptive profit taking
        if pnl_pct >= adaptive_profit_target:
            logger.success(f"Adaptive profit exit for {market_id}: {pnl_pct:.2%} "
                          f"(target: {adaptive_profit_target:.2%})")
            return True
        
        # Dynamic stop loss based on volatility
        volatility_regime = entry_info.get('volatility_regime', 'normal')
        if volatility_regime == 'high_volatility':
            stop_loss = 0.06  # 6% for high volatility
        else:
            stop_loss = 0.04  # 4% for normal/low volatility
        
        if pnl_pct <= -stop_loss:
            logger.info(f"Dynamic stop loss for {market_id}: {pnl_pct:.2%} "
                       f"(stop: {-stop_loss:.2%})")
            return True
        
        # Signal reversal with ML confirmation
        ml_score = signals.get('ml_score', 0.5)
        if signals['signal'] != 'HOLD':
            opposite_signal = 'SELL' if position_side == 'BUY' else 'BUY'
            if (signals['signal'] == opposite_signal and 
                signals['confidence'] > 0.8 and 
                ml_score > 0.7):
                logger.info(f"Enhanced signal reversal exit for {market_id}")
                return True
        
        # Time-based exit for profitable positions
        time_held = datetime.now() - entry_time
        if time_held.total_seconds() > 14400 and pnl_pct > 0.01:  # 4 hours, 1% profit
            logger.info(f"Time-based profitable exit for {market_id}: {pnl_pct:.2%}")
            return True
        
        return False
    
    def update_performance(self, market_id: str, pnl: float):
        """Update performance with enhanced tracking."""
        pnl_pct = pnl / self.position_entry_info.get(market_id, {}).get('size', 100)
        
        # Update performance history
        self.performance_history.append(pnl_pct)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Update win/loss tracking
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
            logger.success(f"Enhanced winning trade: {market_id} +${pnl:.2f} ({pnl_pct:.2%})")
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
            logger.warning(f"Enhanced losing trade: {market_id} ${pnl:.2f} ({pnl_pct:.2%})")
        
        self.total_profit += pnl
        
        # Trigger optimization if needed
        if self.total_trades % self.optimization_frequency == 0:
            asyncio.create_task(self._optimize_parameters())
        
        # Clean up
        if market_id in self.position_entry_info:
            del self.position_entry_info[market_id]
    
    async def _optimize_parameters(self):
        """Real-time parameter optimization based on performance."""
        if not self.adaptive_parameters or len(self.performance_history) < self.performance_window:
            return
        
        logger.info("🔧 Running real-time parameter optimization...")
        
        # Analyze recent performance
        recent_performance = self.performance_history[-self.performance_window:]
        avg_return = np.mean(recent_performance)
        win_rate = sum(1 for p in recent_performance if p > 0) / len(recent_performance)
        
        # Adjust parameters based on performance
        if avg_return < 0 or win_rate < 0.4:
            # Performance declining - be more conservative
            self.signal_strength_threshold = min(self.signal_strength_threshold * 1.1, 0.95)
            self.momentum_threshold = min(self.momentum_threshold * 1.1, 0.025)
            logger.info("📉 Performance declining - increasing thresholds")
        elif avg_return > 0.02 and win_rate > 0.6:
            # Performance good - can be slightly more aggressive
            self.signal_strength_threshold = max(self.signal_strength_threshold * 0.95, 0.6)
            self.momentum_threshold = max(self.momentum_threshold * 0.95, 0.01)
            logger.info("📈 Performance good - decreasing thresholds")
        
        self.last_optimization = datetime.now()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_profit = self.total_profit / self.total_trades if self.total_trades > 0 else 0
        
        # Recent performance
        if self.performance_history:
            recent_avg = np.mean(self.performance_history[-10:])
            recent_volatility = np.std(self.performance_history[-10:])
        else:
            recent_avg = 0
            recent_volatility = 0
        
        return {
            'strategy': 'Enhanced Profitable Strategy',
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'avg_profit_per_trade': avg_profit,
            'active_positions': len(self.position_entry_info),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'recent_avg_return': recent_avg,
            'recent_volatility': recent_volatility,
            'current_parameters': self.current_parameters,
            'last_optimization': self.last_optimization
        }
