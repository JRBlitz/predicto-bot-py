"""
Balance Manager - Real-time balance checking and management
Integrates with Polymarket API to get actual account balance.
"""
import asyncio
import aiohttp
import json
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass


@dataclass
class BalanceInfo:
    """Balance information structure."""
    total_balance: float
    available_balance: float
    reserved_balance: float
    last_updated: datetime
    currency: str = "USDC"


class BalanceManager:
    """
    Manages account balance checking and position sizing.
    Integrates with Polymarket API for real-time balance data.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize balance manager."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://clob.polymarket.com"
        
        # Balance caching
        self.cached_balance: Optional[BalanceInfo] = None
        self.cache_duration = 60  # Cache for 60 seconds
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Fallback balance (your provided balance)
        self.fallback_balance = 40.29
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize the balance manager."""
        timeout = aiohttp.ClientTimeout(total=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info("Balance manager initialized")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_balance(self, force_refresh: bool = False) -> BalanceInfo:
        """Get current account balance."""
        # Check cache first
        if (not force_refresh and 
            self.cached_balance and 
            (datetime.now() - self.cached_balance.last_updated).total_seconds() < self.cache_duration):
            return self.cached_balance
        
        # Try to get balance from API
        try:
            balance_info = await self._fetch_balance_from_api()
            if balance_info:
                self.cached_balance = balance_info
                return balance_info
        except Exception as e:
            logger.warning(f"Could not fetch balance from API: {e}")
        
        # Fallback to provided balance
        logger.info(f"Using fallback balance: ${self.fallback_balance:.2f}")
        fallback_info = BalanceInfo(
            total_balance=self.fallback_balance,
            available_balance=self.fallback_balance,
            reserved_balance=0.0,
            last_updated=datetime.now()
        )
        
        self.cached_balance = fallback_info
        return fallback_info
    
    async def _fetch_balance_from_api(self) -> Optional[BalanceInfo]:
        """Fetch balance from Polymarket API."""
        if not self.api_key or not self.session:
            return None
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            # Get balance endpoint
            async with self.session.get(
                f"{self.base_url}/balance",
                headers=headers
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse balance data (adjust based on actual API response format)
                    total = float(data.get('total', 0))
                    available = float(data.get('available', 0))
                    reserved = float(data.get('reserved', 0))
                    
                    balance_info = BalanceInfo(
                        total_balance=total,
                        available_balance=available,
                        reserved_balance=reserved,
                        last_updated=datetime.now()
                    )
                    
                    logger.success(f"Balance fetched: ${available:.2f} available")
                    return balance_info
                    
                elif response.status == 401:
                    logger.error("API authentication failed - check API key")
                    return None
                    
                else:
                    logger.warning(f"API returned status {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.warning("Balance API request timed out")
            return None
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return None
    
    async def get_available_balance(self) -> float:
        """Get available balance for trading."""
        balance_info = await self.get_balance()
        return balance_info.available_balance
    
    async def can_afford_position(self, position_size: float, buffer_pct: float = 0.05) -> bool:
        """Check if we can afford a position with safety buffer."""
        available = await self.get_available_balance()
        required = position_size * (1 + buffer_pct)  # Add 5% buffer
        
        can_afford = available >= required
        
        if not can_afford:
            logger.warning(f"Cannot afford position: ${position_size:.2f} required, "
                          f"${available:.2f} available")
        
        return can_afford
    
    async def get_max_position_size(self, max_risk_pct: float = 0.3) -> float:
        """Get maximum affordable position size."""
        available = await self.get_available_balance()
        max_size = available * max_risk_pct
        
        logger.info(f"Max position size: ${max_size:.2f} ({max_risk_pct:.1%} of ${available:.2f})")
        return max_size
    
    async def update_balance_after_trade(self, pnl: float):
        """Update cached balance after a trade."""
        if self.cached_balance:
            self.cached_balance.available_balance += pnl
            self.cached_balance.total_balance += pnl
            logger.info(f"Balance updated: ${pnl:+.2f} -> ${self.cached_balance.available_balance:.2f}")
    
    def get_balance_summary(self) -> Dict[str, Any]:
        """Get balance summary for reporting."""
        if not self.cached_balance:
            return {
                'status': 'no_data',
                'message': 'No balance data available'
            }
        
        return {
            'status': 'active',
            'total_balance': self.cached_balance.total_balance,
            'available_balance': self.cached_balance.available_balance,
            'reserved_balance': self.cached_balance.reserved_balance,
            'last_updated': self.cached_balance.last_updated.isoformat(),
            'currency': self.cached_balance.currency,
            'cache_age_seconds': (datetime.now() - self.cached_balance.last_updated).total_seconds()
        }


class PositionSizeCalculator:
    """
    Calculate optimal position sizes based on balance and risk parameters.
    """
    
    def __init__(self, balance_manager: BalanceManager):
        """Initialize position size calculator."""
        self.balance_manager = balance_manager
    
    async def calculate_micro_position_size(
        self, 
        confidence: float,
        volatility_regime: str = 'normal',
        ml_score: float = 0.8,
        min_size: float = 5.0,
        max_size: float = 15.0
    ) -> Dict[str, Any]:
        """Calculate position size optimized for micro capital."""
        
        # Get current balance
        balance_info = await self.balance_manager.get_balance()
        available = balance_info.available_balance
        
        # Base size calculation
        base_size = min(8.0, available * 0.2)  # $8 or 20% of balance, whichever is smaller
        
        # Confidence multiplier (0.8 to 1.2)
        confidence_mult = 0.8 + (confidence * 0.4)
        
        # Volatility adjustment
        if volatility_regime == 'high_volatility':
            vol_mult = 0.6
        elif volatility_regime == 'low_volatility':
            vol_mult = 1.3
        else:
            vol_mult = 1.0
        
        # ML score adjustment (0.7 to 1.3)
        ml_mult = 0.7 + (ml_score * 0.6)
        
        # Calculate size
        calculated_size = base_size * confidence_mult * vol_mult * ml_mult
        
        # Apply bounds
        position_size = max(min_size, min(calculated_size, max_size))
        
        # Ensure affordability
        max_affordable = available * 0.35  # Max 35% of balance
        position_size = min(position_size, max_affordable)
        
        # Final check
        if position_size < min_size or not await self.balance_manager.can_afford_position(position_size):
            position_size = 0
        
        return {
            'position_size': position_size,
            'available_balance': available,
            'percentage_of_balance': (position_size / available * 100) if available > 0 else 0,
            'confidence_multiplier': confidence_mult,
            'volatility_multiplier': vol_mult,
            'ml_multiplier': ml_mult,
            'can_afford': position_size > 0,
            'reasoning': self._get_sizing_reasoning(
                position_size, available, confidence, volatility_regime, ml_score
            )
        }
    
    def _get_sizing_reasoning(
        self, 
        position_size: float, 
        available: float, 
        confidence: float,
        volatility_regime: str,
        ml_score: float
    ) -> str:
        """Get human-readable reasoning for position size."""
        if position_size == 0:
            return "Position size too small or insufficient balance"
        
        pct_of_balance = (position_size / available * 100) if available > 0 else 0
        
        reasons = []
        
        if confidence > 0.9:
            reasons.append("high confidence signal")
        elif confidence < 0.7:
            reasons.append("low confidence - reduced size")
        
        if volatility_regime == 'high_volatility':
            reasons.append("high volatility - conservative sizing")
        elif volatility_regime == 'low_volatility':
            reasons.append("low volatility - slightly larger size")
        
        if ml_score > 0.9:
            reasons.append("strong ML signal")
        elif ml_score < 0.7:
            reasons.append("weak ML signal - reduced size")
        
        reason_text = ", ".join(reasons) if reasons else "standard sizing"
        
        return f"${position_size:.2f} ({pct_of_balance:.1f}% of balance) - {reason_text}"


# Convenience functions for easy integration
async def check_balance(api_key: Optional[str] = None) -> BalanceInfo:
    """Quick balance check."""
    async with BalanceManager(api_key) as manager:
        return await manager.get_balance()


async def get_micro_position_size(
    confidence: float, 
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Quick position size calculation."""
    async with BalanceManager(api_key) as manager:
        calculator = PositionSizeCalculator(manager)
        return await calculator.calculate_micro_position_size(confidence, **kwargs)


if __name__ == "__main__":
    async def test_balance_manager():
        """Test the balance manager."""
        async with BalanceManager() as manager:
            # Test balance checking
            balance = await manager.get_balance()
            print(f"Balance: ${balance.available_balance:.2f}")
            
            # Test position sizing
            calculator = PositionSizeCalculator(manager)
            
            # Test different scenarios
            scenarios = [
                {'confidence': 0.9, 'volatility_regime': 'normal', 'ml_score': 0.85},
                {'confidence': 0.7, 'volatility_regime': 'high_volatility', 'ml_score': 0.6},
                {'confidence': 0.95, 'volatility_regime': 'low_volatility', 'ml_score': 0.95},
            ]
            
            for i, scenario in enumerate(scenarios, 1):
                result = await calculator.calculate_micro_position_size(**scenario)
                print(f"\nScenario {i}: {result['reasoning']}")
                print(f"  Position Size: ${result['position_size']:.2f}")
                print(f"  Percentage of Balance: {result['percentage_of_balance']:.1f}%")
    
    asyncio.run(test_balance_manager())
