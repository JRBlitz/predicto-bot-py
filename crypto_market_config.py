"""
Crypto Market Configuration for Polymarket Trading Bot.
Focuses exclusively on Bitcoin, Ethereum, XRP, and Solana across multiple timeframes.
"""
from typing import Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class CryptoAsset(Enum):
    """Supported crypto assets."""
    BITCOIN = "bitcoin"
    ETHEREUM = "ethereum"
    XRP = "xrp"
    SOLANA = "solana"


class TimeFrame(Enum):
    """Trading timeframes."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class CryptoMarketConfig:
    """Configuration for crypto market trading."""
    asset: CryptoAsset
    timeframe: TimeFrame
    market_id: str
    description: str
    min_price: float = 0.01
    max_price: float = 0.99
    active: bool = True


class CryptoMarketManager:
    """Manages crypto market configurations and mappings."""
    
    def __init__(self):
        """Initialize crypto market configurations."""
        self.markets = self._initialize_crypto_markets()
        self.active_markets = [m for m in self.markets if m.active]
    
    def _initialize_crypto_markets(self) -> List[CryptoMarketConfig]:
        """Initialize all crypto market configurations."""
        markets = []
        
        # Bitcoin Markets
        markets.extend([
            CryptoMarketConfig(
                asset=CryptoAsset.BITCOIN,
                timeframe=TimeFrame.HOURLY,
                market_id="bitcoin_price_hourly",
                description="Bitcoin price prediction - hourly resolution"
            ),
            CryptoMarketConfig(
                asset=CryptoAsset.BITCOIN,
                timeframe=TimeFrame.DAILY,
                market_id="bitcoin_price_daily",
                description="Bitcoin price prediction - daily resolution"
            ),
            CryptoMarketConfig(
                asset=CryptoAsset.BITCOIN,
                timeframe=TimeFrame.WEEKLY,
                market_id="bitcoin_price_weekly",
                description="Bitcoin price prediction - weekly resolution"
            ),
            CryptoMarketConfig(
                asset=CryptoAsset.BITCOIN,
                timeframe=TimeFrame.MONTHLY,
                market_id="bitcoin_price_monthly",
                description="Bitcoin price prediction - monthly resolution"
            )
        ])
        
        # Ethereum Markets
        markets.extend([
            CryptoMarketConfig(
                asset=CryptoAsset.ETHEREUM,
                timeframe=TimeFrame.HOURLY,
                market_id="ethereum_price_hourly",
                description="Ethereum price prediction - hourly resolution"
            ),
            CryptoMarketConfig(
                asset=CryptoAsset.ETHEREUM,
                timeframe=TimeFrame.DAILY,
                market_id="ethereum_price_daily",
                description="Ethereum price prediction - daily resolution"
            ),
            CryptoMarketConfig(
                asset=CryptoAsset.ETHEREUM,
                timeframe=TimeFrame.WEEKLY,
                market_id="ethereum_price_weekly",
                description="Ethereum price prediction - weekly resolution"
            ),
            CryptoMarketConfig(
                asset=CryptoAsset.ETHEREUM,
                timeframe=TimeFrame.MONTHLY,
                market_id="ethereum_price_monthly",
                description="Ethereum price prediction - monthly resolution"
            )
        ])
        
        # XRP Markets
        markets.extend([
            CryptoMarketConfig(
                asset=CryptoAsset.XRP,
                timeframe=TimeFrame.HOURLY,
                market_id="xrp_price_hourly",
                description="XRP price prediction - hourly resolution"
            ),
            CryptoMarketConfig(
                asset=CryptoAsset.XRP,
                timeframe=TimeFrame.DAILY,
                market_id="xrp_price_daily",
                description="XRP price prediction - daily resolution"
            ),
            CryptoMarketConfig(
                asset=CryptoAsset.XRP,
                timeframe=TimeFrame.WEEKLY,
                market_id="xrp_price_weekly",
                description="XRP price prediction - weekly resolution"
            ),
            CryptoMarketConfig(
                asset=CryptoAsset.XRP,
                timeframe=TimeFrame.MONTHLY,
                market_id="xrp_price_monthly",
                description="XRP price prediction - monthly resolution"
            )
        ])
        
        # Solana Markets
        markets.extend([
            CryptoMarketConfig(
                asset=CryptoAsset.SOLANA,
                timeframe=TimeFrame.HOURLY,
                market_id="solana_price_hourly",
                description="Solana price prediction - hourly resolution"
            ),
            CryptoMarketConfig(
                asset=CryptoAsset.SOLANA,
                timeframe=TimeFrame.DAILY,
                market_id="solana_price_daily",
                description="Solana price prediction - daily resolution"
            ),
            CryptoMarketConfig(
                asset=CryptoAsset.SOLANA,
                timeframe=TimeFrame.WEEKLY,
                market_id="solana_price_weekly",
                description="Solana price prediction - weekly resolution"
            ),
            CryptoMarketConfig(
                asset=CryptoAsset.SOLANA,
                timeframe=TimeFrame.MONTHLY,
                market_id="solana_price_monthly",
                description="Solana price prediction - monthly resolution"
            )
        ])
        
        return markets
    
    def get_markets_by_asset(self, asset: CryptoAsset) -> List[CryptoMarketConfig]:
        """Get all markets for a specific crypto asset."""
        return [m for m in self.active_markets if m.asset == asset]
    
    def get_markets_by_timeframe(self, timeframe: TimeFrame) -> List[CryptoMarketConfig]:
        """Get all markets for a specific timeframe."""
        return [m for m in self.active_markets if m.timeframe == timeframe]
    
    def get_market_by_id(self, market_id: str) -> CryptoMarketConfig:
        """Get market configuration by ID."""
        for market in self.markets:
            if market.market_id == market_id:
                return market
        raise ValueError(f"Market ID {market_id} not found")
    
    def get_all_market_ids(self) -> List[str]:
        """Get all active market IDs."""
        return [m.market_id for m in self.active_markets]
    
    def get_diversified_portfolio(self, max_markets: int = 8) -> List[str]:
        """Get a diversified portfolio of markets across assets and timeframes."""
        portfolio = []
        
        # Ensure we have representation across all assets
        for asset in CryptoAsset:
            asset_markets = self.get_markets_by_asset(asset)
            if asset_markets:
                # Prefer daily and weekly for core positions
                daily_markets = [m for m in asset_markets if m.timeframe == TimeFrame.DAILY]
                weekly_markets = [m for m in asset_markets if m.timeframe == TimeFrame.WEEKLY]
                
                if daily_markets:
                    portfolio.append(daily_markets[0].market_id)
                if weekly_markets and len(portfolio) < max_markets:
                    portfolio.append(weekly_markets[0].market_id)
        
        # Fill remaining slots with hourly markets for more trading opportunities
        if len(portfolio) < max_markets:
            hourly_markets = self.get_markets_by_timeframe(TimeFrame.HOURLY)
            for market in hourly_markets:
                if market.market_id not in portfolio and len(portfolio) < max_markets:
                    portfolio.append(market.market_id)
        
        return portfolio[:max_markets]
    
    def get_strategy_config_for_timeframe(self, timeframe: TimeFrame) -> Dict[str, Any]:
        """Get optimized strategy configuration for specific timeframe."""
        base_config = {
            'max_position_size': 0.02,  # 2% per position
            'max_portfolio_risk': 0.08,  # 8% total portfolio risk
            'confidence_threshold': 0.5,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06
        }
        
        # Timeframe-specific adjustments
        timeframe_configs = {
            TimeFrame.HOURLY: {
                'lookback_period': 12,  # 12 hours
                'deviation_threshold': 0.02,  # More sensitive for short-term
                'momentum_threshold': 0.008,
                'position_hold_time_hours': 2,
                'update_frequency_minutes': 15
            },
            TimeFrame.DAILY: {
                'lookback_period': 14,  # 14 days
                'deviation_threshold': 0.03,
                'momentum_threshold': 0.012,
                'position_hold_time_hours': 24,
                'update_frequency_minutes': 60
            },
            TimeFrame.WEEKLY: {
                'lookback_period': 8,  # 8 weeks
                'deviation_threshold': 0.04,
                'momentum_threshold': 0.015,
                'position_hold_time_hours': 168,  # 1 week
                'update_frequency_minutes': 240  # 4 hours
            },
            TimeFrame.MONTHLY: {
                'lookback_period': 6,  # 6 months
                'deviation_threshold': 0.05,
                'momentum_threshold': 0.02,
                'position_hold_time_hours': 720,  # 30 days
                'update_frequency_minutes': 1440  # 24 hours
            }
        }
        
        config = base_config.copy()
        config.update(timeframe_configs.get(timeframe, timeframe_configs[TimeFrame.DAILY]))
        
        return config


# Global crypto market manager instance
crypto_markets = CryptoMarketManager()


def get_crypto_trading_config() -> Dict[str, Any]:
    """Get comprehensive crypto trading configuration."""
    return {
        'supported_assets': [asset.value for asset in CryptoAsset],
        'supported_timeframes': [tf.value for tf in TimeFrame],
        'total_markets': len(crypto_markets.active_markets),
        'diversified_portfolio': crypto_markets.get_diversified_portfolio(),
        'market_configs': {
            market.market_id: {
                'asset': market.asset.value,
                'timeframe': market.timeframe.value,
                'description': market.description,
                'strategy_config': crypto_markets.get_strategy_config_for_timeframe(market.timeframe)
            }
            for market in crypto_markets.active_markets
        }
    }


if __name__ == "__main__":
    # Display configuration
    config = get_crypto_trading_config()
    
    print("🚀 CRYPTO TRADING BOT CONFIGURATION")
    print("=" * 50)
    print(f"Supported Assets: {', '.join(config['supported_assets'])}")
    print(f"Supported Timeframes: {', '.join(config['supported_timeframes'])}")
    print(f"Total Markets: {config['total_markets']}")
    print(f"\n📊 Diversified Portfolio ({len(config['diversified_portfolio'])} markets):")
    
    for market_id in config['diversified_portfolio']:
        market_config = config['market_configs'][market_id]
        print(f"  • {market_id}: {market_config['asset'].upper()} ({market_config['timeframe']})")
    
    print(f"\n⚙️  Strategy Configurations by Timeframe:")
    for timeframe in TimeFrame:
        strategy_config = crypto_markets.get_strategy_config_for_timeframe(timeframe)
        print(f"\n{timeframe.value.upper()}:")
        print(f"  • Lookback: {strategy_config['lookback_period']} periods")
        print(f"  • Deviation Threshold: {strategy_config['deviation_threshold']:.3f}")
        print(f"  • Update Frequency: {strategy_config['update_frequency_minutes']} minutes")
        print(f"  • Position Hold Time: {strategy_config['position_hold_time_hours']} hours")
