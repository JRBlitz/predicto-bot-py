"""
Historical data provider for backtesting.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from loguru import logger
import json
import os
from pathlib import Path


class HistoricalDataProvider:
    """Provider for historical market data."""
    
    def __init__(self, data_dir: str = "data/historical"):
        """Initialize the data provider."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
    
    async def get_historical_data(
        self,
        market_id: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1h"
    ) -> pd.DataFrame:
        """Get historical OHLCV data for a market."""
        cache_key = f"{market_id}_{timeframe}_{start_date.date()}_{end_date.date()}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try to load from file first
        data_file = self.data_dir / f"{market_id}_{timeframe}.csv"
        
        if data_file.exists():
            try:
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                # Filter by date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                self.cache[cache_key] = df
                logger.info(f"Loaded {len(df)} historical data points for {market_id}")
                return df
            except Exception as e:
                logger.error(f"Error loading data file {data_file}: {e}")
        
        # If no file exists, generate synthetic data for demonstration
        logger.warning(f"No historical data file found for {market_id}, generating synthetic data")
        df = self._generate_synthetic_data(market_id, start_date, end_date, timeframe)
        
        # Save synthetic data for future use
        try:
            df.to_csv(data_file)
            logger.info(f"Saved synthetic data to {data_file}")
        except Exception as e:
            logger.error(f"Error saving synthetic data: {e}")
        
        self.cache[cache_key] = df
        return df
    
    def _generate_synthetic_data(
        self,
        market_id: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing purposes."""
        # Parse timeframe
        timeframe_minutes = self._parse_timeframe(timeframe)
        
        # Generate time series
        timestamps = pd.date_range(
            start=start_date,
            end=end_date,
            freq=f"{timeframe_minutes}min"
        )
        
        # Generate price data using geometric Brownian motion
        np.random.seed(hash(market_id) % 2**32)  # Consistent seed per market
        
        # Parameters for price simulation
        initial_price = 0.5  # Start at 50% probability
        volatility = 0.02  # 2% volatility per period
        drift = 0.0001  # Slight upward drift
        
        # Generate random returns
        returns = np.random.normal(drift, volatility, len(timestamps))
        
        # Calculate cumulative price
        log_prices = np.cumsum(returns)
        prices = initial_price * np.exp(log_prices)
        
        # Ensure prices stay within [0.01, 0.99] bounds for prediction markets
        prices = np.clip(prices, 0.01, 0.99)
        
        # Generate OHLC data
        data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            # Add some intrabar volatility
            high_low_range = price * 0.01  # 1% range
            
            open_price = price + np.random.normal(0, high_low_range * 0.3)
            close_price = price + np.random.normal(0, high_low_range * 0.3)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, high_low_range * 0.5))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, high_low_range * 0.5))
            
            # Ensure logical OHLC relationships and bounds
            high_price = min(high_price, 0.99)
            low_price = max(low_price, 0.01)
            open_price = np.clip(open_price, low_price, high_price)
            close_price = np.clip(close_price, low_price, high_price)
            
            # Volume (random but correlated with price movement)
            price_change = abs(close_price - open_price)
            base_volume = 1000
            volume = base_volume * (1 + price_change * 10) * np.random.lognormal(0, 0.5)
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=timestamps)
        return df
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to minutes."""
        timeframe = timeframe.lower()
        if timeframe.endswith('min') or timeframe.endswith('m'):
            return int(timeframe.rstrip('min').rstrip('m'))
        elif timeframe.endswith('h'):
            return int(timeframe.rstrip('h')) * 60
        elif timeframe.endswith('d'):
            return int(timeframe.rstrip('d')) * 24 * 60
        else:
            # Default to 1 hour
            return 60
    
    async def save_historical_data(
        self,
        market_id: str,
        data: pd.DataFrame,
        timeframe: str = "1h"
    ):
        """Save historical data to file."""
        data_file = self.data_dir / f"{market_id}_{timeframe}.csv"
        try:
            data.to_csv(data_file)
            logger.info(f"Saved {len(data)} data points for {market_id} to {data_file}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def get_available_markets(self) -> List[str]:
        """Get list of available markets with historical data."""
        markets = []
        for file_path in self.data_dir.glob("*.csv"):
            # Extract market_id from filename (format: market_id_timeframe.csv)
            filename = file_path.stem
            if '_' in filename:
                market_id = '_'.join(filename.split('_')[:-1])
                if market_id not in markets:
                    markets.append(market_id)
        return markets
    
    def create_sample_data_files(self):
        """Create sample historical data files for testing."""
        sample_markets = [
            "trump_2024_election",
            "bitcoin_100k_2024",
            "recession_2024",
            "ai_agi_2025",
            "climate_goals_2030"
        ]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 3 months of data
        
        for market_id in sample_markets:
            logger.info(f"Creating sample data for {market_id}")
            data = self._generate_synthetic_data(market_id, start_date, end_date, "1h")
            # Save synchronously to avoid nested event loop issues
            data_file = self.data_dir / f"{market_id}_1h.csv"
            try:
                data.to_csv(data_file)
                logger.info(f"Saved {len(data)} data points for {market_id} to {data_file}")
            except Exception as e:
                logger.error(f"Error saving data: {e}")
        
        logger.info(f"Created sample data files for {len(sample_markets)} markets")


class RealTimeDataProvider:
    """Provider for real-time market data (placeholder for future implementation)."""
    
    def __init__(self, client):
        """Initialize with trading client."""
        self.client = client
    
    async def get_current_market_data(self, market_id: str) -> Dict[str, Any]:
        """Get current market data."""
        try:
            orderbook = await self.client.get_orderbook(market_id)
            return {
                'market': {'id': market_id},
                'orderbook': orderbook,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting real-time data for {market_id}: {e}")
            return {}
    
    async def stream_market_data(self, market_ids: List[str], callback):
        """Stream real-time market data (placeholder)."""
        # This would implement WebSocket streaming in a real implementation
        while True:
            for market_id in market_ids:
                data = await self.get_current_market_data(market_id)
                if data:
                    await callback(market_id, data)
            await asyncio.sleep(1)  # 1 second updates
