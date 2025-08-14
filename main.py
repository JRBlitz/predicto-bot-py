"""
Main entry point for the Polymarket trading bot.
"""
import asyncio
import os
from loguru import logger
from src.config.settings import settings
from src.utils.polymarket_client import PolymarketTradingClient
from src.strategies.mean_reversion_strategy import MeanReversionStrategy


async def main():
    """Main function to run the trading bot."""
    logger.info("Starting Polymarket Trading Bot")
    
    # Check if private key is configured
    if not settings.private_key:
        logger.error("Private key not configured. Please set PRIVATE_KEY in .env file")
        return
    
    # Initialize trading client
    client = PolymarketTradingClient(
        private_key=settings.private_key,
        host=settings.polymarket_api_url
    )
    
    try:
        await client.initialize()
        logger.info("Trading client initialized successfully")
        
        # Strategy configuration
        strategy_config = {
            'lookback_period': 20,
            'deviation_threshold': 0.05,
            'position_size_pct': 0.1,
            'base_position_size': 100,
            'interval': 60  # Check every 60 seconds
        }
        
        # Initialize strategy
        strategy = MeanReversionStrategy(client, strategy_config)
        
        # Run the strategy
        await strategy.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        if 'strategy' in locals():
            strategy.stop()
    except Exception as e:
        logger.error(f"Error running bot: {e}")
    finally:
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    # Configure logging
    logger.add(
        settings.log_file,
        level=settings.log_level,
        rotation="1 day",
        retention="7 days"
    )
    
    # Run the bot
    asyncio.run(main())
