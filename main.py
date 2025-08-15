"""
Main entry point for the Polymarket trading bot.
"""
import asyncio
import os
import uvicorn
from loguru import logger
from src.config.settings import settings
from src.utils.polymarket_client import PolymarketTradingClient
from src.strategies.mean_reversion_strategy import MeanReversionStrategy
from src.ui.web_server import web_server


class TradingBot:
    """Main trading bot class with UI integration."""
    
    def __init__(self):
        self.client = None
        self.strategy = None
        self.is_running = False
        
    async def initialize(self):
        """Initialize the trading bot."""
        logger.info("Initializing Polymarket Trading Bot")
        
        # Check if private key is configured
        if not settings.private_key:
            logger.error("Private key not configured. Please set PRIVATE_KEY in .env file")
            raise ValueError("Private key not configured")
        
        # Initialize trading client in TEST MODE by default
        self.client = PolymarketTradingClient(
            private_key=settings.private_key,
            host=settings.polymarket_api_url,
            test_mode=getattr(settings, 'test_mode', True)
        )
        
        await self.client.initialize()
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
        self.strategy = MeanReversionStrategy(self.client, strategy_config)
        
        # Set bot instance in web server
        web_server.set_bot_instance(self.strategy)
        
        logger.info("Bot initialized successfully")
    
    async def run(self):
        """Run the trading bot."""
        if not self.strategy:
            raise RuntimeError("Bot not initialized. Call initialize() first.")
        
        self.is_running = True
        web_server.update_bot_status(running=True, strategy=self.strategy.name)
        
        try:
            await self.strategy.run()
        except Exception as e:
            logger.error(f"Error in bot execution: {e}")
            web_server.update_bot_status(running=False)
            raise
    
    def stop(self):
        """Stop the trading bot."""
        if self.strategy:
            self.strategy.stop()
        self.is_running = False
        web_server.update_bot_status(running=False)
        logger.info("Bot stopped")


async def run_web_server():
    """Run the web server."""
    config = uvicorn.Config(
        web_server.app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    """Main function to run the trading bot with web UI."""
    # Initialize bot
    bot = TradingBot()
    
    try:
        await bot.initialize()
        logger.info("🚀 Predicto Bot initialized successfully")
        logger.info("🌐 Web dashboard will be available at: http://localhost:8000")
        
        # Run web server and bot concurrently
        await asyncio.gather(
            run_web_server(),
            return_exceptions=True
        )
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        bot.stop()
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        if bot:
            bot.stop()
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
