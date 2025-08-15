#!/usr/bin/env python3
"""
Run just the web UI without the trading strategy for testing
"""
import asyncio
import sys
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ui.web_server import web_server
from loguru import logger

async def main():
    """Run just the web server for UI testing."""
    print("🌐 Starting Predicto Bot Web Dashboard (UI Only Mode)")
    print("📊 Dashboard will be available at: http://localhost:8000")
    print("🔥 Press Ctrl+C to stop")
    print("-" * 50)
    
    # Configure logging
    logger.add("ui.log", level="INFO", rotation="1 day", retention="7 days")
    
    try:
        # Set some dummy status for testing
        web_server.update_bot_status(
            running=False,
            strategy="Demo Mode",
            total_trades=0,
            pnl=0.0,
            positions=[],
            orders=[],
            markets=[
                {
                    "id": "demo_market_1",
                    "question": "Will Bitcoin reach $100k by end of 2024?",
                    "volume": 50000.0,
                    "price": 0.65
                },
                {
                    "id": "demo_market_2", 
                    "question": "Will Ethereum reach $5k by end of 2024?",
                    "volume": 25000.0,
                    "price": 0.42
                }
            ]
        )
        
        # Run web server
        config = uvicorn.Config(
            web_server.app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
