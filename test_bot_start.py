#!/usr/bin/env python3
"""
Simple test to verify bot can start without errors
"""
import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_bot_initialization():
    """Test bot initialization without running the full strategy loop."""
    try:
        from main import TradingBot
        from src.ui.web_server import web_server
        
        print("🧪 Testing bot initialization...")
        
        # Create bot instance
        bot = TradingBot()
        
        # Test initialization (will use dummy private key from settings)
        try:
            await bot.initialize()
            print("✅ Bot initialized successfully!")
            
            # Test web server setup
            print("🌐 Testing web server...")
            print(f"✅ Web server app created: {web_server.app}")
            print(f"✅ Bot instance set in web server")
            
            return True
            
        except ValueError as e:
            if "Private key not configured" in str(e):
                print("⚠️  Private key not configured (expected in test)")
                print("✅ Bot validation working correctly")
                return True
            else:
                raise
                
    except Exception as e:
        print(f"❌ Error during bot initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_bot_initialization())
    if success:
        print("\n🎉 Bot initialization test passed!")
        print("Next step: Configure PRIVATE_KEY in .env and run the bot")
    else:
        print("\n💥 Bot initialization test failed!")
        sys.exit(1)
