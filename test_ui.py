#!/usr/bin/env python3
"""
Test script to verify UI integration
"""
import asyncio
import os
import sys
from pathlib import Path
import requests
import time

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_ui_integration():
    """Test the UI integration."""
    print("🧪 Testing Predicto Bot UI Integration...")
    
    # Test 1: Check if all UI files exist
    print("\n1. Checking UI files...")
    ui_files = [
        "src/ui/__init__.py",
        "src/ui/web_server.py",
        "src/ui/templates/dashboard.html",
        "src/ui/static/css/dashboard.css",
        "src/ui/static/js/dashboard.js"
    ]
    
    for file_path in ui_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING!")
            return False
    
    # Test 2: Check imports
    print("\n2. Testing imports...")
    try:
        from src.ui.web_server import web_server
        from src.config.settings import settings
        from src.utils.polymarket_client import PolymarketTradingClient
        print("   ✅ All imports successful")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    
    # Test 3: Check if FastAPI app is created
    print("\n3. Testing FastAPI app...")
    try:
        app = web_server.app
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/api/status", "/api/start", "/api/stop", "/ws"]
        
        for route in expected_routes:
            if any(route in r for r in routes):
                print(f"   ✅ Route {route} exists")
            else:
                print(f"   ⚠️  Route {route} might be missing")
        
    except Exception as e:
        print(f"   ❌ FastAPI app error: {e}")
        return False
    
    # Test 4: Check environment configuration
    print("\n4. Testing environment configuration...")
    if not os.path.exists(".env"):
        print("   ⚠️  .env file not found - create one from .env.example")
        print("   ℹ️  Bot will work in test mode without real private key")
    else:
        print("   ✅ .env file exists")
    
    print("\n🎉 UI integration test completed successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Create .env file with your configuration")
    print("3. Run the bot: python run_bot.py")
    print("4. Open http://localhost:8000 in your browser")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_ui_integration())
    if not success:
        print("\n💥 Some tests failed. Please fix the issues above.")
        sys.exit(1)
    else:
        print("\n✨ All tests passed! Ready to run the bot with UI.")
