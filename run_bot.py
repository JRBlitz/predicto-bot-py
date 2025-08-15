#!/usr/bin/env python3
"""
Startup script for Predicto Bot with Web UI
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import main

if __name__ == "__main__":
    print("🤖 Starting Predicto Bot with Web Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8000")
    print("🔥 Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
