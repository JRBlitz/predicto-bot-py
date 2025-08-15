"""
FastAPI web server for Predicto Bot UI
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import asyncio
from typing import Dict, List, Any
from datetime import datetime
from loguru import logger
from ..config.settings import settings


class BotWebServer:
    """Web server for bot UI and API."""
    
    def __init__(self, bot_instance=None):
        self.app = FastAPI(title="Predicto Bot Dashboard", version="1.0.0")
        self.bot = bot_instance
        self.active_connections: List[WebSocket] = []
        self.bot_status = {
            "running": False,
            "strategy": None,
            "last_update": None,
            "total_trades": 0,
            "pnl": 0.0,
            "positions": [],
            "orders": [],
            "markets": []
        }
        
        # Setup templates
        self.templates = Jinja2Templates(directory="src/ui/templates")
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="src/ui/static"), name="static")
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup all API routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Main dashboard page."""
            return self.templates.TemplateResponse("dashboard.html", {
                "request": request,
                "bot_status": self.bot_status
            })
        
        @self.app.get("/api/status")
        async def get_status():
            """Get bot status."""
            return self.bot_status
        
        @self.app.post("/api/start")
        async def start_bot():
            """Start the bot."""
            if self.bot and not self.bot_status["running"]:
                # Start bot in background task
                asyncio.create_task(self._start_bot_background())
                self.bot_status["running"] = True
                self.bot_status["last_update"] = datetime.now().isoformat()
                await self._broadcast_status()
                return {"status": "started"}
            return {"status": "already_running"}
        
        @self.app.post("/api/stop")
        async def stop_bot():
            """Stop the bot."""
            if self.bot and self.bot_status["running"]:
                self.bot.stop()
                self.bot_status["running"] = False
                self.bot_status["last_update"] = datetime.now().isoformat()
                await self._broadcast_status()
                return {"status": "stopped"}
            return {"status": "already_stopped"}
        
        @self.app.get("/api/markets")
        async def get_markets():
            """Get available markets."""
            if self.bot and self.bot.client:
                try:
                    markets = await self.bot.client.get_markets()
                    self.bot_status["markets"] = markets[:10]  # Limit to first 10
                    return {"markets": markets}
                except Exception as e:
                    logger.error(f"Error getting markets: {e}")
                    return {"markets": [], "error": str(e)}
            return {"markets": []}
        
        @self.app.get("/api/positions")
        async def get_positions():
            """Get current positions."""
            if self.bot and self.bot.client:
                try:
                    positions = await self.bot.client.get_positions()
                    self.bot_status["positions"] = positions
                    return {"positions": positions}
                except Exception as e:
                    logger.error(f"Error getting positions: {e}")
                    return {"positions": [], "error": str(e)}
            return {"positions": []}
        
        @self.app.get("/api/orders")
        async def get_orders():
            """Get open orders."""
            if self.bot and self.bot.client:
                try:
                    orders = await self.bot.client.get_orders()
                    self.bot_status["orders"] = orders
                    return {"orders": orders}
                except Exception as e:
                    logger.error(f"Error getting orders: {e}")
                    return {"orders": [], "error": str(e)}
            return {"orders": []}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                # Send initial status
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "data": self.bot_status
                }))
                
                # Keep connection alive
                while True:
                    await websocket.receive_text()
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
    
    async def _start_bot_background(self):
        """Start bot in background."""
        try:
            if self.bot:
                await self.bot.run()
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            self.bot_status["running"] = False
            await self._broadcast_status()
    
    async def _broadcast_status(self):
        """Broadcast status update to all connected clients."""
        if self.active_connections:
            message = json.dumps({
                "type": "status",
                "data": self.bot_status
            })
            
            # Send to all connections
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except:
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for connection in disconnected:
                self.active_connections.remove(connection)
    
    def update_bot_status(self, **kwargs):
        """Update bot status and broadcast to clients."""
        self.bot_status.update(kwargs)
        self.bot_status["last_update"] = datetime.now().isoformat()
        
        # Broadcast in background
        asyncio.create_task(self._broadcast_status())
    
    def set_bot_instance(self, bot):
        """Set the bot instance."""
        self.bot = bot
        if hasattr(bot, 'name'):
            self.bot_status["strategy"] = bot.name


# Global web server instance
web_server = BotWebServer()
