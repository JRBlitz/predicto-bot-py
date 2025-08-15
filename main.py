#!/usr/bin/env python3
"""
Polymarket Trading Bot - Live Trading Entry Point
Focuses on profitable crypto trading with live Polymarket data.
"""
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.polymarket_client import PolymarketTradingClient
from src.strategies.profitable_momentum_strategy import ProfitableMomentumStrategy
from src.strategies.enhanced_profitable_strategy import EnhancedProfitableStrategy
from crypto_market_config import crypto_markets, get_crypto_trading_config
from config import TradingConfig


class LiveTradingBot:
    """Live Polymarket trading bot focused on crypto markets."""
    
    def __init__(self, config: TradingConfig):
        """Initialize the live trading bot."""
        self.config = config
        self.client = None
        self.strategy = None
        self.running = False
        
        # Performance tracking
        self.session_trades = 0
        self.session_profit = 0.0
        self.session_start = datetime.now()
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the bot."""
        logger.remove()
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
        )
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # File logging
        logger.add(
            "logs/trading_bot_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            level="INFO"
        )
    
    async def initialize(self):
        """Initialize the trading bot."""
        logger.info("🚀 INITIALIZING LIVE TRADING BOT")
        logger.info("=" * 60)
        
        # Validate configuration
        if not self.config.private_key:
            raise ValueError("Private key is required for live trading. Set PRIVATE_KEY environment variable.")
        
        # Initialize Polymarket client
        logger.info("Connecting to Polymarket...")
        self.client = PolymarketTradingClient(
            private_key=self.config.private_key,
            host=self.config.polymarket_api_url,
            test_mode=self.config.test_mode
        )
        await self.client.initialize()
        logger.success("✅ Connected to Polymarket")
        
        # Initialize strategy based on configuration
        if self.config.use_enhanced_strategy:
            self.strategy = EnhancedProfitableStrategy(self.client, self.config.get_strategy_config())
            logger.info("Using Enhanced Profitable Strategy")
        else:
            self.strategy = ProfitableMomentumStrategy(self.client, self.config.get_strategy_config())
            logger.info("Using Profitable Momentum Strategy")
        
        # Display configuration
        logger.info(f"Target Markets: {len(crypto_markets.get_diversified_portfolio())} crypto markets")
        logger.info(f"Base Position Size: ${self.config.base_position_size}")
        logger.info(f"Max Positions: {self.config.max_positions}")
        logger.info(f"Test Mode: {self.config.test_mode}")
        
        logger.success("✅ Bot initialized successfully")
    
    async def run(self):
        """Run the live trading bot."""
        logger.info("🎯 STARTING LIVE TRADING SESSION")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        self.running = True
        
        try:
            while self.running:
                await self._trading_cycle()
                await asyncio.sleep(self.config.update_frequency_seconds)
                
        except KeyboardInterrupt:
            logger.info("Received stop signal. Shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
        finally:
            await self._shutdown()
    
    async def _trading_cycle(self):
        """Execute one trading cycle."""
        try:
            # Get available crypto markets
            markets = await self.client.get_markets()
            if not markets:
                logger.warning("No markets available")
                return
            
            # Focus on diversified crypto portfolio [[memory:6262089]]
            target_markets = crypto_markets.get_diversified_portfolio(max_markets=self.config.max_markets)
            
            logger.info(f"🔄 Trading cycle - analyzing {len(target_markets)} markets")
            
            for market_id in target_markets:
                await self._analyze_and_trade_market(market_id, markets)
            
            # Display session stats
            self._display_session_stats()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    async def _analyze_and_trade_market(self, market_id: str, available_markets: list):
        """Analyze and potentially trade a specific market."""
        try:
            # Find market in available markets
            market_data = None
            for market in available_markets:
                if market.get('id') == market_id or market.get('market_id') == market_id:
                    market_data = market
                    break
            
            if not market_data:
                return
            
            # Get orderbook
            orderbook = await self.client.get_orderbook(market_id)
            if not orderbook:
                return
            
            # Prepare market data for strategy
            strategy_input = {
                'market': market_data,
                'orderbook': orderbook
            }
            
            # Analyze market
            signals = await self.strategy.analyze_market(strategy_input)
            signals['market_id'] = market_id
            signals['current_price'] = self._get_mid_price(orderbook)
            
            # Check for entry signals
            if signals['signal'] in ['BUY', 'SELL']:
                should_enter = await self.strategy.should_enter_position(signals)
                
                if should_enter:
                    await self._execute_trade(signals)
            
            # Check existing positions for exits
            await self._check_position_exits(market_id, signals)
            
        except Exception as e:
            logger.error(f"Error analyzing market {market_id}: {e}")
    
    def _get_mid_price(self, orderbook: dict) -> float:
        """Calculate mid price from orderbook."""
        try:
            if not orderbook.get('bids') or not orderbook.get('asks'):
                return None
            
            best_bid = float(orderbook['bids'][0]['price'])
            best_ask = float(orderbook['asks'][0]['price'])
            return (best_bid + best_ask) / 2
        except:
            return None
    
    async def _execute_trade(self, signals: dict):
        """Execute a trade based on signals."""
        try:
            market_id = signals['market_id']
            side = signals['signal']
            current_price = signals['current_price']
            
            if not current_price:
                logger.warning(f"No valid price for {market_id}")
                return
            
            # Calculate position size
            position_size = await self.strategy.calculate_position_size(signals)
            
            # Calculate order parameters
            if side == 'BUY':
                order_price = current_price * 1.001  # Slightly above mid
            else:
                order_price = current_price * 0.999  # Slightly below mid
            
            # Convert to order size (shares)
            order_size = str(int(position_size / order_price))
            order_price_str = f"{order_price:.6f}"
            
            logger.info(f"📈 EXECUTING {side} ORDER")
            logger.info(f"   Market: {market_id}")
            logger.info(f"   Size: {order_size} shares")
            logger.info(f"   Price: {order_price_str}")
            logger.info(f"   Value: ${position_size:.2f}")
            logger.info(f"   Confidence: {signals['confidence']:.2f}")
            
            # Place order
            order_result = await self.client.place_order(
                market_id=market_id,
                side=side.lower(),
                size=order_size,
                price=order_price_str,
                post_only=True
            )
            
            if order_result:
                self.session_trades += 1
                logger.success(f"✅ Order placed: {order_result.get('order_id', 'N/A')}")
            else:
                logger.error("❌ Failed to place order")
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def _check_position_exits(self, market_id: str, signals: dict):
        """Check if any positions should be exited."""
        try:
            # Get current positions
            positions = await self.client.get_positions()
            
            for position in positions:
                if position.get('market_id') == market_id:
                    should_exit = await self.strategy.should_exit_position(position, signals)
                    
                    if should_exit:
                        await self._exit_position(position, signals)
                        
        except Exception as e:
            logger.error(f"Error checking position exits: {e}")
    
    async def _exit_position(self, position: dict, signals: dict):
        """Exit a position."""
        try:
            market_id = position.get('market_id')
            current_price = signals.get('current_price')
            
            if not current_price:
                return
            
            # Calculate exit order
            side = 'SELL' if position.get('side') == 'BUY' else 'BUY'
            size = str(abs(int(position.get('size', 0))))
            
            if side == 'SELL':
                exit_price = current_price * 0.999  # Slightly below mid
            else:
                exit_price = current_price * 1.001  # Slightly above mid
            
            logger.info(f"🚪 EXITING POSITION")
            logger.info(f"   Market: {market_id}")
            logger.info(f"   Side: {side}")
            logger.info(f"   Size: {size}")
            logger.info(f"   Price: {exit_price:.6f}")
            
            # Place exit order
            exit_result = await self.client.place_order(
                market_id=market_id,
                side=side.lower(),
                size=size,
                price=f"{exit_price:.6f}",
                post_only=True
            )
            
            if exit_result:
                logger.success(f"✅ Exit order placed: {exit_result.get('order_id', 'N/A')}")
                
                # Update performance tracking
                pnl = self._calculate_pnl(position, current_price)
                self.session_profit += pnl
                self.strategy.update_performance(market_id, pnl)
            
        except Exception as e:
            logger.error(f"Error exiting position: {e}")
    
    def _calculate_pnl(self, position: dict, current_price: float) -> float:
        """Calculate P&L for a position."""
        try:
            entry_price = float(position.get('entry_price', 0))
            size = float(position.get('size', 0))
            side = position.get('side', 'BUY')
            
            if side == 'BUY':
                pnl = (current_price - entry_price) * abs(size)
            else:
                pnl = (entry_price - current_price) * abs(size)
            
            return pnl
        except:
            return 0.0
    
    def _display_session_stats(self):
        """Display current session statistics."""
        session_duration = datetime.now() - self.session_start
        hours_running = session_duration.total_seconds() / 3600
        
        if self.session_trades > 0:
            avg_profit = self.session_profit / self.session_trades
            logger.info(f"📊 Session: {self.session_trades} trades, "
                       f"${self.session_profit:.2f} P&L, "
                       f"${avg_profit:.2f} avg/trade, "
                       f"{hours_running:.1f}h")
    
    async def _shutdown(self):
        """Shutdown the bot gracefully."""
        logger.info("🛑 SHUTTING DOWN TRADING BOT")
        
        # Cancel all open orders
        try:
            orders = await self.client.get_orders()
            for order in orders:
                await self.client.cancel_order(order.get('order_id'))
                logger.info(f"Cancelled order: {order.get('order_id')}")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
        
        # Display final stats
        logger.info("Final Session Summary:")
        logger.info(f"  • Total Trades: {self.session_trades}")
        logger.info(f"  • Total P&L: ${self.session_profit:.2f}")
        logger.info(f"  • Session Duration: {datetime.now() - self.session_start}")
        
        if self.strategy:
            stats = self.strategy.get_performance_stats()
            logger.info(f"  • Strategy Stats: {stats}")
        
        self.running = False
        logger.success("✅ Bot shutdown complete")


async def main():
    """Main function."""
    try:
        # Load configuration
        config = TradingConfig()
        
        # Create and run bot
        bot = LiveTradingBot(config)
        await bot.initialize()
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())