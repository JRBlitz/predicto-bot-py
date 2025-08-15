"""
Backtesting engine for testing trading strategies with historical data.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass, field
from enum import Enum

from ..strategies.base_strategy import BaseStrategy
from .portfolio import Portfolio
from .data_provider import HistoricalDataProvider
from .metrics import PerformanceMetrics


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class BacktestOrder:
    """Represents a backtesting order."""
    id: str
    market_id: str
    side: str  # BUY or SELL
    size: float
    price: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_timestamp: Optional[datetime] = None


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: List[Dict[str, Any]] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)


class BacktestingEngine:
    """Engine for backtesting trading strategies."""
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        """Initialize the backtesting engine."""
        self.initial_capital = initial_capital
        self.commission = commission
        self.portfolio = Portfolio(initial_capital)
        self.data_provider = HistoricalDataProvider()
        self.metrics = PerformanceMetrics()
        
        # State tracking
        self.current_time: Optional[datetime] = None
        self.orders: List[BacktestOrder] = []
        self.order_counter = 0
        
    async def run_backtest(
        self,
        strategy: BaseStrategy,
        market_ids: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1h"
    ) -> BacktestResult:
        """Run a backtest for a given strategy."""
        logger.info(f"Starting backtest for {strategy.name}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Markets: {market_ids}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        
        # Reset state
        self.portfolio = Portfolio(self.initial_capital)
        self.orders = []
        self.order_counter = 0
        
        # Get historical data
        historical_data = {}
        for market_id in market_ids:
            data = await self.data_provider.get_historical_data(
                market_id, start_date, end_date, timeframe
            )
            if not data.empty:
                historical_data[market_id] = data
            else:
                logger.warning(f"No historical data available for market {market_id}")
        
        if not historical_data:
            raise ValueError("No historical data available for any markets")
        
        # Create time index from all available data
        all_timestamps = set()
        for data in historical_data.values():
            all_timestamps.update(data.index)
        
        timestamps = sorted(all_timestamps)
        
        # Initialize strategy with mock client
        mock_client = MockTradingClient(self)
        original_client = strategy.client
        strategy.client = mock_client
        
        try:
            # Run backtest simulation
            for timestamp in timestamps:
                self.current_time = timestamp
                
                # Process orders first
                await self._process_pending_orders(historical_data, timestamp)
                
                # Update portfolio value
                self.portfolio.update_timestamp(timestamp)
                
                # Get current market data for this timestamp
                current_market_data = {}
                for market_id, data in historical_data.items():
                    if timestamp in data.index:
                        row = data.loc[timestamp]
                        current_market_data[market_id] = {
                            'market': {'id': market_id},
                            'orderbook': self._create_orderbook_from_ohlc(row),
                            'ohlc': row
                        }
                
                # Run strategy analysis for each market
                for market_id, market_data in current_market_data.items():
                    try:
                        # Analyze market
                        signals = await strategy.analyze_market(market_data)
                        
                        # Check for entry signals
                        if await strategy.should_enter_position(signals):
                            size = await strategy.calculate_position_size(signals)
                            await mock_client.place_order(
                                market_id=market_id,
                                side=signals.get('side', 'BUY'),
                                size=str(size),
                                price=str(signals.get('price', market_data['ohlc']['close']))
                            )
                        
                        # Check for exit signals
                        positions = self.portfolio.get_positions()
                        for position in positions:
                            if position['market_id'] == market_id:
                                if await strategy.should_exit_position(position, signals):
                                    await mock_client.place_order(
                                        market_id=market_id,
                                        side='SELL' if position['side'] == 'BUY' else 'BUY',
                                        size=str(abs(position['size'])),
                                        price=str(signals.get('exit_price', market_data['ohlc']['close']))
                                    )
                    
                    except Exception as e:
                        logger.error(f"Error processing market {market_id} at {timestamp}: {e}")
                        continue
                
                # Update equity curve
                equity = self.portfolio.get_total_value()
                self.portfolio.equity_curve.append((timestamp, equity))
        
        finally:
            # Restore original client
            strategy.client = original_client
        
        # Calculate final results
        result = await self._calculate_results(strategy.name, start_date, end_date)
        
        logger.info(f"Backtest completed for {strategy.name}")
        logger.info(f"Final capital: ${result.final_capital:,.2f}")
        logger.info(f"Total return: {result.total_return_pct:.2f}%")
        logger.info(f"Total trades: {result.total_trades}")
        logger.info(f"Win rate: {result.win_rate:.1f}%")
        
        return result
    
    def _create_orderbook_from_ohlc(self, ohlc_row: pd.Series) -> Dict[str, Any]:
        """Create a mock orderbook from OHLC data."""
        close_price = float(ohlc_row['close'])
        spread = 0.01  # 1% spread
        
        bid_price = close_price * (1 - spread / 2)
        ask_price = close_price * (1 + spread / 2)
        
        return {
            'bids': [{'price': str(bid_price), 'size': 1000}],
            'asks': [{'price': str(ask_price), 'size': 1000}]
        }
    
    async def _process_pending_orders(self, historical_data: Dict[str, pd.DataFrame], timestamp: datetime):
        """Process pending orders at the current timestamp."""
        for order in self.orders:
            if order.status != OrderStatus.PENDING:
                continue
            
            if order.market_id not in historical_data:
                continue
            
            data = historical_data[order.market_id]
            if timestamp not in data.index:
                continue
            
            row = data.loc[timestamp]
            
            # Simple fill logic: order fills if price is within OHLC range
            if row['low'] <= order.price <= row['high']:
                # Fill the order
                order.status = OrderStatus.FILLED
                order.fill_price = order.price
                order.fill_timestamp = timestamp
                
                # Update portfolio
                commission_cost = order.size * order.price * self.commission
                
                if order.side == 'BUY':
                    self.portfolio.add_position(
                        market_id=order.market_id,
                        side='BUY',
                        size=order.size,
                        price=order.price,
                        timestamp=timestamp,
                        commission=commission_cost
                    )
                else:  # SELL
                    self.portfolio.add_position(
                        market_id=order.market_id,
                        side='SELL',
                        size=-order.size,  # Negative for sell
                        price=order.price,
                        timestamp=timestamp,
                        commission=commission_cost
                    )
    
    async def _calculate_results(self, strategy_name: str, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Calculate backtest results."""
        final_capital = self.portfolio.get_total_value()
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Get all filled orders
        filled_orders = [o for o in self.orders if o.status == OrderStatus.FILLED]
        
        # Calculate trade statistics
        trades = self.portfolio.get_closed_trades()
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        # Calculate drawdown and Sharpe ratio
        equity_curve = self.portfolio.equity_curve
        if len(equity_curve) > 1:
            equity_values = [eq[1] for eq in equity_curve]
            peak = np.maximum.accumulate(equity_values)
            drawdown = (peak - equity_values) / peak
            max_drawdown = np.max(drawdown) * 100
            max_drawdown_abs = np.max(peak - equity_values)
            
            # Daily returns for Sharpe ratio
            daily_returns = np.diff(equity_values) / equity_values[:-1]
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        else:
            max_drawdown = 0
            max_drawdown_abs = 0
            sharpe_ratio = 0
            daily_returns = []
        
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown_abs,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=[{
                'timestamp': t['timestamp'],
                'market_id': t['market_id'],
                'side': t['side'],
                'size': t['size'],
                'entry_price': t.get('entry_price', t.get('price', 0)),
                'exit_price': t.get('exit_price', t.get('price', 0)),
                'pnl': t['pnl'],
                'commission': t['commission']
            } for t in trades],
            daily_returns=daily_returns.tolist() if len(daily_returns) > 0 else [],
            equity_curve=equity_curve
        )


class MockTradingClient:
    """Mock trading client for backtesting."""
    
    def __init__(self, engine: BacktestingEngine):
        self.engine = engine
    
    async def place_order(self, market_id: str, side: str, size: str, price: str, **kwargs) -> Dict[str, Any]:
        """Place a mock order."""
        order_id = f"backtest_{self.engine.order_counter}"
        self.engine.order_counter += 1
        
        order = BacktestOrder(
            id=order_id,
            market_id=market_id,
            side=side,
            size=float(size),
            price=float(price),
            timestamp=self.engine.current_time
        )
        
        self.engine.orders.append(order)
        
        return {
            "order_id": order_id,
            "status": "pending",
            "backtest": True
        }
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        return self.engine.portfolio.get_positions()
    
    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get orders."""
        return [
            {
                "id": order.id,
                "market_id": order.market_id,
                "side": order.side,
                "size": order.size,
                "price": order.price,
                "status": order.status.value
            }
            for order in self.engine.orders
        ]
    
    async def get_markets(self) -> List[Dict[str, Any]]:
        """Get markets (not used in backtesting)."""
        return []
    
    async def get_orderbook(self, market_id: str) -> Dict[str, Any]:
        """Get orderbook (not used in backtesting)."""
        return {}
