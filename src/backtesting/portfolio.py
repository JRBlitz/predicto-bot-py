"""
Portfolio simulator for backtesting.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from loguru import logger


@dataclass
class Position:
    """Represents a trading position."""
    market_id: str
    side: str  # BUY or SELL
    size: float
    entry_price: float
    entry_timestamp: datetime
    commission: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class Trade:
    """Represents a completed trade."""
    market_id: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    entry_timestamp: datetime
    exit_timestamp: datetime
    pnl: float
    commission: float
    duration_minutes: float


class Portfolio:
    """Portfolio simulator for tracking positions and P&L during backtesting."""
    
    def __init__(self, initial_capital: float):
        """Initialize the portfolio."""
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.current_timestamp: Optional[datetime] = None
        
        # Performance tracking
        self.total_commission_paid = 0.0
        self.max_portfolio_value = initial_capital
        self.max_drawdown = 0.0
    
    def update_timestamp(self, timestamp: datetime):
        """Update current timestamp."""
        self.current_timestamp = timestamp
    
    def add_position(
        self,
        market_id: str,
        side: str,
        size: float,
        price: float,
        timestamp: datetime,
        commission: float = 0.0
    ):
        """Add or modify a position."""
        self.total_commission_paid += commission
        
        if market_id in self.positions:
            existing_pos = self.positions[market_id]
            
            # Check if this is closing a position
            if (existing_pos.side == 'BUY' and side == 'SELL') or \
               (existing_pos.side == 'SELL' and side == 'BUY'):
                
                # Closing position (fully or partially)
                close_size = min(abs(existing_pos.size), abs(size))
                
                # Calculate P&L for closed portion
                if existing_pos.side == 'BUY':
                    pnl = (price - existing_pos.entry_price) * close_size
                else:  # SELL
                    pnl = (existing_pos.entry_price - price) * close_size
                
                pnl -= commission  # Subtract commission from P&L
                
                # Create trade record
                duration = (timestamp - existing_pos.entry_timestamp).total_seconds() / 60
                trade = Trade(
                    market_id=market_id,
                    side=existing_pos.side,
                    size=close_size,
                    entry_price=existing_pos.entry_price,
                    exit_price=price,
                    entry_timestamp=existing_pos.entry_timestamp,
                    exit_timestamp=timestamp,
                    pnl=pnl,
                    commission=existing_pos.commission + commission,
                    duration_minutes=duration
                )
                self.closed_trades.append(trade)
                
                # Update cash
                self.cash += pnl
                
                # Update position size
                remaining_size = abs(existing_pos.size) - close_size
                if remaining_size > 0:
                    # Partial close - update existing position
                    existing_pos.size = remaining_size if existing_pos.side == 'BUY' else -remaining_size
                else:
                    # Full close - remove position
                    del self.positions[market_id]
                
                logger.debug(f"Closed position: {market_id} {side} {close_size} @ {price}, P&L: ${pnl:.2f}")
                
            else:
                # Same side - add to position
                total_size = abs(existing_pos.size) + abs(size)
                weighted_price = (
                    (existing_pos.entry_price * abs(existing_pos.size)) +
                    (price * abs(size))
                ) / total_size
                
                existing_pos.size = total_size if side == 'BUY' else -total_size
                existing_pos.entry_price = weighted_price
                existing_pos.commission += commission
                
                # Update cash for additional position
                cash_change = -(abs(size) * price + commission)
                self.cash += cash_change
                
                logger.debug(f"Added to position: {market_id} {side} {size} @ {price}")
        else:
            # New position
            position = Position(
                market_id=market_id,
                side=side,
                size=abs(size) if side == 'BUY' else -abs(size),
                entry_price=price,
                entry_timestamp=timestamp,
                commission=commission,
                current_price=price
            )
            self.positions[market_id] = position
            
            # Update cash
            cash_change = -(abs(size) * price + commission)
            self.cash += cash_change
            
            logger.debug(f"New position: {market_id} {side} {size} @ {price}")
    
    def update_position_prices(self, market_prices: Dict[str, float]):
        """Update current prices for all positions."""
        for market_id, position in self.positions.items():
            if market_id in market_prices:
                position.current_price = market_prices[market_id]
                
                # Calculate unrealized P&L
                if position.side == 'BUY':
                    position.unrealized_pnl = (position.current_price - position.entry_price) * abs(position.size)
                else:  # SELL
                    position.unrealized_pnl = (position.entry_price - position.current_price) * abs(position.size)
    
    def get_total_value(self, market_prices: Optional[Dict[str, float]] = None) -> float:
        """Get total portfolio value."""
        if market_prices:
            self.update_position_prices(market_prices)
        
        # Cash + unrealized P&L from positions
        total_value = self.cash
        for position in self.positions.values():
            total_value += position.unrealized_pnl
        
        return total_value
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions in dictionary format."""
        positions = []
        for position in self.positions.values():
            positions.append({
                'market_id': position.market_id,
                'side': position.side,
                'size': position.size,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'entry_timestamp': position.entry_timestamp,
                'commission': position.commission
            })
        return positions
    
    def get_closed_trades(self) -> List[Dict[str, Any]]:
        """Get closed trades in dictionary format."""
        trades = []
        for trade in self.closed_trades:
            trades.append({
                'market_id': trade.market_id,
                'side': trade.side,
                'size': trade.size,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'timestamp': trade.exit_timestamp,
                'pnl': trade.pnl,
                'commission': trade.commission,
                'duration_minutes': trade.duration_minutes
            })
        return trades
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get portfolio performance summary."""
        current_value = self.get_total_value()
        total_return = current_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Trade statistics
        trades = self.closed_trades
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in trades)
        avg_win = sum(t.pnl for t in trades if t.pnl > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(t.pnl for t in trades if t.pnl < 0) / losing_trades if losing_trades > 0 else 0
        
        # Calculate max drawdown
        if len(self.equity_curve) > 1:
            equity_values = [eq[1] for eq in self.equity_curve]
            peak = max(equity_values)
            current_dd = (peak - current_value) / peak * 100 if peak > 0 else 0
            self.max_drawdown = max(self.max_drawdown, current_dd)
        
        return {
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'cash': self.cash,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_commission': self.total_commission_paid,
            'max_drawdown_pct': self.max_drawdown,
            'open_positions': len(self.positions)
        }
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_curve = []
        self.current_timestamp = None
        self.total_commission_paid = 0.0
        self.max_portfolio_value = self.initial_capital
        self.max_drawdown = 0.0
