"""
Performance metrics calculator for backtesting results.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import math
from dataclasses import dataclass


@dataclass
class RiskMetrics:
    """Risk-adjusted performance metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional Value at Risk (95%)
    volatility: float
    downside_volatility: float


@dataclass
class TradeMetrics:
    """Trade-based performance metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    expectancy: float
    avg_trade_duration: float
    avg_bars_held: float


@dataclass
class ReturnMetrics:
    """Return-based performance metrics."""
    total_return: float
    total_return_pct: float
    annualized_return: float
    monthly_returns: List[float]
    best_month: float
    worst_month: float
    positive_months: int
    negative_months: int
    consecutive_wins: int
    consecutive_losses: int


class PerformanceMetrics:
    """Calculator for comprehensive backtesting performance metrics."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def calculate_all_metrics(
        self,
        equity_curve: List[Tuple[datetime, float]],
        trades: List[Dict[str, Any]],
        initial_capital: float,
        risk_free_rate: float = 0.02  # 2% annual risk-free rate
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not equity_curve or len(equity_curve) < 2:
            return self._empty_metrics()
        
        # Convert equity curve to DataFrame
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        df.set_index('timestamp', inplace=True)
        df['returns'] = df['equity'].pct_change().fillna(0)
        
        # Calculate metrics
        risk_metrics = self._calculate_risk_metrics(df, risk_free_rate)
        trade_metrics = self._calculate_trade_metrics(trades)
        return_metrics = self._calculate_return_metrics(df, initial_capital)
        
        return {
            'risk': risk_metrics.__dict__,
            'trades': trade_metrics.__dict__,
            'returns': return_metrics.__dict__,
            'summary': self._create_summary(risk_metrics, trade_metrics, return_metrics)
        }
    
    def _calculate_risk_metrics(self, df: pd.DataFrame, risk_free_rate: float) -> RiskMetrics:
        """Calculate risk-adjusted metrics."""
        returns = df['returns'].dropna()
        equity = df['equity']
        
        if len(returns) == 0:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Annualization factor (assuming daily data)
        periods_per_year = 252  # Trading days
        if len(returns) > 1:
            avg_period = (df.index[-1] - df.index[0]).total_seconds() / (len(returns) * 86400)
            if avg_period > 0:
                periods_per_year = 365.25 / avg_period
        
        # Basic statistics
        mean_return = returns.mean()
        volatility = returns.std()
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Drawdown calculation
        peak = equity.expanding().max()
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max()
        max_drawdown_abs = (peak - equity).max()
        
        # Sharpe Ratio
        excess_returns = mean_return - (risk_free_rate / periods_per_year)
        sharpe_ratio = (excess_returns / volatility * np.sqrt(periods_per_year)) if volatility > 0 else 0
        
        # Sortino Ratio
        sortino_ratio = (excess_returns / downside_volatility * np.sqrt(periods_per_year)) if downside_volatility > 0 else 0
        
        # Calmar Ratio
        annualized_return = (1 + mean_return) ** periods_per_year - 1
        calmar_ratio = (annualized_return / max_drawdown) if max_drawdown > 0 else 0
        
        # Value at Risk (VaR) and Conditional VaR (CVaR)
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        
        return RiskMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown_abs,
            max_drawdown_pct=max_drawdown * 100,
            var_95=var_95 * 100,
            cvar_95=cvar_95 * 100,
            volatility=volatility * np.sqrt(periods_per_year) * 100,
            downside_volatility=downside_volatility * np.sqrt(periods_per_year) * 100
        )
    
    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> TradeMetrics:
        """Calculate trade-based metrics."""
        if not trades:
            return TradeMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        total_trades = len(trades)
        pnls = [trade['pnl'] for trade in trades]
        winning_trades = len([pnl for pnl in pnls if pnl > 0])
        losing_trades = len([pnl for pnl in pnls if pnl < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        expectancy = np.mean(pnls) if pnls else 0
        
        # Duration metrics
        durations = [trade.get('duration_minutes', 0) for trade in trades]
        avg_trade_duration = np.mean(durations) if durations else 0
        avg_bars_held = avg_trade_duration / 60  # Convert to hours (assuming 1-hour bars)
        
        return TradeMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_trade_duration=avg_trade_duration,
            avg_bars_held=avg_bars_held
        )
    
    def _calculate_return_metrics(self, df: pd.DataFrame, initial_capital: float) -> ReturnMetrics:
        """Calculate return-based metrics."""
        if len(df) == 0:
            return ReturnMetrics(0, 0, 0, [], 0, 0, 0, 0, 0, 0)
        
        final_equity = df['equity'].iloc[-1]
        total_return = final_equity - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # Annualized return
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25 if days > 0 else 1
        annualized_return = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Monthly returns
        monthly_equity = df['equity'].resample('M').last()
        monthly_returns = monthly_equity.pct_change().fillna(0) * 100
        monthly_returns_list = monthly_returns.tolist()
        
        best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0
        worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0
        positive_months = len(monthly_returns[monthly_returns > 0])
        negative_months = len(monthly_returns[monthly_returns < 0])
        
        # Consecutive wins/losses
        returns = df['returns']
        consecutive_wins = self._max_consecutive(returns > 0)
        consecutive_losses = self._max_consecutive(returns < 0)
        
        return ReturnMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            monthly_returns=monthly_returns_list,
            best_month=best_month,
            worst_month=worst_month,
            positive_months=positive_months,
            negative_months=negative_months,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses
        )
    
    def _max_consecutive(self, boolean_series: pd.Series) -> int:
        """Calculate maximum consecutive True values."""
        if len(boolean_series) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for value in boolean_series:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _create_summary(self, risk: RiskMetrics, trades: TradeMetrics, returns: ReturnMetrics) -> Dict[str, Any]:
        """Create a summary of key metrics."""
        return {
            'total_return_pct': returns.total_return_pct,
            'annualized_return': returns.annualized_return,
            'max_drawdown_pct': risk.max_drawdown_pct,
            'sharpe_ratio': risk.sharpe_ratio,
            'win_rate': trades.win_rate,
            'total_trades': trades.total_trades,
            'profit_factor': trades.profit_factor,
            'volatility': risk.volatility
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics for cases with insufficient data."""
        return {
            'risk': RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0).__dict__,
            'trades': TradeMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).__dict__,
            'returns': ReturnMetrics(0, 0, 0, [], 0, 0, 0, 0, 0, 0).__dict__,
            'summary': {
                'total_return_pct': 0,
                'annualized_return': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'total_trades': 0,
                'profit_factor': 0,
                'volatility': 0
            }
        }
    
    def compare_strategies(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple strategy results."""
        if not results:
            return {}
        
        comparison = {
            'strategies': [],
            'rankings': {}
        }
        
        # Extract key metrics for comparison
        metrics_to_compare = [
            'total_return_pct',
            'annualized_return',
            'max_drawdown_pct',
            'sharpe_ratio',
            'win_rate',
            'profit_factor',
            'total_trades'
        ]
        
        for result in results:
            strategy_summary = result.get('summary', {})
            strategy_name = result.get('strategy_name', 'Unknown')
            
            comparison['strategies'].append({
                'name': strategy_name,
                'metrics': {metric: strategy_summary.get(metric, 0) for metric in metrics_to_compare}
            })
        
        # Rank strategies by different metrics
        for metric in metrics_to_compare:
            values = [(i, result.get('summary', {}).get(metric, 0)) for i, result in enumerate(results)]
            
            # Sort by metric (higher is better for most metrics, except max_drawdown_pct)
            reverse = metric != 'max_drawdown_pct'
            ranked = sorted(values, key=lambda x: x[1], reverse=reverse)
            
            comparison['rankings'][metric] = [
                {
                    'rank': i + 1,
                    'strategy': results[idx].get('strategy_name', 'Unknown'),
                    'value': value
                }
                for i, (idx, value) in enumerate(ranked)
            ]
        
        # Overall score (weighted combination of metrics)
        weights = {
            'total_return_pct': 0.25,
            'sharpe_ratio': 0.25,
            'max_drawdown_pct': -0.2,  # Negative weight (lower is better)
            'win_rate': 0.15,
            'profit_factor': 0.15
        }
        
        overall_scores = []
        for i, result in enumerate(results):
            summary = result.get('summary', {})
            score = sum(
                weights.get(metric, 0) * summary.get(metric, 0)
                for metric in weights.keys()
            )
            overall_scores.append((i, score))
        
        overall_ranked = sorted(overall_scores, key=lambda x: x[1], reverse=True)
        comparison['rankings']['overall'] = [
            {
                'rank': i + 1,
                'strategy': results[idx].get('strategy_name', 'Unknown'),
                'score': score
            }
            for i, (idx, score) in enumerate(overall_ranked)
        ]
        
        return comparison
