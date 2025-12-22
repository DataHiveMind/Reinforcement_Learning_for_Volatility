"""
Backtesting utilities for trading strategies.

Simulates trading with realistic market conditions and transaction costs.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from src.models.ddpg_agent import DDPGAgent
from src.models.ppo_agent import PPOAgent
from src.utils.evaluation import PerformanceMetrics
from src.utils.plotting import plot_portfolio_value, plot_returns

logger = logging.getLogger(__name__)


class Backtester:
    """Backtest trading strategies."""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
    ):
        """
        Initialize backtester.

        Args:
            data: Market data
            initial_capital: Starting capital
            transaction_cost: Transaction cost per trade (fraction)
            slippage: Slippage per trade (fraction)
        """
        self.data = data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        # Backtest state
        self.capital = initial_capital
        self.positions = np.zeros(5)  # 5 volatility strategies
        self.portfolio_history = []
        self.trade_history = []

        logger.info(
            f"Initialized Backtester with ${initial_capital:,.0f} capital, " f"{len(data)} periods"
        )

    def run_backtest(
        self,
        agent: Union[PPOAgent, DDPGAgent],
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest with agent.

        Args:
            agent: Trained RL agent
            start_idx: Start index
            end_idx: End index

        Returns:
            Backtest results
        """
        if end_idx is None:
            end_idx = len(self.data)

        logger.info(f"Running backtest from {start_idx} to {end_idx}...")

        # Reset state
        self.capital = self.initial_capital
        self.positions = np.zeros(5)
        self.portfolio_history = [self.initial_capital]
        self.trade_history = []

        # Run simulation
        for t in range(start_idx, end_idx):
            # Get observation (simplified - would need proper state building)
            obs = self._build_observation(t)

            # Get action from agent
            action = agent.predict(obs, deterministic=True)

            # Execute trade
            self._execute_trade(action, t)

            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(t)
            self.portfolio_history.append(portfolio_value)

        # Calculate metrics
        results = self._calculate_backtest_metrics()

        logger.info(
            f"Backtest completed: Final value = ${results['final_value']:,.0f}, "
            f"Return = {results['total_return']:.2%}"
        )

        return results

    def run_strategy_backtest(
        self,
        strategy_weights: pd.DataFrame,
        rebalance_freq: int = 1,
    ) -> Dict[str, Any]:
        """
        Backtest with predefined strategy weights.

        Args:
            strategy_weights: DataFrame with strategy allocation over time
            rebalance_freq: Rebalancing frequency

        Returns:
            Backtest results
        """
        logger.info("Running strategy backtest...")

        # Reset state
        self.capital = self.initial_capital
        self.positions = np.zeros(5)
        self.portfolio_history = [self.initial_capital]
        self.trade_history = []

        for t in range(len(strategy_weights)):
            if t % rebalance_freq == 0:
                # Get target weights
                target_weights = strategy_weights.iloc[t].to_numpy()

                # Rebalance
                self._execute_trade(target_weights, t)

            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(t)
            self.portfolio_history.append(portfolio_value)

        # Calculate metrics
        results = self._calculate_backtest_metrics()

        return results

    def _build_observation(self, t: int) -> np.ndarray:
        """Build observation vector (simplified)."""
        # This is simplified - actual implementation would use StateBuilder
        lookback = 20
        start_t = max(0, t - lookback)

        # Get recent data
        recent_data = self.data.iloc[start_t : t + 1]

        # Extract features (placeholder)
        obs = np.zeros(50)  # Placeholder state dimension

        # Fill with some basic features if available
        if "realized_volatility" in recent_data.columns:
            obs[0] = recent_data["realized_volatility"].iloc[-1]

        if "returns" in recent_data.columns:
            obs[1] = recent_data["returns"].iloc[-1]

        # Add position information
        obs[10:15] = self.positions

        return obs.astype(np.float32)

    def _execute_trade(self, target_weights: np.ndarray, t: int) -> None:
        """Execute trade with transaction costs."""
        # Calculate position changes
        position_change = target_weights - self.positions

        # Calculate costs
        turnover = np.sum(np.abs(position_change))
        costs = turnover * self.capital * (self.transaction_cost + self.slippage)

        # Apply costs
        self.capital -= costs

        # Update positions
        self.positions = target_weights.copy()

        # Record trade
        if turnover > 0.01:  # Only record significant trades
            self.trade_history.append(
                {
                    "time": t,
                    "turnover": turnover,
                    "costs": costs,
                    "positions": self.positions.copy(),
                }
            )

    def _calculate_portfolio_value(self, t: int) -> float:
        """Calculate current portfolio value."""
        # Get strategy returns
        strategy_returns = self._get_strategy_returns(t)

        # Calculate position values
        position_value = np.sum(self.positions * strategy_returns * self.capital)

        # Total value
        portfolio_value = self.capital + position_value

        return portfolio_value

    def _get_strategy_returns(self, t: int) -> np.ndarray:
        """Get returns for each strategy at time t."""
        # Placeholder - would use actual strategy returns from data
        strategy_cols = [
            "skew_return",
            "convexity_return",
            "dispersion_return",
            "vol_carry_return",
            "breakout_return",
        ]

        returns = np.zeros(5)
        for i, col in enumerate(strategy_cols):
            if col in self.data.columns and t < len(self.data):
                returns[i] = self.data[col].iloc[t]

        return returns

    def _calculate_backtest_metrics(self) -> Dict[str, Any]:
        """Calculate backtest performance metrics."""
        portfolio_series = pd.Series(self.portfolio_history)
        returns = portfolio_series.pct_change().dropna()

        # Use PerformanceMetrics
        metrics_calc = PerformanceMetrics(
            returns=returns.to_numpy(),
            prices=portfolio_series.to_numpy(),
        )

        metrics = metrics_calc.calculate_all_metrics()

        # Add backtest-specific metrics
        metrics["initial_value"] = self.initial_capital
        metrics["final_value"] = self.portfolio_history[-1]
        metrics["total_return"] = (self.portfolio_history[-1] / self.initial_capital) - 1.0
        metrics["num_trades"] = len(self.trade_history)

        if self.trade_history:
            total_costs = sum(trade["costs"] for trade in self.trade_history)
            metrics["total_costs"] = total_costs
            metrics["costs_pct"] = total_costs / self.initial_capital

        return metrics

    def save_results(
        self,
        output_dir: Union[str, Path],
        plot: bool = True,
    ) -> None:
        """
        Save backtest results.

        Args:
            output_dir: Output directory
            plot: Whether to generate plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics = self._calculate_backtest_metrics()

        import json

        with open(output_dir / "backtest_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save portfolio history
        portfolio_df = pd.DataFrame({"portfolio_value": self.portfolio_history})
        portfolio_df.to_csv(output_dir / "portfolio_history.csv")

        # Save trades
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            trades_df.to_csv(output_dir / "trade_history.csv")

        # Generate plots
        if plot:
            portfolio_series = pd.Series(self.portfolio_history)
            returns = portfolio_series.pct_change().dropna()

            plot_portfolio_value(
                portfolio_series,
                save_path=str(output_dir / "portfolio_value.png"),
            )

            plot_returns(
                returns,
                save_path=str(output_dir / "returns.png"),
            )

        logger.info(f"Saved backtest results to {output_dir}")
