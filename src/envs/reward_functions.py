"""
Reward function components for RL environment.

Implements multi-component reward: PnL, risk, costs, and bonuses.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RewardCalculator:
    """Calculate rewards from portfolio performance."""

    def __init__(self, config: Dict):
        """
        Initialize reward calculator.

        Args:
            config: Reward configuration
        """
        self.config = config
        self.primary_objective = config.get("primary_objective", "risk_adjusted_pnl")

        # Component weights
        components = config.get("components", {})
        self.pnl_weight = components.get("pnl", {}).get("weight", 1.0)
        self.pnl_scale = components.get("pnl", {}).get("scale", 10000)

        self.risk_weight = components.get("risk", {}).get("weight", -0.5)
        self.risk_metric = components.get("risk", {}).get("metric", "volatility")
        self.target_vol = components.get("risk", {}).get("target_vol", 0.15)

        self.cost_weight = components.get("costs", {}).get("weight", -1.0)

        self.turnover_weight = components.get("turnover", {}).get("weight", -0.1)

        self.breakout_bonus_weight = components.get("breakout_bonus", {}).get("weight", 2.0)
        self.breakout_threshold = components.get("breakout_bonus", {}).get("threshold", 0.02)

    def calculate_reward(
        self,
        pnl: float,
        positions: np.ndarray,
        old_positions: np.ndarray,
        capital: float,
        data: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Calculate total reward from components.

        Args:
            pnl: Period PnL
            positions: Current positions
            old_positions: Previous positions
            capital: Current capital
            data: Recent market data

        Returns:
            Dictionary with reward components
        """
        rewards = {}

        # 1. PnL component (scaled)
        rewards["pnl"] = (pnl / self.pnl_scale) * self.pnl_weight

        # 2. Risk penalty
        if self.risk_metric == "volatility":
            # Estimate portfolio volatility
            portfolio_return = pnl / capital if capital > 0 else 0.0
            # Simple penalty based on deviation from target
            risk_penalty = abs(portfolio_return) / self.target_vol
            rewards["risk"] = -risk_penalty * self.risk_weight
        else:
            rewards["risk"] = 0.0

        # 3. Transaction costs (already deducted from PnL, but add penalty)
        position_change = np.sum(np.abs(positions - old_positions))
        rewards["costs"] = -position_change * self.cost_weight

        # 4. Turnover penalty
        turnover = np.sum(np.abs(positions - old_positions))
        rewards["turnover"] = -turnover * self.turnover_weight

        # 5. Breakout detection bonus
        breakout_bonus = self._calculate_breakout_bonus(data, positions)
        rewards["breakout_bonus"] = breakout_bonus * self.breakout_bonus_weight

        # Total reward
        rewards["total_reward"] = sum(rewards.values())

        return rewards

    def _calculate_breakout_bonus(self, data: pd.DataFrame, positions: np.ndarray) -> float:
        """
        Calculate bonus for correctly positioning before volatility breakout.

        Args:
            data: Recent market data
            positions: Current positions

        Returns:
            Breakout bonus
        """
        # Check if volatility increased significantly
        if "realized_volatility" in data.columns and len(data) > 1:
            current_vol = data["realized_volatility"].iloc[-1]
            previous_vol = data["realized_volatility"].iloc[-2]

            vol_change = (current_vol - previous_vol) / (previous_vol + 1e-8)

            # If volatility spiked and we were positioned correctly
            if vol_change > self.breakout_threshold:
                # Reward having long volatility positions
                # Assume strategies 0-2 benefit from vol increase
                long_vol_exposure = np.sum(positions[:3])
                return max(0.0, long_vol_exposure) * vol_change
            elif vol_change < -self.breakout_threshold:
                # Reward having short volatility positions
                short_vol_exposure = np.sum(positions[3:])
                return max(0.0, short_vol_exposure) * abs(vol_change)

        return 0.0
