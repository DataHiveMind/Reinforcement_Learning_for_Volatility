"""
Custom Gymnasium environment for volatility strategy trading.

State: Market microstructure, options, and portfolio features
Actions: Allocation weights to volatility strategies
Rewards: Risk-adjusted PnL with breakout detection bonuses
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.envs.reward_functions import RewardCalculator
from src.envs.state_representation import StateBuilder
from src.utils.config_parser import ConfigParser

logger = logging.getLogger(__name__)


class VolatilityTradingEnv(gym.Env):
    """Custom environment for volatility strategy allocation."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
        self,
        data: pd.DataFrame,
        config: Union[str, Path, Dict],
        initial_capital: float = 100000.0,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize volatility trading environment.

        Args:
            data: Market data with features
            config: Environment configuration
            initial_capital: Starting capital
            render_mode: Rendering mode for visualization
        """
        super().__init__()

        # Load config
        if isinstance(config, (str, Path)):
            self.config = ConfigParser(config).config
        else:
            self.config = config

        # Store data
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.render_mode = render_mode

        # Extract environment settings
        env_config = self.config.get("environment", {})
        state_config = self.config.get("state_space", {})
        action_config = self.config.get("action_space", {})
        reward_config = self.config.get("rewards", {})

        # State builder
        self.state_builder = StateBuilder(state_config)
        self.history_length = state_config.get("history_length", 20)

        # Reward calculator
        self.reward_calculator = RewardCalculator(reward_config)

        # Action space - continuous allocation to volatility strategies
        self.num_strategies = action_config.get("num_strategies", 5)
        self.allocation_bounds = action_config.get("allocation_bounds", [-1.0, 1.0])
        self.max_position_size = action_config.get("max_position_size", 0.5)
        self.leverage_constraint = action_config.get("leverage_constraint", 2.0)

        self.action_space = spaces.Box(
            low=self.allocation_bounds[0],
            high=self.allocation_bounds[1],
            shape=(self.num_strategies,),
            dtype=np.float32,
        )

        # Observation space - will be set after processing data
        sample_obs = self.state_builder.build_state(
            self.data.iloc[: self.history_length], self.portfolio_state
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_obs.shape,
            dtype=np.float32,
        )

        # Episode tracking
        self.current_step = 0
        self.max_steps = len(self.data) - self.history_length - 1
        self.episode = 0

        # Portfolio state
        self.capital = initial_capital
        self.positions = np.zeros(self.num_strategies)
        self.portfolio_value_history = []
        self.action_history = []
        self.reward_history = []

        logger.info(
            f"Initialized VolatilityTradingEnv: "
            f"{self.num_strategies} strategies, "
            f"{self.max_steps} max steps, "
            f"state dim={sample_obs.shape}"
        )

    @property
    def portfolio_state(self) -> Dict[str, float]:
        """Get current portfolio state."""
        portfolio_value = self.capital + np.sum(self.positions * self._get_strategy_values())

        return {
            "current_positions": self.positions.tolist(),
            "cash_balance": self.capital,
            "portfolio_value": portfolio_value,
            "realized_pnl": portfolio_value - self.initial_capital,
            "unrealized_pnl": np.sum(self.positions * self._get_strategy_values()),
            "position_age": self.current_step,
        }

    def _get_strategy_values(self) -> np.ndarray:
        """Get current values of volatility strategies."""
        # Get current market data
        current_data = self.data.iloc[self.current_step : self.current_step + self.history_length]

        # Extract strategy returns from data
        # Assuming data has columns like 'skew_return', 'convexity_return', etc.
        strategy_cols = [
            "skew_return",
            "convexity_return",
            "dispersion_return",
            "vol_carry_return",
            "breakout_return",
        ]

        strategy_returns = []
        for col in strategy_cols:
            if col in current_data.columns:
                strategy_returns.append(current_data[col].iloc[-1])
            else:
                # If column doesn't exist, use zero return
                strategy_returns.append(0.0)

        return np.array(strategy_returns, dtype=np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)

        # Reset state
        self.current_step = 0
        self.capital = self.initial_capital
        self.positions = np.zeros(self.num_strategies)
        self.portfolio_value_history = [self.initial_capital]
        self.action_history = []
        self.reward_history = []
        self.episode += 1

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        logger.debug(f"Reset environment (episode {self.episode})")

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Allocation weights for strategies

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip and normalize action
        action = np.clip(action, self.allocation_bounds[0], self.allocation_bounds[1])
        action = self._apply_constraints(action)

        # Store action
        self.action_history.append(action.copy())

        # Execute trades
        position_change = action - self.positions
        trading_costs = self._calculate_trading_costs(position_change)

        # Update positions
        old_positions = self.positions.copy()
        self.positions = action

        # Get strategy returns
        strategy_values = self._get_strategy_values()
        strategy_pnl = np.sum(self.positions * strategy_values * self.capital)

        # Update capital
        self.capital = self.capital + strategy_pnl - trading_costs

        # Calculate reward
        reward_info = self.reward_calculator.calculate_reward(
            pnl=strategy_pnl,
            positions=self.positions,
            old_positions=old_positions,
            capital=self.capital,
            data=self.data.iloc[self.current_step : self.current_step + self.history_length + 1],
        )
        reward = reward_info["total_reward"]

        # Update tracking
        portfolio_value = self.capital + np.sum(self.positions * strategy_values)
        self.portfolio_value_history.append(portfolio_value)
        self.reward_history.append(reward)

        # Move to next step
        self.current_step += 1

        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_steps

        # Get new observation
        obs = self._get_observation()
        info = self._get_info()
        info["reward_components"] = reward_info

        return obs, reward, terminated, truncated, info

    def _apply_constraints(self, action: np.ndarray) -> np.ndarray:
        """Apply position and leverage constraints."""
        # Clip individual positions
        action = np.clip(action, -self.max_position_size, self.max_position_size)

        # Apply leverage constraint
        total_leverage = np.sum(np.abs(action))
        if total_leverage > self.leverage_constraint:
            action = action * (self.leverage_constraint / total_leverage)

        return action

    def _calculate_trading_costs(self, position_change: np.ndarray) -> float:
        """Calculate transaction costs from position changes."""
        # Get cost parameters from config
        cost_config = self.config.get("rewards", {}).get("costs", {})
        bid_ask_spread = cost_config.get("bid_ask_spread", 0.001)

        # Simple linear cost model
        total_turnover = np.sum(np.abs(position_change))
        costs = total_turnover * bid_ask_spread * self.capital

        return costs

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Get historical data window
        data_window = self.data.iloc[self.current_step : self.current_step + self.history_length]

        # Build state representation
        obs = self.state_builder.build_state(data_window, self.portfolio_state)

        return obs.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state."""
        portfolio_value = self.portfolio_value_history[-1]

        info = {
            "step": self.current_step,
            "episode": self.episode,
            "portfolio_value": portfolio_value,
            "capital": self.capital,
            "positions": self.positions.copy(),
            "total_return": (portfolio_value / self.initial_capital) - 1.0,
        }

        if len(self.portfolio_value_history) > 1:
            returns = np.diff(self.portfolio_value_history) / self.portfolio_value_history[:-1]
            info["sharpe_ratio"] = (
                np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
            )

        return info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Portfolio Value: ${self.portfolio_value_history[-1]:,.2f}")
            print(f"Positions: {self.positions}")
            print(f"Capital: ${self.capital:,.2f}")
        elif self.render_mode == "rgb_array":
            # Return RGB array for video recording (placeholder)
            return np.zeros((400, 600, 3), dtype=np.uint8)
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass
