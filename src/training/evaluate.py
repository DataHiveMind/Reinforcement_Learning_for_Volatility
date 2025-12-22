"""
Model evaluation utilities.

Evaluates trained RL agents on test data and computes performance metrics.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from src.envs.volatility_env import VolatilityTradingEnv
from src.models.ddpg_agent import DDPGAgent
from src.models.ppo_agent import PPOAgent
from src.utils.evaluation import PerformanceMetrics
from src.utils.plotting import plot_drawdown, plot_portfolio_value, plot_returns

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate RL agent performance."""

    def __init__(
        self,
        agent: Union[PPOAgent, DDPGAgent],
        test_data: pd.DataFrame,
        env_config: Union[str, Path, Dict],
        output_dir: Union[str, Path] = "experiments/eval/",
    ):
        """
        Initialize evaluator.

        Args:
            agent: Trained RL agent
            test_data: Test data
            env_config: Environment configuration
            output_dir: Directory for outputs
        """
        self.agent = agent
        self.test_data = test_data
        self.env_config = env_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluation results
        self.episode_returns = []
        self.episode_actions = []
        self.episode_rewards = []
        self.portfolio_values = []

        logger.info(f"Initialized Evaluator with {len(test_data)} test samples")

    def evaluate(
        self,
        n_episodes: int = 1,
        deterministic: bool = True,
        render: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate agent on test data.

        Args:
            n_episodes: Number of episodes to run
            deterministic: Use deterministic policy
            render: Render environment

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating agent for {n_episodes} episodes...")

        all_returns = []
        all_portfolio_values = []

        for episode in range(n_episodes):
            # Create environment
            env = VolatilityTradingEnv(
                self.test_data,
                self.env_config,
                render_mode="human" if render else None,
            )

            # Run episode
            episode_data = self._run_episode(
                env,
                deterministic=deterministic,
                render=render,
            )

            all_returns.extend(episode_data["returns"])
            all_portfolio_values.extend(episode_data["portfolio_values"])

            logger.info(
                f"Episode {episode + 1}/{n_episodes}: "
                f"Total return = {episode_data['total_return']:.2%}, "
                f"Sharpe = {episode_data['sharpe_ratio']:.2f}"
            )

        # Calculate aggregate metrics
        metrics = self._calculate_metrics(all_returns, all_portfolio_values)

        # Generate plots
        self._generate_plots(all_returns, all_portfolio_values)

        # Save results
        self._save_results(metrics)

        logger.info("Evaluation completed")
        return metrics

    def _run_episode(
        self,
        env: VolatilityTradingEnv,
        deterministic: bool = True,
        render: bool = False,
    ) -> Dict[str, Any]:
        """Run single evaluation episode."""
        obs, info = env.reset()

        done = False
        truncated = False
        episode_returns = []
        episode_actions = []
        episode_rewards = []
        portfolio_values = [env.initial_capital]

        while not (done or truncated):
            # Predict action
            action = self.agent.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)

            # Track metrics
            episode_returns.append(info.get("total_return", 0.0))
            episode_actions.append(action)
            episode_rewards.append(reward)
            portfolio_values.append(info.get("portfolio_value", env.initial_capital))

            if render:
                env.render()

        # Episode summary
        total_return = (portfolio_values[-1] / env.initial_capital) - 1.0
        returns_series = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = (
            np.mean(returns_series) / np.std(returns_series) * np.sqrt(252)
            if np.std(returns_series) > 0
            else 0.0
        )

        return {
            "returns": returns_series.tolist(),
            "actions": episode_actions,
            "rewards": episode_rewards,
            "portfolio_values": portfolio_values,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
        }

    def _calculate_metrics(
        self,
        returns: List[float],
        portfolio_values: List[float],
    ) -> Dict[str, Any]:
        """Calculate performance metrics."""
        returns_array = np.array(returns)
        prices_array = np.array(portfolio_values)

        # Use PerformanceMetrics class
        metrics_calculator = PerformanceMetrics(
            returns=returns_array,
            prices=prices_array,
        )

        metrics = metrics_calculator.calculate_all_metrics()

        # Add additional metrics
        metrics["final_portfolio_value"] = portfolio_values[-1]
        metrics["total_return"] = (portfolio_values[-1] / portfolio_values[0]) - 1.0
        metrics["num_periods"] = len(returns)

        return metrics

    def _generate_plots(
        self,
        returns: List[float],
        portfolio_values: List[float],
    ) -> None:
        """Generate evaluation plots."""
        logger.info("Generating evaluation plots...")

        returns_series = pd.Series(returns)
        portfolio_series = pd.Series(portfolio_values)

        # Portfolio value plot
        plot_portfolio_value(
            portfolio_series,
            save_path=str(self.output_dir / "portfolio_value.png"),
        )

        # Returns plot
        plot_returns(
            returns_series,
            save_path=str(self.output_dir / "returns.png"),
        )

        # Drawdown plot
        plot_drawdown(
            portfolio_series,
            save_path=str(self.output_dir / "drawdown.png"),
        )

    def _save_results(self, metrics: Dict[str, Any]) -> None:
        """Save evaluation results."""
        results_path = self.output_dir / "evaluation_metrics.json"

        import json

        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved evaluation metrics to {results_path}")


class RollingEvaluator:
    """Evaluate agent with rolling window."""

    def __init__(
        self,
        agent: Union[PPOAgent, DDPGAgent],
        data: pd.DataFrame,
        env_config: Union[str, Path, Dict],
        window_size: int = 252,
        step_size: int = 21,
    ):
        """
        Initialize rolling evaluator.

        Args:
            agent: Trained agent
            data: Full dataset
            env_config: Environment configuration
            window_size: Evaluation window size
            step_size: Step size for rolling window
        """
        self.agent = agent
        self.data = data
        self.env_config = env_config
        self.window_size = window_size
        self.step_size = step_size

    def evaluate_rolling(self) -> pd.DataFrame:
        """
        Evaluate agent on rolling windows.

        Returns:
            DataFrame with rolling metrics
        """
        logger.info("Starting rolling window evaluation...")

        results = []

        for start_idx in range(0, len(self.data) - self.window_size, self.step_size):
            end_idx = start_idx + self.window_size
            window_data = self.data.iloc[start_idx:end_idx]

            # Evaluate on window
            evaluator = Evaluator(
                self.agent,
                window_data,
                self.env_config,
            )

            metrics = evaluator.evaluate(n_episodes=1, deterministic=True)

            # Store results
            results.append(
                {
                    "start_date": window_data.index[0],
                    "end_date": window_data.index[-1],
                    **metrics,
                }
            )

            logger.info(
                f"Window {len(results)}: "
                f"{window_data.index[0]} to {window_data.index[-1]}, "
                f"Sharpe = {metrics.get('sharpe_ratio', 0):.2f}"
            )

        results_df = pd.DataFrame(results)
        logger.info("Rolling evaluation completed")

        return results_df
