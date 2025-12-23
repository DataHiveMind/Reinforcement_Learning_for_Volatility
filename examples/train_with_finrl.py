"""
Example: Training Volatility Trading Agent with FinRL

This script demonstrates how to use FinRL agents for volatility trading.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.volatility_env import VolatilityTradingEnv
from src.models.finrl_agent import FinRLEnsembleAgent, VolatilityFinRLAgent
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Prepare sample data for training.

    In production, replace this with your actual data loading logic.
    """
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        return pd.read_csv(data_path)

    # Generate sample data for demonstration
    logger.warning("Using synthetic data for demonstration. Replace with real market data.")

    n_days = 1000
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")

    # Synthetic price data
    np.random.seed(42)
    price = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.02))

    # Synthetic features
    data = pd.DataFrame(
        {
            "date": dates,
            "close": price,
            "high": price * (1 + np.abs(np.random.randn(n_days) * 0.01)),
            "low": price * (1 - np.abs(np.random.randn(n_days) * 0.01)),
            "volume": np.random.randint(1000000, 10000000, n_days),
            "volatility": np.abs(np.random.randn(n_days) * 0.2 + 0.15),
            "bid_ask_spread": np.abs(np.random.randn(n_days) * 0.002 + 0.001),
            "order_imbalance": np.random.randn(n_days) * 0.1,
        }
    )

    return data


def train_single_agent(
    env: gym.Env,
    model_name: str = "ppo",
    total_timesteps: int = 100000,
    save_path: str = "./models/finrl_single",
):
    """
    Train a single FinRL agent.

    Args:
        env: Training environment
        model_name: Name of RL algorithm
        total_timesteps: Number of training steps
        save_path: Path to save the model
    """
    logger.info(f"Training single {model_name.upper()} agent...")

    # Model-specific hyperparameters
    model_kwargs = {
        "ppo": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        },
        "sac": {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "ent_coef": "auto",
        },
        "a2c": {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
        },
    }

    # Initialize agent
    agent = VolatilityFinRLAgent(
        env=env,
        model_name=model_name,
        model_kwargs=model_kwargs.get(model_name, {}),
    )

    # Train
    agent.train(
        total_timesteps=total_timesteps,
        tb_log_name=f"finrl_{model_name}",
    )

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)

    logger.info(f"Model saved to {save_path}")

    return agent


def train_ensemble(
    env: gym.Env,
    total_timesteps: int = 100000,
    save_dir: str = "./models/finrl_ensemble",
):
    """
    Train an ensemble of FinRL agents.

    Args:
        env: Training environment
        total_timesteps: Number of training steps per agent
        save_dir: Directory to save ensemble models
    """
    logger.info("Training FinRL ensemble...")

    # Initialize ensemble
    ensemble = FinRLEnsembleAgent(
        env=env,
        model_names=["ppo", "a2c", "sac"],
    )

    # Train all agents
    ensemble.train_all(total_timesteps=total_timesteps)

    # Save ensemble
    ensemble.save_all(save_dir)

    logger.info(f"Ensemble saved to {save_dir}")

    return ensemble


def evaluate_agent(
    agent: VolatilityFinRLAgent,
    env: gym.Env,
    n_episodes: int = 10,
):
    """
    Evaluate a trained agent.

    Args:
        agent: Trained agent
        env: Evaluation environment
        n_episodes: Number of episodes to evaluate
    """
    logger.info(f"Evaluating agent for {n_episodes} episodes...")

    episode_rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward: float = 0.0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)
        logger.info(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    logger.info(f"Evaluation Results: Mean Reward = {mean_reward:.2f} ± {std_reward:.2f}")

    return mean_reward, std_reward


def main():
    parser = argparse.ArgumentParser(description="Train volatility trading agent with FinRL")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "ensemble"],
        help="Training mode: single agent or ensemble",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ppo",
        choices=["ppo", "a2c", "ddpg", "td3", "sac"],
        help="RL algorithm to use (for single mode)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/env/volatility_env.yaml",
        help="Path to environment config",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training data",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./models/finrl",
        help="Path to save trained model(s)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate after training",
    )

    args = parser.parse_args()

    # Load environment config
    if os.path.exists(args.env_config):
        env_config = load_config(args.env_config)
    else:
        logger.warning(f"Config not found at {args.env_config}, using defaults")
        env_config = {}

    # Prepare data
    data = prepare_data(args.data_path)

    # Create environment
    # Note: Adapt this to your actual environment initialization
    env = VolatilityTradingEnv(data=data, **env_config.get("env_params", {}))

    # Train based on mode
    if args.mode == "single":
        agent = train_single_agent(
            env=env,
            model_name=args.model,
            total_timesteps=args.timesteps,
            save_path=args.save_path,
        )

        if args.evaluate:
            evaluate_agent(agent, env)

    elif args.mode == "ensemble":
        ensemble = train_ensemble(
            env=env,
            total_timesteps=args.timesteps,
            save_dir=args.save_path,
        )

        if args.evaluate:
            logger.info("Ensemble evaluation not implemented in this example")

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
