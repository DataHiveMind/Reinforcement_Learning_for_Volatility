"""
FinRL Agent Integration for Volatility Trading

This module provides wrappers and utilities to integrate FinRL agents
with the volatility trading environment.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from finrl.agents.stablebaselines3 import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class VolatilityFinRLAgent:
    """
    Wrapper class for FinRL agents adapted for volatility trading.

    This class provides a bridge between FinRL's trading agents and
    the custom volatility environment.
    """

    def __init__(
        self,
        env: gym.Env,
        model_name: str = "ppo",
        policy: str = "MlpPolicy",
        model_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
    ):
        """
        Initialize FinRL agent wrapper.

        Args:
            env: Gymnasium environment
            model_name: Name of RL algorithm (ppo, a2c, ddpg, td3, sac)
            policy: Policy network architecture
            model_kwargs: Additional arguments for the model
            verbose: Verbosity level
        """
        self.env = env
        self.model_name = model_name.lower()
        self.policy = policy
        self.verbose = verbose

        # Default model kwargs
        self.model_kwargs = {
            "learning_rate": 3e-4,
            "batch_size": 128,
            "gamma": 0.99,
            "tensorboard_log": "./tensorboard_logs/finrl",
        }
        if model_kwargs:
            self.model_kwargs.update(model_kwargs)

        # Initialize DRL agent
        self.agent = DRLAgent(env=env)
        self.model = None

        logger.info(f"Initialized FinRL {model_name.upper()} agent")

    def train(
        self,
        total_timesteps: int = 100000,
        eval_env: Optional[gym.Env] = None,
        callback: Optional[BaseCallback] = None,
        tb_log_name: Optional[str] = None,
    ) -> None:
        """
        Train the FinRL agent.

        Args:
            total_timesteps: Total number of training timesteps
            eval_env: Optional evaluation environment
            callback: Optional callback for training
            tb_log_name: TensorBoard log name
        """
        logger.info(f"Training {self.model_name.upper()} agent for {total_timesteps} steps")

        # Get the appropriate model from FinRL
        if self.model_name == "ppo":
            self.model = self.agent.get_model("ppo", policy=self.policy, **self.model_kwargs)
        elif self.model_name == "a2c":
            self.model = self.agent.get_model("a2c", policy=self.policy, **self.model_kwargs)
        elif self.model_name == "ddpg":
            self.model = self.agent.get_model(
                "ddpg",
                policy=self.policy if self.policy != "MlpPolicy" else "MlpPolicy",
                **self.model_kwargs,
            )
        elif self.model_name == "td3":
            self.model = self.agent.get_model(
                "td3",
                policy=self.policy if self.policy != "MlpPolicy" else "MlpPolicy",
                **self.model_kwargs,
            )
        elif self.model_name == "sac":
            self.model = self.agent.get_model(
                "sac",
                policy=self.policy if self.policy != "MlpPolicy" else "MlpPolicy",
                **self.model_kwargs,
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        # Train the model
        trained_model = self.agent.train_model(
            model=self.model,
            tb_log_name=tb_log_name or f"finrl_{self.model_name}",
            total_timesteps=total_timesteps,
        )

        self.model = trained_model
        logger.info(f"Training completed for {self.model_name.upper()}")

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Any]:
        """
        Predict action given observation.

        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, state)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str) -> None:
        """
        Save the trained model.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load a trained model.

        Args:
            path: Path to the saved model
        """
        logger.info(f"Loading model from {path}")

        # Load based on model type
        if self.model_name == "ppo":
            from stable_baselines3 import PPO

            self.model = PPO.load(path, env=self.env)
        elif self.model_name == "a2c":
            from stable_baselines3 import A2C

            self.model = A2C.load(path, env=self.env)
        elif self.model_name == "ddpg":
            from stable_baselines3 import DDPG

            self.model = DDPG.load(path, env=self.env)
        elif self.model_name == "td3":
            from stable_baselines3 import TD3

            self.model = TD3.load(path, env=self.env)
        elif self.model_name == "sac":
            from stable_baselines3 import SAC

            self.model = SAC.load(path, env=self.env)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        logger.info(f"Model loaded from {path}")


class FinRLEnsembleAgent:
    """
    Ensemble of FinRL agents for improved performance.

    This class trains multiple agents and combines their predictions
    for more robust trading decisions.
    """

    def __init__(
        self,
        env: gym.Env,
        model_names: Optional[list] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ensemble of FinRL agents.

        Args:
            env: Gymnasium environment
            model_names: List of model names to use in ensemble
            model_kwargs: Kwargs for each model
        """
        self.env = env
        self.model_names = model_names or ["ppo", "a2c", "sac"]

        # Initialize agents
        self.agents = {
            name: VolatilityFinRLAgent(
                env=env,
                model_name=name,
                model_kwargs=model_kwargs,
            )
            for name in self.model_names
        }

        logger.info(f"Initialized ensemble with models: {', '.join(self.model_names)}")

    def train_all(
        self,
        total_timesteps: int = 100000,
        **train_kwargs,
    ) -> None:
        """
        Train all agents in the ensemble.

        Args:
            total_timesteps: Training timesteps for each agent
            **train_kwargs: Additional training arguments
        """
        for name, agent in self.agents.items():
            logger.info(f"Training {name.upper()} in ensemble...")
            agent.train(total_timesteps=total_timesteps, **train_kwargs)

    def predict(
        self,
        observation: np.ndarray,
        method: str = "voting",
        deterministic: bool = True,
    ) -> Any:
        """
        Predict action using ensemble.

        Args:
            observation: Current observation
            method: Ensemble method ('voting', 'averaging', 'weighted')
            deterministic: Use deterministic policies

        Returns:
            Ensemble action
        """
        predictions = []

        for agent in self.agents.values():
            action, _ = agent.predict(observation, deterministic=deterministic)
            predictions.append(action)

        predictions = np.array(predictions)

        # Combine predictions
        if method == "averaging":
            return np.mean(predictions, axis=0)
        elif method == "voting":
            # For discrete actions
            return np.argmax(np.bincount(predictions.flatten().astype(int)))
        elif method == "weighted":
            # Could be extended with performance-based weighting
            return np.mean(predictions, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

    def save_all(self, directory: str) -> None:
        """
        Save all agents in the ensemble.

        Args:
            directory: Directory to save models
        """
        import os

        os.makedirs(directory, exist_ok=True)

        for name, agent in self.agents.items():
            path = os.path.join(directory, f"{name}_model")
            agent.save(path)

        logger.info(f"All ensemble models saved to {directory}")

    def load_all(self, directory: str) -> None:
        """
        Load all agents in the ensemble.

        Args:
            directory: Directory containing saved models
        """
        import os

        for name, agent in self.agents.items():
            path = os.path.join(directory, f"{name}_model")
            if os.path.exists(path + ".zip"):
                agent.load(path)
            else:
                logger.warning(f"Model {name} not found at {path}")
