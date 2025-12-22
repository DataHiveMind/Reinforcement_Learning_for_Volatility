"""
PPO (Proximal Policy Optimization) agent implementation.

Wrapper around Stable-Baselines3 PPO with custom configurations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from src.envs.volatility_env import VolatilityTradingEnv
from src.models.networks import CustomActorCriticPolicy
from src.utils.config_parser import ConfigParser

logger = logging.getLogger(__name__)


class PPOAgent:
    """PPO agent for volatility trading."""

    def __init__(
        self,
        env_config: Union[str, Path, Dict],
        training_config: Union[str, Path, Dict],
        output_dir: Union[str, Path] = "experiments/ppo/",
    ):
        """
        Initialize PPO agent.

        Args:
            env_config: Environment configuration
            training_config: Training configuration
            output_dir: Directory for outputs
        """
        # Load configs
        if isinstance(env_config, (str, Path)):
            self.env_config = ConfigParser(env_config).config
        else:
            self.env_config = env_config

        if isinstance(training_config, (str, Path)):
            self.training_config = ConfigParser(training_config).config
        else:
            self.training_config = training_config

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract PPO hyperparameters
        ppo_config = self.training_config.get("ppo", {})
        self.learning_rate = ppo_config.get("learning_rate", 3e-4)
        self.n_steps = ppo_config.get("n_steps", 2048)
        self.batch_size = ppo_config.get("batch_size", 64)
        self.n_epochs = ppo_config.get("n_epochs", 10)
        self.gamma = ppo_config.get("gamma", 0.99)
        self.gae_lambda = ppo_config.get("gae_lambda", 0.95)
        self.clip_range = ppo_config.get("clip_range", 0.2)
        self.ent_coef = ppo_config.get("ent_coef", 0.01)
        self.vf_coef = ppo_config.get("vf_coef", 0.5)
        self.max_grad_norm = ppo_config.get("max_grad_norm", 0.5)

        # Model placeholder
        self.model: Optional[PPO] = None
        self.env = None

        logger.info(
            f"Initialized PPOAgent with lr={self.learning_rate}, "
            f"n_steps={self.n_steps}, batch_size={self.batch_size}"
        )

    def build_model(
        self,
        train_data,
        policy: str = "MlpPolicy",
        verbose: int = 1,
    ) -> None:
        """
        Build PPO model.

        Args:
            train_data: Training data for environment
            policy: Policy architecture
            verbose: Verbosity level
        """
        # Create environment
        env_settings = self.training_config.get("environment", {})
        n_envs = env_settings.get("n_envs", 8)
        vec_env_type = env_settings.get("vec_env_type", "subproc")

        # Create vectorized environment
        def make_env():
            return VolatilityTradingEnv(train_data, self.env_config)

        self.env = make_vec_env(
            make_env,
            n_envs=n_envs,
            vec_env_cls=VecNormalize if vec_env_type == "dummy" else None,
        )

        # Network architecture
        network_config = self.training_config.get("network", {})
        policy_network = network_config.get("policy_network", {})
        net_arch = policy_network.get("net_arch", [dict(pi=[256, 256], vf=[256, 256])])

        policy_kwargs = {
            "net_arch": net_arch,
            "activation_fn": self._get_activation_fn(policy_network.get("activation_fn", "tanh")),
        }

        # Create PPO model
        self.model = PPO(
            policy=policy,
            env=self.env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(self.output_dir / "tensorboard"),
        )

        logger.info(f"Built PPO model with {n_envs} parallel environments")

    def train(
        self,
        total_timesteps: int,
        eval_env=None,
        eval_freq: int = 10000,
        checkpoint_freq: int = 50000,
    ) -> None:
        """
        Train PPO agent.

        Args:
            total_timesteps: Total training timesteps
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            checkpoint_freq: Checkpoint save frequency
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Setup callbacks
        callbacks = []

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(self.output_dir / "checkpoints"),
            name_prefix="ppo_model",
        )
        callbacks.append(checkpoint_callback)

        # Evaluation callback
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.output_dir / "best_model"),
                log_path=str(self.output_dir / "eval"),
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)

        callback_list = CallbackList(callbacks)

        # Train
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
        )

        # Save final model
        self.save(self.output_dir / "final_model.zip")
        logger.info("Training completed")

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Predict action for observation.

        Args:
            observation: Current observation
            deterministic: Use deterministic policy

        Returns:
            Action
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: Union[str, Path]) -> None:
        """Save model."""
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(str(path))
        logger.info(f"Saved model to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load model."""
        self.model = PPO.load(str(path))
        logger.info(f"Loaded model from {path}")

    @staticmethod
    def _get_activation_fn(name: str):
        """Get activation function by name."""
        import torch.nn as nn

        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
        }
        return activations.get(name.lower(), nn.Tanh)
