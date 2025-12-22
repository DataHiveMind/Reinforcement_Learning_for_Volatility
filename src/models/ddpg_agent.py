"""
DDPG (Deep Deterministic Policy Gradient) agent implementation.

Wrapper around Stable-Baselines3 DDPG for continuous control.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from src.envs.volatility_env import VolatilityTradingEnv
from src.utils.config_parser import ConfigParser

logger = logging.getLogger(__name__)


class DDPGAgent:
    """DDPG agent for volatility trading."""

    def __init__(
        self,
        env_config: Union[str, Path, Dict],
        training_config: Union[str, Path, Dict],
        output_dir: Union[str, Path] = "experiments/ddpg/",
    ):
        """
        Initialize DDPG agent.

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

        # Extract DDPG hyperparameters
        ddpg_config = self.training_config.get("ddpg", {})
        self.learning_rate = ddpg_config.get("learning_rate", 1e-3)
        self.buffer_size = ddpg_config.get("buffer_size", 1000000)
        self.learning_starts = ddpg_config.get("learning_starts", 100)
        self.batch_size = ddpg_config.get("batch_size", 100)
        self.tau = ddpg_config.get("tau", 0.005)
        self.gamma = ddpg_config.get("gamma", 0.99)
        self.train_freq = ddpg_config.get("train_freq", 1)
        self.gradient_steps = ddpg_config.get("gradient_steps", 1)

        # Action noise
        self.action_noise_type = ddpg_config.get("action_noise", {}).get(
            "type", "ornstein_uhlenbeck"
        )
        self.noise_std = ddpg_config.get("action_noise", {}).get("std", 0.1)

        # Model placeholder
        self.model: Optional[DDPG] = None
        self.env = None

        logger.info(
            f"Initialized DDPGAgent with lr={self.learning_rate}, "
            f"buffer_size={self.buffer_size}, batch_size={self.batch_size}"
        )

    def build_model(
        self,
        train_data,
        policy: str = "MlpPolicy",
        verbose: int = 1,
    ) -> None:
        """
        Build DDPG model.

        Args:
            train_data: Training data for environment
            policy: Policy architecture
            verbose: Verbosity level
        """
        # Create environment
        self.env = VolatilityTradingEnv(train_data, self.env_config)

        # Setup action noise
        n_actions = self.env.action_space.shape[0]
        if self.action_noise_type == "ornstein_uhlenbeck":
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=self.noise_std * np.ones(n_actions),
            )
        else:
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=self.noise_std * np.ones(n_actions),
            )

        # Network architecture
        network_config = self.training_config.get("network", {})
        policy_network = network_config.get("policy_network", {})
        net_arch = policy_network.get("net_arch", [256, 256])

        policy_kwargs = {
            "net_arch": net_arch,
        }

        # Create DDPG model
        self.model = DDPG(
            policy=policy,
            env=self.env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            action_noise=action_noise,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(self.output_dir / "tensorboard"),
        )

        logger.info("Built DDPG model")

    def train(
        self,
        total_timesteps: int,
        eval_env=None,
        eval_freq: int = 10000,
        checkpoint_freq: int = 50000,
    ) -> None:
        """
        Train DDPG agent.

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
            name_prefix="ddpg_model",
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
        logger.info(f"Starting DDPG training for {total_timesteps} timesteps")
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
        self.model = DDPG.load(str(path))
        logger.info(f"Loaded model from {path}")
