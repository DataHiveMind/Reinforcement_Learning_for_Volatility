"""
RL agent training pipeline.

Orchestrates data loading, environment setup, agent training, and logging.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.data.feature_engineering import FeatureEngineeringPipeline
from src.data.loader import DataLoader
from src.data.preprocess import DataPreprocessor
from src.envs.volatility_env import VolatilityTradingEnv
from src.models.ddpg_agent import DDPGAgent
from src.models.ppo_agent import PPOAgent
from src.utils.config_parser import ConfigParser
from src.utils.logging_utils import ExperimentLogger

logger = logging.getLogger(__name__)


class Trainer:
    """Training pipeline for RL agents."""

    def __init__(
        self,
        config_path: Union[str, Path],
        data_path: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            config_path: Path to training configuration
            data_path: Path to training data
            experiment_name: Name for experiment tracking
        """
        self.config = ConfigParser(config_path).config
        if self.config is None:
            raise ValueError("Failed to load configuration")

        self.data_path = data_path
        self.experiment_name = experiment_name or self.config.get("experiment", {}).get(
            "name", "rl_training"
        )

        # Extract config sections
        self.experiment_config = self.config.get("experiment", {})
        self.algorithm_config = self.config.get("algorithm", {})
        self.env_config_path = self.config.get("environment", {}).get("config")
        self.training_config = self.config.get("training", {})

        # Output directory
        self.output_dir = Path(self.experiment_config.get("output_dir", "experiments/default/"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize experiment logger
        self.logger = ExperimentLogger(
            experiment_name=self.experiment_name,
        )

        # Agent placeholder
        self.agent = None
        self.train_data = None
        self.val_data = None

        logger.info(f"Initialized Trainer for experiment: {self.experiment_name}")

    def load_data(
        self,
        train_split: float = 0.8,
        apply_preprocessing: bool = True,
    ) -> None:
        """
        Load and preprocess training data.

        Args:
            train_split: Fraction of data for training
            apply_preprocessing: Whether to apply preprocessing
        """
        logger.info("Loading training data...")

        # Load data
        loader = DataLoader()
        if self.data_path:
            data = loader.load_parquet(self.data_path)
        else:
            raise ValueError("data_path must be provided")

        # Preprocess
        if apply_preprocessing:
            preprocessor = DataPreprocessor()
            data = preprocessor.handle_missing_values(data, method="forward_fill")  # type: ignore[arg-type]
            # Note: handle_outliers not implemented in DataPreprocessor yet

        # Ensure pandas DataFrame
        if hasattr(data, "to_pandas"):
            data = data.to_pandas()  # type: ignore[attr-defined]

        # Feature engineering
        feature_config = self.config.get("features", {})  # type: ignore[union-attr]
        if feature_config.get("microstructure", {}).get("enabled", False):
            pipeline = FeatureEngineeringPipeline(
                microstructure_config=feature_config.get("microstructure", {}).get("config"),
                options_config=feature_config.get("options", {}).get("config"),
            )
            data = pipeline.engineer_features(data)  # type: ignore[arg-type]

        # Train/val split
        split_idx = int(len(data) * train_split)
        self.train_data = data.iloc[:split_idx]  # type: ignore[attr-defined]
        self.val_data = data.iloc[split_idx:]  # type: ignore[attr-defined]

        logger.info(
            f"Loaded data: {len(self.train_data)} train samples, "
            f"{len(self.val_data)} val samples"
        )

        # Log dataset info
        self.logger.log_params(
            {
                "train_samples": len(self.train_data),
                "val_samples": len(self.val_data),
                "train_start": str(self.train_data.index[0]),
                "train_end": str(self.train_data.index[-1]),
            }
        )

    def setup_agent(self) -> None:
        """Setup RL agent based on configuration."""
        algorithm_type = self.algorithm_config.get("type", "PPO").upper()

        logger.info(f"Setting up {algorithm_type} agent...")

        if algorithm_type == "PPO":
            self.agent = PPOAgent(
                env_config=self.env_config_path,
                training_config=dict(self.config),  # type: ignore[arg-type]
                output_dir=self.output_dir,
            )
        elif algorithm_type == "DDPG":
            self.agent = DDPGAgent(
                env_config=self.env_config_path,
                training_config=dict(self.config),  # type: ignore[arg-type]
                output_dir=self.output_dir,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_type}")

        # Build model
        policy_type = self.algorithm_config.get("policy_type", "MlpPolicy")
        self.agent.build_model(self.train_data, policy=policy_type)

        # Log hyperparameters
        self.logger.log_params(
            {
                "algorithm": algorithm_type,
                "policy": policy_type,
                **self._extract_hyperparams(),
            }
        )

    def train(
        self,
        total_timesteps: Optional[int] = None,
        eval_freq: int = 10000,
        checkpoint_freq: int = 50000,
    ) -> None:
        """
        Train agent.

        Args:
            total_timesteps: Total training timesteps (overrides config)
            eval_freq: Evaluation frequency
            checkpoint_freq: Checkpoint save frequency
        """
        if self.agent is None:
            raise ValueError("Agent not set up. Call setup_agent() first.")

        if total_timesteps is None:
            total_timesteps = self.training_config.get("total_timesteps", 1000000)

        assert isinstance(total_timesteps, int), "total_timesteps must be int"

        logger.info(f"Starting training for {total_timesteps} timesteps...")

        # Create eval environment
        eval_env = None
        if self.val_data is not None:
            eval_env = VolatilityTradingEnv(self.val_data, self.env_config_path)

        # Start MLflow run
        if self.logger:
            with self.logger.start_run():  # type: ignore[union-attr]
                # Train
                self.agent.train(
                    total_timesteps=total_timesteps,
                    eval_env=eval_env,
                    eval_freq=eval_freq,
                    checkpoint_freq=checkpoint_freq,
                )
        else:
            # Train without MLflow
            self.agent.train(
                total_timesteps=total_timesteps,
                eval_env=eval_env,
                eval_freq=eval_freq,
                checkpoint_freq=checkpoint_freq,
            )

        logger.info("Training completed successfully")

    def run_full_pipeline(
        self,
        train_split: float = 0.8,
        total_timesteps: Optional[int] = None,
    ) -> None:
        """
        Run complete training pipeline.

        Args:
            train_split: Train/val split ratio
            total_timesteps: Total training timesteps
        """
        logger.info("Starting full training pipeline...")

        # Load data
        self.load_data(train_split=train_split)

        # Setup agent
        self.setup_agent()

        # Train
        self.train(total_timesteps=total_timesteps)

        logger.info("Full pipeline completed")

    def _extract_hyperparams(self) -> Dict[str, Any]:
        """Extract hyperparameters for logging."""
        algorithm_type = self.algorithm_config.get("type", "PPO").upper()

        if algorithm_type == "PPO":
            config_key = "ppo"
        elif algorithm_type == "DDPG":
            config_key = "ddpg"
        else:
            return {}

        hyperparams = self.config.get(config_key, {})  # type: ignore[union-attr]

        # Flatten nested dicts
        flat_params = {}
        for key, value in hyperparams.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_params[f"{key}_{subkey}"] = subvalue
            else:
                flat_params[key] = value

        return flat_params
