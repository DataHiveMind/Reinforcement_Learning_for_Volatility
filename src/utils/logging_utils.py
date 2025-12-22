"""
Logging utilities for training and evaluation.

Provides:
- Structured logging setup
- MLflow integration
- Tensorboard logging
- Metrics tracking
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlflow
import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str = __name__,
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    use_rich: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        use_rich: Use rich formatting for console output

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler
    if use_rich:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)

    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get or create a logger.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """Logger for RL experiments with MLflow integration."""

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            run_name: Name of the run
            tracking_uri: MLflow tracking URI
            artifact_location: Artifact storage location
        """
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Set up MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

        if artifact_location:
            self.artifact_location = Path(artifact_location)
            self.artifact_location.mkdir(parents=True, exist_ok=True)
        else:
            self.artifact_location = None

        self.run = None
        self.step = 0

    def start_run(self, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Start MLflow run.

        Args:
            tags: Optional tags for the run
        """
        self.run = mlflow.start_run(run_name=self.run_name)

        if tags:
            mlflow.set_tags(tags)

    def end_run(self) -> None:
        """End MLflow run."""
        if self.run:
            mlflow.end_run()
            self.run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters
        """
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics
            step: Step/iteration number
        """
        if step is None:
            step = self.step
            self.step += 1

        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Step number
        """
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, local_path: Union[str, Path]) -> None:
        """
        Log artifact to MLflow.

        Args:
            local_path: Path to artifact file
        """
        mlflow.log_artifact(str(local_path))

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        **kwargs,
    ) -> None:
        """
        Log model to MLflow.

        Args:
            model: Model object
            artifact_path: Path within artifact store
            **kwargs: Additional arguments
        """
        mlflow.pytorch.log_model(model, artifact_path, **kwargs)

    def log_figure(
        self,
        figure: Any,
        artifact_file: str,
    ) -> None:
        """
        Log matplotlib figure.

        Args:
            figure: Matplotlib figure
            artifact_file: File name for artifact
        """
        mlflow.log_figure(figure, artifact_file)

    def log_dataframe(
        self,
        df: pd.DataFrame,
        artifact_file: str,
    ) -> None:
        """
        Log DataFrame as CSV artifact.

        Args:
            df: DataFrame to log
            artifact_file: File name for artifact
        """
        if self.artifact_location:
            filepath = self.artifact_location / artifact_file
            df.to_csv(filepath, index=False)
            self.log_artifact(filepath)

    @staticmethod
    def _flatten_dict(
        d: Dict,
        parent_key: str = "",
        sep: str = ".",
    ) -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ExperimentLogger._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    mlflow_logging: bool = True,
) -> None:
    """
    Log metrics to console and MLflow.

    Args:
        metrics: Dictionary of metrics
        step: Step number
        logger: Logger instance
        mlflow_logging: Whether to log to MLflow
    """
    # Console logging
    if logger:
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {step}: {metrics_str}" if step else metrics_str)

    # MLflow logging
    if mlflow_logging and mlflow.active_run():
        mlflow.log_metrics(metrics, step=step)


class MetricsTracker:
    """Track and aggregate metrics over time."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {}
        self.history = {}

    def update(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Update metrics.

        Args:
            metrics: Dictionary of metrics
            step: Step number
        """
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []

            if step is not None:
                self.history[key].append((step, value))
            else:
                self.history[key].append(value)

        self.metrics.update(metrics)

    def get_metric(self, key: str) -> Optional[float]:
        """Get current value of a metric."""
        return self.metrics.get(key)

    def get_history(self, key: str) -> list:
        """Get history of a metric."""
        return self.history.get(key, [])

    def get_statistics(self, key: str) -> Dict[str, float]:
        """
        Get statistics for a metric.

        Args:
            key: Metric name

        Returns:
            Dictionary with mean, std, min, max
        """
        history = self.get_history(key)

        if not history:
            return {}

        values = [v if isinstance(v, (int, float)) else v[1] for v in history]

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {}
        self.history = {}

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to DataFrame."""
        data = {}

        for key, history in self.history.items():
            if history and isinstance(history[0], tuple):
                # Has step information
                steps, values = zip(*history)
                data[key] = pd.Series(values, index=steps)
            else:
                data[key] = pd.Series(history)

        return pd.DataFrame(data)


class TensorboardLogger:
    """Wrapper for Tensorboard logging."""

    def __init__(self, log_dir: Union[str, Path]):
        """
        Initialize Tensorboard logger.

        Args:
            log_dir: Directory for Tensorboard logs
        """
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalars."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log histogram."""
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, img_tensor: np.ndarray, step: int) -> None:
        """Log image."""
        self.writer.add_image(tag, img_tensor, step)

    def close(self) -> None:
        """Close writer."""
        self.writer.close()
