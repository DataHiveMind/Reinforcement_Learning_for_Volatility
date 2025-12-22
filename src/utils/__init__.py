"""
Utility modules for configuration, evaluation, logging, and plotting.

This package provides:
- Configuration parsing and validation
- Performance evaluation metrics
- Structured logging utilities
- Visualization and plotting tools
"""

from .config_parser import ConfigParser, load_config, merge_configs
from .evaluation import (
    PerformanceMetrics,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from .logging_utils import get_logger, log_metrics, setup_logger
from .plotting import plot_drawdown, plot_portfolio_value, plot_returns, plot_training_curves

__all__ = [
    # Config
    "ConfigParser",
    "load_config",
    "merge_configs",
    # Evaluation
    "PerformanceMetrics",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_calmar_ratio",
    # Logging
    "setup_logger",
    "get_logger",
    "log_metrics",
    # Plotting
    "plot_training_curves",
    "plot_returns",
    "plot_drawdown",
    "plot_portfolio_value",
]
