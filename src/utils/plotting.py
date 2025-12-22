"""
Visualization and plotting utilities.

Provides functions for:
- Training curves
- Performance analysis
- Portfolio visualization
- Feature importance plots
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def plot_training_curves(
    metrics_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    title: str = "Training Curves",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot training curves from metrics DataFrame.

    Args:
        metrics_df: DataFrame with training metrics
        metrics: List of metrics to plot (None = all)
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = metrics_df.columns.tolist()

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for idx, metric in enumerate(metrics):
        if metric in metrics_df.columns:
            ax = axes[idx]
            metrics_df[metric].plot(ax=ax, linewidth=2)
            ax.set_title(f"{metric}")
            ax.set_xlabel("Step")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)

    # Remove extra subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_returns(
    returns: Union[np.ndarray, pd.Series],
    cumulative: bool = True,
    title: str = "Returns",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot returns (cumulative or periodic).

    Args:
        returns: Return series
        cumulative: Whether to plot cumulative returns
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    fig, ax = plt.subplots(figsize=(12, 6))

    if cumulative:
        cumulative_returns = (1 + returns).cumprod() - 1
        cumulative_returns.plot(ax=ax, linewidth=2, color="blue")
        ax.set_ylabel("Cumulative Return")
        ax.set_title(f"{title} - Cumulative")
    else:
        returns.plot(ax=ax, linewidth=1, alpha=0.7, color="blue")
        ax.set_ylabel("Return")
        ax.set_title(f"{title} - Periodic")
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)

    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_drawdown(
    prices: Union[np.ndarray, pd.Series],
    title: str = "Drawdown",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot drawdown analysis.

    Args:
        prices: Price or portfolio value series
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)

    # Calculate drawdown
    running_max = prices.expanding().max()
    drawdown = (prices - running_max) / running_max

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Price with running max
    prices.plot(ax=ax1, label="Portfolio Value", linewidth=2, color="blue")
    running_max.plot(
        ax=ax1, label="Running Max", linewidth=2, linestyle="--", color="red", alpha=0.7
    )
    ax1.set_ylabel("Portfolio Value")
    ax1.set_title("Portfolio Value and Running Maximum")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown
    drawdown.plot(ax=ax2, linewidth=2, color="red", alpha=0.7)
    ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color="red")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Time")
    ax2.set_title("Drawdown")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_portfolio_value(
    portfolio_values: Union[np.ndarray, pd.Series],
    benchmark: Optional[Union[np.ndarray, pd.Series]] = None,
    title: str = "Portfolio Value",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot portfolio value over time.

    Args:
        portfolio_values: Portfolio value series
        benchmark: Optional benchmark series
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if isinstance(portfolio_values, np.ndarray):
        portfolio_values = pd.Series(portfolio_values)

    fig, ax = plt.subplots(figsize=(12, 6))

    portfolio_values.plot(ax=ax, label="Portfolio", linewidth=2, color="blue")

    if benchmark is not None:
        if isinstance(benchmark, np.ndarray):
            benchmark = pd.Series(benchmark, index=portfolio_values.index)
        benchmark.plot(ax=ax, label="Benchmark", linewidth=2, linestyle="--", color="orange")

    ax.set_ylabel("Value")
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_positions(
    positions: pd.DataFrame,
    title: str = "Strategy Positions",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot strategy positions over time.

    Args:
        positions: DataFrame with position allocations
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    positions.plot(ax=ax, linewidth=2, alpha=0.7)

    ax.set_ylabel("Position Size")
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_feature_importance(
    feature_importance: Dict[str, float],
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot feature importance.

    Args:
        feature_importance: Dictionary of feature names and importance scores
        top_n: Number of top features to show
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Sort by importance
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    features, importances = zip(*sorted_features)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_correlation_matrix(
    data: pd.DataFrame,
    title: str = "Correlation Matrix",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot correlation matrix heatmap.

    Args:
        data: DataFrame with features
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    corr = data.corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_volatility_forecast(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
    title: str = "Volatility Forecast",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot actual vs predicted volatility.

    Args:
        actual: Actual volatility
        predicted: Predicted volatility
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if isinstance(actual, np.ndarray):
        actual = pd.Series(actual)
    if isinstance(predicted, np.ndarray):
        predicted = pd.Series(predicted, index=actual.index)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Time series plot
    actual.plot(ax=ax1, label="Actual", linewidth=2, color="blue", alpha=0.7)
    predicted.plot(ax=ax1, label="Predicted", linewidth=2, color="red", linestyle="--", alpha=0.7)
    ax1.set_ylabel("Volatility")
    ax1.set_title("Volatility Time Series")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter plot
    ax2.scatter(actual, predicted, alpha=0.5, s=10)
    ax2.plot([actual.min(), actual.max()], [actual.min(), actual.max()], "r--", linewidth=2)
    ax2.set_xlabel("Actual Volatility")
    ax2.set_ylabel("Predicted Volatility")
    ax2.set_title("Actual vs Predicted")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_distribution(
    data: Union[np.ndarray, pd.Series],
    bins: int = 50,
    title: str = "Distribution",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot data distribution with histogram and KDE.

    Args:
        data: Data series
        bins: Number of histogram bins
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if isinstance(data, pd.Series):
        data = data.values

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(data, bins=bins, alpha=0.7, density=True, color="blue", edgecolor="black")

    # Add KDE
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(data[~np.isnan(data)])
    x_range = np.linspace(data.min(), data.max(), 1000)
    ax.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig
