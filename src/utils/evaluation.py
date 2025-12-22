"""
Performance evaluation metrics for RL trading agents.

Includes:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Win rate and profit factor
- Volatility prediction metrics
"""

import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_returns(prices: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Calculate simple returns from prices.

    Args:
        prices: Price series

    Returns:
        Array of returns
    """
    if isinstance(prices, pd.Series):
        prices = prices.to_numpy()

    returns = np.diff(prices) / prices[:-1]
    return returns


def calculate_log_returns(prices: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Calculate log returns from prices.

    Args:
        prices: Price series

    Returns:
        Array of log returns
    """
    if isinstance(prices, pd.Series):
        prices = prices.to_numpy()

    log_returns = np.diff(np.log(prices))
    return log_returns


def calculate_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        Sharpe ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_numpy()

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year)

    return float(sharpe)


def calculate_sortino_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sortino ratio (uses downside deviation).

    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year

    Returns:
        Sortino ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_numpy()

    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return np.inf

    downside_std = np.std(downside_returns)

    if downside_std == 0:
        return 0.0

    sortino = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)

    return float(sortino)


def calculate_max_drawdown(
    prices: Union[np.ndarray, pd.Series],
    return_series: bool = False,
) -> Union[float, pd.Series]:
    """
    Calculate maximum drawdown.

    Args:
        prices: Price or portfolio value series
        return_series: If True, return drawdown series

    Returns:
        Maximum drawdown (float) or drawdown series
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)

    # Calculate running maximum
    running_max = prices.expanding().max()

    # Calculate drawdown
    drawdown = (prices - running_max) / running_max

    if return_series:
        return drawdown

    return float(drawdown.min())


def calculate_calmar_ratio(
    returns: Union[np.ndarray, pd.Series],
    prices: Union[np.ndarray, pd.Series],
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).

    Args:
        returns: Return series
        prices: Price series
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_numpy()

    returns = returns[~np.isnan(returns)]

    annual_return = np.mean(returns) * periods_per_year
    max_dd_value = calculate_max_drawdown(prices, return_series=False)
    assert isinstance(max_dd_value, float), "Expected float from calculate_max_drawdown"
    max_dd = abs(max_dd_value)

    if max_dd == 0:
        return np.inf

    calmar = annual_return / max_dd
    return float(calmar)


def calculate_win_rate(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate win rate (fraction of positive returns).

    Args:
        returns: Return series

    Returns:
        Win rate (0 to 1)
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_numpy()

    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    win_rate = np.sum(returns > 0) / len(returns)
    return float(win_rate)


def calculate_profit_factor(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        returns: Return series

    Returns:
        Profit factor
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_numpy()

    returns = returns[~np.isnan(returns)]

    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))

    if gross_loss == 0:
        return np.inf

    profit_factor = gross_profit / gross_loss
    return float(profit_factor)


def calculate_information_ratio(
    returns: Union[np.ndarray, pd.Series],
    benchmark_returns: Union[np.ndarray, pd.Series],
    periods_per_year: int = 252,
) -> float:
    """
    Calculate information ratio.

    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year

    Returns:
        Information ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_numpy()
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.to_numpy()

    # Active returns
    active_returns = returns - benchmark_returns
    active_returns = active_returns[~np.isnan(active_returns)]

    if len(active_returns) == 0 or np.std(active_returns) == 0:
        return 0.0

    ir = np.mean(active_returns) / np.std(active_returns) * np.sqrt(periods_per_year)
    return float(ir)


def calculate_turnover(positions: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate portfolio turnover.

    Args:
        positions: Position series

    Returns:
        Average turnover per period
    """
    if isinstance(positions, pd.Series):
        positions = positions.to_numpy()

    position_changes = np.abs(np.diff(positions))
    turnover = np.mean(position_changes)

    return float(turnover)


def calculate_volatility_forecast_metrics(
    actual_vol: Union[np.ndarray, pd.Series],
    predicted_vol: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """
    Calculate volatility forecasting metrics.

    Args:
        actual_vol: Actual realized volatility
        predicted_vol: Predicted volatility

    Returns:
        Dictionary of metrics (MAE, RMSE, R²)
    """
    if isinstance(actual_vol, pd.Series):
        actual_vol = actual_vol.to_numpy()
    if isinstance(predicted_vol, pd.Series):
        predicted_vol = predicted_vol.to_numpy()

    # Remove NaN values
    mask = ~(np.isnan(actual_vol) | np.isnan(predicted_vol))
    actual_vol = actual_vol[mask]
    predicted_vol = predicted_vol[mask]

    if len(actual_vol) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": 0.0}

    # Mean Absolute Error
    mae = np.mean(np.abs(actual_vol - predicted_vol))

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((actual_vol - predicted_vol) ** 2))

    # R-squared
    ss_res = np.sum((actual_vol - predicted_vol) ** 2)
    ss_tot = np.sum((actual_vol - np.mean(actual_vol)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


class PerformanceMetrics:
    """Comprehensive performance metrics calculator."""

    def __init__(
        self,
        returns: Optional[Union[np.ndarray, pd.Series]] = None,
        prices: Optional[Union[np.ndarray, pd.Series]] = None,
        periods_per_year: int = 252,
    ):
        """
        Initialize performance metrics calculator.

        Args:
            returns: Return series
            prices: Price/portfolio value series
            periods_per_year: Number of periods per year
        """
        self.returns = returns
        self.prices = prices
        self.periods_per_year = periods_per_year

        if returns is None and prices is not None:
            self.returns = calculate_returns(prices)

    def calculate_all_metrics(
        self,
        risk_free_rate: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.

        Args:
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Dictionary of all metrics
        """
        if self.returns is None:
            raise ValueError("Returns data not available")

        metrics = {}

        # Return metrics
        metrics["total_return"] = np.prod(1 + self.returns) - 1
        metrics["annual_return"] = np.mean(self.returns) * self.periods_per_year
        metrics["annual_volatility"] = np.std(self.returns) * np.sqrt(self.periods_per_year)

        # Risk-adjusted metrics
        metrics["sharpe_ratio"] = calculate_sharpe_ratio(
            self.returns, risk_free_rate, self.periods_per_year
        )
        metrics["sortino_ratio"] = calculate_sortino_ratio(
            self.returns, risk_free_rate, self.periods_per_year
        )

        # Drawdown metrics
        if self.prices is not None:
            metrics["max_drawdown"] = calculate_max_drawdown(self.prices)
            metrics["calmar_ratio"] = calculate_calmar_ratio(
                self.returns, self.prices, self.periods_per_year
            )

        # Win/loss metrics
        metrics["win_rate"] = calculate_win_rate(self.returns)
        metrics["profit_factor"] = calculate_profit_factor(self.returns)

        # Additional statistics
        returns_array = (
            self.returns if isinstance(self.returns, np.ndarray) else self.returns.to_numpy()
        )
        metrics["skewness"] = float(stats.skew(returns_array))
        metrics["kurtosis"] = float(stats.kurtosis(returns_array))

        return metrics

    def summary(self, risk_free_rate: float = 0.0) -> pd.DataFrame:
        """
        Generate summary DataFrame of metrics.

        Args:
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            DataFrame with metrics
        """
        metrics = self.calculate_all_metrics(risk_free_rate)

        df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
        df.index.name = "Metric"

        return df

    def to_dict(self, risk_free_rate: float = 0.0) -> Dict[str, float]:
        """Return metrics as dictionary."""
        return self.calculate_all_metrics(risk_free_rate)


def calculate_breakout_detection_metrics(
    actual_breakouts: Union[np.ndarray, pd.Series],
    predicted_breakouts: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """
    Calculate metrics for volatility breakout detection.

    Args:
        actual_breakouts: Actual breakout labels (binary)
        predicted_breakouts: Predicted breakout labels (binary)

    Returns:
        Dictionary with precision, recall, F1
    """
    if isinstance(actual_breakouts, pd.Series):
        actual_breakouts = actual_breakouts.to_numpy()
    if isinstance(predicted_breakouts, pd.Series):
        predicted_breakouts = predicted_breakouts.to_numpy()

    # Convert to binary
    actual = actual_breakouts > 0
    predicted = predicted_breakouts > 0

    # True positives, false positives, false negatives
    tp = np.sum(actual & predicted)
    fp = np.sum(~actual & predicted)
    fn = np.sum(actual & ~predicted)

    # Precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def calculate_regime_performance(
    returns: Union[np.ndarray, pd.Series],
    regimes: Union[np.ndarray, pd.Series],
) -> pd.DataFrame:
    """
    Calculate performance metrics by market regime.

    Args:
        returns: Return series
        regimes: Regime labels

    Returns:
        DataFrame with metrics by regime
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    if isinstance(regimes, np.ndarray):
        regimes = pd.Series(regimes, index=returns.index)

    df = pd.DataFrame({"returns": returns, "regime": regimes})

    results = []
    for regime in df["regime"].unique():
        regime_returns = df[df["regime"] == regime]["returns"].to_numpy()

        metrics = {
            "regime": regime,
            "mean_return": np.mean(regime_returns),
            "volatility": np.std(regime_returns),
            "sharpe": calculate_sharpe_ratio(regime_returns),
            "win_rate": calculate_win_rate(regime_returns),
        }
        results.append(metrics)

    return pd.DataFrame(results)
