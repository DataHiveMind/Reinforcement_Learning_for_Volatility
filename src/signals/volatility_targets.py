"""
Volatility target and label generation.

Creates target variables for volatility prediction and breakout detection.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolatilityTargets:
    """Generate volatility prediction targets and labels."""

    def __init__(
        self,
        forecast_horizon: int = 5,
        breakout_threshold: float = 1.5,
        vol_window: int = 20,
    ):
        """
        Initialize volatility target generator.

        Args:
            forecast_horizon: Number of periods ahead to forecast
            breakout_threshold: Threshold for volatility breakout (in std devs)
            vol_window: Window for volatility calculation
        """
        self.forecast_horizon = forecast_horizon
        self.breakout_threshold = breakout_threshold
        self.vol_window = vol_window

    def generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all volatility targets.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with target columns
        """
        targets = pd.DataFrame(index=df.index)

        # Future realized volatility
        targets["future_realized_vol"] = self._future_realized_volatility(df)

        # Volatility change
        targets["vol_change"] = self._volatility_change(df)

        # Breakout labels
        targets["breakout_label"] = self._breakout_labels(df)

        # Volatility regime
        targets["vol_regime"] = self._volatility_regime(df)

        logger.info(f"Generated volatility targets for {len(df)} rows")
        return targets

    def _future_realized_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate future realized volatility.

        Args:
            df: DataFrame with price/return data

        Returns:
            Future volatility
        """
        if "returns" not in df.columns:
            return pd.Series(np.nan, index=df.index)

        returns = df["returns"]

        # Calculate forward-looking volatility
        future_vol = returns.shift(-self.forecast_horizon).rolling(self.vol_window).std() * np.sqrt(
            252
        )

        return future_vol

    def _volatility_change(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate change in volatility.

        Args:
            df: DataFrame with volatility data

        Returns:
            Volatility change
        """
        if "realized_volatility" in df.columns:
            vol = df["realized_volatility"]
        elif "returns" in df.columns:
            vol = df["returns"].rolling(self.vol_window).std() * np.sqrt(252)
        else:
            return pd.Series(np.nan, index=df.index)

        # Percentage change in volatility
        vol_change = vol.pct_change(self.forecast_horizon)

        return vol_change

    def _breakout_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate binary labels for volatility breakouts.

        Args:
            df: DataFrame with price/return data

        Returns:
            Binary breakout labels (1=breakout, 0=no breakout)
        """
        if "returns" not in df.columns:
            return pd.Series(0, index=df.index)

        returns = df["returns"]

        # Calculate rolling volatility
        rolling_vol = returns.rolling(self.vol_window).std()
        rolling_mean = rolling_vol.rolling(self.vol_window).mean()
        rolling_std = rolling_vol.rolling(self.vol_window).std()

        # Future volatility
        future_vol = returns.shift(-self.forecast_horizon).rolling(self.vol_window).std()

        # Breakout = future vol significantly higher than historical
        z_score = (future_vol - rolling_mean) / (rolling_std + 1e-8)
        breakout = (z_score > self.breakout_threshold).astype(int)

        return pd.Series(breakout.fillna(0), index=df.index)

    def _volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify volatility regime.

        Args:
            df: DataFrame with volatility data

        Returns:
            Regime labels (0=low, 1=normal, 2=high)
        """
        if "realized_volatility" in df.columns:
            vol = df["realized_volatility"]
        elif "returns" in df.columns:
            vol = df["returns"].rolling(self.vol_window).std() * np.sqrt(252)
        else:
            return pd.Series(1, index=df.index)

        # Calculate percentiles
        vol_25 = vol.rolling(252).quantile(0.25)
        vol_75 = vol.rolling(252).quantile(0.75)

        # Classify regime
        regime = pd.Series(1, index=df.index)  # Normal
        regime[vol < vol_25] = 0  # Low vol
        regime[vol > vol_75] = 2  # High vol

        return regime.fillna(1).astype(int)


class BreakoutDetector:
    """Detect volatility breakouts in real-time."""

    def __init__(
        self,
        lookback: int = 20,
        threshold: float = 2.0,
        min_duration: int = 3,
    ):
        """
        Initialize breakout detector.

        Args:
            lookback: Lookback window
            threshold: Detection threshold (in std devs)
            min_duration: Minimum breakout duration
        """
        self.lookback = lookback
        self.threshold = threshold
        self.min_duration = min_duration

    def detect_breakouts(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Detect volatility breakouts.

        Args:
            df: DataFrame with price/return data

        Returns:
            Breakout signals and details
        """
        if "returns" not in df.columns:
            return pd.Series(0, index=df.index), pd.DataFrame()

        returns = df["returns"]

        # Calculate rolling statistics
        rolling_vol = returns.rolling(self.lookback).std()
        vol_mean = rolling_vol.rolling(self.lookback * 2).mean()
        vol_std = rolling_vol.rolling(self.lookback * 2).std()

        # Z-score
        z_score = (rolling_vol - vol_mean) / (vol_std + 1e-8)

        # Detect breakouts
        breakout_signal = (z_score > self.threshold).astype(int)

        # Filter by minimum duration
        breakout_filtered = self._filter_by_duration(breakout_signal)

        # Breakout details
        details = pd.DataFrame(
            {
                "rolling_vol": rolling_vol,
                "vol_mean": vol_mean,
                "vol_std": vol_std,
                "z_score": z_score,
                "breakout": breakout_filtered,
            },
            index=df.index,
        )

        return breakout_filtered, details

    def _filter_by_duration(self, signal: pd.Series) -> pd.Series:
        """Filter signals by minimum duration."""
        filtered = signal.copy()

        # Find signal transitions
        signal_diff = signal.diff()
        starts = signal_diff[signal_diff == 1].index
        ends = signal_diff[signal_diff == -1].index

        # Check duration of each signal
        for start in starts:
            # Find corresponding end
            future_ends = ends[ends > start]
            if len(future_ends) > 0:
                end = future_ends[0]
                duration = len(signal.loc[start:end])
            else:
                duration = len(signal.loc[start:])

            # Filter short signals
            if duration < self.min_duration:
                filtered.loc[start:] = 0
                if len(future_ends) > 0:
                    filtered.loc[:end] = 0

        return filtered
