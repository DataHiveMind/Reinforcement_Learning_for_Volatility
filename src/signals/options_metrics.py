"""
Options-based trading signals.

Generates signals from implied volatility and options Greeks.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OptionsSignals:
    """Generate trading signals from options data."""

    def __init__(
        self,
        skew_threshold: float = 0.05,
        vrp_threshold: float = 0.02,
        vol_of_vol_threshold: float = 0.5,
    ):
        """
        Initialize options signal generator.

        Args:
            skew_threshold: Threshold for IV skew
            vrp_threshold: Threshold for variance risk premium
            vol_of_vol_threshold: Threshold for vol-of-vol
        """
        self.skew_threshold = skew_threshold
        self.vrp_threshold = vrp_threshold
        self.vol_of_vol_threshold = vol_of_vol_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all options-based signals.

        Args:
            df: DataFrame with options features

        Returns:
            DataFrame with signal columns
        """
        signals = pd.DataFrame(index=df.index)

        # IV skew signal
        signals["skew_signal"] = self._iv_skew_signal(df)

        # Term structure signal
        signals["term_signal"] = self._term_structure_signal(df)

        # Variance risk premium signal
        signals["vrp_signal"] = self._vrp_signal(df)

        # Vol-of-vol signal
        signals["vov_signal"] = self._vol_of_vol_signal(df)

        # Greeks exposure signal
        signals["greeks_signal"] = self._greeks_signal(df)

        # Combined options signal
        signals["options_signal"] = self._combine_signals(signals)

        logger.info(f"Generated options signals for {len(df)} rows")
        return signals

    def _iv_skew_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signal from IV skew.

        Positive skew = tail risk, negative skew = unusual
        """
        if "iv_skew_30d" not in df.columns:
            return pd.Series(0.0, index=df.index)

        skew = df["iv_skew_30d"]

        # Normalize
        signal = np.tanh(skew / self.skew_threshold)

        return pd.Series(signal, index=df.index)

    def _term_structure_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signal from term structure.

        Steep term structure (contango) = carry opportunity
        Inverted = potential vol spike
        """
        if "term_slope" not in df.columns:
            return pd.Series(0.0, index=df.index)

        term_slope = df["term_slope"]

        # Positive signal for contango (carry trade)
        # Negative signal for backwardation (vol spike risk)
        signal = np.tanh(term_slope / 0.05)

        return pd.Series(signal, index=df.index)

    def _vrp_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signal from variance risk premium.

        Positive VRP = implied > realized, sell vol
        Negative VRP = implied < realized, buy vol
        """
        if "variance_risk_premium" not in df.columns:
            return pd.Series(0.0, index=df.index)

        vrp = df["variance_risk_premium"]

        # Positive signal when VRP is positive (sell vol opportunity)
        signal = np.tanh(vrp / self.vrp_threshold)

        return pd.Series(signal, index=df.index)

    def _vol_of_vol_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signal from volatility of volatility.

        High vol-of-vol = unstable, high gamma opportunity
        """
        if "vol_of_vol_30d" not in df.columns:
            return pd.Series(0.0, index=df.index)

        vov = df["vol_of_vol_30d"]

        # Normalize - high vol-of-vol suggests convexity trades
        signal = np.tanh(vov / self.vol_of_vol_threshold)

        return pd.Series(signal, index=df.index)

    def _greeks_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signal from Greeks exposure.

        Aggregate delta, gamma, vega to determine directional bias
        """
        signal = pd.Series(0.0, index=df.index)

        # Delta exposure
        if "delta_exposure" in df.columns:
            delta = df["delta_exposure"]
            signal += np.tanh(delta / 0.5) * 0.3

        # Gamma exposure (positive = long convexity)
        if "gamma_exposure" in df.columns:
            gamma = df["gamma_exposure"]
            signal += np.tanh(gamma / 0.5) * 0.3

        # Vega exposure
        if "vega_exposure" in df.columns:
            vega = df["vega_exposure"]
            signal += np.tanh(vega / 0.5) * 0.4

        return signal

    def _combine_signals(self, signals: pd.DataFrame) -> pd.Series:
        """Combine individual signals into composite signal."""
        # Weighted combination
        weights = {
            "skew_signal": 0.25,
            "term_signal": 0.20,
            "vrp_signal": 0.30,
            "vov_signal": 0.15,
            "greeks_signal": 0.10,
        }

        combined = pd.Series(0.0, index=signals.index)

        for signal_name, weight in weights.items():
            if signal_name in signals.columns:
                combined += signals[signal_name] * weight

        return combined


class VolatilityRegimeSignals:
    """Generate volatility regime-based signals."""

    def __init__(
        self,
        low_vol_threshold: float = 0.15,
        high_vol_threshold: float = 0.35,
    ):
        """
        Initialize regime signal generator.

        Args:
            low_vol_threshold: Threshold for low volatility regime
            high_vol_threshold: Threshold for high volatility regime
        """
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold

    def classify_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify volatility regime.

        Args:
            df: DataFrame with volatility data

        Returns:
            Regime classification (0=low, 1=normal, 2=high)
        """
        if "realized_volatility" not in df.columns:
            return pd.Series(1, index=df.index)

        vol = df["realized_volatility"]

        regime = pd.Series(1, index=df.index)  # Normal regime
        regime[vol < self.low_vol_threshold] = 0  # Low vol
        regime[vol > self.high_vol_threshold] = 2  # High vol

        return regime

    def generate_regime_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signal based on regime.

        Args:
            df: DataFrame with volatility data

        Returns:
            Regime-based signal
        """
        regime = self.classify_regime(df)

        # Low vol regime: sell vol (negative signal)
        # High vol regime: buy vol (positive signal)
        signal = (regime - 1) * 0.5  # Maps to [-0.5, 0, 0.5]

        return signal
