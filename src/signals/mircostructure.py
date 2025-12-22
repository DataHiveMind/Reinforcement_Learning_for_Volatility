"""
Microstructure-based trading signals.

Generates signals from order book and trade flow data.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MicrostructureSignals:
    """Generate trading signals from microstructure features."""

    def __init__(
        self,
        imbalance_threshold: float = 0.3,
        vpin_threshold: float = 0.7,
        spread_threshold: float = 0.002,
    ):
        """
        Initialize microstructure signal generator.

        Args:
            imbalance_threshold: Threshold for order book imbalance
            vpin_threshold: Threshold for VPIN toxicity
            spread_threshold: Threshold for relative spread
        """
        self.imbalance_threshold = imbalance_threshold
        self.vpin_threshold = vpin_threshold
        self.spread_threshold = spread_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all microstructure signals.

        Args:
            df: DataFrame with microstructure features

        Returns:
            DataFrame with signal columns
        """
        signals = pd.DataFrame(index=df.index)

        # Order book imbalance signal
        signals["imbalance_signal"] = self._order_book_imbalance_signal(df)

        # VPIN toxicity signal
        signals["vpin_signal"] = self._vpin_signal(df)

        # Spread dynamics signal
        signals["spread_signal"] = self._spread_signal(df)

        # Volume profile signal
        signals["volume_signal"] = self._volume_signal(df)

        # Combined microstructure signal
        signals["micro_signal"] = self._combine_signals(signals)

        logger.info(f"Generated microstructure signals for {len(df)} rows")
        return signals

    def _order_book_imbalance_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signal from order book imbalance.

        Positive signal = buying pressure, negative = selling pressure
        """
        if "imbalance_level_5" not in df.columns:
            return pd.Series(0.0, index=df.index)

        imbalance = df["imbalance_level_5"]

        # Normalize to [-1, 1]
        signal = np.tanh(imbalance / self.imbalance_threshold)

        return pd.Series(signal, index=df.index)

    def _vpin_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signal from VPIN (Volume-Synchronized Probability of Informed Trading).

        High VPIN = toxic flow, avoid trading or reduce position
        """
        if "vpin" not in df.columns:
            return pd.Series(0.0, index=df.index)

        vpin = df["vpin"]

        # Negative signal when VPIN is high (toxic flow)
        signal = -np.tanh((vpin - 0.5) / self.vpin_threshold)

        return pd.Series(signal, index=df.index)

    def _spread_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signal from bid-ask spread dynamics.

        Widening spread = liquidity shock, reduce position
        """
        if "relative_spread" not in df.columns:
            return pd.Series(0.0, index=df.index)

        spread = df["relative_spread"]

        # Negative signal when spread is wide
        signal = -np.tanh((spread - self.spread_threshold) / self.spread_threshold)

        return pd.Series(signal, index=df.index)

    def _volume_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signal from volume patterns.

        High volume with price movement = momentum
        """
        if "total_volume" not in df.columns or "returns" not in df.columns:
            return pd.Series(0.0, index=df.index)

        # Normalize volume
        volume_zscore = (df["total_volume"] - df["total_volume"].rolling(20).mean()) / (
            df["total_volume"].rolling(20).std() + 1e-8
        )

        # Volume-weighted returns
        signal = np.sign(df["returns"]) * np.tanh(volume_zscore / 2.0)

        return pd.Series(signal.fillna(0.0), index=df.index)

    def _combine_signals(self, signals: pd.DataFrame) -> pd.Series:
        """Combine individual signals into composite signal."""
        # Equal-weighted combination
        signal_cols = [
            "imbalance_signal",
            "vpin_signal",
            "spread_signal",
            "volume_signal",
        ]

        available_signals = [col for col in signal_cols if col in signals.columns]

        if not available_signals:
            return pd.Series(0.0, index=signals.index)

        combined = signals[available_signals].mean(axis=1)

        return combined


class LiquiditySignals:
    """Generate liquidity-based signals."""

    def __init__(self, depth_threshold: float = 100000):
        """
        Initialize liquidity signal generator.

        Args:
            depth_threshold: Threshold for market depth
        """
        self.depth_threshold = depth_threshold

    def generate_liquidity_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate liquidity availability signal.

        Args:
            df: DataFrame with order book data

        Returns:
            Liquidity signal (-1 to 1)
        """
        if "total_bid_size" not in df.columns or "total_ask_size" not in df.columns:
            return pd.Series(0.0, index=df.index)

        # Total market depth
        total_depth = df["total_bid_size"] + df["total_ask_size"]

        # Normalize
        signal = np.tanh((total_depth - self.depth_threshold) / self.depth_threshold)

        return pd.Series(signal, index=df.index)


class FlowToxicitySignals:
    """Generate trade flow toxicity signals."""

    def __init__(self, lookback: int = 50):
        """
        Initialize flow toxicity signal generator.

        Args:
            lookback: Lookback window for calculations
        """
        self.lookback = lookback

    def generate_toxicity_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate flow toxicity signal using ELO-style rating.

        Args:
            df: DataFrame with trade data

        Returns:
            Toxicity signal
        """
        if "buy_volume_ratio" not in df.columns:
            return pd.Series(0.0, index=df.index)

        # Deviation from 50-50 buy/sell balance
        buy_ratio = df["buy_volume_ratio"]
        imbalance = abs(buy_ratio - 0.5)

        # Rolling toxicity
        toxicity = imbalance.rolling(self.lookback).mean()

        # Normalize to [-1, 1]
        signal = -np.tanh(toxicity * 4)  # Negative signal for high toxicity

        return pd.Series(signal.fillna(0.0), index=df.index)
