"""
State representation for RL environment.

Builds observation vectors from market data and portfolio state.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StateBuilder:
    """Build state representations from market data."""

    def __init__(self, config: Dict):
        """
        Initialize state builder.

        Args:
            config: State space configuration
        """
        self.config = config
        self.components = config.get("components", {})
        self.normalization = config.get("normalization", "rolling_zscore")
        self.normalization_window = config.get("normalization_window", 252)
        self.clip_range = config.get("clip_range", [-5, 5])
        self.history_length = config.get("history_length", 20)
        self.use_lstm_state = config.get("use_lstm_state", False)

        # Feature tracking
        self.feature_columns: List[str] = []
        self.state_dim = 0

    def build_state(
        self,
        data: pd.DataFrame,
        portfolio_state: Dict[str, float],
    ) -> np.ndarray:
        """
        Build observation vector from data and portfolio state.

        Args:
            data: Historical market data window
            portfolio_state: Current portfolio information

        Returns:
            State vector as numpy array
        """
        features = []

        # 1. Microstructure features
        if self.components.get("microstructure", {}).get("enabled", True):
            micro_features = self._extract_microstructure_features(data)
            features.append(micro_features)

        # 2. Options features
        if self.components.get("options", {}).get("enabled", True):
            options_features = self._extract_options_features(data)
            features.append(options_features)

        # 3. Market features
        if self.components.get("market", {}).get("enabled", True):
            market_features = self._extract_market_features(data)
            features.append(market_features)

        # 4. Portfolio features
        if self.components.get("portfolio", {}).get("enabled", True):
            portfolio_features = self._extract_portfolio_features(portfolio_state)
            features.append(portfolio_features)

        # Concatenate all features
        state = np.concatenate(features)

        # Normalize
        state = self._normalize(state)

        # Clip outliers
        state = np.clip(state, self.clip_range[0], self.clip_range[1])

        return state.astype(np.float32)

    def _extract_microstructure_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract microstructure features from data."""
        feature_names = [
            # Order book imbalance
            "imbalance_level_1",
            "imbalance_level_5",
            "imbalance_level_10",
            # Spread
            "bid_ask_spread",
            "relative_spread",
            # Volume
            "total_volume",
            "buy_volume_ratio",
            # VPIN
            "vpin",
            # Realized volatility
            "realized_vol_5",
            "realized_vol_20",
        ]

        features = []
        for name in feature_names:
            if name in data.columns:
                # Use last value from window
                features.append(data[name].iloc[-1])
            else:
                features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _extract_options_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract options-implied features."""
        feature_names = [
            # IV skew
            "iv_skew_30d",
            "iv_skew_60d",
            # Term structure
            "term_slope",
            "term_curvature",
            # Vol of vol
            "vol_of_vol_30d",
            # Greeks exposure
            "delta_exposure",
            "gamma_exposure",
            "vega_exposure",
            # VRP
            "variance_risk_premium",
        ]

        features = []
        for name in feature_names:
            if name in data.columns:
                features.append(data[name].iloc[-1])
            else:
                features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _extract_market_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract general market features."""
        features = []

        # Realized volatility (if available)
        if "realized_volatility" in data.columns:
            features.append(data["realized_volatility"].iloc[-1])
        else:
            features.append(0.0)

        # Historical returns
        if "returns" in data.columns:
            returns = data["returns"].values
            features.extend(
                [
                    np.mean(returns),  # mean return
                    np.std(returns),  # volatility
                    np.min(returns),  # worst return
                    np.max(returns),  # best return
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # Market regime (if available)
        if "market_regime" in data.columns:
            features.append(data["market_regime"].iloc[-1])
        else:
            features.append(0.0)

        # Time features
        if len(data) > 0 and hasattr(data.index, "hour"):
            # Cyclical encoding of time
            hour = data.index[-1].hour if hasattr(data.index[-1], "hour") else 12
            features.extend(
                [
                    np.sin(2 * np.pi * hour / 24),
                    np.cos(2 * np.pi * hour / 24),
                ]
            )
        else:
            features.extend([0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def _extract_portfolio_features(self, portfolio_state: Dict[str, float]) -> np.ndarray:
        """Extract portfolio state features."""
        features = []

        # Current positions
        positions = portfolio_state.get("current_positions", [0.0] * 5)
        features.extend(positions)

        # Cash balance (normalized)
        cash = portfolio_state.get("cash_balance", 100000.0)
        features.append(cash / 100000.0)  # normalize to initial capital

        # PnL
        realized_pnl = portfolio_state.get("realized_pnl", 0.0)
        unrealized_pnl = portfolio_state.get("unrealized_pnl", 0.0)
        features.extend([realized_pnl / 100000.0, unrealized_pnl / 100000.0])

        # Position age (normalized)
        position_age = portfolio_state.get("position_age", 0)
        features.append(position_age / 252.0)  # normalize to year

        return np.array(features, dtype=np.float32)

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize state vector."""
        if self.normalization == "rolling_zscore":
            # Z-score normalization (placeholder - should use running stats)
            mean = np.mean(state)
            std = np.std(state) + 1e-8
            return (state - mean) / std
        elif self.normalization == "minmax":
            # Min-max normalization
            min_val = np.min(state)
            max_val = np.max(state)
            return (state - min_val) / (max_val - min_val + 1e-8)
        else:
            # No normalization
            return state
