"""
Feature engineering for market microstructure and options data.

Implements:
- Microstructure features (order book, trade flow, volatility)
- Options-implied features (IV, skew, Greeks)
- Feature engineering pipelines
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class MicrostructureFeatureEngineer:
    """Engineer features from market microstructure data."""

    def __init__(self, config: Union[str, Path, Dict]):
        """
        Initialize microstructure feature engineer.

        Args:
            config: Path to config file or config dictionary
        """
        if isinstance(config, (str, Path)):
            with open(config, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

        logger.info("Initialized MicrostructureFeatureEngineer")

    def compute_order_book_imbalance(
        self,
        df: pd.DataFrame,
        levels: List[int] = [1, 5, 10],
    ) -> pd.DataFrame:
        """
        Compute order book imbalance at different levels.

        Args:
            df: DataFrame with bid/ask prices and sizes
            levels: Depth levels to compute imbalance

        Returns:
            DataFrame with imbalance features
        """
        features = pd.DataFrame(index=df.index)

        for level in levels:
            bid_cols = [f"bid_size_{i}" for i in range(1, level + 1)]
            ask_cols = [f"ask_size_{i}" for i in range(1, level + 1)]

            if all(col in df.columns for col in bid_cols + ask_cols):
                bid_volume = df[bid_cols].sum(axis=1)
                ask_volume = df[ask_cols].sum(axis=1)

                # Imbalance ratio
                features[f"imbalance_level_{level}"] = (bid_volume - ask_volume) / (
                    bid_volume + ask_volume
                )

                # Weighted mid price
                features[f"weighted_mid_level_{level}"] = (
                    df["bid_price_1"] * ask_volume + df["ask_price_1"] * bid_volume
                ) / (bid_volume + ask_volume)

        logger.info("Computed order book imbalance for levels: %s", levels)
        return features

    def compute_microprice(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute microprice (weighted average of best bid/ask).

        Args:
            df: DataFrame with best bid/ask

        Returns:
            Series with microprice
        """
        bid_price = df["bid_price_1"]
        ask_price = df["ask_price_1"]
        bid_size = df["bid_size_1"]
        ask_size = df["ask_size_1"]

        microprice = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)

        return microprice

    def compute_spread_features(
        self,
        df: pd.DataFrame,
        rolling_window: int = 60,
    ) -> pd.DataFrame:
        """
        Compute spread-related features.

        Args:
            df: DataFrame with bid/ask prices
            rolling_window: Window for rolling statistics

        Returns:
            DataFrame with spread features
        """
        features = pd.DataFrame(index=df.index)

        # Bid-ask spread
        features["bid_ask_spread"] = df["ask_price_1"] - df["bid_price_1"]
        features["relative_spread"] = features["bid_ask_spread"] / (
            (df["bid_price_1"] + df["ask_price_1"]) / 2
        )

        # Rolling spread statistics
        features["spread_volatility"] = features["bid_ask_spread"].rolling(rolling_window).std()
        features["spread_mean"] = features["bid_ask_spread"].rolling(rolling_window).mean()

        logger.info("Computed spread features")
        return features

    def compute_vpin(
        self,
        df: pd.DataFrame,
        bucket_size: int = 50,
        estimation_window: int = 50,
    ) -> pd.Series:
        """
        Compute Volume-Synchronized Probability of Informed Trading (VPIN).

        Args:
            df: DataFrame with trade data (volume, direction)
            bucket_size: Volume bucket size
            estimation_window: Number of buckets for estimation

        Returns:
            Series with VPIN values
        """
        if "volume" not in df.columns or "direction" not in df.columns:
            logger.warning("VPIN requires 'volume' and 'direction' columns")
            return pd.Series(np.nan, index=df.index)

        # Create volume buckets
        df = df.copy()
        df["cum_volume"] = df["volume"].cumsum()
        df["bucket"] = (df["cum_volume"] // bucket_size).astype(int)

        # Aggregate by bucket
        bucket_data = df.groupby("bucket").agg(
            {
                "volume": "sum",
                "direction": lambda x: (x * df.loc[x.index, "volume"]).sum(),
            }
        )

        # Compute buy/sell volume
        bucket_data["buy_volume"] = bucket_data["direction"].clip(lower=0)
        bucket_data["sell_volume"] = (-bucket_data["direction"]).clip(lower=0)

        # VPIN = rolling average of |buy - sell| / total volume
        bucket_data["vpin"] = (
            abs(bucket_data["buy_volume"] - bucket_data["sell_volume"])
            .rolling(estimation_window)
            .mean()
            / bucket_data["volume"].rolling(estimation_window).mean()
        )

        # Map back to original index
        df["vpin"] = df["bucket"].map(bucket_data["vpin"])

        logger.info("Computed VPIN")
        return df["vpin"]

    def compute_order_flow_imbalance(
        self,
        df: pd.DataFrame,
        window: int = 100,
    ) -> pd.DataFrame:
        """
        Compute order flow imbalance metrics.

        Args:
            df: DataFrame with trade data
            window: Rolling window size

        Returns:
            DataFrame with order flow features
        """
        features = pd.DataFrame(index=df.index)

        if "direction" in df.columns and "volume" in df.columns:
            # Signed volume
            df["signed_volume"] = df["direction"] * df["volume"]

            # Rolling order flow imbalance
            features["order_flow_imbalance"] = (
                df["signed_volume"].rolling(window).sum() / df["volume"].rolling(window).sum()
            )

            # Buy/sell ratio
            buy_volume = df["signed_volume"].clip(lower=0).rolling(window).sum()
            sell_volume = (-df["signed_volume"]).clip(lower=0).rolling(window).sum()
            features["buy_sell_ratio"] = buy_volume / (sell_volume + 1e-10)

        logger.info("Computed order flow imbalance")
        return features

    def compute_realized_volatility(
        self,
        df: pd.DataFrame,
        frequencies: List[int] = [5, 10, 30, 60],
        price_col: str = "price",
    ) -> pd.DataFrame:
        """
        Compute realized volatility at different frequencies.

        Args:
            df: DataFrame with price data
            frequencies: List of window sizes (in periods)
            price_col: Name of price column

        Returns:
            DataFrame with realized volatility features
        """
        features = pd.DataFrame(index=df.index)

        # Log returns - ensure we have pandas Series for rolling operations
        log_returns = pd.Series(np.log(df[price_col] / df[price_col].shift(1)), index=df.index)

        for freq in frequencies:
            # Standard realized volatility
            features[f"realized_vol_{freq}"] = log_returns.rolling(freq).std() * np.sqrt(freq)

            # Parkinson volatility (high-low)
            if "high" in df.columns and "low" in df.columns:
                hl_ratio = pd.Series(np.log(df["high"] / df["low"]) ** 2, index=df.index)
                features[f"parkinson_vol_{freq}"] = np.sqrt(
                    hl_ratio.rolling(freq).mean() / (4 * np.log(2))
                )

        logger.info(f"Computed realized volatility for frequencies: {frequencies}")
        return features

    def compute_liquidity_measures(
        self,
        df: pd.DataFrame,
        rolling_window: int = 300,
    ) -> pd.DataFrame:
        """
        Compute liquidity measures.

        Args:
            df: DataFrame with price and volume data
            rolling_window: Window for rolling statistics

        Returns:
            DataFrame with liquidity features
        """
        features = pd.DataFrame(index=df.index)

        # Amihud illiquidity
        if "volume" in df.columns and "price" in df.columns:
            returns = df["price"].pct_change().abs()
            dollar_volume = df["price"] * df["volume"]
            features["amihud_illiquidity"] = (
                (returns / (dollar_volume + 1e-10)).rolling(rolling_window).mean()
            )

        # Roll spread estimator
        if "price" in df.columns:
            price_changes = df["price"].diff()
            covariance = price_changes.rolling(rolling_window).cov(price_changes.shift(1))
            features["roll_spread"] = 2 * np.sqrt(-covariance.clip(upper=0))

        # Liquidity ratio (volume / volatility)
        if "volume" in df.columns:
            vol = df["price"].pct_change().rolling(rolling_window).std()
            features["liquidity_ratio"] = df["volume"].rolling(rolling_window).mean() / (
                vol + 1e-10
            )

        logger.info("Computed liquidity measures")
        return features

    def engineer_all_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Engineer all microstructure features based on config.

        Args:
            df: Input DataFrame with market data

        Returns:
            DataFrame with all engineered features
        """
        features = pd.DataFrame(index=df.index)

        config = self.config.get("order_book", {})
        if config.get("imbalance", {}).get("enabled", True):
            levels = config.get("imbalance", {}).get("levels", [1, 5, 10])
            imb_features = self.compute_order_book_imbalance(df, levels=levels)
            features = pd.concat([features, imb_features], axis=1)

        if config.get("spread", {}).get("enabled", True):
            window = config.get("spread", {}).get("rolling_window", 60)
            spread_features = self.compute_spread_features(df, rolling_window=window)
            features = pd.concat([features, spread_features], axis=1)

        # Trade flow features
        flow_config = self.config.get("trade_flow", {})
        if flow_config.get("toxicity", {}).get("enabled", True):
            vpin = self.compute_vpin(df)
            features["vpin"] = vpin

        if flow_config.get("volume", {}).get("enabled", True):
            flow_features = self.compute_order_flow_imbalance(df)
            features = pd.concat([features, flow_features], axis=1)

        # Volatility features
        vol_config = self.config.get("volatility", {})
        if vol_config.get("realized_vol", {}).get("enabled", True):
            frequencies = vol_config.get("realized_vol", {}).get("frequencies", [5, 10, 30, 60])
            vol_features = self.compute_realized_volatility(df, frequencies=frequencies)
            features = pd.concat([features, vol_features], axis=1)

        # Liquidity features
        liq_config = self.config.get("liquidity", {})
        if liq_config.get("measures", {}).get("enabled", True):
            window = liq_config.get("measures", {}).get("rolling_window", 300)
            liq_features = self.compute_liquidity_measures(df, rolling_window=window)
            features = pd.concat([features, liq_features], axis=1)

        logger.info(f"Engineered {len(features.columns)} microstructure features")
        return features


class OptionsFeatureEngineer:
    """Engineer features from options market data."""

    def __init__(self, config: Union[str, Path, Dict]):
        """
        Initialize options feature engineer.

        Args:
            config: Path to config file or config dictionary
        """
        if isinstance(config, (str, Path)):
            with open(config, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

        logger.info("Initialized OptionsFeatureEngineer")

    def compute_iv_skew(
        self,
        df: pd.DataFrame,
        maturity: int = 30,
    ) -> pd.DataFrame:
        """
        Compute implied volatility skew metrics.

        Args:
            df: DataFrame with IV surface data
            maturity: Days to maturity

        Returns:
            DataFrame with skew features
        """
        features = pd.DataFrame(index=df.index)

        # Filter for specific maturity
        maturity_df = df[df["days_to_expiry"] == maturity]

        if len(maturity_df) > 0:
            # Put-call skew
            otm_puts = maturity_df[maturity_df["moneyness"] < 1.0]["iv"]
            otm_calls = maturity_df[maturity_df["moneyness"] > 1.0]["iv"]

            if len(otm_puts) > 0 and len(otm_calls) > 0:
                features[f"put_call_skew_{maturity}d"] = otm_puts.mean() - otm_calls.mean()

            # Fit polynomial to smile and extract curvature
            if "moneyness" in maturity_df.columns and "iv" in maturity_df.columns:
                moneyness = maturity_df["moneyness"].to_numpy()
                iv = maturity_df["iv"].to_numpy()

                if len(moneyness) > 2:
                    coeffs = np.polyfit(moneyness, iv, deg=2)
                    features[f"smile_curvature_{maturity}d"] = coeffs[0]

        logger.info(f"Computed IV skew for {maturity}d maturity")
        return features

    def compute_term_structure(
        self,
        df: pd.DataFrame,
        strike_type: str = "atm",
    ) -> pd.DataFrame:
        """
        Compute term structure of implied volatility.

        Args:
            df: DataFrame with IV data across maturities
            strike_type: Strike type ('atm', 'otm', 'itm')

        Returns:
            DataFrame with term structure features
        """
        features = pd.DataFrame(index=df.index)

        # Filter for ATM options
        atm_df = df[abs(df["moneyness"] - 1.0) < 0.05]

        if len(atm_df) > 0:
            # Sort by maturity
            atm_df = atm_df.sort_values("days_to_expiry")

            # Term slope (short vs long)
            short_iv = atm_df[atm_df["days_to_expiry"] <= 30]["iv"].mean()
            long_iv = atm_df[atm_df["days_to_expiry"] >= 60]["iv"].mean()

            features["term_slope"] = long_iv - short_iv

            # Term structure curvature
            if len(atm_df["days_to_expiry"].unique()) > 2:
                days = atm_df["days_to_expiry"].to_numpy()
                iv = atm_df["iv"].to_numpy()
                coeffs = np.polyfit(days, iv, deg=2)
                features["term_curvature"] = coeffs[0]

        logger.info("Computed term structure features")
        return features

    def compute_vol_of_vol(
        self,
        df: pd.DataFrame,
        window: int = 30,
    ) -> pd.Series:
        """
        Compute volatility of volatility.

        Args:
            df: DataFrame with IV time series
            window: Rolling window for estimation

        Returns:
            Series with vol-of-vol
        """
        if "iv" in df.columns:
            iv_returns = df["iv"].pct_change()
            vol_of_vol = iv_returns.rolling(window).std()
            logger.info("Computed vol-of-vol")
            return vol_of_vol
        else:
            logger.warning("No 'iv' column found for vol-of-vol computation")
            return pd.Series(np.nan, index=df.index)

    def compute_greeks_exposure(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute aggregated Greeks exposure.

        Args:
            df: DataFrame with options Greeks

        Returns:
            DataFrame with Greeks exposure features
        """
        features = pd.DataFrame(index=df.index)

        # Aggregate Greeks by type
        greek_cols = ["gamma", "vanna", "vomma", "delta", "vega"]

        for greek in greek_cols:
            if greek in df.columns:
                # Total exposure
                features[f"total_{greek}"] = df[greek].sum()

                # Net exposure (calls - puts)
                if "option_type" in df.columns:
                    calls = df[df["option_type"] == "call"][greek].sum()
                    puts = df[df["option_type"] == "put"][greek].sum()
                    features[f"{greek}_imbalance"] = calls - puts

        logger.info("Computed Greeks exposure")
        return features

    def compute_variance_risk_premium(
        self,
        df: pd.DataFrame,
        horizon: int = 30,
    ) -> pd.Series:
        """
        Compute variance risk premium (implied - realized variance).

        Args:
            df: DataFrame with IV and realized vol
            horizon: Forecast horizon in days

        Returns:
            Series with VRP
        """
        if "iv" in df.columns and "realized_vol" in df.columns:
            # Variance risk premium
            implied_var = (df["iv"] ** 2) * horizon / 252
            realized_var = (df["realized_vol"] ** 2) * horizon / 252

            vrp = implied_var - realized_var

            logger.info("Computed variance risk premium")
            return vrp
        else:
            logger.warning("Missing columns for VRP computation")
            return pd.Series(np.nan, index=df.index)

    def engineer_all_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Engineer all options features based on config.

        Args:
            df: Input DataFrame with options data

        Returns:
            DataFrame with all engineered features
        """
        features = pd.DataFrame(index=df.index)

        # IV skew features
        skew_config = self.config.get("implied_volatility", {}).get("skew", {})
        if skew_config.get("enabled", True):
            maturities = skew_config.get("maturities", [30, 60, 90])
            for maturity in maturities:
                skew_features = self.compute_iv_skew(df, maturity=maturity)
                features = pd.concat([features, skew_features], axis=1)

        # Term structure
        term_config = self.config.get("implied_volatility", {}).get("term_structure", {})
        if term_config.get("enabled", True):
            term_features = self.compute_term_structure(df)
            features = pd.concat([features, term_features], axis=1)

        # Vol-of-vol
        vvol_config = self.config.get("volatility_of_volatility", {})
        if vvol_config.get("enabled", True):
            window = vvol_config.get("estimation", {}).get("window", 30)
            features["vol_of_vol"] = self.compute_vol_of_vol(df, window=window)

        # Greeks
        greeks_config = self.config.get("greeks", {})
        if greeks_config.get("enabled", True):
            greeks_features = self.compute_greeks_exposure(df)
            features = pd.concat([features, greeks_features], axis=1)

        # Variance risk premium
        rvs_config = self.config.get("realized_vs_implied", {})
        if rvs_config.get("enabled", True):
            horizons = rvs_config.get("horizons", [7, 14, 30])
            for horizon in horizons:
                features[f"vrp_{horizon}d"] = self.compute_variance_risk_premium(
                    df, horizon=horizon
                )

        logger.info(f"Engineered {len(features.columns)} options features")
        return features


class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline."""

    def __init__(
        self,
        microstructure_config: Optional[Union[str, Path, Dict]] = None,
        options_config: Optional[Union[str, Path, Dict]] = None,
    ):
        """
        Initialize feature engineering pipeline.

        Args:
            microstructure_config: Config for microstructure features
            options_config: Config for options features
        """
        self.microstructure_engineer = None
        self.options_engineer = None

        if microstructure_config:
            self.microstructure_engineer = MicrostructureFeatureEngineer(microstructure_config)

        if options_config:
            self.options_engineer = OptionsFeatureEngineer(options_config)

        logger.info("Initialized FeatureEngineeringPipeline")

    def engineer_features(
        self,
        microstructure_data: Optional[pd.DataFrame] = None,
        options_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Engineer all features from available data sources.

        Args:
            microstructure_data: Market microstructure data
            options_data: Options market data

        Returns:
            Combined DataFrame with all features
        """
        all_features = []

        if microstructure_data is not None and self.microstructure_engineer:
            micro_features = self.microstructure_engineer.engineer_all_features(microstructure_data)
            all_features.append(micro_features)

        if options_data is not None and self.options_engineer:
            options_features = self.options_engineer.engineer_all_features(options_data)
            all_features.append(options_features)

        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            logger.info(f"Combined pipeline produced {len(combined_features.columns)} features")
            return combined_features
        else:
            logger.warning("No features engineered")
            return pd.DataFrame()

    def save_features(
        self,
        features: pd.DataFrame,
        filepath: Union[str, Path],
        format: str = "parquet",
    ) -> None:
        """
        Save engineered features.

        Args:
            features: Features DataFrame
            filepath: Output file path
            format: File format ('parquet', 'csv', 'hdf5')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            features.to_parquet(filepath)
        elif format == "csv":
            features.to_csv(filepath)
        elif format == "hdf5":
            features.to_hdf(filepath, key="features", mode="w")
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved {len(features.columns)} features to {filepath}")
