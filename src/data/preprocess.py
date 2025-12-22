"""
Data preprocessing and cleaning utilities.

Handles:
- Missing values
- Outlier detection and treatment
- Normalization and scaling
- Time-series specific preprocessing
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Base preprocessor for market data."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scalers = {}
        self.stats = {}

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "forward_fill",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Handle missing values.

        Args:
            df: Input DataFrame
            method: Method for handling missing values
                   ('forward_fill', 'backward_fill', 'interpolate', 'drop')
            limit: Maximum number of consecutive NaNs to fill

        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        initial_missing = df.isnull().sum().sum()

        if method == "forward_fill":
            df = df.fillna(method="ffill", limit=limit)
        elif method == "backward_fill":
            df = df.fillna(method="bfill", limit=limit)
        elif method == "interpolate":
            df = df.interpolate(method="linear", limit=limit)
        elif method == "drop":
            df = df.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")

        final_missing = df.isnull().sum().sum()
        logger.info(
            f"Handled missing values: {initial_missing} -> {final_missing} " f"(method: {method})"
        )

        return df

    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "zscore",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Detect outliers in data.

        Args:
            df: Input DataFrame
            columns: Columns to check (None = all numeric)
            method: Detection method ('zscore', 'iqr', 'isolation_forest')
            threshold: Threshold for outlier detection

        Returns:
            Boolean DataFrame indicating outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outliers = pd.DataFrame(False, index=df.index, columns=columns)

        if method == "zscore":
            z_scores = np.abs(stats.zscore(df[columns], nan_policy="omit"))
            outliers[columns] = z_scores > threshold

        elif method == "iqr":
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outliers[col] = (df[col] < lower) | (df[col] > upper)

        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(df[columns].fillna(0))
            outliers[columns] = predictions == -1

        else:
            raise ValueError(f"Unknown method: {method}")

        n_outliers = outliers.sum().sum()
        logger.info(f"Detected {n_outliers} outliers using {method} method")

        return outliers

    def treat_outliers(
        self,
        df: pd.DataFrame,
        method: str = "winsorize",
        percentiles: Tuple[float, float] = (1, 99),
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Treat outliers in data.

        Args:
            df: Input DataFrame
            method: Treatment method ('winsorize', 'clip', 'remove')
            percentiles: Lower and upper percentiles for winsorization
            columns: Columns to treat (None = all numeric)

        Returns:
            DataFrame with outliers treated
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if method == "winsorize":
            from scipy.stats.mstats import winsorize

            for col in columns:
                lower = (100 - percentiles[0]) / 100
                upper = (100 - percentiles[1]) / 100
                df[col] = winsorize(df[col], limits=(lower, upper))

        elif method == "clip":
            for col in columns:
                lower = df[col].quantile(percentiles[0] / 100)
                upper = df[col].quantile(percentiles[1] / 100)
                df[col] = df[col].clip(lower=lower, upper=upper)

        elif method == "remove":
            outliers = self.detect_outliers(df, columns=columns)
            mask = ~outliers.any(axis=1)
            df = df[mask]
            logger.info(f"Removed {(~mask).sum()} rows with outliers")

        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Treated outliers using {method} method")
        return df

    def normalize(
        self,
        df: pd.DataFrame,
        method: str = "zscore",
        columns: Optional[List[str]] = None,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Normalize/scale data.

        Args:
            df: Input DataFrame
            method: Scaling method ('zscore', 'robust', 'minmax')
            columns: Columns to scale (None = all numeric)
            fit: Whether to fit scaler (True) or use existing

        Returns:
            Scaled DataFrame
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if method not in self.scalers or fit:
            if method == "zscore":
                scaler = StandardScaler()
            elif method == "robust":
                scaler = RobustScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown method: {method}")

            if fit:
                scaler.fit(df[columns])
                self.scalers[method] = scaler

        scaler = self.scalers[method]
        df[columns] = scaler.transform(df[columns])

        logger.info(f"Normalized {len(columns)} columns using {method}")
        return df

    def clip_values(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        clip_range: Tuple[float, float] = (-5, 5),
    ) -> pd.DataFrame:
        """
        Clip values to a range.

        Args:
            df: Input DataFrame
            columns: Columns to clip
            clip_range: (min, max) range

        Returns:
            Clipped DataFrame
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        df[columns] = df[columns].clip(lower=clip_range[0], upper=clip_range[1])

        logger.info(f"Clipped {len(columns)} columns to range {clip_range}")
        return df


class TimeSeriesPreprocessor(DataPreprocessor):
    """Specialized preprocessor for time-series data."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)

    def resample(
        self,
        df: pd.DataFrame,
        freq: str = "1H",
        agg_func: Union[str, Dict] = "last",
    ) -> pd.DataFrame:
        """
        Resample time-series data.

        Args:
            df: Input DataFrame with datetime index
            freq: Resampling frequency ('1H', '1D', etc.)
            agg_func: Aggregation function or dict of functions per column

        Returns:
            Resampled DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        logger.info(f"Resampling to {freq}")
        df_resampled = df.resample(freq).agg(agg_func)

        logger.info(f"Resampled from {len(df)} to {len(df_resampled)} rows")
        return df_resampled

    def add_time_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Add time-based features.

        Args:
            df: Input DataFrame with datetime index
            features: List of features to add
                     (hour, day, dayofweek, month, quarter, year)

        Returns:
            DataFrame with time features
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        df = df.copy()

        if features is None:
            features = ["hour", "dayofweek", "month"]

        if "hour" in features:
            df["hour"] = df.index.hour
        if "day" in features:
            df["day"] = df.index.day
        if "dayofweek" in features:
            df["dayofweek"] = df.index.dayofweek
        if "month" in features:
            df["month"] = df.index.month
        if "quarter" in features:
            df["quarter"] = df.index.quarter
        if "year" in features:
            df["year"] = df.index.year

        logger.info(f"Added time features: {features}")
        return df

    def add_lags(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int],
        prefix: str = "lag",
    ) -> pd.DataFrame:
        """
        Add lagged features.

        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            prefix: Prefix for lag column names

        Returns:
            DataFrame with lag features
        """
        df = df.copy()

        for col in columns:
            for lag in lags:
                df[f"{prefix}_{col}_{lag}"] = df[col].shift(lag)

        logger.info(f"Added {len(columns) * len(lags)} lag features")
        return df

    def add_rolling_stats(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int],
        stats: List[str] = ["mean", "std"],
    ) -> pd.DataFrame:
        """
        Add rolling window statistics.

        Args:
            df: Input DataFrame
            columns: Columns to compute rolling stats for
            windows: List of window sizes
            stats: List of statistics ('mean', 'std', 'min', 'max', 'skew')

        Returns:
            DataFrame with rolling features
        """
        df = df.copy()

        for col in columns:
            for window in windows:
                rolling = df[col].rolling(window=window)

                if "mean" in stats:
                    df[f"{col}_rolling_mean_{window}"] = rolling.mean()
                if "std" in stats:
                    df[f"{col}_rolling_std_{window}"] = rolling.std()
                if "min" in stats:
                    df[f"{col}_rolling_min_{window}"] = rolling.min()
                if "max" in stats:
                    df[f"{col}_rolling_max_{window}"] = rolling.max()
                if "skew" in stats:
                    df[f"{col}_rolling_skew_{window}"] = rolling.skew()

        n_features = len(columns) * len(windows) * len(stats)
        logger.info(f"Added {n_features} rolling window features")
        return df

    def add_diff_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        periods: List[int] = [1],
    ) -> pd.DataFrame:
        """
        Add differenced features.

        Args:
            df: Input DataFrame
            columns: Columns to difference
            periods: Difference periods

        Returns:
            DataFrame with diff features
        """
        df = df.copy()

        for col in columns:
            for period in periods:
                df[f"{col}_diff_{period}"] = df[col].diff(period)

        logger.info(f"Added {len(columns) * len(periods)} diff features")
        return df

    def split_train_val_test(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time-series data into train/val/test sets.

        Args:
            df: Input DataFrame
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        logger.info(
            f"Split data: train={len(train_df)}, " f"val={len(val_df)}, test={len(test_df)}"
        )

        return train_df, val_df, test_df

    def forward_fill_limit(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        limit: int = 5,
    ) -> pd.DataFrame:
        """
        Forward fill with a limit to prevent stale data.

        Args:
            df: Input DataFrame
            columns: Columns to fill
            limit: Maximum consecutive fills

        Returns:
            Forward-filled DataFrame
        """
        df = df.copy()

        if columns is None:
            columns = df.columns.tolist()

        for col in columns:
            df[col] = df[col].fillna(method="ffill", limit=limit)

        return df
