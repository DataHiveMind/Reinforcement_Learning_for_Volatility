"""
Data processing, loading, and feature engineering modules.

This package provides functionality for:
- Loading market data from various sources (yfinance, local files)
- Preprocessing and cleaning data
- Engineering microstructure and options features
- Handling time-series data with ArcticDB
"""

from .feature_engineering import (
    FeatureEngineeringPipeline,
    MicrostructureFeatureEngineer,
    OptionsFeatureEngineer,
)
from .loader import ArcticDataLoader, DataLoader
from .preprocess import DataPreprocessor, TimeSeriesPreprocessor
from .yfinance_loader import YFinanceDataLoader, create_sample_data

__all__ = [
    "DataLoader",
    "YFinanceDataLoader",
    "create_sample_data",
    "ArcticDataLoader",
    "DataPreprocessor",
    "TimeSeriesPreprocessor",
    "MicrostructureFeatureEngineer",
    "OptionsFeatureEngineer",
    "FeatureEngineeringPipeline",
]
