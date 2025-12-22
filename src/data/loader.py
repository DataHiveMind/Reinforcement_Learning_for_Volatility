"""
Data loading utilities for market data.

Supports loading from multiple sources:
- Local files (CSV, Parquet, HDF5)
- ArcticDB for high-performance time-series storage
- Market data APIs (Yahoo Finance, etc.)
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from arcticdb import Arctic

logger = logging.getLogger(__name__)


class DataLoader:
    """Base class for loading market data from various sources."""

    def __init__(self, data_dir: Union[str, Path] = "data/"):
        """
        Initialize DataLoader.

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(
        self,
        filepath: Union[str, Path],
        parse_dates: bool = True,
        index_col: Optional[str] = "timestamp",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file
            parse_dates: Whether to parse date columns
            index_col: Column to use as index
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame with loaded data
        """
        filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.data_dir / filepath

        logger.info(f"Loading CSV from {filepath}")

        df = pd.read_csv(
            filepath,
            parse_dates=parse_dates,
            index_col=index_col,
            **kwargs,
        )

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def load_parquet(
        self,
        filepath: Union[str, Path],
        use_polars: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Load data from Parquet file.

        Args:
            filepath: Path to Parquet file
            use_polars: Whether to use Polars instead of Pandas
            **kwargs: Additional arguments for read_parquet

        Returns:
            DataFrame with loaded data
        """
        filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.data_dir / filepath

        logger.info(f"Loading Parquet from {filepath}")

        if use_polars:
            df = pl.read_parquet(filepath, **kwargs)
        else:
            df = pd.read_parquet(filepath, **kwargs)

        logger.info(f"Loaded {len(df)} rows")
        return df

    def load_hdf5(
        self,
        filepath: Union[str, Path],
        key: str = "data",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from HDF5 file.

        Args:
            filepath: Path to HDF5 file
            key: Key/table name in HDF5 file
            **kwargs: Additional arguments for pd.read_hdf

        Returns:
            DataFrame with loaded data
        """
        filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.data_dir / filepath

        logger.info(f"Loading HDF5 from {filepath}, key={key}")

        df = pd.read_hdf(filepath, key=key, **kwargs)

        logger.info(f"Loaded {len(df)} rows")
        return df

    def load_multiple(
        self,
        filepaths: List[Union[str, Path]],
        concat: bool = True,
        **kwargs,
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Load multiple data files.

        Args:
            filepaths: List of file paths
            concat: Whether to concatenate into single DataFrame
            **kwargs: Additional loading arguments

        Returns:
            Single DataFrame if concat=True, else list of DataFrames
        """
        dfs = []

        for filepath in filepaths:
            filepath = Path(filepath)
            suffix = filepath.suffix.lower()

            if suffix == ".csv":
                df = self.load_csv(filepath, **kwargs)
            elif suffix in [".parquet", ".pq"]:
                df = self.load_parquet(filepath, **kwargs)
            elif suffix in [".h5", ".hdf5"]:
                df = self.load_hdf5(filepath, **kwargs)
            else:
                logger.warning(f"Unsupported file format: {suffix}")
                continue

            dfs.append(df)

        if concat and dfs:
            logger.info(f"Concatenating {len(dfs)} DataFrames")
            return pd.concat(dfs, axis=0, ignore_index=False)

        return dfs

    def save_csv(
        self,
        df: pd.DataFrame,
        filepath: Union[str, Path],
        **kwargs,
    ) -> None:
        """Save DataFrame to CSV."""
        filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.data_dir / filepath

        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, **kwargs)
        logger.info(f"Saved to {filepath}")

    def save_parquet(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        filepath: Union[str, Path],
        **kwargs,
    ) -> None:
        """Save DataFrame to Parquet."""
        filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.data_dir / filepath

        filepath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(df, pl.DataFrame):
            df.write_parquet(filepath, **kwargs)
        else:
            df.to_parquet(filepath, **kwargs)

        logger.info(f"Saved to {filepath}")


class ArcticDataLoader:
    """High-performance data loader using ArcticDB."""

    def __init__(
        self,
        uri: str = "lmdb://data/arctic",
        library: str = "market_data",
    ):
        """
        Initialize ArcticDB loader.

        Args:
            uri: ArcticDB connection URI
            library: Library name for data storage
        """
        self.uri = uri
        self.library_name = library
        self.arctic = Arctic(uri)

        # Create or get library
        try:
            self.library = self.arctic.get_library(library)
            logger.info(f"Connected to existing library: {library}")
        except Exception:
            self.library = self.arctic.create_library(library)
            logger.info(f"Created new library: {library}")

    def write(
        self,
        symbol: str,
        data: pd.DataFrame,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Write data to ArcticDB.

        Args:
            symbol: Symbol/identifier for the data
            data: DataFrame to store
            metadata: Optional metadata dictionary
        """
        logger.info(f"Writing {len(data)} rows for symbol: {symbol}")
        self.library.write(symbol, data, metadata=metadata)

    def read(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Read data from ArcticDB.

        Args:
            symbol: Symbol to read
            start: Start timestamp (optional)
            end: End timestamp (optional)
            columns: Columns to read (optional)

        Returns:
            DataFrame with requested data
        """
        logger.info(f"Reading symbol: {symbol}")

        # Build query
        query_params = {}
        if start or end:
            query_params["date_range"] = (start, end)
        if columns:
            query_params["columns"] = columns

        versioned_item = self.library.read(symbol, **query_params)

        # ArcticDB returns a VersionedItem with a .data attribute
        data = versioned_item.data

        # Ensure we have a pandas DataFrame (ArcticDB stores pandas DataFrames)
        if not isinstance(data, pd.DataFrame):
            # If somehow we get something else, try to convert
            data = pd.DataFrame(data)  # type: ignore[arg-type]

        logger.info(f"Loaded {len(data)} rows")
        return data

    def append(
        self,
        symbol: str,
        data: pd.DataFrame,
    ) -> None:
        """
        Append data to existing symbol.

        Args:
            symbol: Symbol to append to
            data: DataFrame to append
        """
        logger.info(f"Appending {len(data)} rows to symbol: {symbol}")
        self.library.append(symbol, data)

    def update(
        self,
        symbol: str,
        data: pd.DataFrame,
    ) -> None:
        """
        Update existing data.

        Args:
            symbol: Symbol to update
            data: DataFrame with updates
        """
        logger.info(f"Updating {len(data)} rows for symbol: {symbol}")
        self.library.update(symbol, data)

    def delete(self, symbol: str) -> None:
        """Delete symbol from library."""
        logger.info(f"Deleting symbol: {symbol}")
        self.library.delete(symbol)

    def list_symbols(self) -> List[str]:
        """List all symbols in library."""
        return self.library.list_symbols()

    def has_symbol(self, symbol: str) -> bool:
        """Check if symbol exists."""
        return symbol in self.list_symbols()

    def snapshot(
        self,
        symbol: str,
        snapshot_name: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Create a snapshot of the data.

        Args:
            symbol: Symbol to snapshot
            snapshot_name: Name for the snapshot
            metadata: Optional metadata
        """
        logger.info(f"Creating snapshot '{snapshot_name}' for symbol: {symbol}")
        self.library.snapshot(symbol, snapshot_name)


class MarketDataLoader(DataLoader):
    """Specialized loader for market data from various sources."""

    def __init__(self, data_dir: Union[str, Path] = "data/"):
        super().__init__(data_dir)

    def load_yahoo_finance(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Load data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

        logger.info(f"Downloading {ticker} from Yahoo Finance: {start} to {end}")

        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start, end=end, interval=interval)

        logger.info(f"Downloaded {len(df)} rows")
        return df

    def load_orderbook_snapshot(
        self,
        filepath: Union[str, Path],
        level: int = 10,
    ) -> pd.DataFrame:
        """
        Load order book snapshot data.

        Args:
            filepath: Path to order book data
            level: Number of order book levels

        Returns:
            DataFrame with order book data
        """
        df = self.load_parquet(filepath)

        # Validate order book columns
        expected_cols = []
        for i in range(1, level + 1):
            expected_cols.extend(
                [
                    f"bid_price_{i}",
                    f"bid_size_{i}",
                    f"ask_price_{i}",
                    f"ask_size_{i}",
                ]
            )

        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing order book columns: {missing_cols}")

        # Ensure we return pandas DataFrame
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        return df

    def load_trades(
        self,
        filepath: Union[str, Path],
        parse_direction: bool = True,
    ) -> pd.DataFrame:
        """
        Load trade data.

        Args:
            filepath: Path to trade data
            parse_direction: Whether to classify trade direction

        Returns:
            DataFrame with trade data
        """
        df = self.load_parquet(filepath)

        # Ensure we're working with pandas DataFrame
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        if parse_direction and "direction" not in df.columns:
            # Simple tick rule
            price_diff = df["price"].diff()
            df["direction"] = np.sign(price_diff)
            # Use forward fill (fillna method parameter is deprecated in pandas 2.0+)
            df.loc[df["direction"] == 0, "direction"] = np.nan
            df["direction"] = df["direction"].ffill()

        return df
