"""
Simple data loader using yfinance for equities data.

This replaces the OpenBB integration with a simpler, more reliable yfinance-based loader.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf
from pandas import DataFrame

logger = logging.getLogger(__name__)


class YFinanceDataLoader:
    """
    Load equities data using Yahoo Finance (yfinance).

    Features:
    - Historical OHLCV data
    - Multiple timeframes (1m, 1h, 1d, etc.)
    - Multiple symbols at once
    - Automatic data validation
    - Built-in error handling
    """

    def __init__(self):
        """Initialize YFinance data loader."""
        self.cache: Dict[str, DataFrame] = {}

    def load_historical(
        self,
        symbol: Union[str, List[str]],
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        period: str = "1y",
        interval: str = "1d",
        auto_adjust: bool = True,
        prepost: bool = False,
    ) -> DataFrame:
        """
        Load historical OHLCV data from Yahoo Finance.

        Args:
            symbol: Ticker symbol(s) to load
            start: Start date (YYYY-MM-DD or datetime)
            end: End date (YYYY-MM-DD or datetime)
            period: Period to download (e.g., '1d', '5d', '1mo', '1y', 'max')
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            auto_adjust: Adjust OHLC automatically
            prepost: Include pre and post market data

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, (Adj Close)

        Example:
            >>> loader = YFinanceDataLoader()
            >>> data = loader.load_historical('SPY', period='1y', interval='1d')
            >>> data = loader.load_historical(['SPY', 'AAPL'], start='2023-01-01', end='2024-01-01')
        """
        try:
            # Convert symbol to list if single string
            symbols = [symbol] if isinstance(symbol, str) else symbol

            # Determine which time parameters to use (can't use both period and start/end)
            if start is not None or end is not None:
                # Use start/end dates, ignore period
                use_period = None
                use_start = start
                use_end = end
            else:
                # Use period, no start/end
                use_period = period
                use_start = None
                use_end = None

            logger.info(
                f"Loading data for {symbols} (period={use_period}, start={use_start}, end={use_end}, interval={interval})"
            )

            # Download data
            if len(symbols) == 1:
                ticker = yf.Ticker(symbols[0])
                data = ticker.history(
                    start=use_start,
                    end=use_end,
                    period=use_period,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    prepost=prepost,
                )
                data["symbol"] = symbols[0]
            else:
                # Download multiple symbols
                data = yf.download(
                    symbols,
                    start=use_start,
                    end=use_end,
                    period=use_period,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    prepost=prepost,
                    group_by="ticker",
                    threads=True,
                )

            # Validate data
            if data.empty:
                logger.warning(f"No data returned for {symbols}")
                return pd.DataFrame()

            # Clean column names
            data.columns = [col.lower().replace(" ", "_") for col in data.columns]

            logger.info(f"Loaded {len(data)} rows for {symbols}")
            return data

        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()

    def load_realtime_quote(self, symbol: Union[str, List[str]]) -> DataFrame:
        """
        Get real-time quote data.

        Args:
            symbol: Ticker symbol(s)

        Returns:
            DataFrame with current market data
        """
        try:
            symbols = [symbol] if isinstance(symbol, str) else symbol

            quotes = []
            for sym in symbols:
                ticker = yf.Ticker(sym)
                info = ticker.info

                quotes.append(
                    {
                        "symbol": sym,
                        "price": info.get("currentPrice", info.get("regularMarketPrice")),
                        "bid": info.get("bid"),
                        "ask": info.get("ask"),
                        "volume": info.get("volume"),
                        "market_cap": info.get("marketCap"),
                        "pe_ratio": info.get("trailingPE"),
                        "52w_high": info.get("fiftyTwoWeekHigh"),
                        "52w_low": info.get("fiftyTwoWeekLow"),
                    }
                )

            return pd.DataFrame(quotes)

        except Exception as e:
            logger.error(f"Error loading quote for {symbol}: {e}")
            return pd.DataFrame()

    def load_company_info(self, symbol: str) -> Dict:
        """
        Get company information.

        Args:
            symbol: Ticker symbol

        Returns:
            Dictionary with company details
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error loading info for {symbol}: {e}")
            return {}

    def load_dividends(self, symbol: str) -> DataFrame:
        """
        Get dividend history.

        Args:
            symbol: Ticker symbol

        Returns:
            DataFrame with dividend history
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.dividends
        except Exception as e:
            logger.error(f"Error loading dividends for {symbol}: {e}")
            return pd.DataFrame()

    def load_splits(self, symbol: str) -> DataFrame:
        """
        Get stock split history.

        Args:
            symbol: Ticker symbol

        Returns:
            DataFrame with split history
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.splits
        except Exception as e:
            logger.error(f"Error loading splits for {symbol}: {e}")
            return pd.DataFrame()

    def load_options_chain(self, symbol: str, date: Optional[str] = None) -> Dict[str, DataFrame]:
        """
        Get options chain data.

        Args:
            symbol: Ticker symbol
            date: Expiration date (YYYY-MM-DD). If None, uses nearest expiration.

        Returns:
            Dictionary with 'calls' and 'puts' DataFrames
        """
        try:
            ticker = yf.Ticker(symbol)

            if date is None:
                # Get nearest expiration
                expirations = ticker.options
                if not expirations:
                    logger.warning(f"No options available for {symbol}")
                    return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}
                date = expirations[0]

            opt = ticker.option_chain(date)

            return {
                "calls": opt.calls,
                "puts": opt.puts,
                "expiration": date,
            }

        except Exception as e:
            logger.error(f"Error loading options for {symbol}: {e}")
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}

    def load_multiple_symbols(
        self,
        symbols: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d",
    ) -> Dict[str, DataFrame]:
        """
        Load data for multiple symbols as separate DataFrames.

        Args:
            symbols: List of ticker symbols
            start: Start date
            end: End date
            period: Period to download
            interval: Data interval

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}

        for symbol in symbols:
            data = self.load_historical(
                symbol=symbol,
                start=start,
                end=end,
                period=period,
                interval=interval,
            )
            if not data.empty:
                results[symbol] = data

        return results

    def calculate_technical_indicators(self, data: DataFrame) -> DataFrame:
        """
        Add common technical indicators to price data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        df = data.copy()

        # Simple Moving Averages
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["sma_200"] = df["close"].rolling(window=200).mean()

        # Exponential Moving Average
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

        # Volatility (20-day)
        df["volatility"] = df["close"].pct_change().rolling(window=20).std() * np.sqrt(252)

        # Volume indicators
        df["volume_sma"] = df["volume"].rolling(window=20).mean()

        return df


def create_sample_data(
    symbols: List[str] = ["SPY", "AAPL", "MSFT"],
    period: str = "1y",
    interval: str = "1d",
    add_indicators: bool = True,
) -> Dict[str, DataFrame]:
    """
    Quick helper to create sample dataset for testing.

    Args:
        symbols: List of ticker symbols
        period: Time period
        interval: Data interval
        add_indicators: Whether to add technical indicators

    Returns:
        Dictionary mapping symbol to DataFrame with data
    """
    loader = YFinanceDataLoader()
    data = loader.load_multiple_symbols(symbols, period=period, interval=interval)

    if add_indicators:
        for symbol in data:
            data[symbol] = loader.calculate_technical_indicators(data[symbol])

    return data


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    loader = YFinanceDataLoader()

    # Load single symbol
    spy_data = loader.load_historical("SPY", period="6mo", interval="1d")
    print(f"\nSPY data shape: {spy_data.shape}")
    print(spy_data.head())

    # Load multiple symbols
    data = loader.load_multiple_symbols(["SPY", "AAPL", "MSFT"], period="1mo")
    for symbol, df in data.items():
        print(f"\n{symbol}: {len(df)} rows")

    # Get real-time quote
    quote = loader.load_realtime_quote("SPY")
    print(f"\nSPY Quote:")
    print(quote)

    # Load with indicators
    spy_with_ta = loader.calculate_technical_indicators(spy_data)
    print(f"\nSPY with indicators:")
    print(spy_with_ta[["close", "sma_20", "rsi", "volatility"]].tail())
