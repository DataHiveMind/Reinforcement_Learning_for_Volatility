"""
Example: Load equities data using yfinance

This script demonstrates how to load and process stock market data
using Yahoo Finance (yfinance) for the volatility trading project.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.yfinance_loader import YFinanceDataLoader, create_sample_data


def main():
    """Demonstrate yfinance data loading capabilities."""

    print("=" * 70)
    print("YFinance Data Loader - Examples")
    print("=" * 70)

    loader = YFinanceDataLoader()

    # Example 1: Load single symbol
    print("\n📊 Example 1: Load SPY data for 6 months")
    print("-" * 70)
    spy_data = loader.load_historical("SPY", period="6mo", interval="1d")
    print(f"Shape: {spy_data.shape}")
    print(f"\nFirst 5 rows:")
    print(spy_data.head())
    print(f"\nLast 5 rows:")
    print(spy_data.tail())

    # Example 2: Load multiple symbols
    print("\n\n📈 Example 2: Load multiple symbols")
    print("-" * 70)
    symbols = ["SPY", "AAPL", "MSFT", "NVDA", "GOOGL"]
    multi_data = loader.load_multiple_symbols(symbols, period="3mo", interval="1d")

    for symbol, data in multi_data.items():
        latest_close = data["close"].iloc[-1]
        pct_change = ((data["close"].iloc[-1] / data["close"].iloc[0]) - 1) * 100
        print(
            f"{symbol:6s}: {len(data):4d} days | Latest: ${latest_close:8.2f} | "
            f"3M Change: {pct_change:+6.2f}%"
        )

    # Example 3: Technical indicators
    print("\n\n📉 Example 3: Add technical indicators")
    print("-" * 70)
    spy_with_ta = loader.calculate_technical_indicators(spy_data)

    indicators = ["close", "sma_20", "sma_50", "rsi", "macd", "volatility"]
    print(spy_with_ta[indicators].tail(10))

    # Example 4: Real-time quotes
    print("\n\n💹 Example 4: Real-time quotes")
    print("-" * 70)
    quotes = loader.load_realtime_quote(["SPY", "AAPL", "MSFT"])
    print(quotes[["symbol", "price", "bid", "ask", "volume"]])

    # Example 5: Options data
    print("\n\n🎯 Example 5: Options chain")
    print("-" * 70)
    options = loader.load_options_chain("SPY")

    if not options["calls"].empty:
        print(f"Expiration: {options.get('expiration', 'N/A')}")
        print(f"\nCalls: {len(options['calls'])} contracts")
        print(
            options["calls"][
                ["strike", "lastPrice", "bid", "ask", "volume", "impliedVolatility"]
            ].head()
        )

        print(f"\nPuts: {len(options['puts'])} contracts")
        print(
            options["puts"][
                ["strike", "lastPrice", "bid", "ask", "volume", "impliedVolatility"]
            ].head()
        )
    else:
        print("⚠️  No options data available")

    # Example 6: Company information
    print("\n\n🏢 Example 6: Company information")
    print("-" * 70)
    info = loader.load_company_info("AAPL")

    key_info = {
        "Name": info.get("longName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Market Cap": f"${info.get('marketCap', 0) / 1e9:.2f}B",
        "P/E Ratio": info.get("trailingPE"),
        "Dividend Yield": (
            f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get("dividendYield") else "N/A"
        ),
        "52W High": f"${info.get('fiftyTwoWeekHigh', 0):.2f}",
        "52W Low": f"${info.get('fiftyTwoWeekLow', 0):.2f}",
    }

    for key, value in key_info.items():
        print(f"{key:20s}: {value}")

    # Example 7: Create sample dataset
    print("\n\n📦 Example 7: Create sample dataset for training")
    print("-" * 70)
    sample_data = create_sample_data(
        symbols=["SPY", "QQQ", "IWM"], period="1y", interval="1d", add_indicators=True
    )

    print("Sample dataset created with technical indicators:")
    for symbol, data in sample_data.items():
        print(f"  {symbol}: {len(data)} rows, {len(data.columns)} columns")
        print(f"    Columns: {', '.join(data.columns[:10])}...")

    # Example 8: Save to CSV
    print("\n\n💾 Example 8: Save data to CSV")
    print("-" * 70)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol, data in sample_data.items():
        filepath = output_dir / f"{symbol}_daily_1y.csv"
        data.to_csv(filepath)
        print(f"Saved {symbol} data to {filepath}")

    print("\n" + "=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)

    return sample_data


if __name__ == "__main__":
    data = main()
