"""
Example script demonstrating OpenBB Terminal integration for loading market data.

This script shows how to:
1. Load historical OHLCV data
2. Get order book snapshots
3. Fetch options chains and Greeks
4. Save data for feature engineering
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import MarketDataLoader


def main():
    """Main function to demonstrate OpenBB data loading."""

    # Initialize loader
    loader = MarketDataLoader(data_dir=project_root / "data" / "raw")

    # Configuration
    SYMBOL = "SPY"
    START_DATE = "2024-01-01"
    END_DATE = "2024-12-01"

    print("=" * 60)
    print("OpenBB Terminal Data Loading Example")
    print("=" * 60)

    # 1. Load historical price data
    print(f"\n1. Loading historical data for {SYMBOL}...")
    print("-" * 60)

    try:
        prices_df = loader.load_openbb_historical(
            symbol=SYMBOL, start=START_DATE, end=END_DATE, provider="yfinance", interval="1d"
        )

        print(f"✅ Loaded {len(prices_df)} price bars")
        print(f"   Columns: {list(prices_df.columns)}")
        print(f"   Date range: {prices_df.index.min()} to {prices_df.index.max()}")
        print(f"\nFirst few rows:")
        print(prices_df.head())

        # Save to disk
        output_path = project_root / "data" / "raw" / f"{SYMBOL}_historical.parquet"
        loader.save_parquet(prices_df, output_path)
        print(f"\n💾 Saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error loading historical data: {e}")

    # 2. Load current order book
    print(f"\n2. Loading order book snapshot for {SYMBOL}...")
    print("-" * 60)

    try:
        orderbook_df = loader.load_openbb_orderbook(symbol=SYMBOL, provider="polygon")

        print(f"✅ Loaded order book snapshot")
        print(f"   Columns: {list(orderbook_df.columns)}")
        print(f"\nSnapshot:")
        print(orderbook_df.head())

        # Save to disk
        output_path = project_root / "data" / "raw" / f"{SYMBOL}_orderbook_snapshot.parquet"
        loader.save_parquet(orderbook_df, output_path)
        print(f"\n💾 Saved to: {output_path}")

    except Exception as e:
        print(f"⚠️ Order book not available: {e}")
        print("   (Requires premium data subscription)")

    # 3. Load options data
    print(f"\n3. Loading options chain for {SYMBOL}...")
    print("-" * 60)

    try:
        options_df = loader.load_openbb_options(symbol=SYMBOL, provider="cboe")

        print(f"✅ Loaded {len(options_df)} options contracts")
        print(f"   Columns: {list(options_df.columns)[:10]}...")

        if "expiration" in options_df.columns:
            print(f"   Expirations: {options_df['expiration'].nunique()}")

        print(f"\nFirst few contracts:")
        print(options_df.head(10))

        # Save to disk
        output_path = project_root / "data" / "raw" / f"{SYMBOL}_options_chain.parquet"
        loader.save_parquet(options_df, output_path)
        print(f"\n💾 Saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error loading options: {e}")

    # 4. Load Greeks data
    print(f"\n4. Loading options Greeks for {SYMBOL}...")
    print("-" * 60)

    try:
        greeks_df = loader.load_openbb_greeks(symbol=SYMBOL, provider="cboe")

        print(f"✅ Loaded Greeks data")
        print(f"   Shape: {greeks_df.shape}")
        print(f"   Columns: {list(greeks_df.columns)[:10]}...")
        print(f"\nSample Greeks:")
        print(greeks_df.head(10))

        # Save to disk
        output_path = project_root / "data" / "raw" / f"{SYMBOL}_greeks.parquet"
        loader.save_parquet(greeks_df, output_path)
        print(f"\n💾 Saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error loading Greeks: {e}")

    print("\n" + "=" * 60)
    print("✅ Data loading complete!")
    print("=" * 60)
    print(f"\nData saved to: {project_root / 'data' / 'raw'}")
    print("\nNext steps:")
    print("1. Run the feature engineering notebook: notebook/01_feature_engineering.ipynb")
    print("2. Train RL models with the processed features")


if __name__ == "__main__":
    main()
