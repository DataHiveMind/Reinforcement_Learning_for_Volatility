# Quick Start: OpenBB Integration

## Installation

### Option 1: Update Existing Environment (Recommended)

```bash
# Update conda environment with new dependencies
conda env update -f environment.yml --prune

# Activate environment
conda activate rl-volatility

# Verify OpenBB installation
python -c "from openbb import obb; print('✅ OpenBB installed successfully')"
```

### Option 2: Fresh Install

```bash
# Remove old environment
conda env remove -n rl-volatility

# Create new environment
conda env create -f environment.yml

# Activate
conda activate rl-volatility
```

### Option 3: Pip Only

```bash
pip install -r requirements.txt
```

## Test the Integration

### 1. Run the Example Script

```bash
python examples/load_openbb_data.py
```

This will:
- ✅ Load SPY historical data from Yahoo Finance
- ✅ Attempt to fetch order book snapshot
- ✅ Load options chain with IVs and Greeks
- 💾 Save data to `data/raw/` directory

### 2. Run the Notebook

```bash
jupyter lab notebook/01_feature_engineering.ipynb
```

Execute all cells to:
- Load real market data using OpenBB
- Engineer microstructure features
- Calculate options-based features
- Generate visualizations

## Configuration (Optional)

### Get Free API Keys

1. **Polygon.io** (Real-time market data)
   - Sign up: https://polygon.io/
   - Free tier: 5 API calls/minute
   - Premium: $200/month for Level 2 data

2. **Alpha Vantage** (Stock data)
   - Sign up: https://www.alphavantage.co/
   - Free tier: 5 API calls/minute

3. **CBOE** (Options data)
   - Sign up: https://www.cboe.com/data/
   - Free delayed data available

### Set Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc
export OPENBB_POLYGON_API_KEY="your_polygon_key_here"
export OPENBB_ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
export OPENBB_CBOE_API_KEY="your_cboe_key"

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

### Or Configure in Python

```python
import os
os.environ["OPENBB_POLYGON_API_KEY"] = "your_key"
os.environ["OPENBB_ALPHA_VANTAGE_API_KEY"] = "your_key"
```

## Quick Usage Example

```python
from src.data.loader import MarketDataLoader

# Initialize
loader = MarketDataLoader()

# Example 1: Load historical data (free - no API key needed)
prices = loader.load_openbb_historical(
    symbol="AAPL",
    start="2024-01-01",
    end="2024-12-01",
    provider="yfinance",
    interval="1d"
)

print(f"Loaded {len(prices)} price bars")
print(prices.head())

# Example 2: Load options chain (free with CBOE)
options = loader.load_openbb_options(
    symbol="AAPL",
    provider="cboe"
)

print(f"Loaded {len(options)} options contracts")
print(options.head())

# Example 3: Order book (requires Polygon API key)
try:
    orderbook = loader.load_openbb_orderbook(
        symbol="AAPL",
        provider="polygon"
    )
    print("Order book loaded!")
except Exception as e:
    print(f"Order book unavailable: {e}")
    print("(This requires premium Polygon subscription)")
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'openbb'"

```bash
# Install OpenBB
pip install openbb openbb-core

# Or update conda environment
conda env update -f environment.yml
```

### "API rate limit exceeded"

- You're hitting the free tier limit
- Wait a minute and try again
- Or get an API key and configure it

### "Order book data unavailable"

- Level 2 order book requires premium Polygon subscription
- Notebook will automatically fall back to approximation method
- This is normal if you don't have a paid subscription

### "ImportError: cannot import name 'obb' from 'openbb'"

```bash
# Make sure you have the latest version
pip install --upgrade openbb openbb-core
```

## What's Next?

1. ✅ Run the example script: `python examples/load_openbb_data.py`
2. ✅ Open the notebook: `jupyter lab notebook/01_feature_engineering.ipynb`
3. ✅ Load your favorite ticker (SPY, AAPL, TSLA, etc.)
4. ✅ Train RL models on real market data
5. ✅ Compare with synthetic data baseline

## Need Help?

- OpenBB Docs: https://docs.openbb.co/
- OpenBB Discord: https://discord.gg/openbb
- Project Issues: Create an issue in this repository

## Features Now Available

✅ **Real market data** instead of synthetic
✅ **Multiple data providers** (Yahoo, Polygon, Alpha Vantage, etc.)
✅ **Options chains** with real IVs and Greeks
✅ **Order book approximation** from OHLCV (free)
✅ **Level 2 order book** with premium subscription
✅ **Production-ready** data pipeline

Enjoy building better trading strategies with real data! 🚀
