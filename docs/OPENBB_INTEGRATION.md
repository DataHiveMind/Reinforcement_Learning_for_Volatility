# OpenBB Terminal Integration

This project now uses [OpenBB Terminal](https://openbb.co/) to fetch real market data instead of relying solely on synthetic data generation.

## Features

- **Historical Price Data**: Load OHLCV data from multiple providers (Yahoo Finance, Polygon, Alpha Vantage, etc.)
- **Order Book Data**: Fetch Level 2 market depth data (requires premium data subscription)
- **Options Chains**: Get real-time options data with strikes, IVs, and Greeks
- **Multiple Providers**: Switch between data providers seamlessly

## Installation

### 1. Update Conda Environment

```bash
conda env update -f environment.yml
```

### 2. Install OpenBB (if using pip)

```bash
pip install openbb openbb-core
```

### 3. Configure API Keys (Optional but Recommended)

OpenBB supports multiple data providers. For premium features, configure your API keys:

```bash
# Run OpenBB configuration
python -c "from openbb import obb; obb"

# Or set environment variables
export OPENBB_POLYGON_API_KEY="your_polygon_key"
export OPENBB_ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
export OPENBB_CBOE_API_KEY="your_cboe_key"
```

Get free API keys:
- [Polygon.io](https://polygon.io/) - Market data and order book
- [Alpha Vantage](https://www.alphavantage.co/) - Stock data
- [CBOE](https://www.cboe.com/data/) - Options data

## Usage

### Quick Start Example

```python
from src.data.loader import MarketDataLoader

# Initialize loader
loader = MarketDataLoader()

# Load historical data
df = loader.load_openbb_historical(
    symbol="SPY",
    start="2024-01-01",
    end="2024-12-01",
    provider="yfinance",
    interval="1d"
)

# Load options chain
options = loader.load_openbb_options(
    symbol="SPY",
    provider="cboe"
)

# Load order book (requires premium subscription)
orderbook = loader.load_openbb_orderbook(
    symbol="SPY",
    provider="polygon"
)
```

### Run Example Script

```bash
python examples/load_openbb_data.py
```

### Updated Notebook

The feature engineering notebook (`notebook/01_feature_engineering.ipynb`) now uses OpenBB to:
1. Load real historical price data
2. Construct order book levels from OHLCV data
3. Fetch real options chains with implied volatilities
4. Calculate Greeks and volatility surfaces

## Data Providers

### Free Providers
- **Yahoo Finance**: Daily/intraday OHLCV data
- **FRED**: Economic indicators
- **FMP**: Financial statements

### Premium Providers (API Key Required)
- **Polygon.io**: Real-time/historical market data, Level 2 order book
- **Alpha Vantage**: Stocks, forex, crypto
- **IEX Cloud**: Real-time market data
- **Intrinio**: Options, fundamentals
- **CBOE**: Options chains and Greeks

## Available Methods

### MarketDataLoader Class

| Method | Description | Provider |
|--------|-------------|----------|
| `load_openbb_historical()` | Historical OHLCV data | yfinance, polygon, alpha_vantage |
| `load_openbb_orderbook()` | Level 2 order book snapshot | polygon, cboe |
| `load_openbb_options()` | Options chains | cboe, tradier, intrinio |
| `load_openbb_greeks()` | Options Greeks (delta, gamma, vega, etc.) | cboe, tradier |

## Order Book Data

### Level 2 Data (Premium)
For real Level 2 order book data, you need:
1. Polygon.io subscription ($200/month for real-time)
2. Or CBOE data subscription

```python
# Load real-time order book
orderbook = loader.load_openbb_orderbook(
    symbol="SPY",
    provider="polygon"
)
```

### Approximated Order Book (Free)
The notebook includes logic to construct approximate order book levels from OHLCV data:
- Estimates bid-ask spread from high-low range
- Distributes volume across multiple levels
- Suitable for backtesting and feature engineering

## Troubleshooting

### "OpenBB not found"
```bash
pip install openbb openbb-core
```

### "API rate limit exceeded"
- Get a free API key from the provider
- Configure it in OpenBB settings
- Or switch to a different provider

### "Order book data unavailable"
- Order book requires premium subscription
- Notebook will fall back to approximation method
- Consider Polygon.io for Level 2 data access

## Benefits Over Synthetic Data

✅ **Real Market Microstructure**: Actual bid-ask spreads and order flow
✅ **Realistic Volatility**: True implied volatility surfaces
✅ **Market Events**: Captures real market dynamics and regime changes
✅ **Better Backtesting**: More reliable performance estimates
✅ **Production Ready**: Same data source for training and live trading

## Next Steps

1. Run the updated feature engineering notebook
2. Train RL agents on real market data
3. Compare performance vs. synthetic data
4. Add more data sources (sentiment, news, etc.)

## Resources

- [OpenBB Documentation](https://docs.openbb.co/)
- [OpenBB GitHub](https://github.com/OpenBB-finance/OpenBBTerminal)
- [Data Providers List](https://docs.openbb.co/terminal/usage/data-sources)
