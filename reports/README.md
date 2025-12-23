# Reports Directory

This directory contains all generated reports, visualizations, and analysis outputs.

## Directory Structure

```
reports/
├── figures/               # Plots and visualizations
│   ├── technical_analysis/
│   ├── feature_importance/
│   └── performance_charts/
├── data_quality/          # Data validation reports
│   ├── missing_values/
│   └── correlation_matrices/
├── backtest_results/      # Backtesting performance reports
│   ├── equity_curves/
│   └── trade_logs/
├── model_evaluations/     # Model performance metrics
│   ├── training_curves/
│   └── evaluation_metrics/
└── README.md              # This file
```

## Usage

All notebooks and scripts should save their outputs to the appropriate subdirectory:

- **figures/**: Technical charts, price plots, indicator visualizations
- **data_quality/**: Data validation reports, statistics summaries
- **backtest_results/**: Backtesting performance, P&L curves, drawdown charts
- **model_evaluations/**: Training metrics, evaluation results, model comparisons

## File Naming Convention

Use the following naming pattern for consistency:
```
{date}_{ticker}_{analysis_type}_{description}.{ext}
```

Examples:
- `20241223_SPY_technical_analysis_dashboard.html`
- `20241223_AAPL_feature_correlation_heatmap.png`
- `20241223_portfolio_backtest_equity_curve.png`
- `20241223_ppo_model_training_metrics.csv`

## Cleanup

Old reports can be archived or deleted manually. Consider keeping:
- Latest version of each report type
- Key milestone reports for model comparison
- Final backtest results for documentation
