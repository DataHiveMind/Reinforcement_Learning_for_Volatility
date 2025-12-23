# 📘 Reinforcement Learning for Volatility Alpha Capture

RL‑Driven Volatility Forecasting and Allocation Using Market Microstructure & Unstructured Data

> **🆕 Now with FinRL Integration!** - Advanced financial RL library with pre-built agents, ensemble methods, and optimization tools. See [FINRL_INTEGRATION.md](docs/FINRL_INTEGRATION.md) for details.
>
> **📊 Data Source**: Uses Yahoo Finance (yfinance) for reliable, free equities data.

## 📌 Overview

This project develops a reinforcement learning framework for capturing short‑horizon volatility alpha using:

Market microstructure features (order‑book imbalance, trade flow toxicity, queue dynamics)

Options‑implied metrics (IV skew, term‑structure curvature, vol‑of‑vol)

Unstructured data signals (news sentiment, embeddings, macro‑uncertainty topics)

The goal is to build an RL agent that allocates capital across volatility strategies (skew, convexity, dispersion, vol‑carry) and predicts volatility breakouts before they occur.

This project is designed to mirror the research workflows used at top quant funds and PhD‑level ML labs.

## 🚀 Quick Start

### 1. Create Conda Environment

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate rl-volatility

# Register Jupyter kernel
python -m ipykernel install --user --name=rl-volatility --display-name="Python (RL-Volatility)"
```

Or use the setup script:

```bash
./setup_env.sh
```

### 2. Install Project

```bash
# Install in development mode
pip install -e ".[all]"
```

### 3. VS Code Setup

The workspace is pre-configured to use the `rl-volatility` conda environment. After creating the environment:

- Reload VS Code window (Cmd/Ctrl + Shift + P → "Developer: Reload Window")
- Jupyter notebooks will automatically use the correct kernel

See [CONDA_SETUP.md](CONDA_SETUP.md) for detailed instructions.

## 🎯 Research Objectives

Build a custom RL environment that simulates volatility dynamics using microstructure + options data

Engineer high‑frequency alpha signals from order‑book and trade‑flow data

Integrate unstructured text features from news and macro transcripts

Train PPO/DDPG agents to learn volatility‑aware allocation policies

Evaluate performance using:

Sharpe ratio

Volatility prediction accuracy

Breakout detection recall

Regime‑dependent performance

Produce a reproducible, modular research pipeline suitable for real‑world quant research

## 📂 Project Structure

rl-volatility-alpha/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── training/
│   ├── data/
│   └── env/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_feature_engineering.ipynb
│   ├── 02_volatility_labels.ipynb
│   ├── 03_microstructure_signals.ipynb
│   ├── 04_env_sanity_checks.ipynb
│   └── 05_model_experiments.ipynb
│
├── src/
│   ├── data/
│   ├── envs/
│   ├── models/
│   ├── signals/
│   ├── training/
│   └── utils/
│
├── experiments/
│   ├── ppo_microstructure/
│   └── ddpg_microstructure/
│
└── tests/

## 🧠 Key Components

1. Market Microstructure Features
    - Extracted from order‑book & trade‑flow data:

    - Order‑book imbalance

    - Queue position & depth

    - Trade flow toxicity (ELO/VPIN‑style)

    - Short‑horizon realized volatility

    - Spread dynamics & liquidity shocks

2. Options‑Implied Volatility Features
    - IV skew & term structure

    - Vol‑of‑vol

    - Smirk curvature

    - Realized vs implied spreads

3. Unstructured Data Signals
    - Using NLP + transformers:

    - News sentiment

    - Macro‑uncertainty embeddings

    - Topic‑model‑based volatility drivers

    - FOMC / earnings call latent factors

4. Reinforcement Learning Environment
    - Custom OpenAI‑style environment:

    - State = microstructure + options + NLP features

    - Actions = volatility strategy allocation

    - Reward = volatility‑adjusted PnL, convexity capture, breakout detection

5. RL Agents
    - Implemented agents:

    - **PPO** (stable, robust for noisy signals)

    - **DDPG** (continuous action space for allocation weights)

    - **FinRL Integration** 🆕 (PPO, A2C, SAC, TD3, DDPG + ensembles)

    - Optional: SAC, TD3

## 📈 Evaluation Metrics

1. Sharpe ratio

2. Volatility prediction accuracy

3. Breakout detection recall/precision

4. Regime‑dependent performance

5. Turnover & transaction cost impact

6. Signal decay & horizon analysis

## 🧪 Experiment Tracking

All experiments are logged under:
/experiments/
    /ppo_microstructure/
    /ddpg_microstructure/

Each experiment contains:

logs/
checkpoints/
metrics.json

## 🚀 Getting Started

Summary

## 📜 Research Motivation

Volatility is driven by:

microstructure imbalances

options‑implied expectations

macro‑uncertainty shocks

sentiment‑driven flow

Traditional models (GARCH, HAR, linear factor models) struggle with:

nonlinear interactions

regime shifts

high‑frequency microstructure noise

unstructured data

Reinforcement learning provides a framework for:

dynamic allocation

nonlinear state representations

adaptive policy learning

regime‑aware decision making

This project explores whether RL can capture volatility alpha more effectively than traditional models.
