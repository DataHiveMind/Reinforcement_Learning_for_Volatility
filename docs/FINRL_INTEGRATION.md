# FinRL Integration Guide

## Overview

This project now includes integration with [FinRL](https://github.com/AI4Finance-Foundation/FinRL) (Financial Reinforcement Learning), a comprehensive deep reinforcement learning library specifically designed for quantitative finance and automated trading.

## What is FinRL?

FinRL provides:
- **Pre-built RL Agents**: PPO, A2C, SAC, DDPG, TD3 and more
- **Financial Data Processing**: Tools for market data preparation
- **Trading Environments**: Standard interfaces for trading simulations
- **Ensemble Methods**: Combine multiple agents for robust trading decisions
- **Integration with Stable-Baselines3**: Industry-standard RL implementations

## Installation

FinRL has been added to all dependency files. Install using one of these methods:

### Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate rl-volatility
```

### Using pip
```bash
pip install -r requirements.txt
```

### Using setuptools
```bash
pip install -e '.[all]'
```

## Project Structure

```
src/models/
├── finrl_agent.py          # FinRL wrapper classes
├── ppo_agent.py            # Original PPO implementation
├── ddpg_agent.py           # Original DDPG implementation
└── networks.py             # Neural network architectures

examples/
├── train_with_finrl.py     # Training script using FinRL
└── load_openbb_data.py     # Data loading example

configs/trainings/
├── finrl_config.yaml       # FinRL-specific configuration
├── ppo_mircostructure.yaml # Original PPO config
└── ddpg_microstructure.yaml # Original DDPG config

notebook/
└── 06_finrl_integration.ipynb  # Interactive FinRL tutorial
```

## Quick Start

### 1. Train a Single Agent

```python
from src.models.finrl_agent import VolatilityFinRLAgent
from src.envs.volatility_env import VolatilityTradingEnv

# Create environment
env = VolatilityTradingEnv(data=your_data)

# Initialize FinRL agent
agent = VolatilityFinRLAgent(
    env=env,
    model_name="ppo",  # or "sac", "a2c", "ddpg", "td3"
    model_kwargs={
        "learning_rate": 3e-4,
        "batch_size": 64,
    }
)

# Train
agent.train(total_timesteps=100000)

# Save model
agent.save("./models/finrl_ppo_model")
```

### 2. Train an Ensemble

```python
from src.models.finrl_agent import FinRLEnsembleAgent

# Create ensemble with multiple algorithms
ensemble = FinRLEnsembleAgent(
    env=env,
    model_names=["ppo", "a2c", "sac"]
)

# Train all agents
ensemble.train_all(total_timesteps=100000)

# Save ensemble
ensemble.save_all("./models/finrl_ensemble")
```

### 3. Use from Command Line

```bash
# Train single agent
python examples/train_with_finrl.py \
    --mode single \
    --model ppo \
    --timesteps 100000 \
    --env-config configs/env/volatility_env.yaml \
    --save-path ./models/finrl_ppo

# Train ensemble
python examples/train_with_finrl.py \
    --mode ensemble \
    --timesteps 100000 \
    --env-config configs/env/volatility_env.yaml \
    --save-path ./models/finrl_ensemble \
    --evaluate
```

## Available Algorithms

### On-Policy Algorithms
- **PPO (Proximal Policy Optimization)**: Stable and sample-efficient
- **A2C (Advantage Actor-Critic)**: Fast training with synchronous updates

### Off-Policy Algorithms
- **SAC (Soft Actor-Critic)**: Maximum entropy RL, good for continuous actions
- **DDPG (Deep Deterministic Policy Gradient)**: Classic continuous control
- **TD3 (Twin Delayed DDPG)**: Improved DDPG with reduced overestimation

## Configuration

### Using YAML Config

Edit `configs/trainings/finrl_config.yaml`:

```yaml
model:
  name: "ppo"
  policy: "MlpPolicy"

training:
  total_timesteps: 100000
  eval_freq: 10000

ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  gamma: 0.99
  ent_coef: 0.01
```

### Programmatic Configuration

```python
model_kwargs = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}

agent = VolatilityFinRLAgent(
    env=env,
    model_name="ppo",
    model_kwargs=model_kwargs
)
```

## Advanced Features

### 1. Custom Callbacks

```python
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/checkpoints/",
    name_prefix="finrl_model"
)

# Train with callbacks
agent.train(
    total_timesteps=100000,
    callback=[eval_callback, checkpoint_callback]
)
```

### 2. Hyperparameter Tuning with Optuna

```python
import optuna

def objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Create and train agent
    agent = VolatilityFinRLAgent(
        env=env,
        model_name="ppo",
        model_kwargs={"learning_rate": lr, "batch_size": batch_size}
    )
    agent.train(total_timesteps=50000)

    # Evaluate
    mean_reward = evaluate_agent(agent, env)
    return mean_reward

# Optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print("Best params:", study.best_params)
```

### 3. Ensemble Prediction Strategies

```python
# Different ensemble methods
ensemble = FinRLEnsembleAgent(env=env)

# Averaging (default for continuous actions)
action = ensemble.predict(obs, method="averaging")

# Voting (for discrete actions)
action = ensemble.predict(obs, method="voting")

# Weighted averaging (can be customized)
action = ensemble.predict(obs, method="weighted")
```

## Integration with Existing Code

FinRL complements your existing implementations:

### Use with Custom Environments

```python
from src.envs.volatility_env import VolatilityTradingEnv
from src.models.finrl_agent import VolatilityFinRLAgent

# Your existing environment works with FinRL
env = VolatilityTradingEnv(...)
agent = VolatilityFinRLAgent(env=env, model_name="sac")
```

### Compare with Existing Agents

```python
# Compare FinRL PPO with your custom PPO
from src.models.ppo_agent import PPOAgent  # Your implementation
from src.models.finrl_agent import VolatilityFinRLAgent

# Custom implementation
custom_ppo = PPOAgent(...)
custom_ppo.train(...)

# FinRL implementation
finrl_ppo = VolatilityFinRLAgent(env=env, model_name="ppo")
finrl_ppo.train(...)

# Evaluate both
custom_reward = evaluate(custom_ppo, env)
finrl_reward = evaluate(finrl_ppo, env)
```

## Monitoring and Logging

### TensorBoard Integration

FinRL automatically logs to TensorBoard:

```bash
# Start TensorBoard
tensorboard --logdir ./tensorboard_logs/finrl --port 6006
```

View training metrics at `http://localhost:6006`

### MLflow Integration

```python
import mlflow

with mlflow.start_run(run_name="finrl_ppo"):
    mlflow.log_params({
        "algorithm": "ppo",
        "learning_rate": 3e-4,
        "total_timesteps": 100000,
    })

    agent.train(total_timesteps=100000)
    mean_reward = evaluate_agent(agent, env)

    mlflow.log_metric("mean_reward", mean_reward)
    mlflow.log_artifact("./models/finrl_ppo_model.zip")
```

## Best Practices

### 1. Algorithm Selection

- **PPO**: Best default choice, stable and sample-efficient
- **SAC**: For continuous action spaces with exploration needs
- **A2C**: When you need fast training
- **DDPG/TD3**: For deterministic continuous control
- **Ensemble**: When you need robust performance

### 2. Hyperparameter Tuning

Start with these defaults and tune:
- Learning rate: 1e-5 to 1e-3
- Batch size: 64 to 256
- Gamma: 0.95 to 0.999
- Network architecture: [256, 256] or [512, 512]

### 3. Training Tips

- Use vectorized environments for faster training
- Monitor training with TensorBoard
- Save checkpoints regularly
- Use evaluation callbacks for model selection
- Start with shorter training runs for debugging

## Examples

### Interactive Notebook

Open `notebook/06_finrl_integration.ipynb` for an interactive tutorial covering:
- Data preparation
- Single agent training
- Ensemble methods
- Model saving/loading
- Hyperparameter optimization

### Command-Line Script

See `examples/train_with_finrl.py` for a complete training pipeline.

## Troubleshooting

### Common Issues

**Issue**: FinRL not found
```bash
# Solution: Install FinRL
pip install finrl
```

**Issue**: Incompatible gym/gymnasium versions
```bash
# Solution: Use gymnasium (not gym)
pip install gymnasium>=0.28.0
```

**Issue**: CUDA errors with PyTorch
```bash
# Solution: Check PyTorch CUDA compatibility
python -c "import torch; print(torch.cuda.is_available())"
```

### Getting Help

- FinRL Documentation: https://finrl.readthedocs.io/
- FinRL GitHub: https://github.com/AI4Finance-Foundation/FinRL
- Stable-Baselines3 Docs: https://stable-baselines3.readthedocs.io/

## Performance Comparison

Expected training performance (on synthetic data):

| Algorithm | Training Time | Sample Efficiency | Stability | Best For |
|-----------|---------------|-------------------|-----------|----------|
| PPO       | Medium        | High              | High      | General purpose |
| A2C       | Fast          | Medium            | Medium    | Quick prototyping |
| SAC       | Medium        | High              | High      | Continuous control |
| DDPG      | Fast          | Medium            | Low       | Deterministic tasks |
| TD3       | Medium        | High              | Medium    | Improved DDPG |
| Ensemble  | Slow          | Highest           | Highest   | Production |

## Next Steps

1. ✅ FinRL is now integrated into your project
2. 📊 Run the tutorial notebook: `06_finrl_integration.ipynb`
3. 🎯 Train your first agent: `python examples/train_with_finrl.py`
4. 🔬 Compare FinRL with your existing implementations
5. 🚀 Deploy the best-performing model

## Contributing

To extend FinRL integration:

1. Add new algorithms in `src/models/finrl_agent.py`
2. Create algorithm-specific configs in `configs/trainings/`
3. Add examples in `examples/`
4. Update this documentation

## License

FinRL is licensed under MIT License. See FinRL repository for details.
