# FinRL Quick Reference

## Installation
```bash
# Already added to dependencies - just reinstall
conda env update -f environment.yml
# or
pip install -r requirements.txt
```

## 1-Minute Quick Start

### Train PPO Agent
```python
from src.models.finrl_agent import VolatilityFinRLAgent
from src.envs.volatility_env import VolatilityTradingEnv

env = VolatilityTradingEnv(data=your_data)
agent = VolatilityFinRLAgent(env=env, model_name="ppo")
agent.train(total_timesteps=100000)
agent.save("./models/my_ppo_model")
```

### Train Ensemble
```python
from src.models.finrl_agent import FinRLEnsembleAgent

ensemble = FinRLEnsembleAgent(env=env, model_names=["ppo", "a2c", "sac"])
ensemble.train_all(total_timesteps=100000)
ensemble.save_all("./models/my_ensemble")
```

### Load and Predict
```python
agent = VolatilityFinRLAgent(env=env, model_name="ppo")
agent.load("./models/my_ppo_model")
action, _ = agent.predict(observation)
```

## Command Line

```bash
# Single agent
python examples/train_with_finrl.py --mode single --model ppo --timesteps 100000

# Ensemble
python examples/train_with_finrl.py --mode ensemble --timesteps 100000 --evaluate
```

## Available Algorithms

| Algorithm | Type | Best For | Speed |
|-----------|------|----------|-------|
| PPO | On-policy | General purpose, stable | ⚡⚡ |
| A2C | On-policy | Fast prototyping | ⚡⚡⚡ |
| SAC | Off-policy | Continuous control, exploration | ⚡⚡ |
| DDPG | Off-policy | Deterministic continuous | ⚡⚡⚡ |
| TD3 | Off-policy | Improved DDPG | ⚡⚡ |

## Key Files

- `src/models/finrl_agent.py` - FinRL wrapper classes
- `examples/train_with_finrl.py` - Training script
- `configs/trainings/finrl_config.yaml` - Configuration
- `notebook/06_finrl_integration.ipynb` - Interactive tutorial
- `docs/FINRL_INTEGRATION.md` - Full documentation

## Hyperparameters Cheat Sheet

### PPO
```python
model_kwargs = {
    "learning_rate": 3e-4,      # [1e-5, 1e-3]
    "n_steps": 2048,            # [512, 2048, 4096]
    "batch_size": 64,           # [32, 64, 128, 256]
    "gamma": 0.99,              # [0.95, 0.999]
    "ent_coef": 0.01,          # [0.0, 0.1]
}
```

### SAC
```python
model_kwargs = {
    "learning_rate": 3e-4,      # [1e-5, 1e-3]
    "buffer_size": 100000,      # [10000, 1000000]
    "batch_size": 256,          # [128, 256, 512]
    "gamma": 0.99,              # [0.95, 0.999]
    "tau": 0.005,               # [0.001, 0.02]
}
```

## Monitoring

```bash
# TensorBoard
tensorboard --logdir ./tensorboard_logs/finrl --port 6006

# MLflow
mlflow ui --port 5000
```

## Common Tasks

### Compare Multiple Algorithms
```python
algorithms = ["ppo", "a2c", "sac"]
results = {}

for algo in algorithms:
    agent = VolatilityFinRLAgent(env=env, model_name=algo)
    agent.train(total_timesteps=50000)
    results[algo] = evaluate(agent, env)
```

### Hyperparameter Tuning
```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    agent = VolatilityFinRLAgent(
        env=env,
        model_name="ppo",
        model_kwargs={"learning_rate": lr, "batch_size": batch_size}
    )
    agent.train(total_timesteps=50000)
    return evaluate(agent, env)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
```

### Custom Training Callback
```python
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best",
    log_path="./logs/",
    eval_freq=10000
)

agent.train(total_timesteps=100000, callback=eval_callback)
```

## Troubleshooting

**Q: Import error for finrl**
```bash
pip install finrl
```

**Q: Gym vs Gymnasium**
```bash
# Use gymnasium (installed by default)
pip install gymnasium>=0.28.0
```

**Q: CUDA not available**
```python
import torch
print(torch.cuda.is_available())
# If False, install CUDA-enabled PyTorch
```

## Next Steps

1. ✅ Read [FINRL_INTEGRATION.md](FINRL_INTEGRATION.md)
2. 📓 Run [06_finrl_integration.ipynb](../notebook/06_finrl_integration.ipynb)
3. 🎯 Train your first model
4. 📊 Compare with existing implementations
5. 🚀 Deploy to production

## Resources

- FinRL Docs: https://finrl.readthedocs.io/
- FinRL GitHub: https://github.com/AI4Finance-Foundation/FinRL
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- Our Docs: [FINRL_INTEGRATION.md](FINRL_INTEGRATION.md)
