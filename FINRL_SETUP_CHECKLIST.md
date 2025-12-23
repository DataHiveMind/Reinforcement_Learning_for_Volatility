# FinRL Installation and Setup Checklist

## ✅ Step 1: Install Dependencies

Choose one of the following methods:

### Option A: Using Conda (Recommended)
```bash
# Update your existing conda environment
conda activate rl-volatility
conda env update -f environment.yml --prune

# Verify installation
python -c "import finrl; print(f'FinRL version: {finrl.__version__}')"
```

### Option B: Using pip
```bash
# Activate your virtual environment
# then install/update requirements
pip install -r requirements.txt

# Or install FinRL directly
pip install finrl

# Verify installation
python -c "import finrl; print(f'FinRL version: {finrl.__version__}')"
```

### Option C: Using setuptools
```bash
# Install in development mode with all dependencies
pip install -e '.[all]'

# Verify installation
python -c "import finrl; print(f'FinRL version: {finrl.__version__}')"
```

## ✅ Step 2: Verify Integration

Run this quick test:

```python
# test_finrl_integration.py
from src.models.finrl_agent import VolatilityFinRLAgent, FinRLEnsembleAgent

print("✅ FinRL integration successful!")
print("Available classes:")
print("  - VolatilityFinRLAgent")
print("  - FinRLEnsembleAgent")
```

```bash
python test_finrl_integration.py
```

## ✅ Step 3: Read Documentation

Priority order:
1. ✅ `FINRL_INTEGRATION_SUMMARY.md` (You are here!)
2. ✅ `docs/FINRL_QUICKSTART.md` (5-minute quick start)
3. ✅ `docs/FINRL_INTEGRATION.md` (Complete guide)

## ✅ Step 4: Run Tutorial Notebook

```bash
# Start Jupyter Lab
jupyter lab

# Then open:
# notebook/06_finrl_integration.ipynb
```

Work through the notebook cells to:
- Learn the API
- Train your first agent
- Build an ensemble
- Save and load models

## ✅ Step 5: Train Your First Model

### Quick Test (5 minutes)
```bash
python examples/train_with_finrl.py \
    --mode single \
    --model ppo \
    --timesteps 10000 \
    --save-path ./models/test_finrl
```

### Full Training (30-60 minutes)
```bash
python examples/train_with_finrl.py \
    --mode single \
    --model ppo \
    --timesteps 100000 \
    --env-config configs/env/volatility_env.yaml \
    --save-path ./models/finrl_ppo \
    --evaluate
```

### Ensemble Training (1-2 hours)
```bash
python examples/train_with_finrl.py \
    --mode ensemble \
    --timesteps 100000 \
    --env-config configs/env/volatility_env.yaml \
    --save-path ./models/finrl_ensemble \
    --evaluate
```

## ✅ Step 6: Monitor Training

### TensorBoard
```bash
# In a separate terminal
tensorboard --logdir ./tensorboard_logs/finrl --port 6006
```
Then open: http://localhost:6006

### MLflow (if using)
```bash
# In a separate terminal
mlflow ui --port 5000
```
Then open: http://localhost:5000

## ✅ Step 7: Compare Algorithms

Create a simple comparison script:

```python
# compare_algorithms.py
from src.models.finrl_agent import VolatilityFinRLAgent
from src.envs.volatility_env import VolatilityTradingEnv

# Prepare your data
env = VolatilityTradingEnv(...)

algorithms = ["ppo", "a2c", "sac"]
results = {}

for algo in algorithms:
    print(f"\nTraining {algo.upper()}...")
    agent = VolatilityFinRLAgent(env=env, model_name=algo)
    agent.train(total_timesteps=50000)

    # Evaluate
    mean_reward = evaluate(agent, env)
    results[algo] = mean_reward
    print(f"{algo.upper()} mean reward: {mean_reward:.2f}")

# Print comparison
print("\n" + "="*50)
print("Algorithm Comparison Results")
print("="*50)
for algo, reward in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{algo.upper():8s}: {reward:8.2f}")
```

## ✅ Step 8: Integrate with Your Workflow

### Update Your Training Pipeline
```python
# In your existing training scripts
from src.models.finrl_agent import VolatilityFinRLAgent

# Add FinRL as an option
if config['agent_type'] == 'finrl':
    agent = VolatilityFinRLAgent(
        env=env,
        model_name=config['finrl_model'],
        model_kwargs=config['finrl_kwargs']
    )
```

### Add to Evaluation Scripts
```python
# In your existing evaluation scripts
from src.models.finrl_agent import VolatilityFinRLAgent

# Load and evaluate FinRL models
agent = VolatilityFinRLAgent(env=env, model_name="ppo")
agent.load("./models/finrl_ppo_model")
performance = evaluate(agent, test_env)
```

## 📋 Optional Advanced Steps

### Hyperparameter Tuning with Optuna
```bash
# Create a tuning script (see docs/FINRL_INTEGRATION.md)
python scripts/tune_finrl_hyperparams.py --model ppo --trials 50
```

### Ensemble Model Evaluation
```python
from src.models.finrl_agent import FinRLEnsembleAgent

ensemble = FinRLEnsembleAgent(env=env)
ensemble.load_all("./models/finrl_ensemble")

# Test different voting methods
for method in ["averaging", "voting", "weighted"]:
    performance = evaluate_ensemble(ensemble, env, method=method)
    print(f"{method}: {performance}")
```

### Custom Callback Development
```python
from stable_baselines3.common.callbacks import BaseCallback

class VolatilityCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Custom logic for volatility trading
        return True

agent.train(total_timesteps=100000, callback=VolatilityCallback())
```

## 🐛 Troubleshooting

### Common Issues

**1. FinRL import errors**
```bash
# Make sure FinRL is installed
pip install finrl

# Check version
python -c "import finrl; print(finrl.__version__)"
```

**2. Gymnasium vs Gym compatibility**
```bash
# FinRL uses gymnasium (not gym)
pip install gymnasium>=0.28.0
pip uninstall gym  # if you have old gym installed
```

**3. CUDA/GPU issues**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False and you need GPU, reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**4. Memory issues during training**
```python
# Reduce batch size or buffer size
model_kwargs = {
    "batch_size": 32,  # instead of 256
    "buffer_size": 10000,  # instead of 100000
}
```

## 📊 Expected Results

After completing these steps, you should have:

✅ FinRL successfully installed and integrated
✅ Understanding of FinRL API through tutorial notebook
✅ At least one trained model (PPO, SAC, or A2C)
✅ Performance comparison between algorithms
✅ Saved models ready for evaluation
✅ TensorBoard logs for monitoring
✅ Knowledge of how to build ensembles

## 🎓 Next Learning Steps

1. **Deep Dive**: Read `docs/FINRL_INTEGRATION.md` for advanced features
2. **Optimization**: Try hyperparameter tuning with Optuna
3. **Production**: Deploy best model for live trading
4. **Research**: Compare FinRL with your custom implementations
5. **Contribution**: Extend the integration with new features

## 📞 Getting Help

- **Quick Questions**: See `docs/FINRL_QUICKSTART.md`
- **Integration Issues**: See `docs/FINRL_INTEGRATION.md`
- **FinRL Specific**: https://github.com/AI4Finance-Foundation/FinRL/issues
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/

## 🎉 You're Ready!

FinRL is now fully integrated. Start with the tutorial notebook and work your way through the examples. Good luck with your volatility trading research!

---

**Remember**:
- Start small (10K timesteps for testing)
- Monitor with TensorBoard
- Save checkpoints frequently
- Compare multiple algorithms
- Use ensembles for production
