# RL Volatility - Conda Environment Setup

This project uses a conda environment for dependency management.

## Quick Start

### Create the environment

```bash
conda env create -f environment.yml
```

Or use the setup script:

```bash
./setup_env.sh
```

### Activate the environment

```bash
conda activate rl-volatility
```

### Register with Jupyter

```bash
python -m ipykernel install --user --name=rl-volatility --display-name="Python (RL-Volatility)"
```

## VS Code Integration

The workspace is configured to automatically use the `rl-volatility` conda environment for:

- Python files
- Jupyter notebooks
- Terminal sessions

After creating the environment, restart VS Code or reload the window (Cmd/Ctrl + Shift + P → "Developer: Reload Window").

## Environment Management

### Update environment after changes to environment.yml

```bash
conda env update -f environment.yml --prune
```

### Export current environmen

```bash
conda env export > environment.yml
```

### Remove environment

```bash
conda env remove -n rl-volatility
```

### List installed packages

```bash
conda list
```

## Jupyter Kernel

After activating the environment and installing ipykernel, you can select the kernel in VS Code:

1. Open a `.ipynb` file
2. Click "Select Kernel" in the top right
3. Choose "Python (RL-Volatility)"

The kernel will now be available for all notebooks in this workspace.
