"""
Configuration parsing and management utilities.

Supports:
- YAML configuration loading
- Config merging and inheritance
- Environment variable substitution
- Config validation
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf


class ConfigParser:
    """Parse and manage configuration files."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize ConfigParser.

        Args:
            config_path: Path to main config file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = None

        if self.config_path and self.config_path.exists():
            self.config = self.load(self.config_path)

    @staticmethod
    def load(config_path: Union[str, Path]) -> DictConfig:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file

        Returns:
            OmegaConf DictConfig object
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load YAML
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert to OmegaConf for advanced features
        config = OmegaConf.create(config_dict)

        # Resolve environment variables
        config = ConfigParser.resolve_env_vars(config)

        return config

    @staticmethod
    def resolve_env_vars(config: Union[Dict, DictConfig]) -> DictConfig:
        """
        Resolve environment variable references in config.

        Supports ${ENV_VAR} and ${ENV_VAR:default_value} syntax.

        Args:
            config: Configuration dict or DictConfig

        Returns:
            Config with resolved environment variables
        """
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        # Register custom resolver for environment variables
        OmegaConf.register_new_resolver(
            "env",
            lambda var, default=None: os.getenv(var, default),
            replace=True,
        )

        # Resolve all interpolations
        OmegaConf.resolve(config)

        return config

    @staticmethod
    def merge(*configs: Union[Dict, DictConfig, str, Path]) -> DictConfig:
        """
        Merge multiple configs with priority (later overrides earlier).

        Args:
            *configs: Config dicts, DictConfigs, or paths to config files

        Returns:
            Merged configuration
        """
        merged = OmegaConf.create({})

        for config in configs:
            if isinstance(config, (str, Path)):
                config = ConfigParser.load(config)
            elif isinstance(config, dict):
                config = OmegaConf.create(config)

            merged = OmegaConf.merge(merged, config)

        return merged

    @staticmethod
    def save(config: Union[Dict, DictConfig], output_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: Configuration to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict if OmegaConf
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config

        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def get_nested(config: Union[Dict, DictConfig], key: str, default: Any = None) -> Any:
        """
        Get nested config value using dot notation.

        Args:
            config: Configuration dict
            key: Dot-separated key (e.g., 'model.learning_rate')
            default: Default value if key not found

        Returns:
            Config value or default
        """
        if isinstance(config, DictConfig):
            return OmegaConf.select(config, key, default=default)

        keys = key.split(".")
        value = config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    @staticmethod
    def set_nested(config: Union[Dict, DictConfig], key: str, value: Any) -> None:
        """
        Set nested config value using dot notation.

        Args:
            config: Configuration dict
            key: Dot-separated key
            value: Value to set
        """
        if isinstance(config, DictConfig):
            OmegaConf.update(config, key, value)
            return

        keys = key.split(".")
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    @staticmethod
    def validate_required_keys(
        config: Union[Dict, DictConfig],
        required_keys: List[str],
    ) -> None:
        """
        Validate that required keys exist in config.

        Args:
            config: Configuration dict
            required_keys: List of required keys (dot notation supported)

        Raises:
            ValueError: If required key is missing
        """
        missing_keys = []

        for key in required_keys:
            value = ConfigParser.get_nested(config, key)
            if value is None:
                missing_keys.append(key)

        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

    def load_base_config(self, base_key: str = "base_config") -> DictConfig:
        """
        Load and merge with base config if specified.

        Args:
            base_key: Key in config that points to base config path

        Returns:
            Merged configuration
        """
        if not self.config:
            raise ValueError("No config loaded")

        base_path = ConfigParser.get_nested(self.config, base_key)

        if base_path:
            base_path = Path(base_path)
            if not base_path.is_absolute() and self.config_path:
                base_path = self.config_path.parent / base_path

            base_config = ConfigParser.load(base_path)
            self.config = ConfigParser.merge(base_config, self.config)

        return self.config

    @staticmethod
    def interpolate_strings(config: Union[Dict, DictConfig]) -> DictConfig:
        """
        Interpolate string references in config.

        Args:
            config: Configuration dict

        Returns:
            Config with interpolated strings
        """
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        OmegaConf.resolve(config)
        return config


def load_config(
    config_path: Union[str, Path],
    base_config: Optional[Union[str, Path]] = None,
) -> DictConfig:
    """
    Load configuration file with optional base config.

    Args:
        config_path: Path to main config file
        base_config: Path to base config to merge with

    Returns:
        Loaded and merged configuration
    """
    if base_config:
        return ConfigParser.merge(base_config, config_path)
    else:
        return ConfigParser.load(config_path)


def merge_configs(*config_paths: Union[str, Path, Dict, DictConfig]) -> DictConfig:
    """
    Merge multiple configuration files or dicts.

    Args:
        *config_paths: Paths to config files or config dicts

    Returns:
        Merged configuration
    """
    return ConfigParser.merge(*config_paths)


def load_experiment_config(
    experiment_dir: Union[str, Path],
    config_name: str = "config.yaml",
) -> DictConfig:
    """
    Load experiment configuration from experiment directory.

    Args:
        experiment_dir: Path to experiment directory
        config_name: Name of config file

    Returns:
        Experiment configuration
    """
    experiment_dir = Path(experiment_dir)
    config_path = experiment_dir / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    return ConfigParser.load(config_path)
