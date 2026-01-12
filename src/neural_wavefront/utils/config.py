"""Configuration utilities for loading and validating config files."""

from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration class for loading and accessing YAML config files."""

    def __init__(self, config_path: str | Path):
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, "r") as f:
            self._config: Dict[str, Any] = yaml.safe_load(f)

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value by key."""
        self._config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self._config.get(key, default)

    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._config

    def save(self, save_path: str | Path) -> None:
        """
        Save current configuration to a new file.

        Args:
            save_path: Path where to save the configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)


def load_config(config_path: str | Path = "configs/config.yaml") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Config object with loaded parameters
    """
    return Config(config_path)
