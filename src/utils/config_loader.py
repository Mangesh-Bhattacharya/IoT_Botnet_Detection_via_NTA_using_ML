"""
Configuration Loader
Utilities for loading and managing project configuration
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the IoT Botnet Detection project"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration YAML file
        """
        if config_path is None:
            # Default to config/config.yaml
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation)
        
        Args:
            key: Configuration key (e.g., 'data.test_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)
    
    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"


def create_directories(config: Config):
    """
    Create necessary project directories if they don't exist
    
    Args:
        config: Configuration object
    """
    directories = [
        config.get('data.raw_data_path'),
        config.get('data.processed_data_path'),
        config.get('paths.models_dir'),
        config.get('paths.results_dir'),
        config.get('paths.figures_dir'),
        config.get('paths.reports_dir'),
        config.get('paths.logs_dir'),
    ]
    
    for directory in directories:
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✓ Created/verified directory: {directory}")


if __name__ == "__main__":
    # Test configuration loading
    config = Config()
    print(config.get('data.test_size'))
    print(config.get('models.random_forest.n_estimators'))
    create_directories(config)
