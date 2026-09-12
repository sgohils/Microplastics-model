"""Configuration loading utilities."""
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Dictionary containing configuration parameters
    """
    config_path = Path(config_path)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_data_path(config: Dict[str, Any], *subpaths: str) -> Path:
    """Get a path relative to the project data directory.
    
    Args:
        config: Configuration dictionary
        subpaths: Subdirectory paths
    
    Returns:
        Full path object
    """
    base = Path(config['data']['raw_dir'])
    for subpath in subpaths:
        base = base / subpath
    return base


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
