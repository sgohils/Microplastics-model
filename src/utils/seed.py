"""Utility functions for seeding random generators for reproducibility.

This module provides functions to ensure reproducible results across
different random number generators used in the project.
"""
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all frameworks.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(preferred: str = "auto"):
    """Get the appropriate compute device.
    
    Args:
        preferred: 'auto', 'cpu', or 'cuda'
    
    Returns:
        torch.device object
    """
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif preferred == "cpu":
        return torch.device("cpu")
    elif preferred == "cuda":
        return torch.device("cuda")
    else:
        return torch.device(preferred)
