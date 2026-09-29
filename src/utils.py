"""
utils.py - General utility functions for reproducibility and logging.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducible execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def format_header(title: str, width: int = 70) -> str:
    """Returns a formatted banner header string."""
    line = "=" * width
    centered = title.center(width)
    return f"{line}\n{centered}\n{line}"
