"""
Settings module for Bitcoin prediction system.

This module centralizes all configuration settings and provides
easy imports for other modules.
"""

from .config import (
    ENVIRONMENT,
    IS_DEBUG,
    DATABASE_URL,
    ValidationConfig,
    ModelConfig,
    DataConfig,
    PREDICTION_HORIZONS,
    MIN_PRICE,
    MAX_PRICE,
    create_custom_config
)

__all__ = [
    'ENVIRONMENT',
    'IS_DEBUG', 
    'DATABASE_URL',
    'ValidationConfig',
    'ModelConfig',
    'DataConfig',
    'PREDICTION_HORIZONS',
    'MIN_PRICE',
    'MAX_PRICE',
    'create_custom_config'
]
