"""
Configuration settings for Bitcoin prediction system.

This module centralizes all configuration settings including:
- Environment variables
- Validation criteria
- Model hyperparameters
- Application constants
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment settings
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
IS_DEBUG = os.getenv('IS_DEBUG', 'true').lower() == 'true'

# Database settings
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/bitcoin_predictions')

@dataclass
class ValidationConfig:
    """Configuration for prediction validation criteria"""
    price_threshold_percent: float = 5.0  # Success if within 5% of actual
    direction_weight: float = 0.4  # 40% weight on direction accuracy
    price_weight: float = 0.6  # 60% weight on price accuracy
    min_confidence_threshold: float = 0.6  # Minimum confidence for success
    flat_threshold_percent: float = 1.0  # Consider flat if change < 1%

@dataclass
class ModelConfig:
    """Default model configuration parameters"""
    sequence_length: int = 60
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    model_type: str = 'lstm'
    
@dataclass
class DataConfig:
    """Data processing configuration"""
    default_features: list = None
    target_column: str = 'close'
    
    def __post_init__(self):
        if self.default_features is None:
            self.default_features = ['close', 'volume', 'volatility', 'sma_20', 'rsi']

# Application constants
PREDICTION_HORIZONS = ['1day', '3day', '7day']
MIN_PRICE = 10000  # Minimum realistic BTC price
MAX_PRICE = 200000  # Maximum realistic BTC price

def create_custom_config() -> ValidationConfig:
    """Create custom validation configuration with user input"""
    print("\n⚙️ VALIDATION CONFIGURATION")
    print("=" * 30)
    
    config = ValidationConfig()
    
    print(f"Current settings:")
    print(f"  Price threshold: {config.price_threshold_percent}%")
    print(f"  Direction weight: {config.direction_weight}")
    print(f"  Price weight: {config.price_weight}")
    print(f"  Min confidence: {config.min_confidence_threshold}")
    
    # Ask user if they want to customize
    customize = input("\nCustomize settings? (y/n): ").lower().strip()
    
    if customize == 'y':
        try:
            threshold = float(input(f"Price threshold % (current {config.price_threshold_percent}): ") or config.price_threshold_percent)
            config.price_threshold_percent = threshold
            
            dir_weight = float(input(f"Direction weight 0-1 (current {config.direction_weight}): ") or config.direction_weight)
            config.direction_weight = max(0, min(1, dir_weight))
            config.price_weight = 1 - config.direction_weight
            
            min_conf = float(input(f"Min confidence 0-1 (current {config.min_confidence_threshold}): ") or config.min_confidence_threshold)
            config.min_confidence_threshold = max(0, min(1, min_conf))
            
        except ValueError:
            print("Invalid input, using defaults")
    
    return config 