"""
Model loading utilities for Bitcoin prediction system.

This module provides utilities to load trained models and handle model-related
operations across the prediction system.
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch

from .models import BitcoinPredictor
from core.setup_path import OUTPUT_DIR

warnings.filterwarnings('ignore')


class ModelLoader:
    """
    Utility class to load trained models from saved files.
    
    This class provides methods to load the latest trained model or a specific
    model file, handling all the necessary configuration and device setup.
    """

    @staticmethod
    def load_latest_model(base_dir: Optional[str] = None) -> Optional[Tuple]:
        """
        Load the latest trained model from the output directory.
        
        Args:
            base_dir: Base directory to search for models. If None, uses OUTPUT_DIR.
            
        Returns:
            Tuple of (model, model_data, config, device) or None if no models found.
        """
        base_path = Path(base_dir) if base_dir else OUTPUT_DIR

        if not base_path.exists():
            print(f"❌ Output directory not found: {base_path}")
            return None

        # Find latest model
        latest_model = None
        latest_time = datetime.min

        for model_file in base_path.rglob('final_model_*.pth'):
            try:
                stem_parts = model_file.stem.split('_')
                timestamp_str = f"{stem_parts[-2]}_{stem_parts[-1]}"
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')

                if timestamp > latest_time:
                    latest_time = timestamp
                    latest_model = model_file
            except Exception:
                continue

        if not latest_model:
            print("❌ No models found")
            return None

        return ModelLoader.load_model(latest_model)

    @staticmethod
    def load_model(model_path: Path) -> Tuple:
        """
        Load a specific model from file.
        
        Args:
            model_path: Path to the model file.
            
        Returns:
            Tuple of (model, model_data, config, device).
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model data
        model_data = torch.load(model_path, map_location=device,
                                weights_only=False)
        config = model_data['model_config']

        # Recreate model
        model = BitcoinPredictor(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            model_type=config['model_type']
        ).to(device)

        # Load weights
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()

        return model, model_data, config, device 