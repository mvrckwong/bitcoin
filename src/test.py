"""
test.py - Bitcoin Model Testing & Validation Script

This standalone script tests and validates trained Bitcoin prediction models.
It automatically finds the latest model, loads it, and provides comprehensive evaluation.

Usage: python test.py

Note: Handles PyTorch 2.6+ compatibility with weights_only=False for sklearn objects.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from tqdm import tqdm
from models import BitcoinPredictor, BitcoinDataset, calculate_rsi

# Set style for better plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

@dataclass
class ModelInfo:
    """Information about a saved model"""
    path: Path
    name: str
    timestamp: datetime
    model_type: str
    features: List[str]
    best_val_loss: float
    config: Dict

class BitcoinPredictor(nn.Module):
    """
    Recreated model architecture for loading saved models
    """
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2, 
                 model_type: str = 'lstm'):
        super(BitcoinPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type.lower()
        
        # Choose architecture
        if self.model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.model_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError("model_type must be 'lstm' or 'gru'")
        
        # Attention mechanism (optional enhancement)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # RNN forward pass
        rnn_out, _ = self.rnn(x)
        
        # Use the last output
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Only squeeze the last dimension (feature dimension), keep batch dimension
        return out.squeeze(-1)

class BitcoinDataset(Dataset):
    """
    Recreated dataset class for testing
    """
    def __init__(self, data: pd.DataFrame, sequence_length: int = 60, 
                 target_col: str = 'close', feature_cols: Optional[List[str]] = None,
                 scalers: Optional[Dict] = None):
        self.sequence_length = sequence_length
        self.target_col = target_col
        
        # If no feature columns specified, use only the target column
        if feature_cols is None:
            feature_cols = [target_col]
        
        self.feature_cols = feature_cols
        self.data = data[feature_cols + [target_col] if target_col not in feature_cols else feature_cols].copy()
        
        # Use provided scalers or create new ones
        if scalers:
            self.scalers = scalers
            self.scaled_data = self.data.copy()
            for col in self.data.columns:
                if col in self.scalers:
                    self.scaled_data[col] = self.scalers[col].transform(self.data[[col]])
        else:
            # Create new scalers (for new test data)
            self.scalers = {}
            self.scaled_data = self.data.copy()
            for col in self.data.columns:
                scaler = MinMaxScaler()
                self.scaled_data[col] = scaler.fit_transform(self.data[[col]])
                self.scalers[col] = scaler
        
        # Create sequences
        self.sequences, self.targets = self._create_sequences()
    
    def _create_sequences(self):
        sequences = []
        targets = []
        
        print("🔄 Creating test sequences...")
        data_range = range(len(self.scaled_data) - self.sequence_length)
        
        for i in tqdm(data_range, desc="Processing sequences", unit="seq"):
            # Features for sequence
            seq_features = self.scaled_data[self.feature_cols].iloc[i:i+self.sequence_length].values
            sequences.append(seq_features)
            
            # Target (next price)
            target = self.scaled_data[self.target_col].iloc[i+self.sequence_length]
            targets.append(target)
        
        print("📦 Converting to tensors...")
        # Convert to numpy arrays first, then to tensors (more efficient)
        sequences_array = np.array(sequences, dtype=np.float32)
        targets_array = np.array(targets, dtype=np.float32)
        
        return torch.FloatTensor(sequences_array), torch.FloatTensor(targets_array)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
    
    def inverse_transform_target(self, scaled_target):
        """Convert scaled target back to original price"""
        return self.scalers[self.target_col].inverse_transform([[scaled_target]])[0][0]

class ModelFinder:
    """Utility class to find and manage saved models"""
    
    def __init__(self, base_output_dir: Union[str, Path] = '.outputs'):
        self.base_output_dir = Path(base_output_dir)
    
    def find_all_models(self) -> List[ModelInfo]:
        """Find all saved models in output directory"""
        models = []
        
        if not self.base_output_dir.exists():
            print(f"❌ Output directory not found: {self.base_output_dir}")
            return models
        
        # Search all subdirectories for final models
        for model_file in self.base_output_dir.rglob('*.pth'):
            if model_file.name.startswith('final_model_'):
                try:
                    model_info = self._extract_model_info(model_file)
                    models.append(model_info)
                except Exception as e:
                    print(f"⚠️ Could not load model info from {model_file}")
                    print(f"   Error: {str(e)[:100]}...")
                    continue
        
        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x.timestamp, reverse=True)
        return models
    
    def find_latest_model(self, model_type: Optional[str] = None) -> Optional[ModelInfo]:
        """Find the latest model, optionally filtered by type"""
        models = self.find_all_models()
        
        if model_type:
            models = [m for m in models if model_type.lower() in m.path.parent.name.lower()]
        
        return models[0] if models else None
    
    def find_best_model(self) -> Optional[ModelInfo]:
        """Find the best model based on validation loss"""
        models = self.find_all_models()
        
        if not models:
            return None
        
        # Sort by validation loss (lowest first)
        best_model = min(models, key=lambda x: x.best_val_loss)
        return best_model
    
    def _extract_model_info(self, model_path: Path) -> ModelInfo:
        """Extract information from a model file"""
        # Load model data with weights_only=False to handle sklearn objects
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        config = model_data['model_config']
        metrics = model_data['training_metrics']
        
        # Extract timestamp from filename
        timestamp_str = model_path.stem.split('_')[-2] + '_' + model_path.stem.split('_')[-1]
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        
        return ModelInfo(
            path=model_path,
            name=model_path.name,
            timestamp=timestamp,
            model_type=config.get('model_type', 'unknown'),
            features=config.get('feature_cols', []),
            best_val_loss=metrics.get('best_val_loss', float('inf')),
            config=config
        )
    
    def list_models(self) -> None:
        """Print a formatted list of all available models"""
        models = self.find_all_models()
        
        if not models:
            print("❌ No models found in output directory")
            print("💡 Make sure you have trained models first!")
            return
        
        print("📋 AVAILABLE MODELS")
        print("=" * 80)
        print(f"{'#':<3} {'Name':<25} {'Type':<8} {'Features':<6} {'Val Loss':<10} {'Date':<16}")
        print("-" * 80)
        
        for i, model in enumerate(models, 1):
            features_count = len(model.features)
            date_str = model.timestamp.strftime('%Y-%m-%d %H:%M')
            
            print(f"{i:<3} {model.name[:24]:<25} {model.model_type:<8} "
                  f"{features_count:<6} {model.best_val_loss:<10.6f} {date_str:<16}")

class ModelTester:
    """Class for testing and validating trained models"""
    
    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info
        self.model = None
        self.model_data = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the model from file"""
        print(f"🔄 Loading model: {self.model_info.name}")
        
        # Load model data
        self.model_data = torch.load(self.model_info.path, map_location=self.device, weights_only=False)
        config = self.model_data['model_config']
        
        # Recreate model
        self.model = BitcoinPredictor(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            model_type=config['model_type']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(self.model_data['model_state_dict'])
        self.model.eval()
        
        print("✅ Model loaded successfully")
        print(f"   📊 Features: {config['feature_cols']}")
        print(f"   🎯 Best validation loss: {self.model_data['training_metrics']['best_val_loss']:.6f}")
    
    def test_on_data(self, data: pd.DataFrame, target_col: str = 'close') -> Dict:
        """Test the model on provided data"""
        print("\n🧪 Testing model on provided data...")
        
        # Create dataset using the same configuration as training
        config = self.model_info.config
        
        # Use training scalers for consistency
        training_scalers = self.model_data.get('dataset_scalers', {})
        
        dataset = BitcoinDataset(
            data=data,
            sequence_length=config['sequence_length'],
            target_col=target_col,
            feature_cols=config['feature_cols'],
            scalers=training_scalers if training_scalers else None
        )
        
        # Make predictions
        predictions = []
        actuals = []
        
        self.model.eval()
        print("🔮 Making predictions...")
        
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc="Evaluating", unit="sample"):
                sequence, target = dataset[i]
                sequence = sequence.unsqueeze(0).to(self.device)  # Add batch dimension
                
                pred = self.model(sequence).cpu().numpy()
                if pred.ndim == 0:  # Scalar
                    pred = pred.item()
                else:
                    pred = pred[0]
                
                predictions.append(dataset.inverse_transform_target(pred))
                actuals.append(dataset.inverse_transform_target(target.numpy().item()))
        
        # Calculate metrics
        metrics = self._calculate_metrics(actuals, predictions)
        
        print("\n📊 Test Results:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'actuals': actuals,
            'dataset': dataset
        }
    
    def _calculate_metrics(self, actuals: List[float], predictions: List[float]) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        actuals = np.array(actuals)
        predictions = np.array(predictions)
        
        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        r2 = r2_score(actuals, predictions)
        
        # Directional accuracy (predict direction correctly)
        if len(actuals) > 1:
            actual_directions = np.diff(actuals) > 0
            pred_directions = np.diff(predictions) > 0
            directional_accuracy = np.mean(actual_directions == pred_directions) * 100
        else:
            directional_accuracy = 0.0
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R²': r2,
            'Directional_Accuracy_%': directional_accuracy
        }
    
    def visualize_predictions(self, test_results: Dict, save_plot: bool = True, 
                            show_plot: bool = True, max_points: int = 200):
        """Create visualizations of model predictions"""
        actuals = test_results['actuals']
        predictions = test_results['predictions']
        
        # Limit points for better visualization
        if len(actuals) > max_points:
            step = len(actuals) // max_points
            actuals = actuals[::step]
            predictions = predictions[::step]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Predictions Analysis: {self.model_info.name}', fontsize=16)
        
        # 1. Time series comparison
        ax1 = axes[0, 0]
        x = range(len(actuals))
        ax1.plot(x, actuals, label='Actual', color='blue', alpha=0.7, linewidth=2)
        ax1.plot(x, predictions, label='Predicted', color='red', alpha=0.7, linewidth=2)
        ax1.set_title('Actual vs Predicted Prices')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Scatter plot
        ax2 = axes[0, 1]
        ax2.scatter(actuals, predictions, alpha=0.6, color='green', s=30)
        
        # Perfect prediction line
        min_val = min(min(actuals), min(predictions))
        max_val = max(max(actuals), max(predictions))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
        
        ax2.set_title('Prediction Scatter Plot')
        ax2.set_xlabel('Actual Price ($)')
        ax2.set_ylabel('Predicted Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals
        ax3 = axes[1, 0]
        residuals = np.array(predictions) - np.array(actuals)
        ax3.scatter(actuals, residuals, alpha=0.6, color='orange', s=30)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax3.set_title('Residuals Plot')
        ax3.set_xlabel('Actual Price ($)')
        ax3.set_ylabel('Residuals (Predicted - Actual)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Metrics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        metrics = test_results['metrics']
        
        # Create metrics text with better formatting
        metrics_text = []
        for k, v in metrics.items():
            if k == 'MAPE' or 'Accuracy' in k:
                metrics_text.append(f'{k}: {v:.2f}%')
            elif k == 'R²':
                metrics_text.append(f'{k}: {v:.4f}')
            else:
                metrics_text.append(f'{k}: ${v:,.2f}')
        
        ax4.text(0.1, 0.9, 'Performance Metrics:', transform=ax4.transAxes, 
                fontsize=14, fontweight='bold')
        ax4.text(0.1, 0.7, '\n'.join(metrics_text), transform=ax4.transAxes, 
                fontsize=12, fontfamily='monospace')
        
        # Model info
        model_info_text = (
            f"Model: {self.model_info.model_type.upper()}\n"
            f"Features: {len(self.model_info.features)}\n"
            f"Sequence Length: {self.model_info.config['sequence_length']}\n"
            f"Hidden Size: {self.model_info.config['hidden_size']}\n"
            f"Layers: {self.model_info.config['num_layers']}"
        )
        ax4.text(0.1, 0.3, model_info_text, transform=ax4.transAxes, 
                fontsize=10, fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.model_info.path.parent / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved: {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def generate_report(self, test_results: Dict, save_report: bool = True, include_emojis: bool = True) -> str:
        """Generate a comprehensive test report"""
        report_lines = []
        
        # Choose symbols based on emoji preference
        if include_emojis:
            symbols = {
                'title': "🔍", 'info': "📋", 'results': "📊", 
                'analysis': "🎯", 'recommendations': "💡"
            }
        else:
            symbols = {
                'title': "[ANALYSIS]", 'info': "[INFO]", 'results': "[RESULTS]", 
                'analysis': "[ANALYSIS]", 'recommendations': "[RECOMMENDATIONS]"
            }
        
        report_lines.append("=" * 80)
        report_lines.append(f"{symbols['title']} BITCOIN MODEL VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Model information
        report_lines.append(f"{symbols['info']} MODEL INFORMATION")
        report_lines.append("-" * 40)
        report_lines.append(f"Model Name: {self.model_info.name}")
        report_lines.append(f"Model Type: {self.model_info.model_type.upper()}")
        report_lines.append(f"Training Date: {self.model_info.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Features Used: {', '.join(self.model_info.features)}")
        report_lines.append(f"Feature Count: {len(self.model_info.features)}")
        report_lines.append(f"Sequence Length: {self.model_info.config['sequence_length']}")
        report_lines.append(f"Hidden Size: {self.model_info.config['hidden_size']}")
        report_lines.append(f"Number of Layers: {self.model_info.config['num_layers']}")
        report_lines.append(f"Training Validation Loss: {self.model_info.best_val_loss:.6f}")
        report_lines.append("")
        
        # Test results
        report_lines.append(f"{symbols['results']} TEST RESULTS")
        report_lines.append("-" * 40)
        metrics = test_results['metrics']
        
        for metric, value in metrics.items():
            if metric == 'MAPE' or 'Accuracy' in metric:
                report_lines.append(f"{metric:.<30} {value:.2f}%")
            elif metric == 'R²':
                report_lines.append(f"{metric:.<30} {value:.6f}")
            else:
                report_lines.append(f"{metric:.<30} ${value:,.2f}")
        
        report_lines.append("")
        
        # Performance analysis
        report_lines.append(f"{symbols['analysis']} PERFORMANCE ANALYSIS")
        report_lines.append("-" * 40)
        
        mae = metrics['MAE']
        mape = metrics['MAPE']
        r2 = metrics['R²']
        dir_acc = metrics['Directional_Accuracy_%']
        
        # Performance ratings
        if mape < 1:
            mape_rating = "Excellent"
        elif mape < 3:
            mape_rating = "Very Good"
        elif mape < 5:
            mape_rating = "Good"
        elif mape < 10:
            mape_rating = "Fair"
        else:
            mape_rating = "Poor"
        
        if r2 > 0.9:
            r2_rating = "Excellent"
        elif r2 > 0.8:
            r2_rating = "Very Good"
        elif r2 > 0.7:
            r2_rating = "Good"
        elif r2 > 0.5:
            r2_rating = "Fair"
        else:
            r2_rating = "Poor"
        
        if dir_acc > 70:
            dir_rating = "Excellent"
        elif dir_acc > 60:
            dir_rating = "Good"
        elif dir_acc > 55:
            dir_rating = "Fair"
        else:
            dir_rating = "Poor"
        
        report_lines.append(f"MAPE Performance: {mape_rating} ({mape:.2f}%)")
        report_lines.append(f"R² Performance: {r2_rating} ({r2:.4f})")
        report_lines.append(f"Directional Accuracy: {dir_rating} ({dir_acc:.1f}%)")
        report_lines.append("")
        
        # Recommendations
        report_lines.append(f"{symbols['recommendations']} RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        recommendations = []
        if mape > 5:
            recommendations.append("• Consider adding more features or increasing model complexity")
        if r2 < 0.7:
            recommendations.append("• Model may be underfitting - try larger network or more training")
        if dir_acc < 60:
            recommendations.append("• Poor directional accuracy - review feature engineering")
        if mae > 2000:  # High dollar error
            recommendations.append("• High absolute error - consider price normalization or different loss function")
        
        if mape < 3 and r2 > 0.8 and dir_acc > 65:
            recommendations.append("• Model performance is satisfactory for production use")
            recommendations.append("• Consider testing on longer time horizons")
            recommendations.append("• Monitor performance on real market data")
        
        if not recommendations:
            recommendations.append("• Model shows reasonable performance")
            recommendations.append("• Continue monitoring and validation on new data")
        
        for rec in recommendations:
            report_lines.append(rec)
        
        report_lines.append("")
        report_lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        
        report_text = '\n'.join(report_lines)
        
        if save_report:
            report_path = self.model_info.path.parent / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                print(f"📄 Report saved: {report_path}")
            except UnicodeEncodeError:
                # Fallback: save without emojis
                print("⚠️ Unicode error detected, saving report without emojis...")
                report_no_emoji = self.generate_report(test_results, save_report=False, include_emojis=False)
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_no_emoji)
                print(f"📄 Report saved (no emojis): {report_path}")
        
        return report_text

def load_trained_model(model_path: str, device: Optional[torch.device] = None) -> Tuple[BitcoinPredictor, Dict]:
    """
    Load a trained model from file
    
    Args:
        model_path: Path to the saved model
        device: Device to load model on
        
    Returns:
        Tuple of (model, model_data)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model_data = torch.load(model_file, map_location=device, weights_only=False)
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
    
    print(f"✅ Model loaded from: {model_file}")
    print(f"   📊 Features: {config['feature_cols']}")
    print(f"   🎯 Best validation loss: {model_data['training_metrics']['best_val_loss']:.6f}")
    
    return model, model_data

def evaluate_model(model: BitcoinPredictor, test_data: pd.DataFrame, 
                  model_data: Dict) -> Dict[str, float]:
    """
    Evaluate model performance on test data
    
    Args:
        model: Trained model
        test_data: Test dataset
        model_data: Model configuration and scalers
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create test dataset
    test_dataset = BitcoinDataset(
        data=test_data,
        sequence_length=model_data['model_config']['sequence_length'],
        target_col='close',
        feature_cols=model_data['model_config']['feature_cols']
    )
    
    # Make predictions
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequence, target in test_dataset:
            sequence = sequence.unsqueeze(0)  # Add batch dimension
            prediction = model(sequence)
            
            # Convert back to original scale
            pred_price = test_dataset.inverse_transform_target(prediction.item())
            actual_price = test_dataset.inverse_transform_target(target.item())
            
            predictions.append(pred_price)
            actuals.append(actual_price)
    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    mse = np.mean((predictions - actuals)**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def generate_test_data(days: int = 200, start_price: float = 50000) -> pd.DataFrame:
    """
    Generate synthetic test data
    
    Args:
        days: Number of days to generate
        start_price: Starting price
        
    Returns:
        DataFrame with synthetic price data
    """
    dates = pd.date_range('2024-01-01', periods=days, freq='D')
    np.random.seed(42)
    
    # Generate prices with some trend and seasonality
    trend = np.linspace(0, 1000, days)
    seasonality = 100 * np.sin(np.linspace(0, 4*np.pi, days))
    noise = np.random.randn(days) * 100
    
    prices = start_price + trend + seasonality + noise
    prices = np.maximum(prices, 0)  # Ensure prices are positive
    
    # Generate synthetic volume data
    base_volume = 1000000  # Base volume in USD
    volume_trend = np.linspace(0, 500000, days)  # Increasing trend
    volume_seasonality = 200000 * np.sin(np.linspace(0, 4*np.pi, days))  # Weekly seasonality
    volume_noise = np.random.randn(days) * 100000  # Random noise
    volumes = base_volume + volume_trend + volume_seasonality + volume_noise
    volumes = np.maximum(volumes, 100000)  # Ensure minimum volume
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': volumes
    })
    
    # Add technical indicators with proper NaN handling
    # Calculate RSI
    df['rsi'] = calculate_rsi(df['close'])
    df['rsi'] = df['rsi'].fillna(50)  # Fill initial NaN with neutral RSI value
    
    # Calculate SMA
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_20'] = df['sma_20'].fillna(method='bfill').fillna(method='ffill')
    
    # Calculate volatility
    df['volatility'] = df['close'].rolling(20).std()
    df['volatility'] = df['volatility'].fillna(method='bfill').fillna(method='ffill')
    
    # Ensure no NaN values remain
    assert not df.isna().any().any(), "NaN values found in generated test data"
    
    return df

def compare_models(finder: ModelFinder, test_data: pd.DataFrame, max_models: int = 3):
    """Compare multiple models on the same test data"""
    models = finder.find_all_models()[:max_models]
    
    if len(models) < 2:
        print("❌ Need at least 2 models for comparison")
        return
    
    print(f"\n🏆 COMPARING TOP {len(models)} MODELS")
    print("=" * 60)
    
    results = []
    
    for i, model in enumerate(models, 1):
        print(f"\n🔍 Testing Model {i}: {model.name}")
        try:
            tester = ModelTester(model)
            test_results = tester.test_on_data(test_data)
            
            results.append({
                'name': model.name[:20] + '...' if len(model.name) > 20 else model.name,
                'type': model.model_type,
                'features': len(model.features),
                'mape': test_results['metrics']['MAPE'],
                'r2': test_results['metrics']['R²'],
                'dir_acc': test_results['metrics']['Directional_Accuracy_%'],
                'mae': test_results['metrics']['MAE']
            })
        except Exception as e:
            print(f"❌ Failed to test {model.name}: {e}")
    
    if results:
        print(f"\n📊 MODEL COMPARISON RESULTS")
        print("=" * 90)
        print(f"{'Model':<25} {'Type':<6} {'Feat':<4} {'MAPE%':<8} {'R²':<8} {'Dir%':<6} {'MAE$':<10}")
        print("-" * 90)
        
        for r in results:
            print(f"{r['name']:<25} {r['type']:<6} {r['features']:<4} "
                  f"{r['mape']:<8.2f} {r['r2']:<8.4f} {r['dir_acc']:<6.1f} {r['mae']:<10,.0f}")
        
        # Find best model
        best_mape = min(results, key=lambda x: x['mape'])
        best_r2 = max(results, key=lambda x: x['r2'])
        best_dir = max(results, key=lambda x: x['dir_acc'])
        
        print(f"\n🏆 WINNERS:")
        print(f"   📈 Best MAPE: {best_mape['name']} ({best_mape['mape']:.2f}%)")
        print(f"   📊 Best R²: {best_r2['name']} ({best_r2['r2']:.4f})")
        print(f"   🎯 Best Direction: {best_dir['name']} ({best_dir['dir_acc']:.1f}%)")

def main():
    """Main function to run model testing"""
    print("🔍 BITCOIN MODEL TESTING & VALIDATION")
    print("=" * 50)
    
    # Find models
    print("🔍 Searching for trained models...")
    finder = ModelFinder()
    finder.list_models()
    
    # Get latest model
    latest_model = finder.find_latest_model()
    if not latest_model:
        print("\n❌ No models found. This could be due to:")
        print("   1. No models have been trained yet")
        print("   2. PyTorch 2.6+ compatibility issues with sklearn objects")
        print("\n💡 Solutions:")
        print("   1. Train a model first by running your main training script")
        print("   2. Make sure model files are in .outputs directory")
        return
    
    print(f"\n🎯 Testing latest model: {latest_model.name}")
    print(f"   📅 Trained: {latest_model.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   🎯 Features: {len(latest_model.features)} ({', '.join(latest_model.features)})")
    print(f"   📊 Training Val Loss: {latest_model.best_val_loss:.6f}")
    
    # Create tester
    tester = ModelTester(latest_model)
    
    # Generate test data
    test_data = generate_test_data(days=150)
    
    # Test the model
    test_results = tester.test_on_data(test_data)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    tester.visualize_predictions(test_results, save_plot=True, show_plot=False)
    
    # Generate report
    print("\n📄 Generating validation report...")
    report = tester.generate_report(test_results)
    
    # Show summary
    metrics = test_results['metrics']
    print(f"\n🎯 QUICK SUMMARY:")
    print(f"   💰 Price Error: ${metrics['MAE']:,.2f} (MAPE: {metrics['MAPE']:.2f}%)")
    print(f"   📊 Correlation: R² = {metrics['R²']:.4f}")
    print(f"   🎯 Direction Accuracy: {metrics['Directional_Accuracy_%']:.1f}%")
    
    # Performance assessment
    if metrics['MAPE'] < 3 and metrics['R²'] > 0.8:
        print("   ✅ Model performance is GOOD for production use")
    elif metrics['MAPE'] < 5 and metrics['R²'] > 0.7:
        print("   ⚠️ Model performance is FAIR - consider improvements")
    else:
        print("   ❌ Model performance needs IMPROVEMENT")
    
    print("\n🎉 Model testing completed!")
    
    # Show available actions
    print("\n💡 Available actions:")
    print("   1. python test.py  # Run this test again")
    print("   2. Test on real data: modify generate_test_data() function")
    print("   3. Compare models: uncomment model comparison section below")
    
    # Optional: Compare all models
    all_models = finder.find_all_models()
    if len(all_models) > 1:
        print(f"\n🤔 Found {len(all_models)} models total.")
        response = input("Want to compare all models? (y/n): ").lower().strip()
        if response == 'y':
            compare_models(finder, test_data)

if __name__ == "__main__":
    main() 