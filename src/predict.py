"""
predict_validate.py - Bitcoin Prediction Validator

This script loads trained models, makes predictions, validates them against actual data,
and exports results to CSV format for analysis and database integration.

Features:
- Configurable success/failure criteria
- Batch prediction processing
- Detailed CSV export with all metrics
- Database-ready format
- Multiple validation scenarios

Usage: python predict_validate.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings
import uuid
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import matplotlib.pyplot as plt
from models import BitcoinPredictor, BitcoinDataset, calculate_rsi
from sqlmodel import SQLModel, Field, Session, create_engine, select
from core.setup_path import OUTPUT_DIR
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Environment settings
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
IS_DEBUG = os.getenv('IS_DEBUG', 'true').lower() == 'true'

# SQLModel database setup
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/bitcoin_predictions')
engine = create_engine(DATABASE_URL)

# Define SQLModel classes at module level
class PredictionBase(SQLModel):
    """Base model for prediction data"""
    timestamp: str
    actual_price: float
    predicted_price: float
    absolute_error: float
    percentage_error: float
    direction_actual: str  # 'UP', 'DOWN', 'FLAT'
    direction_predicted: str
    direction_correct: bool
    within_threshold: bool
    success_status: str  # 'SUCCESS', 'FAILED', 'PARTIAL'
    confidence_score: float
    model_name: str
    prediction_horizon: str  # '1day', '3day', '7day'
    features_used: str
    sequence_length: int

class Prediction(PredictionBase, table=True):
    """Database model for predictions"""
    id: Optional[str] = Field(default=None, primary_key=True)

class PredictionCreate(PredictionBase):
    """Model for creating new predictions"""
    pass

class PredictionRead(PredictionBase):
    """Model for reading predictions"""
    id: str

class PredictionResult(PredictionBase):
    """Structure for individual prediction results"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def create(cls, **kwargs) -> 'PredictionResult':
        """Create a new prediction result"""
        return cls(**kwargs)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return self.dict()

# Create tables
SQLModel.metadata.create_all(engine)

class ValidationConfig:
    """Configuration for validation criteria"""
    price_threshold_percent: float = 5.0  # Success if within 5% of actual
    direction_weight: float = 0.4  # 40% weight on direction accuracy
    price_weight: float = 0.6  # 60% weight on price accuracy
    min_confidence_threshold: float = 0.6  # Minimum confidence for success
    flat_threshold_percent: float = 1.0  # Consider flat if change < 1%

class BitcoinPredictor(nn.Module):
    """Recreated model architecture for loading saved models"""
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2, 
                 model_type: str = 'lstm'):
        super(BitcoinPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type.lower()
        
        if self.model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.model_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_output = rnn_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(-1)

class BitcoinDataset(Dataset):
    """Dataset for prediction validation"""
    def __init__(self, data: pd.DataFrame, sequence_length: int = 60, 
                 target_col: str = 'close', feature_cols: Optional[List[str]] = None,
                 scalers: Optional[Dict] = None):
        self.sequence_length = sequence_length
        self.target_col = target_col
        
        if feature_cols is None:
            feature_cols = [target_col]
        
        self.feature_cols = feature_cols
        self.data = data[feature_cols + [target_col] if target_col not in feature_cols else feature_cols].copy()
        
        # Handle scalers
        if scalers:
            self.scalers = scalers
            self.scaled_data = self.data.copy()
            for col in self.data.columns:
                if col in self.scalers:
                    self.scaled_data[col] = self.scalers[col].transform(self.data[[col]])
        else:
            self.scalers = {}
            self.scaled_data = self.data.copy()
            for col in self.data.columns:
                scaler = MinMaxScaler()
                self.scaled_data[col] = scaler.fit_transform(self.data[[col]])
                self.scalers[col] = scaler
        
        self.sequences, self.targets, self.timestamps = self._create_sequences()
    
    def _create_sequences(self):
        sequences = []
        targets = []
        timestamps = []
        
        for i in range(len(self.scaled_data) - self.sequence_length):
            seq_features = self.scaled_data[self.feature_cols].iloc[i:i+self.sequence_length].values
            sequences.append(seq_features)
            
            target = self.scaled_data[self.target_col].iloc[i+self.sequence_length]
            targets.append(target)
            
            # Store timestamp for the prediction
            if 'date' in self.data.columns:
                timestamp = self.data['date'].iloc[i+self.sequence_length]
            else:
                timestamp = i + self.sequence_length
            timestamps.append(timestamp)
        
        sequences_array = np.array(sequences, dtype=np.float32)
        targets_array = np.array(targets, dtype=np.float32)
        
        return torch.FloatTensor(sequences_array), torch.FloatTensor(targets_array), timestamps
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.timestamps[idx]
    
    def inverse_transform_target(self, scaled_target):
        return self.scalers[self.target_col].inverse_transform([[scaled_target]])[0][0]

class ModelLoader:
    """Utility to load trained models"""
    
    @staticmethod
    def load_latest_model(base_dir: str = None) -> Optional[Tuple]:
        """Load the latest trained model"""
        base_path = Path(base_dir) if base_dir else OUTPUT_DIR
        
        if not base_path.exists():
            print(f"❌ Output directory not found: {base_path}")
            return None
        
        # Find latest model
        latest_model = None
        latest_time = datetime.min
        
        for model_file in base_path.rglob('final_model_*.pth'):
            try:
                timestamp_str = model_file.stem.split('_')[-2] + '_' + model_file.stem.split('_')[-1]
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
                if timestamp > latest_time:
                    latest_time = timestamp
                    latest_model = model_file
            except:
                continue
        
        if not latest_model:
            print("❌ No models found")
            return None
        
        return ModelLoader.load_model(latest_model)
    
    @staticmethod
    def load_model(model_path: Path) -> Tuple:
        """Load a specific model"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model data
        model_data = torch.load(model_path, map_location=device, weights_only=False)
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

class DatabaseManager:
    """Handles database operations for prediction results"""
    
    def __init__(self):
        self.is_production = ENVIRONMENT == 'production'
        print(f"🔧 Database Manager initialized")
        print(f"   Environment: {ENVIRONMENT}")
        print(f"   Is Production: {self.is_production}")
        
        if self.is_production:
            self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        SQLModel.metadata.create_all(engine)
        print("✅ Database tables created/verified")
    
    def verify_predictions(self, prediction_ids: List[str]) -> bool:
        """Verify that predictions were saved to the database"""
        if not self.is_production:
            print("❌ No database connection available for verification")
            return False
            
        try:
            with Session(engine) as session:
                # Query to check if all predictions exist
                statement = select(Prediction).where(Prediction.id.in_(prediction_ids))
                results = session.exec(statement).all()
                
                print(f"\n🔍 Database Verification:")
                print(f"   Expected records: {len(prediction_ids)}")
                print(f"   Found records: {len(results)}")
                
                # Get sample of saved records
                print("\n📋 Sample of saved records:")
                for result in results[:5]:
                    print(f"   ID: {result.id}")
                    print(f"   Timestamp: {result.timestamp}")
                    print(f"   Actual: ${result.actual_price:,.2f}")
                    print(f"   Predicted: ${result.predicted_price:,.2f}")
                    print(f"   Status: {result.success_status}")
                    print("   ---")
                
                return len(results) == len(prediction_ids)
            
        except Exception as e:
            print(f"❌ Error verifying predictions: {str(e)}")
            return False
    
    def save_predictions(self, results: List[PredictionResult]):
        """Save prediction results to database"""
        print(f"\n💾 Attempting to save {len(results)} predictions to database")
        print(f"   Production mode: {self.is_production}")
        
        if not self.is_production:
            print("ℹ️ Skipping database save in development mode")
            return
        
        try:
            with Session(engine) as session:
                # Convert PredictionResult to Prediction (mapped class)
                db_predictions = [
                    Prediction(
                        id=str(uuid.uuid4()),
                        timestamp=result.timestamp,
                        actual_price=result.actual_price,
                        predicted_price=result.predicted_price,
                        absolute_error=result.absolute_error,
                        percentage_error=result.percentage_error,
                        direction_actual=result.direction_actual,
                        direction_predicted=result.direction_predicted,
                        direction_correct=result.direction_correct,
                        within_threshold=result.within_threshold,
                        success_status=result.success_status,
                        confidence_score=result.confidence_score,
                        model_name=result.model_name,
                        prediction_horizon=result.prediction_horizon,
                        features_used=result.features_used,
                        sequence_length=result.sequence_length
                    ) for result in results
                ]
                
                # Add all predictions
                for prediction in db_predictions:
                    session.add(prediction)
                
                session.commit()
                print(f"✅ Successfully saved {len(results)} predictions to database")
                
                # Verify the save
                prediction_ids = [p.id for p in db_predictions]
                if self.verify_predictions(prediction_ids):
                    print("✅ Database verification successful")
                else:
                    print("⚠️ Database verification failed - some records may be missing")
            
        except Exception as e:
            print(f"❌ Error saving to database: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error details: {str(e)}")
    
    def view_predictions(self, limit: int = 10, order_by: str = 'timestamp DESC'):
        """View predictions from the database"""
        if not self.is_production:
            print("❌ No database connection available")
            return None
            
        try:
            with Session(engine) as session:
                statement = select(Prediction).order_by(Prediction.timestamp.desc()).limit(limit)
                results = session.exec(statement).all()
                
                if not results:
                    print("No predictions found in database")
                    return None
                
                # Convert to DataFrame for better display
                df = pd.DataFrame([result.dict() for result in results])
                
                print("\n📊 Database Predictions:")
                print("=" * 100)
                print(df.to_string(index=False))
                print("=" * 100)
                
                # Print summary statistics
                print("\n📈 Summary Statistics:")
                print(f"Total records shown: {len(df)}")
                print(f"Success rate: {(df['success_status'] == 'SUCCESS').mean()*100:.1f}%")
                print(f"Average error: {df['percentage_error'].mean():.2f}%")
                print(f"Direction accuracy: {df['direction_correct'].mean()*100:.1f}%")
                
                return df
                
        except Exception as e:
            print(f"❌ Error viewing predictions: {str(e)}")
            return None

class PredictionValidator:
    """Main class for prediction validation"""
    
    def __init__(self, model_path: Optional[str] = None, config: ValidationConfig = ValidationConfig()):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.db_manager = DatabaseManager()
        
        # Load model
        if model_path:
            self.model, self.model_data, self.model_config, self.device = ModelLoader.load_model(Path(model_path))
        else:
            result = ModelLoader.load_latest_model()
            if result is None:
                raise ValueError("No models found. Train a model first!")
            self.model, self.model_data, self.model_config, self.device = result
        
        print(f"✅ Model loaded: {self.model_config['model_type'].upper()}")
        print(f"   📊 Features: {len(self.model_config['feature_cols'])}")
        print(f"   🎯 Sequence Length: {self.model_config['sequence_length']}")
    
    def predict_and_validate(self, test_data: pd.DataFrame, 
                           prediction_horizons: List[str] = ['1day']) -> List[PredictionResult]:
        """Make predictions and validate against actual data"""
        
        print(f"\n🔮 Starting prediction validation...")
        print(f"   📅 Test data: {len(test_data)} records")
        print(f"   ⏰ Horizons: {', '.join(prediction_horizons)}")
        
        results = []
        
        for horizon in prediction_horizons:
            print(f"\n📈 Processing {horizon} predictions...")
            horizon_results = self._process_horizon(test_data, horizon)
            results.extend(horizon_results)
        
        return results
    
    def _process_horizon(self, data: pd.DataFrame, horizon: str) -> List[PredictionResult]:
        """Process predictions for a specific time horizon"""
        
        # Create dataset
        training_scalers = self.model_data.get('dataset_scalers', {})
        dataset = BitcoinDataset(
            data=data,
            sequence_length=self.model_config['sequence_length'],
            target_col='close',
            feature_cols=self.model_config['feature_cols'],
            scalers=training_scalers
        )
        
        results = []
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc=f"Predicting {horizon}", unit="sample"):
                sequence, target, timestamp = dataset[i]
                sequence = sequence.unsqueeze(0).to(self.device)
                
                # Get prediction
                pred = self.model(sequence).cpu().numpy()
                if pred.ndim == 0:
                    pred = pred.item()
                else:
                    pred = pred[0]
                
                # Convert back to original scale
                predicted_price = dataset.inverse_transform_target(pred)
                actual_price = dataset.inverse_transform_target(target.numpy().item())
                
                # Calculate metrics
                result = self._calculate_prediction_result(
                    timestamp, actual_price, predicted_price, horizon, 
                    i, data, dataset
                )
                
                results.append(result)
        
        return results
    
    def _calculate_prediction_result(self, timestamp, actual_price: float, 
                                   predicted_price: float, horizon: str, 
                                   index: int, data: pd.DataFrame,
                                   dataset: BitcoinDataset) -> PredictionResult:
        """Calculate detailed prediction result"""
        
        # Basic metrics
        absolute_error = abs(predicted_price - actual_price)
        percentage_error = (absolute_error / actual_price) * 100
        
        # Direction analysis
        if index > 0:
            prev_price = data['close'].iloc[index + dataset.sequence_length - 1]
            actual_direction = self._get_direction(prev_price, actual_price)
            predicted_direction = self._get_direction(prev_price, predicted_price)
        else:
            actual_direction = 'FLAT'
            predicted_direction = 'FLAT'
        
        direction_correct = actual_direction == predicted_direction
        
        # Threshold check
        within_threshold = percentage_error <= self.config.price_threshold_percent
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(percentage_error, direction_correct)
        
        # Determine success status
        success_status = self._determine_success(
            within_threshold, direction_correct, confidence_score
        )
        
        return PredictionResult.create(
            timestamp=str(timestamp),
            actual_price=round(actual_price, 2),
            predicted_price=round(predicted_price, 2),
            absolute_error=round(absolute_error, 2),
            percentage_error=round(percentage_error, 4),
            direction_actual=actual_direction,
            direction_predicted=predicted_direction,
            direction_correct=direction_correct,
            within_threshold=within_threshold,
            success_status=success_status,
            confidence_score=round(confidence_score, 4),
            model_name=f"{self.model_config['model_type']}_{len(self.model_config['feature_cols'])}f",
            prediction_horizon=horizon,
            features_used=','.join(self.model_config['feature_cols']),
            sequence_length=self.model_config['sequence_length']
        )
    
    def _get_direction(self, prev_price: float, current_price: float) -> str:
        """Determine price direction"""
        change_percent = ((current_price - prev_price) / prev_price) * 100
        
        if abs(change_percent) < self.config.flat_threshold_percent:
            return 'FLAT'
        elif change_percent > 0:
            return 'UP'
        else:
            return 'DOWN'
    
    def _calculate_confidence(self, percentage_error: float, direction_correct: bool) -> float:
        """Calculate confidence score (0-1)"""
        # Price accuracy component (0-1, higher is better)
        price_component = max(0, 1 - (percentage_error / (self.config.price_threshold_percent * 2)))
        
        # Direction component (0-1)
        direction_component = 1.0 if direction_correct else 0.0
        
        # Weighted combination
        confidence = (
            self.config.price_weight * price_component + 
            self.config.direction_weight * direction_component
        )
        
        return max(0, min(1, confidence))
    
    def _determine_success(self, within_threshold: bool, direction_correct: bool, 
                          confidence_score: float) -> str:
        """Determine overall success status"""
        
        if confidence_score >= self.config.min_confidence_threshold:
            if within_threshold and direction_correct:
                return 'SUCCESS'
            elif within_threshold or direction_correct:
                return 'PARTIAL'
            else:
                return 'FAILED'
        else:
            return 'FAILED'
    
    def export_to_csv(self, results: List[PredictionResult], 
                     filename: Optional[str] = None) -> str:
        """Export results to CSV format and optionally to database"""
        
        print(f"\n📊 Exporting results...")
        print(f"   Environment: {ENVIRONMENT}")
        print(f"   Total results: {len(results)}")
        
        # Create output directory if it doesn't exist
        OUTPUT_DIR.mkdir(exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'prediction_validation_{timestamp}.csv'
        
        # Convert to DataFrame
        df = pd.DataFrame([result.to_dict() for result in results])
        
        # Add summary statistics
        total_predictions = len(df)
        successful_predictions = len(df[df['success_status'] == 'SUCCESS'])
        partial_predictions = len(df[df['success_status'] == 'PARTIAL'])
        failed_predictions = len(df[df['success_status'] == 'FAILED'])
        
        direction_accuracy = df['direction_correct'].mean() * 100
        avg_percentage_error = df['percentage_error'].mean()
        avg_confidence = df['confidence_score'].mean()
        
        # Create summary row
        summary_data = {
            'id': 'SUMMARY',
            'timestamp': 'SUMMARY',
            'actual_price': 0,
            'predicted_price': 0,
            'absolute_error': 0,
            'percentage_error': avg_percentage_error,
            'direction_actual': f'{direction_accuracy:.1f}%_accuracy',
            'direction_predicted': '',
            'direction_correct': direction_accuracy > 70,
            'within_threshold': (successful_predictions + partial_predictions) / total_predictions > 0.7,
            'success_status': f'{successful_predictions}/{total_predictions}',
            'confidence_score': avg_confidence,
            'model_name': df['model_name'].iloc[0] if len(df) > 0 else '',
            'prediction_horizon': 'ALL',
            'features_used': f'SUCCESS:{successful_predictions}_PARTIAL:{partial_predictions}_FAILED:{failed_predictions}',
            'sequence_length': df['sequence_length'].iloc[0] if len(df) > 0 else 0
        }
        
        # Add summary to DataFrame
        summary_df = pd.DataFrame([summary_data])
        final_df = pd.concat([df, summary_df], ignore_index=True)
        
        # Save to CSV in output directory
        output_path = OUTPUT_DIR / filename
        final_df.to_csv(output_path, index=False)
        print(f"✅ CSV saved to: {output_path.absolute()}")
        
        # Save to database if in production mode
        if ENVIRONMENT == 'production':
            print("\n💾 Attempting database save...")
            # Filter out the summary row before saving to database
            db_results = [r for r in results if r.id != 'SUMMARY']
            print(f"   Filtered {len(db_results)} records for database (excluding summary)")
            
            if db_results:
                self.db_manager.save_predictions(db_results)
            else:
                print("⚠️ No valid records to save to database")
        else:
            print("\nℹ️ Skipping database save (not in production mode)")
        
        print(f"\n📊 PREDICTION VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Total Predictions: {total_predictions}")
        print(f"✅ Successful: {successful_predictions} ({successful_predictions/total_predictions*100:.1f}%)")
        print(f"⚠️ Partial: {partial_predictions} ({partial_predictions/total_predictions*100:.1f}%)")
        print(f"❌ Failed: {failed_predictions} ({failed_predictions/total_predictions*100:.1f}%)")
        print(f"🎯 Direction Accuracy: {direction_accuracy:.1f}%")
        print(f"📈 Avg Error: {avg_percentage_error:.2f}%")
        print(f"🔒 Avg Confidence: {avg_confidence:.3f}")
        
        return str(output_path)

def generate_validation_data(days: int = 200, start_price: float = 50000) -> pd.DataFrame:
    """Generate test data for validation"""
    print(f"🔢 Generating {days} days of validation data...")
    
    # Create realistic price data with trends and volatility
    dates = pd.date_range(start='2024-02-01', periods=days, freq='D')
    np.random.seed(456)  # Different seed for validation
    
    # More realistic price simulation
    prices = [start_price]
    for i in range(1, days):
        # Add trend, mean reversion, and volatility
        prev_price = prices[-1]
        
        # Random walk with slight upward bias
        trend = 0.0005  # 0.05% daily upward trend
        volatility = 0.025  # 2.5% daily volatility
        
        # Add some mean reversion
        if len(prices) > 20:
            sma_20 = np.mean(prices[-20:])
            mean_reversion = (sma_20 - prev_price) / prev_price * 0.1
        else:
            mean_reversion = 0
        
        change = np.random.normal(trend + mean_reversion, volatility)
        new_price = prev_price * (1 + change)
        
        # Prevent unrealistic prices
        new_price = max(new_price, 10000)  # Min $10k
        new_price = min(new_price, 200000)  # Max $200k
        
        prices.append(new_price)
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    
    # Add required features
    df['volume'] = np.random.randint(15000, 60000, len(df))
    df['volatility'] = df['close'].rolling(20, min_periods=1).std().fillna(0)
    df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['rsi'] = calculate_rsi(df['close']).fillna(50)
    
    print(f"✅ Validation data generated")
    print(f"   📊 Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
    print(f"   📈 Total change: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.1f}%")
    
    return df

def create_custom_config() -> ValidationConfig:
    """Create custom validation configuration"""
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

def main():
    """Main function"""
    print("🔮 BITCOIN PREDICTION VALIDATOR")
    print("=" * 40)
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Run prediction validation")
    print("2. View database predictions")
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "2":
        # Initialize database manager
        db_manager = DatabaseManager()
        if db_manager.conn:
            limit = input("How many records to view? (default 10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            db_manager.view_predictions(limit=limit)
        db_manager.close()
        return
    
    # Create validation configuration
    config = create_custom_config()
    
    try:
        # Initialize validator
        print(f"\n🤖 Loading model...")
        validator = PredictionValidator(config=config)
        
        # Generate or load test data
        print(f"\n📊 Preparing test data...")
        
        # Option to use custom data
        use_custom = input("Use custom data file? (y/n): ").lower().strip()
        
        if use_custom == 'y':
            file_path = input("Enter CSV file path: ").strip()
            try:
                test_data = pd.read_csv(file_path)
                if 'date' not in test_data.columns:
                    test_data['date'] = pd.date_range(start='2024-01-01', periods=len(test_data), freq='D')
                print(f"✅ Loaded {len(test_data)} records from {file_path}")
            except Exception as e:
                print(f"❌ Error loading file: {e}")
                print("🔄 Falling back to generated data")
                test_data = generate_validation_data()
        else:
            test_data = generate_validation_data()
        
        # Choose prediction horizons
        horizons = ['1day']  # Can extend to ['1day', '3day', '7day']
        
        # Run validation
        results = validator.predict_and_validate(test_data, horizons)
        
        # Export results
        csv_path = validator.export_to_csv(results)
        
        # Show sample results
        if results:
            print(f"\n📋 SAMPLE RESULTS (first 5):")
            print("-" * 70)
            
            sample_df = pd.DataFrame([r.to_dict() for r in results[:5]])
            print(sample_df[['timestamp', 'actual_price', 'predicted_price', 
                           'percentage_error', 'direction_correct', 'success_status']].to_string(index=False))
        
        print(f"\n🎉 Validation completed successfully!")
        print(f"💾 Full results available in: {csv_path}")
        print(f"\n💡 Next steps:")
        print(f"   1. Review CSV file for detailed analysis")
        print(f"   2. Import to database for tracking")
        print(f"   3. Use results to improve model")
        print(f"   4. Set up automated validation pipeline")
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        print(f"💡 Make sure you have trained models in {OUTPUT_DIR} directory")

if __name__ == "__main__":
    main()