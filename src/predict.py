import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from tqdm import tqdm

from models import calculate_rsi
from core.setup_path import OUTPUT_DIR
from database import DatabaseManager, PredictionResult
from settings import (
    ENVIRONMENT, ValidationConfig, ModelConfig,
    DataConfig, PREDICTION_HORIZONS, MIN_PRICE, MAX_PRICE,
    create_custom_config
)

warnings.filterwarnings('ignore')


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
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.model_type == 'gru':
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )
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
                 target_col: str = None,
                 feature_cols: Optional[List[str]] = None,
                 scalers: Optional[Dict] = None):
        # Use DataConfig defaults
        data_config = DataConfig()
        self.sequence_length = sequence_length
        self.target_col = target_col or data_config.target_column

        if feature_cols is None:
            feature_cols = data_config.default_features

        self.feature_cols = feature_cols
        if self.target_col not in feature_cols:
            cols = feature_cols + [self.target_col]
        else:
            cols = feature_cols
        self.data = data[cols].copy()

        # Handle scalers
        if scalers:
            self.scalers = scalers
            self.scaled_data = self.data.copy()
            for col in self.data.columns:
                if col in self.scalers:
                    transformed = self.scalers[col].transform(self.data[[col]])
                    self.scaled_data[col] = transformed
        else:
            self.scalers = {}
            self.scaled_data = self.data.copy()
            for col in self.data.columns:
                scaler = MinMaxScaler()
                transformed = scaler.fit_transform(self.data[[col]])
                self.scaled_data[col] = transformed
                self.scalers[col] = scaler

        self.sequences, self.targets, self.timestamps = self._create_sequences()

    def _create_sequences(self):
        sequences = []
        targets = []
        timestamps = []

        for i in range(len(self.scaled_data) - self.sequence_length):
            seq_start = i
            seq_end = i + self.sequence_length
            seq_features = self.scaled_data[self.feature_cols].iloc[
                seq_start:seq_end
            ].values
            sequences.append(seq_features)

            target = self.scaled_data[self.target_col].iloc[
                i + self.sequence_length
            ]
            targets.append(target)

            # Store timestamp for the prediction
            if 'date' in self.data.columns:
                timestamp = self.data['date'].iloc[i + self.sequence_length]
            else:
                timestamp = i + self.sequence_length
            timestamps.append(timestamp)

        sequences_array = np.array(sequences, dtype=np.float32)
        targets_array = np.array(targets, dtype=np.float32)

        return (torch.FloatTensor(sequences_array),
                torch.FloatTensor(targets_array),
                timestamps)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.timestamps[idx]

    def inverse_transform_target(self, scaled_target):
        transformed = self.scalers[self.target_col].inverse_transform(
            [[scaled_target]]
        )
        return transformed[0][0]


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
        """Load a specific model"""
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


class PredictionValidator:
    """Main class for prediction validation"""

    def __init__(self, model_path: Optional[str] = None,
                 config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.db_manager = DatabaseManager()

        # Load model
        if model_path:
            model_result = ModelLoader.load_model(Path(model_path))
            self.model, self.model_data, self.model_config, self.device = model_result
        else:
            result = ModelLoader.load_latest_model()
            if result is None:
                raise ValueError("No models found. Train a model first!")
            self.model, self.model_data, self.model_config, self.device = result

        model_type = self.model_config['model_type'].upper()
        print(f"✅ Model loaded: {model_type}")
        print(f"   📊 Features: {len(self.model_config['feature_cols'])}")
        print(f"   🎯 Sequence Length: {self.model_config['sequence_length']}")

    def predict_and_validate(self, test_data: pd.DataFrame,
                             prediction_horizons: List[str] = ['1day']
                             ) -> List[PredictionResult]:
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

    def _process_horizon(self, data: pd.DataFrame,
                         horizon: str) -> List[PredictionResult]:
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
            desc = f"Predicting {horizon}"
            for i in tqdm(range(len(dataset)), desc=desc, unit="sample"):
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
                actual_price = dataset.inverse_transform_target(
                    target.numpy().item()
                )

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
                                     dataset: BitcoinDataset
                                     ) -> PredictionResult:
        """Calculate detailed prediction result"""

        # Basic metrics
        absolute_error = abs(predicted_price - actual_price)
        percentage_error = (absolute_error / actual_price) * 100

        # Direction analysis
        if index > 0:
            prev_price = data['close'].iloc[
                index + dataset.sequence_length - 1
            ]
            actual_direction = self._get_direction(prev_price, actual_price)
            predicted_direction = self._get_direction(
                prev_price, predicted_price
            )
        else:
            actual_direction = 'FLAT'
            predicted_direction = 'FLAT'

        direction_correct = actual_direction == predicted_direction

        # Threshold check
        within_threshold = (
            percentage_error <= self.config.price_threshold_percent
        )

        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            percentage_error, direction_correct
        )

        # Determine success status
        success_status = self._determine_success(
            within_threshold, direction_correct, confidence_score
        )

        model_name = (
            f"{self.model_config['model_type']}_"
            f"{len(self.model_config['feature_cols'])}f"
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
            model_name=model_name,
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

    def _calculate_confidence(self, percentage_error: float,
                              direction_correct: bool) -> float:
        """Calculate confidence score (0-1)"""
        # Price accuracy component (0-1, higher is better)
        threshold_factor = self.config.price_threshold_percent * 2
        price_component = max(0, 1 - (percentage_error / threshold_factor))

        # Direction component (0-1)
        direction_component = 1.0 if direction_correct else 0.0

        # Weighted combination
        confidence = (
            self.config.price_weight * price_component +
            self.config.direction_weight * direction_component
        )

        return max(0, min(1, confidence))

    def _determine_success(self, within_threshold: bool,
                           direction_correct: bool,
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
            'within_threshold': (
                (successful_predictions + partial_predictions) /
                total_predictions > 0.7
            ),
            'success_status': f'{successful_predictions}/{total_predictions}',
            'confidence_score': avg_confidence,
            'model_name': df['model_name'].iloc[0] if len(df) > 0 else '',
            'prediction_horizon': 'ALL',
            'features_used': (
                f'SUCCESS:{successful_predictions}_'
                f'PARTIAL:{partial_predictions}_'
                f'FAILED:{failed_predictions}'
            ),
            'sequence_length': (
                df['sequence_length'].iloc[0] if len(df) > 0 else 0
            )
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
            filtered_count = len(db_results)
            print(f"   Filtered {filtered_count} records for database "
                  "(excluding summary)")

            if db_results:
                self.db_manager.save_predictions(db_results)
            else:
                print("⚠️ No valid records to save to database")
        else:
            print("\nℹ️ Skipping database save (not in production mode)")

        print(f"\n📊 PREDICTION VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Total Predictions: {total_predictions}")
        success_pct = successful_predictions / total_predictions * 100
        print(f"✅ Successful: {successful_predictions} ({success_pct:.1f}%)")
        partial_pct = partial_predictions / total_predictions * 100
        print(f"⚠️ Partial: {partial_predictions} ({partial_pct:.1f}%)")
        failed_pct = failed_predictions / total_predictions * 100
        print(f"❌ Failed: {failed_predictions} ({failed_pct:.1f}%)")
        print(f"🎯 Direction Accuracy: {direction_accuracy:.1f}%")
        print(f"📈 Avg Error: {avg_percentage_error:.2f}%")
        print(f"🔒 Avg Confidence: {avg_confidence:.3f}")

        return str(output_path)


def generate_validation_data(days: int = 200,
                             start_price: float = 50000) -> pd.DataFrame:
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

        # Prevent unrealistic prices using settings constants
        new_price = max(new_price, MIN_PRICE)
        new_price = min(new_price, MAX_PRICE)

        prices.append(new_price)

    df = pd.DataFrame({
        'date': dates,
        'close': prices
    })

    # Add required features using DataConfig
    data_config = DataConfig()
    df['volume'] = np.random.randint(15000, 60000, len(df))
    df['volatility'] = df['close'].rolling(20, min_periods=1).std().fillna(0)
    df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['rsi'] = calculate_rsi(df['close']).fillna(50)

    print(f"✅ Validation data generated")
    price_min = df['close'].min()
    price_max = df['close'].max()
    print(f"   📊 Price range: ${price_min:,.2f} - ${price_max:,.2f}")
    price_change = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
    print(f"   📈 Total change: {price_change:.1f}%")

    return df


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
        if hasattr(db_manager, 'conn') and db_manager.conn:
            limit = input("How many records to view? (default 10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            db_manager.view_predictions(limit=limit)
        if hasattr(db_manager, 'close'):
            db_manager.close()
        return

    # Create validation configuration using imported function
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
                    test_data['date'] = pd.date_range(
                        start='2024-01-01',
                        periods=len(test_data),
                        freq='D'
                    )
                print(f"✅ Loaded {len(test_data)} records from {file_path}")
            except Exception as e:
                print(f"❌ Error loading file: {e}")
                print("🔄 Falling back to generated data")
                test_data = generate_validation_data()
        else:
            test_data = generate_validation_data()

        # Choose prediction horizons from settings
        horizons = PREDICTION_HORIZONS[:1]  # Use first horizon from settings

        # Run validation
        results = validator.predict_and_validate(test_data, horizons)

        # Export results
        csv_path = validator.export_to_csv(results)

        # Show sample results
        if results:
            print(f"\n📋 SAMPLE RESULTS (first 5):")
            print("-" * 70)

            sample_df = pd.DataFrame([r.to_dict() for r in results[:5]])
            columns = [
                'timestamp', 'actual_price', 'predicted_price',
                'percentage_error', 'direction_correct', 'success_status'
            ]
            print(sample_df[columns].to_string(index=False))

        print(f"\n🎉 Validation completed successfully!")
        print(f"💾 Full results available in: {csv_path}")
        print(f"\n💡 Next steps:")
        print(f"   1. Review CSV file for detailed analysis")
        print(f"   2. Import to database for tracking")
        print(f"   3. Use results to improve model")
        print(f"   4. Set up automated validation pipeline")

    except Exception as e:
        print(f"❌ Error during validation: {e}")
        output_dir_msg = f"💡 Make sure you have trained models in {OUTPUT_DIR} directory"
        print(output_dir_msg)


if __name__ == "__main__":
    main()