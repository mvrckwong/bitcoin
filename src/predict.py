"""
Optimized Bitcoin Prediction Validator.

This module provides the main prediction validation functionality, leveraging
modularized components for model loading, data handling, and database operations.
The code has been refactored to remove redundancy and improve maintainability.
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models import BitcoinDataset, ModelLoader
from core import generate_validation_data, load_data_from_csv, OUTPUT_DIR
from database import DatabaseManager, PredictionResult
from settings import (
    ENVIRONMENT, ValidationConfig, DataConfig,
    PREDICTION_HORIZONS, create_custom_config
)

warnings.filterwarnings('ignore')


class PredictionValidator:
    """
    Main class for prediction validation.
    
    This class handles the core prediction validation logic, including:
    - Loading trained models
    - Making predictions on test data
    - Calculating validation metrics
    - Exporting results to CSV and database
    """

    def __init__(self, model_path: Optional[str] = None,
                 config: Optional[ValidationConfig] = None):
        """
        Initialize the prediction validator.
        
        Args:
            model_path: Optional path to a specific model file.
            config: Optional validation configuration.
        """
        self.config = config or ValidationConfig()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.db_manager = DatabaseManager()

        # Load model using the modularized ModelLoader
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
        """
        Make predictions and validate against actual data.
        
        Args:
            test_data: DataFrame containing test data.
            prediction_horizons: List of prediction horizons to evaluate.
            
        Returns:
            List of PredictionResult objects containing validation metrics.
        """
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
        """
        Process predictions for a specific time horizon.
        
        Args:
            data: DataFrame containing the data for predictions.
            horizon: Time horizon for predictions.
            
        Returns:
            List of PredictionResult objects for this horizon.
            
        Note:
            Timestamps in results represent when predictions were executed,
            not the time period being predicted.
        """
        # Validate data has sufficient records for sequence creation
        min_required = self.model_config['sequence_length'] + 1
        if len(data) < min_required:
            raise ValueError(
                f"Insufficient data for predictions. "
                f"Need at least {min_required} records, got {len(data)}."
            )
        
        # Create dataset using the modularized BitcoinDataset
        training_scalers = self.model_data.get('dataset_scalers', {})
        dataset = BitcoinDataset(
            data=data,
            sequence_length=self.model_config['sequence_length'],
            target_col='close',
            feature_cols=self.model_config['feature_cols'],
            scalers=training_scalers,
            include_timestamps=False  # Don't use the dataset's timestamp logic
        )

        results = []

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            desc = f"Predicting {horizon}"
            for i in tqdm(range(len(dataset)), desc=desc, unit="sample"):
                # Capture current execution time for this specific prediction
                current_time = datetime.now()
                timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
                
                sequence, target = dataset[i]
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
        """
        Calculate detailed prediction result.
        
        Args:
            timestamp: Timestamp of the prediction.
            actual_price: Actual price value.
            predicted_price: Predicted price value.
            horizon: Prediction horizon.
            index: Index in the dataset.
            data: Original data DataFrame.
            dataset: Dataset object for inverse transforms.
            
        Returns:
            PredictionResult object with all calculated metrics.
        """
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
        """
        Determine price direction.
        
        Args:
            prev_price: Previous price value.
            current_price: Current price value.
            
        Returns:
            Direction string: 'UP', 'DOWN', or 'FLAT'.
        """
        change_percent = ((current_price - prev_price) / prev_price) * 100

        if abs(change_percent) < self.config.flat_threshold_percent:
            return 'FLAT'
        elif change_percent > 0:
            return 'UP'
        else:
            return 'DOWN'

    def _calculate_confidence(self, percentage_error: float,
                              direction_correct: bool) -> float:
        """
        Calculate confidence score (0-1).
        
        Args:
            percentage_error: Percentage error of the prediction.
            direction_correct: Whether direction prediction was correct.
            
        Returns:
            Confidence score between 0 and 1.
        """
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
        """
        Determine overall success status.
        
        Args:
            within_threshold: Whether error is within acceptable threshold.
            direction_correct: Whether direction prediction was correct.
            confidence_score: Calculated confidence score.
            
        Returns:
            Success status: 'SUCCESS', 'PARTIAL', or 'FAILED'.
        """
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
        """
        Export results to CSV format and optionally to database.
        
        Args:
            results: List of PredictionResult objects.
            filename: Optional filename for the CSV file.
            
        Returns:
            Path to the exported CSV file.
        """
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


def main():
    """
    Main function for running prediction validation.
    
    This function provides an interactive interface for users to:
    1. Run prediction validation with custom or generated data
    2. View existing database predictions
    """
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
                test_data = load_data_from_csv(file_path)
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