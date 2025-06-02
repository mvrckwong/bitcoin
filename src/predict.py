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

# Import logging setup
from core.setup_logging import (
    setup_development_logging, setup_production_logging, setup_staging_logging,
    log_function_call, log_performance, log_security_event, log_health_check,
    log_system_metrics, get_logger_for_service
)

warnings.filterwarnings('ignore')

# Setup logging based on environment
if ENVIRONMENT == 'production':
    logger = setup_production_logging("bitcoin-predictor")
elif ENVIRONMENT == 'staging':
    logger = setup_staging_logging("bitcoin-predictor")
else:
    logger = setup_development_logging("bitcoin-predictor")


class PredictionValidator:
    """
    Main class for prediction validation.
    
    This class handles the core prediction validation logic, including:
    - Loading trained models
    - Making predictions on test data
    - Calculating validation metrics
    - Exporting results to CSV and database
    """

    @log_function_call(include_args=True, sanitize=True)
    @log_performance(threshold_ms=5000.0)
    def __init__(self, model_path: Optional[str] = None,
                 config: Optional[ValidationConfig] = None):
        """
        Initialize the prediction validator.
        
        Args:
            model_path: Optional path to a specific model file.
            config: Optional validation configuration.
        """
        logger.info("Initializing PredictionValidator", extra={
            "model_path": model_path,
            "config_provided": config is not None,
            "environment": ENVIRONMENT
        })
        
        self.config = config or ValidationConfig()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        logger.debug("Device configuration", extra={
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available()
        })

        try:
            self.db_manager = DatabaseManager()
            log_health_check("database", "healthy", {"connection": "established"})
        except Exception as e:
            logger.error("Database connection failed", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            log_health_check("database", "unhealthy", {"error": str(e)})
            raise

        # Load model using the modularized ModelLoader
        try:
            if model_path:
                logger.info("Loading specific model", extra={"model_path": model_path})
                model_result = ModelLoader.load_model(Path(model_path))
                self.model, self.model_data, self.model_config, self.device = model_result
            else:
                logger.info("Loading latest available model")
                result = ModelLoader.load_latest_model()
                if result is None:
                    error_msg = "No models found. Train a model first!"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                self.model, self.model_data, self.model_config, self.device = result

            model_type = self.model_config['model_type'].upper()
            feature_count = len(self.model_config['feature_cols'])
            sequence_length = self.model_config['sequence_length']
            
            logger.success("Model loaded successfully", extra={
                "model_type": model_type,
                "feature_count": feature_count,
                "sequence_length": sequence_length,
                "device": str(self.device)
            })
            
            # Log system metrics after model loading
            log_system_metrics({
                "model_loaded": True,
                "model_type": model_type,
                "features_count": feature_count,
                "sequence_length": sequence_length
            })
            
        except Exception as e:
            logger.error("Model loading failed", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "model_path": model_path
            })
            raise

    @log_function_call(include_args=True, sanitize=True)
    @log_performance(threshold_ms=30000.0)
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
        logger.info("Starting prediction validation", extra={
            "test_data_records": len(test_data),
            "prediction_horizons": prediction_horizons,
            "model_type": self.model_config['model_type']
        })

        # Log security event for prediction operation
        log_security_event("prediction_validation_started", {
            "data_records": len(test_data),
            "horizons": prediction_horizons,
            "model_type": self.model_config['model_type']
        })

        results = []

        for horizon in prediction_horizons:
            logger.info(f"Processing horizon: {horizon}", extra={
                "horizon": horizon,
                "data_size": len(test_data)
            })
            
            try:
                horizon_results = self._process_horizon(test_data, horizon)
                results.extend(horizon_results)
                
                logger.success(f"Horizon {horizon} completed", extra={
                    "horizon": horizon,
                    "results_count": len(horizon_results)
                })
                
            except Exception as e:
                logger.error(f"Horizon {horizon} processing failed", extra={
                    "horizon": horizon,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                raise

        logger.success("Prediction validation completed", extra={
            "total_results": len(results),
            "horizons_processed": len(prediction_horizons)
        })

        return results

    @log_function_call(include_args=False, sanitize=True)
    @log_performance(threshold_ms=20000.0)
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
        min_required = self.model_config['sequence_length'] + 1
        
        logger.debug("Validating data requirements", extra={
            "data_length": len(data),
            "min_required": min_required,
            "horizon": horizon
        })
        
        # Validate data has sufficient records for sequence creation
        if len(data) < min_required:
            error_msg = (
                f"Insufficient data for predictions. "
                f"Need at least {min_required} records, got {len(data)}."
            )
            logger.error(error_msg, extra={
                "data_length": len(data),
                "min_required": min_required,
                "horizon": horizon
            })
            raise ValueError(error_msg)
        
        # Create dataset using the modularized BitcoinDataset
        logger.debug("Creating dataset", extra={
            "sequence_length": self.model_config['sequence_length'],
            "feature_cols_count": len(self.model_config['feature_cols']),
            "horizon": horizon
        })
        
        try:
            training_scalers = self.model_data.get('dataset_scalers', {})
            dataset = BitcoinDataset(
                data=data,
                sequence_length=self.model_config['sequence_length'],
                target_col='close',
                feature_cols=self.model_config['feature_cols'],
                scalers=training_scalers,
                include_timestamps=False  # Don't use the dataset's timestamp logic
            )
            
            logger.debug("Dataset created successfully", extra={
                "dataset_length": len(dataset),
                "scalers_available": len(training_scalers),
                "horizon": horizon
            })
            
        except Exception as e:
            logger.error("Dataset creation failed", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "horizon": horizon
            })
            raise

        results = []

        # Make predictions
        logger.debug("Starting model predictions", extra={
            "dataset_length": len(dataset),
            "model_mode": "evaluation",
            "device": str(self.device)
        })
        
        self.model.eval()
        prediction_errors = 0
        
        with torch.no_grad():
            desc = f"Predicting {horizon}"
            for i in tqdm(range(len(dataset)), desc=desc, unit="sample"):
                try:
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
                    
                    # Log every 100th prediction for monitoring
                    if i % 100 == 0:
                        logger.debug("Prediction progress", extra={
                            "processed": i + 1,
                            "total": len(dataset),
                            "horizon": horizon,
                            "current_accuracy": result.percentage_error
                        })
                
                except Exception as e:
                    prediction_errors += 1
                    logger.warning("Individual prediction failed", extra={
                        "index": i,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "horizon": horizon
                    })
                    
                    # Don't fail the entire process for individual prediction errors
                    # but log them for monitoring
                    if prediction_errors > len(dataset) * 0.1:  # More than 10% errors
                        logger.error("Too many prediction errors", extra={
                            "error_count": prediction_errors,
                            "total_predictions": i + 1,
                            "error_rate": prediction_errors / (i + 1)
                        })
                        raise RuntimeError(f"Excessive prediction errors: {prediction_errors}/{i + 1}")

        if prediction_errors > 0:
            logger.warning("Prediction process completed with errors", extra={
                "total_errors": prediction_errors,
                "total_predictions": len(dataset),
                "error_rate": prediction_errors / len(dataset),
                "successful_predictions": len(results)
            })

        return results

    @log_function_call(include_args=False, sanitize=True)
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

        # Log detailed prediction result for monitoring
        logger.trace("Prediction result calculated", extra={
            "timestamp": timestamp,
            "actual_price": actual_price,
            "predicted_price": predicted_price,
            "percentage_error": percentage_error,
            "direction_correct": direction_correct,
            "success_status": success_status,
            "confidence_score": confidence_score,
            "horizon": horizon
        })

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

    @log_function_call(include_args=False, sanitize=True)
    @log_performance(threshold_ms=10000.0)
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
        logger.info("Starting results export", extra={
            "results_count": len(results),
            "environment": ENVIRONMENT,
            "filename": filename
        })

        # Create output directory if it doesn't exist
        try:
            OUTPUT_DIR.mkdir(exist_ok=True)
            logger.debug("Output directory prepared", extra={
                "output_dir": str(OUTPUT_DIR),
                "exists": OUTPUT_DIR.exists()
            })
        except Exception as e:
            logger.error("Failed to create output directory", extra={
                "output_dir": str(OUTPUT_DIR),
                "error": str(e)
            })
            raise

        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'prediction_validation_{timestamp}.csv'

        # Convert to DataFrame
        try:
            df = pd.DataFrame([result.to_dict() for result in results])
            logger.debug("DataFrame created", extra={
                "dataframe_shape": df.shape,
                "columns": list(df.columns)
            })
        except Exception as e:
            logger.error("Failed to create DataFrame", extra={
                "error": str(e),
                "results_count": len(results)
            })
            raise

        # Add summary statistics
        total_predictions = len(df)
        successful_predictions = len(df[df['success_status'] == 'SUCCESS'])
        partial_predictions = len(df[df['success_status'] == 'PARTIAL'])
        failed_predictions = len(df[df['success_status'] == 'FAILED'])

        direction_accuracy = df['direction_correct'].mean() * 100
        avg_percentage_error = df['percentage_error'].mean()
        avg_confidence = df['confidence_score'].mean()

        # Log summary statistics
        logger.info("Prediction summary statistics", extra={
            "total_predictions": total_predictions,
            "successful_predictions": successful_predictions,
            "partial_predictions": partial_predictions,
            "failed_predictions": failed_predictions,
            "direction_accuracy": direction_accuracy,
            "avg_percentage_error": avg_percentage_error,
            "avg_confidence": avg_confidence
        })

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
        
        try:
            final_df.to_csv(output_path, index=False)
            logger.success("CSV file saved successfully", extra={
                "output_path": str(output_path),
                "file_size_bytes": output_path.stat().st_size if output_path.exists() else 0
            })
        except Exception as e:
            logger.error("Failed to save CSV file", extra={
                "output_path": str(output_path),
                "error": str(e)
            })
            raise

        # Save to database if in production mode
        if ENVIRONMENT == 'production':
            logger.info("Attempting database save for production environment")
            
            try:
                # Filter out the summary row before saving to database
                db_results = [r for r in results if r.id != 'SUMMARY']
                filtered_count = len(db_results)
                
                logger.debug("Filtering results for database", extra={
                    "original_count": len(results),
                    "filtered_count": filtered_count
                })

                if db_results:
                    self.db_manager.save_predictions(db_results)
                    logger.success("Database save completed", extra={
                        "records_saved": filtered_count
                    })
                    
                    # Log security event for database operation
                    log_security_event("database_save_completed", {
                        "records_count": filtered_count,
                        "table": "predictions"
                    })
                else:
                    logger.warning("No valid records to save to database")
                    
            except Exception as e:
                logger.error("Database save failed", extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "records_count": len(db_results) if 'db_results' in locals() else 0
                })
                # Don't raise the exception, just log it - CSV export was successful
        else:
            logger.info("Skipping database save", extra={
                "reason": "not in production mode",
                "environment": ENVIRONMENT
            })

        # Log final summary
        success_pct = successful_predictions / total_predictions * 100
        partial_pct = partial_predictions / total_predictions * 100
        failed_pct = failed_predictions / total_predictions * 100
        
        logger.success("Prediction validation summary", extra={
            "total_predictions": total_predictions,
            "success_count": successful_predictions,
            "success_percentage": success_pct,
            "partial_count": partial_predictions,
            "partial_percentage": partial_pct,
            "failed_count": failed_predictions,
            "failed_percentage": failed_pct,
            "direction_accuracy": direction_accuracy,
            "avg_error_percentage": avg_percentage_error,
            "avg_confidence": avg_confidence,
            "output_file": str(output_path)
        })

        return str(output_path)


@log_function_call(include_args=False, sanitize=True)
def main():
    """
    Main function for running prediction validation.
    
    This function provides an interactive interface for users to:
    1. Run prediction validation with custom or generated data
    2. View existing database predictions
    """
    logger.info("Starting Bitcoin Prediction Validator", extra={
        "environment": ENVIRONMENT,
        "service": "bitcoin-predictor"
    })

    # Log system startup metrics
    log_system_metrics({
        "service_started": True,
        "environment": ENVIRONMENT,
        "torch_cuda_available": torch.cuda.is_available()
    })

    # Ask user what they want to do
    logger.debug("Presenting user options")
    print("\n🔮 BITCOIN PREDICTION VALIDATOR")
    print("=" * 40)
    print("\nWhat would you like to do?")
    print("1. Run prediction validation")
    print("2. View database predictions")
    
    try:
        choice = input("Enter choice (1/2): ").strip()
        logger.info("User choice selected", extra={"choice": choice})
    except KeyboardInterrupt:
        logger.info("User cancelled operation")
        return
    except Exception as e:
        logger.error("Error getting user input", extra={"error": str(e)})
        return

    if choice == "2":
        logger.info("User selected database view option")
        
        try:
            # Initialize database manager
            db_manager = DatabaseManager()
            
            if hasattr(db_manager, 'conn') and db_manager.conn:
                logger.debug("Database connection established for viewing")
                
                try:
                    limit_input = input("How many records to view? (default 10): ").strip()
                    limit = int(limit_input) if limit_input.isdigit() else 10
                    
                    logger.info("Viewing database predictions", extra={"limit": limit})
                    db_manager.view_predictions(limit=limit)
                    
                except ValueError as e:
                    logger.warning("Invalid limit input, using default", extra={"error": str(e)})
                    db_manager.view_predictions(limit=10)
                except Exception as e:
                    logger.error("Error viewing database predictions", extra={"error": str(e)})
            else:
                logger.error("Database connection not available")
                print("❌ Database connection not available")
                
            if hasattr(db_manager, 'close'):
                db_manager.close()
                logger.debug("Database connection closed")
                
        except Exception as e:
            logger.error("Database manager initialization failed", extra={"error": str(e)})
            print(f"❌ Database error: {e}")
            
        return

    # Create validation configuration using imported function
    logger.info("Creating validation configuration")
    
    try:
        config = create_custom_config()
        logger.debug("Validation configuration created", extra={
            "config_type": type(config).__name__
        })
    except Exception as e:
        logger.error("Failed to create validation configuration", extra={"error": str(e)})
        print(f"❌ Configuration error: {e}")
        return

    try:
        # Initialize validator
        logger.info("Initializing PredictionValidator")
        validator = PredictionValidator(config=config)

        # Generate or load test data
        logger.info("Preparing test data")

        # Option to use custom data
        try:
            use_custom = input("Use custom data file? (y/n): ").lower().strip()
            logger.info("Custom data choice", extra={"use_custom": use_custom})
        except KeyboardInterrupt:
            logger.info("User cancelled data selection")
            return

        if use_custom == 'y':
            try:
                file_path = input("Enter CSV file path: ").strip()
                logger.info("Loading custom data file", extra={"file_path": file_path})
                
                test_data = load_data_from_csv(file_path)
                logger.success("Custom data loaded successfully", extra={
                    "file_path": file_path,
                    "data_shape": test_data.shape
                })
                
            except FileNotFoundError as e:
                logger.error("Custom file not found", extra={"file_path": file_path, "error": str(e)})
                print(f"❌ File not found: {file_path}")
                logger.info("Falling back to generated data")
                test_data = generate_validation_data()
                
            except Exception as e:
                logger.error("Error loading custom file", extra={"error": str(e)})
                print(f"❌ Error loading file: {e}")
                logger.info("Falling back to generated data")
                test_data = generate_validation_data()
        else:
            logger.info("Using generated validation data")
            test_data = generate_validation_data()

        # Choose prediction horizons from settings
        horizons = PREDICTION_HORIZONS[:1]  # Use first horizon from settings
        logger.info("Selected prediction horizons", extra={"horizons": horizons})

        # Run validation
        logger.info("Starting prediction validation process")
        results = validator.predict_and_validate(test_data, horizons)

        # Export results
        logger.info("Exporting validation results")
        csv_path = validator.export_to_csv(results)

        # Show sample results
        if results:
            logger.info("Displaying sample results")
            print(f"\n📋 SAMPLE RESULTS (first 5):")
            print("-" * 70)

            sample_df = pd.DataFrame([r.to_dict() for r in results[:5]])
            columns = [
                'timestamp', 'actual_price', 'predicted_price',
                'percentage_error', 'direction_correct', 'success_status'
            ]
            print(sample_df[columns].to_string(index=False))

        logger.success("Validation completed successfully", extra={
            "results_count": len(results),
            "csv_path": csv_path
        })

        print(f"\n🎉 Validation completed successfully!")
        print(f"💾 Full results available in: {csv_path}")
        print(f"\n💡 Next steps:")
        print(f"   1. Review CSV file for detailed analysis")
        print(f"   2. Import to database for tracking")
        print(f"   3. Use results to improve model")
        print(f"   4. Set up automated validation pipeline")

        # Log security event for successful completion
        log_security_event("prediction_validation_completed", {
            "results_count": len(results),
            "success": True,
            "output_file": csv_path
        })

    except Exception as e:
        logger.error("Error during validation process", extra={
            "error": str(e),
            "error_type": type(e).__name__
        })
        print(f"❌ Error during validation: {e}")
        
        output_dir_msg = f"💡 Make sure you have trained models in {OUTPUT_DIR} directory"
        print(output_dir_msg)
        logger.info("Validation failed", extra={"suggestion": output_dir_msg})
        
        # Log security event for failed operation
        log_security_event("prediction_validation_failed", {
            "error": str(e),
            "error_type": type(e).__name__
        })


if __name__ == "__main__":
    main()