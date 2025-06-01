"""
Database manager for Bitcoin prediction system.

This module handles database connections, operations, and data persistence.
"""

import uuid
from typing import List
import pandas as pd
from sqlmodel import SQLModel, Session, create_engine, select

from settings.config import ENVIRONMENT, DATABASE_URL
from .schemas import Prediction, PredictionResult

# SQLModel database setup
engine = create_engine(DATABASE_URL)

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