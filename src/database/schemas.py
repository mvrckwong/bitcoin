"""
Database schemas for Bitcoin prediction system.

This module defines all SQLModel classes for database operations including:
- Prediction data models
- Database table definitions
- Data validation schemas
"""

import uuid
from typing import Optional
from sqlmodel import SQLModel, Field

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