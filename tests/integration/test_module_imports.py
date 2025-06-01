"""
Test file to verify the refactored module structure works correctly.
"""

import pytest

# Test settings imports
from settings import ENVIRONMENT, ValidationConfig, create_custom_config
from settings.config import ModelConfig, DataConfig

# Test database imports  
from database import DatabaseManager, PredictionResult, Prediction
from database.schemas import PredictionBase


@pytest.mark.integration
def test_settings_imports():
    """Test that settings imports work correctly"""
    print("🧪 Testing settings imports...")
    
    # Test environment
    assert ENVIRONMENT is not None
    print(f"✅ Environment: {ENVIRONMENT}")
    
    # Test ValidationConfig
    config = ValidationConfig()
    assert hasattr(config, 'price_threshold_percent')
    print(f"✅ ValidationConfig: threshold={config.price_threshold_percent}%")
    
    # Test ModelConfig
    model_config = ModelConfig()
    assert hasattr(model_config, 'sequence_length')
    print(f"✅ ModelConfig: sequence_length={model_config.sequence_length}")


@pytest.mark.integration
def test_database_imports():
    """Test that database imports work correctly"""
    print("🧪 Testing database imports...")
    
    # Test database classes
    db_manager = DatabaseManager()
    assert db_manager is not None
    print(f"✅ DatabaseManager initialized")
    
    # Test schema creation
    result = PredictionResult.create(
        timestamp="2024-01-01",
        actual_price=50000.0,
        predicted_price=49000.0,
        absolute_error=1000.0,
        percentage_error=2.0,
        direction_actual="UP",
        direction_predicted="UP", 
        direction_correct=True,
        within_threshold=True,
        success_status="SUCCESS",
        confidence_score=0.85,
        model_name="lstm_test",
        prediction_horizon="1day",
        features_used="close,volume",
        sequence_length=60
    )
    assert result.success_status == "SUCCESS"
    print(f"✅ PredictionResult created: {result.success_status}")


@pytest.mark.integration
def test_all_imports():
    """Comprehensive test that all imports work correctly"""
    print("\n🎉 All imports working correctly!")
    print("\n📦 New module structure:")
    print("   settings/ - Configuration management")
    print("   database/ - Database operations & schemas")
    print("   📝 Next: Extract data, validation, and utils modules")


if __name__ == "__main__":
    # For direct execution
    test_settings_imports()
    test_database_imports() 
    test_all_imports() 