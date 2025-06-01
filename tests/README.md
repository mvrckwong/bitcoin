# Bitcoin Trading Model Tests

This directory contains all tests for the Bitcoin trading model project, organized into different categories for better maintainability and clarity.

## Directory Structure

```
tests/
├── __init__.py                     # Test package initialization
├── README.md                       # This file
├── unit/                          # Unit tests
│   └── __init__.py
├── integration/                   # Integration tests
│   ├── __init__.py
│   └── test_module_imports.py     # Tests module imports and refactoring
└── model_validation/              # Model validation tests
    ├── __init__.py
    └── test_model_validation.py   # Comprehensive model testing
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Tests for individual components and functions in isolation
- Fast-running tests that don't require external dependencies
- Mock external dependencies when necessary

### Integration Tests (`tests/integration/`)
- Tests for module interactions and system integration
- Tests import functionality and module refactoring
- Ensures different parts of the system work together

### Model Validation Tests (`tests/model_validation/`)
- Comprehensive model performance testing
- Model loading, prediction accuracy, and validation metrics
- Slow-running tests marked with `@pytest.mark.slow`

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only  
pytest tests/integration/

# Model validation tests only
pytest tests/model_validation/
```

### Run Tests with Markers
```bash
# Run only slow tests
pytest -m slow

# Run only integration tests
pytest -m integration

# Run only model validation tests
pytest -m model_validation

# Skip slow tests
pytest -m "not slow"
```

### Run Specific Test Files
```bash
# Test module imports
pytest tests/integration/test_module_imports.py

# Test model validation (this is slow)
pytest tests/model_validation/test_model_validation.py
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

## Test Configuration

The tests are configured via `pytest.ini` in the project root with the following settings:

- **Test Discovery**: Automatically finds tests in the `tests/` directory
- **Python Path**: Adds `src/` to the Python path for imports
- **Markers**: Defines test markers for categorization
- **Output**: Verbose output with short traceback format

## Test Markers

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests  
- `@pytest.mark.model_validation`: Model validation tests
- `@pytest.mark.slow`: Tests that take a long time to run

## Adding New Tests

### Unit Tests
1. Create test files in `tests/unit/` following the pattern `test_*.py`
2. Import the module you want to test from `src/`
3. Use `@pytest.mark.unit` decorator
4. Write focused tests for individual functions/classes

### Integration Tests  
1. Create test files in `tests/integration/` following the pattern `test_*.py`
2. Use `@pytest.mark.integration` decorator
3. Test interactions between modules and components

### Model Validation Tests
1. Create test files in `tests/model_validation/` following the pattern `test_*.py`
2. Use `@pytest.mark.model_validation` and `@pytest.mark.slow` decorators
3. Test model performance, accuracy, and validation metrics

## Example Test Structure

```python
import pytest
from src.your_module import YourClass

@pytest.mark.unit
def test_your_function():
    """Test description"""
    # Arrange
    input_data = ...
    
    # Act
    result = your_function(input_data)
    
    # Assert
    assert result == expected_result
```

## Notes

- Model validation tests require trained models to be present
- Some tests may be skipped if dependencies are not available
- Use appropriate markers to categorize your tests
- Keep tests focused and independent of each other 