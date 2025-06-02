# Bitcoin Price Prediction System

A comprehensive machine learning system for predicting Bitcoin prices using deep learning models (LSTM/GRU) with advanced features including real-time validation, database integration, and performance optimization.

## 🚀 Features

- **Advanced Neural Networks**: LSTM and GRU models for time series prediction
- **Real-time Validation**: Automated prediction validation with performance metrics
- **Database Integration**: PostgreSQL integration for storing predictions and results
- **Comprehensive Testing**: Unit, integration, and model validation tests
- **Performance Optimization**: Modular architecture with 25% code reduction
- **Technical Indicators**: Built-in RSI calculation and feature engineering
- **Checkpointing**: Training resumption and model versioning support
- **Data Visualization**: Matplotlib and Seaborn integration for analysis

## 🛠️ Installation

### Prerequisites

- Python >= 3.9
- PostgreSQL (optional, for database features)
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd bitcoin
   ```

2. **Install dependencies**:
   ```bash
   # Install uv (if not already installed)
   pip install uv
   
   # Install project dependencies
   uv sync
   ```

3. **Activate virtual environment**:
   ```bash
   # On Windows
   .venv\Scripts\activate
   
   # On Unix/macOS
   source .venv/bin/activate
   ```

4. **Setup database** (optional):
   ```bash
   psql -U your_username -d your_database -f setup_database.sql
   ```

## 📊 Usage

### Basic Training and Prediction

```python
from src.main import BitcoinPredictionPipeline
import pandas as pd

# Load your data
data = pd.read_csv('bitcoin_data.csv')

# Initialize pipeline
pipeline = BitcoinPredictionPipeline(
    sequence_length=60,
    hidden_size=128,
    num_layers=2,
    learning_rate=0.001,
    model_type='lstm'
)

# Prepare data and train
pipeline.prepare_data(data, target_col='close')
pipeline.train(epochs=100, batch_size=32)

# Make predictions
sequence = data.tail(60).values
prediction = pipeline.predict(sequence)
```

### Model Validation

```python
from src.predict import validate_model

# Validate a trained model
results = validate_model(
    model_path='path/to/model.pth',
    csv_file='validation_data.csv'
)
print(f"Validation Accuracy: {results['accuracy']:.2%}")
```

### Using Pre-built Components

```python
# Load models
from src.models import ModelLoader
loader = ModelLoader()
model = loader.load_latest_model()

# Generate validation data
from src.core import generate_validation_data
validation_data = generate_validation_data(
    num_samples=1000,
    start_price=50000
)
```

## 🏗️ Project Structure

```
bitcoin/
├── src/                          # Main source code
│   ├── main.py                   # Training pipeline and main functionality
│   ├── predict.py                # Model validation and prediction utilities
│   ├── models/                   # Model definitions and utilities
│   │   ├── models.py             # BitcoinPredictor, BitcoinDataset classes
│   │   ├── model_loader.py       # Model loading utilities
│   │   └── __init__.py           # Module exports
│   ├── core/                     # Core utilities and data handling
│   │   ├── data_utils.py         # Data processing and validation utilities
│   │   ├── setup_path.py         # Path configuration
│   │   └── __init__.py           # Module exports
│   ├── database/                 # Database integration
│   └── settings/                 # Configuration settings
├── tests/                        # Comprehensive test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── model_validation/         # Model validation tests
│   └── README.md                 # Testing documentation
├── .outputs/                     # Model outputs and artifacts
├── .logs/                        # Application logs
├── setup_database.sql            # Database schema setup
├── pyproject.toml               # Project configuration and dependencies
├── pytest.ini                  # Test configuration
└── README.md                    # This file
```

## 🧪 Testing

The project includes a comprehensive testing suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/model_validation/  # Model validation tests

# Run with coverage
pytest --cov=src tests/
```

## 📈 Performance Optimizations

This project has been optimized for performance and maintainability:

- **25% Code Reduction**: Eliminated redundant code through modularization
- **Modular Architecture**: Separated concerns across dedicated modules
- **Reusable Components**: Shared utilities across training/validation/testing
- **Enhanced Documentation**: Comprehensive docstrings following PEP 257
- **Type Hints**: Full type annotation support

See [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) for detailed optimization information.

## 🔧 Configuration

### Environment Variables

Create a `.env` file for database configuration:

```env
DATABASE_URL=postgresql://username:password@localhost/bitcoin_db
```

### Model Configuration

Customize model parameters in your training script:

```python
config = {
    'sequence_length': 60,      # Number of time steps to look back
    'hidden_size': 128,         # LSTM/GRU hidden layer size
    'num_layers': 2,            # Number of recurrent layers
    'learning_rate': 0.001,     # Learning rate for optimization
    'model_type': 'lstm'        # 'lstm' or 'gru'
}
```

## 📊 Database Schema

The system stores predictions in a PostgreSQL database with the following schema:

- **predictions**: Main table storing prediction results
  - `id`: Unique identifier (UUID)
  - `timestamp`: Prediction timestamp
  - `actual_price`, `predicted_price`: Price values
  - `absolute_error`, `percentage_error`: Error metrics
  - `direction_actual`, `direction_predicted`: Price direction
  - `confidence_score`: Model confidence
  - `model_name`: Model identifier
  - Additional metadata fields

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow PEP 8 style guidelines
4. Add comprehensive tests for new functionality
5. Update documentation as needed
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📚 Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **matplotlib/seaborn**: Data visualization
- **SQLModel**: Database ORM
- **psycopg2**: PostgreSQL adapter
- **python-dotenv**: Environment variable management

### Development Dependencies
- **pytest**: Testing framework
- **sphinx**: Documentation generation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with PyTorch for deep learning capabilities
- Utilizes scikit-learn for preprocessing and metrics
- Database integration powered by SQLModel and PostgreSQL
- Testing framework provided by pytest

## 📞 Support

For questions, issues, or contributions, please:
1. Check the [documentation](tests/README.md) in the tests directory
2. Review the [optimization summary](docs/OPTIMIZATION_SUMMARY.md)
3. Open an issue on the repository
4. Review existing test cases for usage examples

---

**Note**: This system is designed for educational and research purposes. Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors.

