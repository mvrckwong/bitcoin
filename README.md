# 🚀 Bitcoin Price Prediction System

A comprehensive machine learning system for predicting Bitcoin prices using deep learning models (LSTM/GRU) with advanced features including real-time validation, database integration, and modern task automation.

## ✨ Features

- **🧠 Advanced Neural Networks**: LSTM and GRU models for time series prediction
- **⚡ Modern Task Runner**: Streamlined workflow with Taskfile automation
- **🔄 Real-time Validation**: Automated prediction validation with performance metrics
- **💾 Database Integration**: PostgreSQL integration for storing predictions and results
- **🧪 Comprehensive Testing**: Unit, integration, and model validation tests
- **📈 Performance Optimization**: Modular architecture with 25% code reduction
- **📊 Technical Indicators**: Built-in RSI calculation and feature engineering
- **💾 Checkpointing**: Training resumption and model versioning support
- **📊 Data Visualization**: Matplotlib and Seaborn integration for analysis

## 🛠️ Quick Start

### 1. Prerequisites

- **Python >= 3.9**
- **Task** (modern task runner) - [Installation Guide](https://taskfile.dev/installation/)
- **PostgreSQL** (optional, for database features)
- **CUDA-compatible GPU** (optional, for faster training)

### 2. Install Task (Required)

Choose your preferred installation method:

```bash
# Go (if you have Go installed)
go install github.com/go-task/task/v3/cmd/task@latest

# macOS (Homebrew)
brew install go-task/tap/go-task

# Windows (Chocolatey)
choco install go-task

# Ubuntu/Debian
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin

# Manual download
# Visit: https://github.com/go-task/task/releases
```

### 3. Setup Project

```bash
# Clone the repository
git clone <repository-url>
cd bitcoin

# View all available tasks
task help

# Setup Python environment
task setup

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix/macOS:
source venv/bin/activate

# Install dependencies
task install

# Check system status
task status
```

## 🚀 Using the Task Runner

The project uses **Taskfile** for streamlined operations. Here's how to use it:

### 📋 View Available Commands

```bash
task help
```

This shows all available tasks with descriptions:

```
🚀 Bitcoin Prediction Pipeline Taskfile
========================================

📊 Training Commands:
  task train              - Run training with default settings
  task train:quick        - Quick test training (10 epochs)
  task train:production   - Full production training (200 epochs)

🔮 Prediction Commands:
  task predict            - Run prediction validation
  task predict:batch      - Run prediction validation non-interactively

🛠️  Utility Commands:
  task setup              - Set up Python environment
  task install            - Install dependencies
  task clean              - Clean output files
  task clean:all          - Clean everything including models
  task models:list        - List saved models
  task gpu:check          - Check GPU availability
  task status             - Show current status

🔧 Development Commands:
  task test:imports       - Test all imports
  task test:quick         - Run quick functionality test
```

### 🏃‍♂️ Common Workflows

#### **Quick Start (First Time Users)**

```bash
# 1. Setup environment and dependencies
task setup
task install

# 2. Check if everything works
task test:quick

# 3. Train a quick model (10 epochs)
task train:quick

# 4. Run prediction validation
task predict:batch

# 5. Check what was created
task status
task models:list
```

#### **Production Training**

```bash
# Full training with 200 epochs (takes time!)
task train:production

# Monitor system during training
task gpu:check
task status
```

#### **Model Development Cycle**

```bash
# Test imports and setup
task test:imports

# Quick training for development
task train:quick

# Validate model performance
task predict

# List saved models
task models:list

# Clean intermediate files (keep models)
task clean
```

### 📊 Training Commands

| Command | Description | Use Case |
|---------|-------------|----------|
| `task train` | Default training (100 epochs) | Standard training |
| `task train:quick` | Fast training (10 epochs) | Testing/development |
| `task train:production` | Full training (200 epochs) | Production models |

**Training outputs:**
- Models saved to `outputs/` directory
- Automatic checkpointing every 5 epochs
- Progress bars and metrics display
- Best model automatically saved

### 🔮 Prediction Commands

| Command | Description | Use Case |
|---------|-------------|----------|
| `task predict` | Interactive validation | Manual testing |
| `task predict:batch` | Non-interactive validation | Automated workflows |

**Prediction features:**
- Automatic test data generation
- Comprehensive metrics calculation
- CSV export of results
- Database integration (if configured)

### 🛠️ Utility Commands

| Command | Description | What it does |
|---------|-------------|-------------|
| `task status` | System overview | Shows models, files, GPU status |
| `task models:list` | List saved models | Shows all .pth files with sizes |
| `task gpu:check` | GPU availability | CUDA status and memory info |
| `task clean` | Clean temp files | Removes logs, CSVs (keeps models) |
| `task clean:all` | Clean everything | Removes all outputs and venv |

### 🔧 Development Commands

| Command | Description | Purpose |
|---------|-------------|---------|
| `task test:imports` | Test all imports | Verify dependencies |
| `task test:quick` | Functionality test | Quick system validation |

## 📊 Manual Usage (Advanced)

For users who prefer direct Python usage:

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
from src.predict import PredictionValidator

# Initialize validator
validator = PredictionValidator()

# Run validation
test_data = generate_validation_data()
results = validator.predict_and_validate(test_data)

# Export results
validator.export_to_csv(results)
```

### Load Existing Models

```python
from src.models import ModelLoader

# Load latest model
loader = ModelLoader()
model, model_data, config, device = loader.load_latest_model()

# Generate test data
from src.core import generate_validation_data
validation_data = generate_validation_data(
    num_samples=1000,
    start_price=50000
)
```

## 🏗️ Project Structure

```
bitcoin/
├── Taskfile.yml                 # 🚀 Task automation (NEW!)
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
├── outputs/                      # Model outputs and artifacts (auto-created)
├── logs/                         # Application logs (auto-created)
├── venv/                         # Virtual environment (auto-created)
├── setup_database.sql            # Database schema setup
├── pyproject.toml               # Project configuration and dependencies
├── pytest.ini                  # Test configuration
└── README.md                    # This file
```

## 🧪 Testing

### Using Taskfile (Recommended)

```bash
# Quick functionality test
task test:quick

# Test imports
task test:imports
```

### Using pytest (Advanced)

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

## 🔧 Configuration

### Model Configuration

The system uses predefined configurations that you can access:

- **quick_test**: 10 epochs, small batch size (for testing)
- **production**: 200 epochs, optimized settings (for production)
- **default**: 100 epochs, balanced settings

### Database Configuration

Create a `.env` file for database integration:

```env
DATABASE_URL=postgresql://username:password@localhost/bitcoin_db
ENVIRONMENT=development  # or 'production'
```

### Advanced Configuration

For custom parameters, modify the configuration in `src/main.py`:

```python
CONFIGS = {
    'custom': {
        'epochs': 50,
        'batch_size': 64,
        'sequence_length': 120,
        'hidden_size': 256,
        'learning_rate': 0.0005,
        'num_layers': 3
    }
}
```

## 📈 Performance Features

### Automated Checkpointing

- Models save every 5 epochs automatically
- Best model saved separately
- Resume training from any checkpoint
- Full training state preservation

### Progress Tracking

- Real-time training progress bars
- Validation metrics display
- GPU memory monitoring
- Automatic early stopping

### Output Management

- Organized output directory structure
- Automatic timestamp naming
- Model metadata preservation
- CSV export for analysis

## 💾 Database Integration

The system includes optional PostgreSQL integration:

### Schema Overview

- **predictions**: Main table for prediction results
  - `id`: Unique identifier (UUID)
  - `timestamp`: Prediction timestamp
  - `actual_price`, `predicted_price`: Price values
  - `absolute_error`, `percentage_error`: Error metrics
  - `direction_actual`, `direction_predicted`: Price direction
  - `confidence_score`: Model confidence
  - `model_name`: Model identifier

### Usage

```bash
# Set environment to production for database saves
export ENVIRONMENT=production
task predict:batch
```

## 🚧 Troubleshooting

### Common Issues

**Task not found:**
```bash
# Install Task first
# See installation section above
task --version  # Verify installation
```

**Import errors:**
```bash
task test:imports  # Check dependencies
task install       # Reinstall dependencies
```

**CUDA issues:**
```bash
task gpu:check     # Check GPU availability
```

**Permission errors:**
```bash
# On Windows, run as administrator
# On Unix, check directory permissions
```

### Getting Help

1. **View all tasks**: `task help`
2. **Check system status**: `task status`
3. **Test functionality**: `task test:quick`
4. **Check dependencies**: `task test:imports`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Follow PEP 8 style guidelines
4. Add comprehensive tests for new functionality
5. Test with: `task test:quick`
6. Update documentation as needed
7. Commit changes: `git commit -m 'Add amazing feature'`
8. Push to branch: `git push origin feature/amazing-feature`
9. Open a Pull Request

### Development Workflow

```bash
# Setup development environment
task setup
task install

# Make changes
# ... edit code ...

# Test changes
task test:quick
task train:quick

# Clean up
task clean
```

## 📚 Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **matplotlib/seaborn**: Data visualization
- **tqdm**: Progress bars
- **pathlib**: Path handling

### Development Dependencies
- **Task**: Modern task runner
- **pytest**: Testing framework

### Optional Dependencies
- **SQLModel**: Database ORM
- **psycopg2**: PostgreSQL adapter
- **python-dotenv**: Environment variable management

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Task**: Modern task runner for streamlined workflows
- **PyTorch**: Deep learning capabilities
- **scikit-learn**: Machine learning preprocessing and metrics
- **pytest**: Comprehensive testing framework

## 📞 Support

### Quick Help

1. **First time setup**: `task help` → `task setup` → `task install` → `task test:quick`
2. **Training issues**: `task status` → `task gpu:check` → `task train:quick`
3. **Model problems**: `task models:list` → `task clean` → retrain

### Detailed Support

For questions, issues, or contributions:
1. Check the task help: `task help`
2. Review system status: `task status`
3. Check the [testing documentation](tests/README.md)
4. Open an issue on the repository
5. Review existing task workflows for usage examples

---

**⚠️ Important Note**: This system is designed for educational and research purposes. Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors.

**🎯 Pro Tip**: Start with `task help` and `task train:quick` for the best first experience!

