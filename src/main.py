import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from models.models import BitcoinPredictor, BitcoinDataset, calculate_rsi
from core.setup_path import OUTPUT_DIR

class BitcoinPredictionPipeline:
    """
    Complete pipeline for training and prediction with checkpointing
    """
    def __init__(self, sequence_length: int = 60, hidden_size: int = 128, 
                 num_layers: int = 2, learning_rate: float = 0.001,
                 model_type: str = 'lstm', output_dir: str = None):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.model_type = model_type
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.dataset = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training state for checkpointing
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
    def get_model_name(self, prefix: str = "bitcoin_model") -> str:
        """Generate a unique model name with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"
        
    def save_checkpoint(self, epoch: int, is_best: bool = False, 
                       checkpoint_name: Optional[str] = None) -> Path:
        """
        Save training checkpoint
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            checkpoint_name: Optional custom name for checkpoint
            
        Returns:
            Path: Path to saved checkpoint
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{epoch}"
            
        checkpoint_path = self.output_dir / f"{checkpoint_name}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'model_config': {
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'learning_rate': self.learning_rate,
                'model_type': self.model_type,
                'input_size': len(self.dataset.feature_cols) if self.dataset else None,
                'feature_cols': self.dataset.feature_cols if self.dataset else None
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_model_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_model_path)
            
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load training checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dict: Checkpoint data
        """
        checkpoint_file = self.output_dir / checkpoint_path if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
            
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        
        # Load model state
        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"✅ Loaded checkpoint from epoch {self.current_epoch}")
        print(f"   📊 Best validation loss: {self.best_val_loss:.6f}")
        
        return checkpoint
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints in output directory"""
        if not self.output_dir.exists():
            return []
            
        checkpoints = [f.name for f in self.output_dir.glob('*.pth')]
        checkpoints.sort()  # Sort by name (which includes timestamp)
        return checkpoints
    
    def save_model_final(self, model_name: Optional[str] = None) -> Path:
        """
        Save final trained model
        
        Args:
            model_name: Optional custom name for the model
            
        Returns:
            Path: Path to saved model
        """
        if model_name is None:
            model_name = self.get_model_name("final_model")
            
        model_path = self.output_dir / f"{model_name}.pth"
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'model_type': self.model_type,
                'input_size': len(self.dataset.feature_cols),
                'feature_cols': self.dataset.feature_cols
            },
            'training_metrics': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss
            },
            'dataset_scalers': {
                col: scaler for col, scaler in self.dataset.scalers.items()
            }
        }
        
        torch.save(model_data, model_path)
        print(f"💾 Final model saved: {model_path}")
        print(f"   💡 Note: Model contains sklearn objects, load with weights_only=False")
        return model_path
        
    def prepare_data(self, data: pd.DataFrame, target_col: str = 'close', 
                    feature_cols: Optional[List[str]] = None, train_split: float = 0.8):
        """
        Prepare data for training with progress indicators
        """
        print("🚀 Starting data preparation...")
        
        # Create dataset
        print("📊 Creating dataset...")
        self.dataset = BitcoinDataset(data, self.sequence_length, target_col, feature_cols)
        
        # Split data
        print("✂️ Splitting data...")
        train_size = int(len(self.dataset) * train_split)
        test_size = len(self.dataset) - train_size
        
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, test_size]
        )
        
        # Initialize model with correct input size
        print("🏗️ Initializing model...")
        input_size = len(self.dataset.feature_cols)
        self.model = BitcoinPredictor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            model_type=self.model_type
        ).to(self.device)
        
        print(f"✅ Data preparation complete!")
        print(f"   📈 Model input size: {input_size}")
        print(f"   🎯 Features: {self.dataset.feature_cols}")
        print(f"   🏋️ Training samples: {len(self.train_dataset)}")
        print(f"   🧪 Test samples: {len(self.test_dataset)}")
        print(f"   💻 Device: {self.device}")
        print(f"   📁 Output directory: {self.output_dir}")
        print()  # Empty line for readability
        
    def train(self, epochs: int = 100, batch_size: int = 32, patience: int = 10,
              save_checkpoint_every: int = 10, resume_from_checkpoint: Optional[str] = None):
        """
        Train the model with early stopping, progress bars, and checkpointing
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            save_checkpoint_every: Save checkpoint every N epochs
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        start_epoch = 0
        patience_counter = 0
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            checkpoint_path = self.output_dir / resume_from_checkpoint
            if checkpoint_path.exists():
                self.load_checkpoint(resume_from_checkpoint)
                start_epoch = self.current_epoch + 1
                print(f"🔄 Resuming training from epoch {start_epoch}")
            else:
                print(f"⚠️ Checkpoint not found: {checkpoint_path}")
                print("🆕 Starting fresh training")
        
        # Main epoch progress bar
        epoch_pbar = tqdm(range(start_epoch, epochs), desc="Training Progress", unit="epoch")
        
        for epoch in epoch_pbar:
            self.current_epoch = epoch
            
            # Training
            self.model.train()
            train_loss = 0
            
            # Training batch progress bar
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", 
                            leave=False, unit="batch")
            
            for sequences, targets in train_pbar:
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(sequences)
                loss = criterion(predictions, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                batch_loss = loss.item()
                train_loss += batch_loss
                
                # Update training progress bar
                train_pbar.set_postfix({'loss': f'{batch_loss:.6f}'})
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            # Validation batch progress bar
            val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", 
                          leave=False, unit="batch")
            
            with torch.no_grad():
                for sequences, targets in val_pbar:
                    sequences, targets = sequences.to(self.device), targets.to(self.device)
                    predictions = self.model(sequences)
                    loss = criterion(predictions, targets)
                    batch_loss = loss.item()
                    val_loss += batch_loss
                    
                    # Update validation progress bar
                    val_pbar.set_postfix({'loss': f'{batch_loss:.6f}'})
            
            train_loss /= len(train_loader)
            val_loss /= len(test_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            # Check for best model
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                is_best = True
                best_indicator = "🏆"
            else:
                patience_counter += 1
                best_indicator = ""
            
            # Save checkpoint periodically and when best
            if (epoch + 1) % save_checkpoint_every == 0 or is_best:
                checkpoint_path = self.save_checkpoint(epoch, is_best)
                if not is_best:  # Don't print for best model (printed by save_checkpoint)
                    tqdm.write(f"💾 Checkpoint saved: {checkpoint_path.name}")
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.6f}',
                'val_loss': f'{val_loss:.6f}',
                'best': best_indicator,
                'patience': f'{patience_counter}/{patience}'
            })
            
            # Early stopping
            if patience_counter >= patience:
                epoch_pbar.set_description(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        best_model_path = self.output_dir / "best_model.pth"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        epoch_pbar.close()
        
        # Save final model
        final_model_path = self.save_model_final()
        
        print(f"\n✅ Training completed! Best validation loss: {self.best_val_loss:.6f}")
        print(f"📁 Models saved in: {self.output_dir}")
        return self.train_losses, self.val_losses
    
    def predict(self, sequence: np.ndarray) -> float:
        """
        Make a single prediction
        """
        self.model.eval()
        with torch.no_grad():
            if len(sequence.shape) == 2:
                sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
            
            sequence_tensor = torch.FloatTensor(sequence).to(self.device)
            prediction_tensor = self.model(sequence_tensor)
            
            # Handle both single and batch predictions
            if prediction_tensor.dim() == 0:  # Scalar
                scaled_prediction = prediction_tensor.cpu().numpy().item()
            else:  # Batch
                scaled_prediction = prediction_tensor.cpu().numpy()[0]
            
            # Convert back to original scale
            prediction = self.dataset.inverse_transform_target(scaled_prediction)
            
        return prediction
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model performance with progress bar
        """
        test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        
        predictions = []
        actuals = []
        
        self.model.eval()
        
        # Evaluation progress bar
        eval_pbar = tqdm(test_loader, desc="Evaluating model", unit="batch")
        
        with torch.no_grad():
            for sequences, targets in eval_pbar:
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                preds = self.model(sequences)
                
                predictions.extend(preds.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
                
                # Update progress bar with batch info
                eval_pbar.set_postfix({'samples': len(predictions)})
        
        # Convert back to original scale
        print("📊 Converting predictions to original scale...")
        predictions = [self.dataset.inverse_transform_target(p) for p in tqdm(predictions, desc="Converting predictions", leave=False)]
        actuals = [self.dataset.inverse_transform_target(a) for a in tqdm(actuals, desc="Converting actuals", leave=False)]
        
        # Calculate metrics
        print("📈 Calculating metrics...")
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        mse = np.mean((np.array(predictions) - np.array(actuals))**2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }

def validate_config(epochs: int, batch_size: int, sequence_length: int, 
                   hidden_size: int, learning_rate: float, num_layers: int) -> bool:
    """
    Validate configuration parameters
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        sequence_length: Length of input sequences
        hidden_size: Size of hidden layers
        learning_rate: Learning rate for optimizer
        num_layers: Number of RNN layers
        
    Returns:
        bool: True if config is valid, False otherwise
    """
    issues = []
    
    if epochs <= 0:
        issues.append("epochs must be positive")
    if batch_size <= 0:
        issues.append("batch_size must be positive")
    if sequence_length <= 0:
        issues.append("sequence_length must be positive")
    if hidden_size <= 0:
        issues.append("hidden_size must be positive")
    if learning_rate <= 0:
        issues.append("learning_rate must be positive")
    if num_layers <= 0:
        issues.append("num_layers must be positive")
    
    # Warnings for potentially problematic values
    if learning_rate > 0.1:
        issues.append("WARNING: learning_rate > 0.1 might be too high")
    if sequence_length > 365:
        issues.append("WARNING: sequence_length > 365 might cause memory issues")
    if hidden_size > 1024:
        issues.append("WARNING: hidden_size > 1024 might be computationally expensive")
    
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True

def example_usage(epochs: int = 50, batch_size: int = 32, sequence_length: int = 60, 
                 hidden_size: int = 128, learning_rate: float = 0.001, num_layers: int = 2,
                 resume_checkpoint: Optional[str] = None):
    """
    Example of how to use the pipeline for different scenarios with checkpointing
    """
    
    print("=" * 60)
    print("🚀 BITCOIN PRICE PREDICTION PIPELINE WITH CHECKPOINTING")
    print("=" * 60)
    
    # Phase 1: Price-only prediction
    print("\n" + "=" * 30)
    print("📈 Phase 1: Price-only prediction")
    print("=" * 30)
    print(f"⚙️ Config: epochs={epochs}, batch_size={batch_size}, seq_len={sequence_length}")
    print(f"         hidden_size={hidden_size}, lr={learning_rate}, layers={num_layers}")
    
    # Generate sample data (replace with real Bitcoin data)
    print("\n📊 Generating sample data...")
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    prices = 30000 + np.cumsum(np.random.randn(len(dates)) * 100)
    
    df_simple = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    print(f"✅ Generated {len(df_simple)} data points")
    
    # Initialize pipeline with custom parameters and output directory
    pipeline = BitcoinPredictionPipeline(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
        output_dir=OUTPUT_DIR / 'price_only'
    )
    
    # Show existing checkpoints
    existing_checkpoints = pipeline.list_checkpoints()
    if existing_checkpoints:
        print(f"\n📋 Found {len(existing_checkpoints)} existing checkpoints:")
        for cp in existing_checkpoints[-3:]:  # Show last 3
            print(f"   📄 {cp}")
    
    # Train on price only
    pipeline.prepare_data(df_simple, target_col='close', feature_cols=['close'])
    train_losses, val_losses = pipeline.train(
        epochs=epochs, 
        batch_size=batch_size,
        save_checkpoint_every=5,  # Save checkpoint every 5 epochs
        resume_from_checkpoint=resume_checkpoint
    )
    
    # Evaluate
    print("\n📊 Evaluating price-only model...")
    metrics = pipeline.evaluate()
    print("📋 Price-only metrics:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Phase 2: Multi-feature prediction
    print("\n" + "=" * 30)
    print("🎯 Phase 2: Multi-feature prediction")
    print("=" * 30)
    
    # Add more features (replace with real data)
    print("🔧 Engineering features...")
    df_enhanced = df_simple.copy()
    
    with tqdm(total=4, desc="Adding features") as pbar:
        df_enhanced['volume'] = np.random.randint(1000, 10000, len(df_enhanced))
        pbar.update(1)
        
        df_enhanced['market_cap'] = df_enhanced['close'] * 19000000  # Approximate circulating supply
        pbar.update(1)
        
        df_enhanced['volatility'] = df_enhanced['close'].rolling(20).std().fillna(0)
        pbar.update(1)
        
        # Add technical indicators
        df_enhanced['sma_20'] = df_enhanced['close'].rolling(20).mean().fillna(df_enhanced['close'])
        df_enhanced['rsi'] = calculate_rsi(df_enhanced['close']).fillna(50)
        pbar.update(1)
    
    # Select features for multi-feature model
    feature_cols = ['close', 'volume', 'volatility', 'sma_20', 'rsi']
    print(f"✅ Using features: {feature_cols}")
    
    # Reinitialize pipeline for multi-feature with larger architecture
    pipeline_enhanced = BitcoinPredictionPipeline(
        sequence_length=sequence_length,
        hidden_size=hidden_size * 2,  # Larger hidden size for multi-feature
        num_layers=num_layers + 1,    # Extra layer for complexity
        learning_rate=learning_rate,
        output_dir=OUTPUT_DIR / 'multi_feature'
    )
    pipeline_enhanced.prepare_data(df_enhanced, target_col='close', feature_cols=feature_cols)
    
    # Train enhanced model
    train_losses_enhanced, val_losses_enhanced = pipeline_enhanced.train(
        epochs=epochs, 
        batch_size=batch_size,
        save_checkpoint_every=5
    )
    
    # Evaluate enhanced model
    print("\n📊 Evaluating multi-feature model...")
    metrics_enhanced = pipeline_enhanced.evaluate()
    print("📋 Multi-feature metrics:")
    for metric, value in metrics_enhanced.items():
        print(f"   {metric}: {value:.4f}")
    
    # Compare results
    print("\n" + "=" * 30)
    print("📊 MODEL COMPARISON")
    print("=" * 30)
    print(f"{'Metric':<10} {'Price-only':<12} {'Multi-feature':<15} {'Improvement':<12}")
    print("-" * 50)
    
    for metric in metrics.keys():
        price_only = metrics[metric]
        multi_feat = metrics_enhanced[metric]
        improvement = ((price_only - multi_feat) / price_only) * 100
        improvement_str = f"{improvement:+.1f}%" if improvement != 0 else "0.0%"
        print(f"{metric:<10} {price_only:<12.4f} {multi_feat:<15.4f} {improvement_str:<12}")
    
    # Show saved models
    print("\n" + "=" * 30)
    print("💾 SAVED MODELS")
    print("=" * 30)
    
    for pipeline_name, pl in [("Price-only", pipeline), ("Multi-feature", pipeline_enhanced)]:
        print(f"\n📁 {pipeline_name} models:")
        checkpoints = pl.list_checkpoints()
        for cp in checkpoints[-3:]:  # Show last 3 checkpoints
            print(f"   📄 {cp}")
    
    print("\n🎉 Pipeline execution completed successfully!")
    print("\n💡 To resume training from a checkpoint, use:")
    print("   pipeline.train(epochs=100, resume_from_checkpoint='checkpoint_epoch_50.pth')")
    print("\n💡 To load a trained model for inference:")
    print(f"   model, data = load_trained_model('{OUTPUT_DIR}/price_only/final_model_*.pth')")
    
    return pipeline, pipeline_enhanced

if __name__ == "__main__":
    print("🔧 BITCOIN PREDICTION CONFIGURATION WITH CHECKPOINTING")
    print("=" * 55)
    
    # Configuration - Modify these hyperparameters as needed
    EPOCHS = 100
    BATCH_SIZE = 32
    SEQUENCE_LENGTH = 60
    HIDDEN_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_LAYERS = 2
    
    # Checkpointing configuration
    RESUME_FROM_CHECKPOINT = None  # Set to checkpoint filename to resume training
    # Example: RESUME_FROM_CHECKPOINT = "checkpoint_epoch_25.pth"
    
    # Optional: Different configs for experimentation
    CONFIGS = {
        'quick_test': {
            'epochs': 10,
            'batch_size': 64,
            'sequence_length': 30,
            'hidden_size': 64,
            'learning_rate': 0.01,
            'num_layers': 1,
            'resume_checkpoint': None
        },
        'production': {
            'epochs': 200,
            'batch_size': 16,
            'sequence_length': 120,
            'hidden_size': 256,
            'learning_rate': 0.0005,
            'num_layers': 3,
            'resume_checkpoint': None
        }
    }
    
    # Choose configuration
    USE_CONFIG = None  # Set to 'quick_test' or 'production' to use preset configs
    
    # Create outputs directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    if USE_CONFIG and USE_CONFIG in CONFIGS:
        config = CONFIGS[USE_CONFIG]
        print(f"📋 Using {USE_CONFIG} configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        # Validate configuration
        config_copy = config.copy()
        resume_checkpoint = config_copy.pop('resume_checkpoint', None)
        
        if validate_config(**config_copy):
            print("✅ Configuration validated successfully")
            simple_pipeline, enhanced_pipeline = example_usage(**config)
        else:
            print("❌ Configuration validation failed")
    else:
        print("📋 Using manual configuration:")
        print(f"   EPOCHS: {EPOCHS}")
        print(f"   BATCH_SIZE: {BATCH_SIZE}")
        print(f"   SEQUENCE_LENGTH: {SEQUENCE_LENGTH}")
        print(f"   HIDDEN_SIZE: {HIDDEN_SIZE}")
        print(f"   LEARNING_RATE: {LEARNING_RATE}")
        print(f"   NUM_LAYERS: {NUM_LAYERS}")
        print(f"   RESUME_FROM_CHECKPOINT: {RESUME_FROM_CHECKPOINT}")
        
        # Validate configuration
        if validate_config(EPOCHS, BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE, LEARNING_RATE, NUM_LAYERS):
            print("✅ Configuration validated successfully")
            
            # Run with manual configuration
            simple_pipeline, enhanced_pipeline = example_usage(
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                sequence_length=SEQUENCE_LENGTH,
                hidden_size=HIDDEN_SIZE,
                learning_rate=LEARNING_RATE,
                num_layers=NUM_LAYERS,
                resume_checkpoint=RESUME_FROM_CHECKPOINT
            )
        else:
            print("❌ Configuration validation failed") 