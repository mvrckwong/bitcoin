"""
models.py - Shared model components for Bitcoin prediction

This module contains the core model architecture and dataset classes used across
the Bitcoin prediction system.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path

class BitcoinPredictor(nn.Module):
    """
    Modular neural network that can handle variable input features
    """
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2, 
                 model_type: str = 'lstm'):
        super(BitcoinPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type.lower()
        
        # Choose architecture
        if self.model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.model_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError("model_type must be 'lstm' or 'gru'")
        
        # Attention mechanism (optional enhancement)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # RNN forward pass
        rnn_out, _ = self.rnn(x)
        
        # Optional: Add attention
        # attended_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
        # rnn_out = rnn_out + attended_out  # Residual connection
        
        # Use the last output
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Only squeeze the last dimension (feature dimension), keep batch dimension
        return out.squeeze(-1)

class BitcoinDataset(Dataset):
    """
    Flexible dataset class that can handle price-only or multi-feature data
    """
    def __init__(self, data: pd.DataFrame, sequence_length: int = 60, 
                 target_col: str = 'close', feature_cols: Optional[List[str]] = None,
                 scalers: Optional[Dict] = None, include_timestamps: bool = False):
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.include_timestamps = include_timestamps
        
        # If no feature columns specified, use only the target column
        if feature_cols is None:
            feature_cols = [target_col]
        
        self.feature_cols = feature_cols
        self.data = data[feature_cols + [target_col] if target_col not in feature_cols else feature_cols].copy()
        
        # Handle scalers
        if scalers:
            self.scalers = scalers
            self.scaled_data = self.data.copy()
            for col in self.data.columns:
                if col in self.scalers:
                    self.scaled_data[col] = self.scalers[col].transform(self.data[[col]])
        else:
            # Create new scalers
            self.scalers = {}
            self.scaled_data = self.data.copy()
            for col in self.data.columns:
                scaler = MinMaxScaler()
                self.scaled_data[col] = scaler.fit_transform(self.data[[col]])
                self.scalers[col] = scaler
        
        # Create sequences
        if include_timestamps:
            self.sequences, self.targets, self.timestamps = self._create_sequences_with_timestamps()
        else:
            self.sequences, self.targets = self._create_sequences()
    
    def _create_sequences(self):
        sequences = []
        targets = []
        
        print("🔄 Creating sequences...")
        data_range = range(len(self.scaled_data) - self.sequence_length)
        
        for i in tqdm(data_range, desc="Processing sequences", unit="seq"):
            # Features for sequence
            seq_features = self.scaled_data[self.feature_cols].iloc[i:i+self.sequence_length].values
            sequences.append(seq_features)
            
            # Target (next price)
            target = self.scaled_data[self.target_col].iloc[i+self.sequence_length]
            targets.append(target)
        
        print("📦 Converting to tensors...")
        # Convert to numpy arrays first, then to tensors (more efficient)
        sequences_array = np.array(sequences, dtype=np.float32)
        targets_array = np.array(targets, dtype=np.float32)
        
        return torch.FloatTensor(sequences_array), torch.FloatTensor(targets_array)
    
    def _create_sequences_with_timestamps(self):
        sequences = []
        targets = []
        timestamps = []
        
        print("🔄 Creating sequences with timestamps...")
        data_range = range(len(self.scaled_data) - self.sequence_length)
        
        for i in tqdm(data_range, desc="Processing sequences", unit="seq"):
            # Features for sequence
            seq_features = self.scaled_data[self.feature_cols].iloc[i:i+self.sequence_length].values
            sequences.append(seq_features)
            
            # Target (next price)
            target = self.scaled_data[self.target_col].iloc[i+self.sequence_length]
            targets.append(target)
            
            # Store timestamp for the prediction
            if 'date' in self.data.columns:
                timestamp = self.data['date'].iloc[i+self.sequence_length]
            else:
                timestamp = i + self.sequence_length
            timestamps.append(timestamp)
        
        print("📦 Converting to tensors...")
        sequences_array = np.array(sequences, dtype=np.float32)
        targets_array = np.array(targets, dtype=np.float32)
        
        return torch.FloatTensor(sequences_array), torch.FloatTensor(targets_array), timestamps
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.include_timestamps:
            return self.sequences[idx], self.targets[idx], self.timestamps[idx]
        return self.sequences[idx], self.targets[idx]
    
    def inverse_transform_target(self, scaled_target):
        """Convert scaled target back to original price"""
        return self.scalers[self.target_col].inverse_transform([[scaled_target]])[0][0]

def calculate_rsi(prices, window=14):
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def load_trained_model(model_path: str, device: Optional[torch.device] = None) -> Tuple[BitcoinPredictor, Dict]:
    """
    Load a trained model from file
    
    Args:
        model_path: Path to the saved model
        device: Device to load model on
        
    Returns:
        Tuple of (model, model_data)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model_data = torch.load(model_file, map_location=device, weights_only=False)
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
    
    print(f"✅ Model loaded from: {model_file}")
    print(f"   📊 Features: {config['feature_cols']}")
    print(f"   🎯 Best validation loss: {model_data['training_metrics']['best_val_loss']:.6f}")
    
    return model, model_data 