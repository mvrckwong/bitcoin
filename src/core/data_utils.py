"""
Data utilities for Bitcoin prediction system.

This module provides utilities for generating test data, data preprocessing,
and other data-related operations used across the prediction system.
"""

import numpy as np
import pandas as pd

from models.models import calculate_rsi
from settings import DataConfig, MIN_PRICE, MAX_PRICE


def generate_validation_data(days: int = 200,
                             start_price: float = 50000) -> pd.DataFrame:
    """
    Generate realistic test data for validation purposes.
    
    This function creates synthetic Bitcoin price data with realistic trends,
    volatility, and technical indicators for testing prediction models.
    
    Args:
        days: Number of days of data to generate.
        start_price: Starting price for the simulation.
        
    Returns:
        DataFrame with datetime (with time), price, and technical indicator columns.
    """
    print(f"🔢 Generating {days} days of validation data with timestamps...")

    # Create realistic datetime data with hourly intervals for more precision
    # Start at a specific time (9:00 AM) to simulate market opening
    start_datetime = pd.Timestamp('2024-02-01 09:00:00')
    datetimes = pd.date_range(
        start=start_datetime, 
        periods=days, 
        freq='D'  # Daily frequency, but we'll add random hour variations
    )
    
    # Add some realistic time variations (market hours typically 9 AM to 4 PM)
    np.random.seed(456)  # Consistent seed for reproducible time variations
    time_variations = np.random.randint(0, 8, days)  # 0-7 hours after 9 AM
    datetimes_with_time = [
        dt + pd.Timedelta(hours=int(variation)) 
        for dt, variation in zip(datetimes, time_variations)
    ]

    # More realistic price simulation
    prices = [start_price]
    for i in range(1, days):
        # Add trend, mean reversion, and volatility
        prev_price = prices[-1]

        # Random walk with slight upward bias
        trend = 0.0005  # 0.05% daily upward trend
        volatility = 0.025  # 2.5% daily volatility

        # Add some mean reversion
        if len(prices) > 20:
            sma_20 = np.mean(prices[-20:])
            mean_reversion = (sma_20 - prev_price) / prev_price * 0.1
        else:
            mean_reversion = 0

        change = np.random.normal(trend + mean_reversion, volatility)
        new_price = prev_price * (1 + change)

        # Prevent unrealistic prices using settings constants
        new_price = max(new_price, MIN_PRICE)
        new_price = min(new_price, MAX_PRICE)

        prices.append(new_price)

    df = pd.DataFrame({
        'date': datetimes_with_time,
        'close': prices
    })

    # Add required features using DataConfig
    data_config = DataConfig()
    df['volume'] = np.random.randint(15000, 60000, len(df))
    df['volatility'] = df['close'].rolling(20, min_periods=1).std().fillna(0)
    df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['rsi'] = calculate_rsi(df['close']).fillna(50)

    print(f"✅ Validation data generated with timestamps")
    price_min = df['close'].min()
    price_max = df['close'].max()
    print(f"   📊 Price range: ${price_min:,.2f} - ${price_max:,.2f}")
    price_change = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
    print(f"   📈 Total change: {price_change:.1f}%")
    print(f"   ⏰ Time range: {df['date'].min()} to {df['date'].max()}")

    return df


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load and prepare data from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        DataFrame with properly formatted data. Date column is optional.
        
    Raises:
        ValueError: If file cannot be loaded or processed.
        FileNotFoundError: If the specified file does not exist.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"📁 Loaded {len(data)} records from {file_path}")
        
        # Optional date column handling
        if 'date' not in data.columns:
            # Check for common date column variations
            date_columns = [col for col in data.columns 
                           if any(date_term in col.lower() 
                                 for date_term in ['date', 'time', 'timestamp', 'datetime'])]
            
            if date_columns:
                # Use the first found date-like column
                date_col = date_columns[0]
                data['date'] = data[date_col]
                print(f"📅 Using column '{date_col}' as date column")
            else:
                print(f"ℹ️ No date column found - this is okay since prediction timestamps use execution time")
        
        # Validate and standardize date column if present
        if 'date' in data.columns:
            try:
                # Ensure dates are proper datetime objects
                data['date'] = pd.to_datetime(data['date'])
                print(f"✅ Date column validated and standardized")
                print(f"   ⏰ DateTime range: {data['date'].min()} to {data['date'].max()}")
                
                # Check if the datetime data includes time information
                has_time_info = (data['date'].dt.time != pd.Timestamp('1900-01-01').time()).any()
                if has_time_info:
                    print(f"   🕐 Time information detected in timestamps")
                else:
                    print(f"   📅 Date-only timestamps detected")
                    
                # Check for missing dates
                missing_dates = data['date'].isna().sum()
                if missing_dates > 0:
                    print(f"⚠️ Warning: Found {missing_dates} missing dates in 'date' column")
                
                # Sort by date to ensure chronological order
                data = data.sort_values('date').reset_index(drop=True)
                print(f"✅ Data sorted chronologically")
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not parse date column: {e}")
                print(f"   Date column will be ignored for predictions")
        
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {file_path}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        raise


def validate_data_columns(data: pd.DataFrame, 
                          required_columns: list = None) -> bool:
    """
    Validate that the DataFrame has required columns for predictions.
    
    Args:
        data: DataFrame to validate.
        required_columns: List of required column names. If None, uses default features.
        
    Returns:
        True if all required columns are present, False otherwise.
        
    Note:
        'date' column is optional since prediction timestamps now use execution time.
    """
    if required_columns is None:
        data_config = DataConfig()
        required_columns = data_config.default_features
        
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return False
        
    print(f"✅ All required columns present: {required_columns}")
    
    # Optional validation for date column if present
    if 'date' in data.columns and data['date'].isna().any():
        print(f"⚠️ Warning: Found missing values in 'date' column")
        return False
    
    return True 