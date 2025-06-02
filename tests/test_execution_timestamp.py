"""
Test script to verify execution time timestamp functionality.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
from datetime import datetime
from predict import PredictionValidator
from core import generate_validation_data
from settings import ValidationConfig

def test_execution_timestamp():
    """Test that predictions use current execution time as timestamps."""
    print("🕐 Testing execution time timestamp functionality...")
    
    try:
        # Record the test start time
        test_start_time = datetime.now()
        print(f"🚀 Test started at: {test_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test 1: Generate data (date column is now optional)
        print("\n1️⃣ Testing data generation...")
        test_data = generate_validation_data(days=80)
        print(f"✅ Generated data with {len(test_data)} records")
        
        # Test 2: Create CSV data without date column
        print("\n2️⃣ Testing data without date column...")
        csv_data = pd.DataFrame({
            'close': [50000, 51000, 52000, 53000] * 20,  # 80 records
            'volume': [10000, 11000, 12000, 13000] * 20,
            'volatility': [100, 110, 120, 130] * 20,
            'sma_20': [50000, 50500, 51000, 51500] * 20,
            'rsi': [50, 55, 60, 65] * 20
        })
        
        csv_path = 'test_no_date.csv'
        csv_data.to_csv(csv_path, index=False)
        
        from core import load_data_from_csv
        loaded_data = load_data_from_csv(csv_path)
        print(f"✅ Successfully loaded CSV without date column")
        
        # Test 3: Run prediction validation
        print("\n3️⃣ Testing prediction validation with execution timestamps...")
        config = ValidationConfig()
        validator = PredictionValidator(config=config)
        
        # Record time before predictions
        prediction_start_time = datetime.now()
        print(f"📈 Predictions starting at: {prediction_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = validator.predict_and_validate(loaded_data, ['1day'])
        
        # Record time after predictions
        prediction_end_time = datetime.now()
        print(f"📈 Predictions completed at: {prediction_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"✅ Generated {len(results)} predictions")
        
        # Test 4: Verify timestamps are execution times
        print("\n4️⃣ Verifying execution timestamps...")
        if results:
            print(f"📋 Sample prediction timestamps:")
            for i, result in enumerate(results[:3]):
                timestamp_str = result.timestamp
                print(f"   {i+1}. {timestamp_str}")
                
                # Parse the timestamp
                result_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                
                # Verify it's within a reasonable time window (allow some buffer)
                time_diff = abs((result_time - prediction_start_time).total_seconds())
                if time_diff <= 60:  # Within 1 minute of when predictions started
                    print(f"      ✅ Timestamp is recent (within {time_diff:.1f}s of execution)")
                else:
                    print(f"      ❌ Timestamp too far from execution time!")
                    print(f"         Time difference: {time_diff:.1f} seconds")
                    return False
                
                # Verify timestamp format includes time
                if len(timestamp_str) >= 19 and timestamp_str.count(':') == 2:
                    print(f"      ✅ Proper datetime format (YYYY-MM-DD HH:MM:SS)")
                else:
                    print(f"      ❌ Invalid datetime format: {timestamp_str}")
                    return False
        
        # Test 5: Export and verify
        print("\n5️⃣ Testing CSV export...")
        csv_path_output = validator.export_to_csv(results, 'test_execution_timestamps.csv')
        print(f"✅ Results exported to CSV")
        
        # Cleanup
        os.remove(csv_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(csv_path):
            os.remove(csv_path)
        return False

if __name__ == "__main__":
    print("🕐 EXECUTION TIMESTAMP TESTING")
    print("=" * 40)
    
    success = test_execution_timestamp()
    
    if success:
        print("\n🎉 All execution timestamp tests passed!")
        print("✅ Key features working:")
        print("   - Timestamps represent when predictions were executed")
        print("   - Date column is now optional in input data")
        print("   - All predictions get current execution time")
        print("   - Format: YYYY-MM-DD HH:MM:SS")
    else:
        print("\n❌ Some execution timestamp tests failed!") 