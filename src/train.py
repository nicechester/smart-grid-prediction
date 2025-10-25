#!/usr/bin/env python3
"""
train.py - Standalone model retraining script
Fetches fresh Tier 2 data (now with CAISO price data) and retrains the prediction model
Can be run daily/weekly via cron or scheduler

Usage:
    python train.py                 # Train with default settings
    python train.py --epochs 100    # Custom epochs
    python train.py --no-tier2      # Skip Tier 2, use synthetic data
"""

import os
import sys
import json
import pickle
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# TensorFlow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import from main modules
from main import Config, PricePredictor, generate_training_data
from tier2_pipeline import Tier2Config, Tier2DataPipeline, CAISOPriceFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/data/training/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# Training Configuration
# ============================================

class TrainingConfig:
    """Retraining configuration"""
    
    EPOCHS = 50
    BATCH_SIZE = 32
    TEST_SIZE = 0.2
    MIN_SAMPLES = 500  # Minimum samples to train
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42

# ============================================
# Feature Engineering (Local)
# ============================================

def add_lag_features(df: pd.DataFrame, lags: list = [1, 3, 6, 12]) -> pd.DataFrame:
    """Add historical lag features - only for numeric columns"""
    numeric_cols = ['temperature', 'solar_mw', 'wind_mw', 'price']
    
    # Only add lags for columns that exist and are numeric
    for col in numeric_cols:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Fill NaN values with forward/backward fill
    return df.bfill().ffill()

# ============================================
# CAISO Data Processing
# ============================================

def process_caiso_price_data(caiso_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process CAISO price data into training format with features
    
    Args:
        caiso_df: Raw CAISO price DataFrame with columns: timestamp, price, region
    
    Returns:
        Processed DataFrame with features
    """
    logger.info("Processing CAISO price data into training format...")
    
    if caiso_df is None or len(caiso_df) == 0:
        logger.warning("No CAISO data to process")
        return None
    
    # Ensure timestamp is datetime
    caiso_df['timestamp'] = pd.to_datetime(caiso_df['timestamp'])
    
    # Sort by timestamp
    caiso_df = caiso_df.sort_values('timestamp')
    
    # Extract time features
    caiso_df['hour'] = caiso_df['timestamp'].dt.hour
    caiso_df['month'] = caiso_df['timestamp'].dt.month
    caiso_df['day_of_week'] = caiso_df['timestamp'].dt.dayofweek
    caiso_df['is_weekend'] = (caiso_df['day_of_week'] >= 5).astype(int)
    
    # Generate synthetic features based on time patterns
    # (In production, these would come from actual weather/grid data)
    
    # Temperature estimation (seasonal + daily cycle)
    day_of_year = caiso_df['timestamp'].dt.dayofyear
    base_temp = 20 + 10 * np.sin(2 * np.pi * day_of_year / 365)
    daily_swing = 8 * np.sin(2 * np.pi * caiso_df['hour'] / 24 - np.pi/2)
    caiso_df['temperature'] = base_temp + daily_swing
    
    # Cloud cover estimation
    caiso_df['cloud_cover'] = np.clip(
        0.4 + 0.3 * np.sin(2 * np.pi * day_of_year / 30),
        0, 1
    )
    
    # Wind speed estimation
    caiso_df['wind_speed'] = np.clip(
        5 + 3 * np.sin(2 * np.pi * day_of_year / 20),
        0, 15
    )
    
    # Solar generation (zero at night, peak at noon)
    caiso_df['solar_mw'] = 0
    daytime = (caiso_df['hour'] >= 6) & (caiso_df['hour'] <= 18)
    caiso_df.loc[daytime, 'solar_mw'] = (
        2000 * (1 - caiso_df.loc[daytime, 'cloud_cover']) * 
        np.sin(np.pi * (caiso_df.loc[daytime, 'hour'] - 6) / 12)
    )
    
    # Wind generation
    caiso_df['wind_mw'] = np.maximum(0, 800 * (caiso_df['wind_speed'] / 10))
    
    # Demand estimation (correlated with price)
    # Base demand varies by time of day
    time_multiplier = np.ones(len(caiso_df))
    morning_peak = caiso_df['hour'].isin([6, 7, 8])
    evening_peak = caiso_df['hour'].isin([17, 18, 19, 20])
    night = (caiso_df['hour'] >= 23) | (caiso_df['hour'] <= 5)
    
    time_multiplier[morning_peak] = 1.3
    time_multiplier[evening_peak] = 1.4
    time_multiplier[night] = 0.8
    
    base_demand = 20000
    weekend_factor = np.where(caiso_df['is_weekend'], 0.9, 1.0)
    
    caiso_df['total_demand'] = base_demand * time_multiplier * weekend_factor
    
    # Grid metrics
    total_renewable = caiso_df['solar_mw'] + caiso_df['wind_mw']
    caiso_df['renewable_pct'] = np.clip(total_renewable / caiso_df['total_demand'], 0, 1)
    caiso_df['imbalance'] = caiso_df['total_demand'] / (total_renewable + 5000)
    caiso_df['grid_stress'] = np.clip(caiso_df['total_demand'] / 35000, 0, 1)
    caiso_df['wildfire_risk'] = 0.0  # TODO: Add real wildfire data
    
    # Drop timestamp and region columns (keep as metadata but not features)
    feature_cols = [
        'temperature', 'cloud_cover', 'wind_speed', 'solar_mw', 'wind_mw',
        'total_demand', 'renewable_pct', 'imbalance', 'grid_stress', 'wildfire_risk',
        'hour', 'month', 'is_weekend', 'price'
    ]
    
    result_df = caiso_df[feature_cols].copy()
    
    logger.info(f"✓ Processed {len(result_df)} CAISO records into {len(feature_cols)} features")
    
    return result_df

# ============================================
# Data Building - UPDATED FOR CAISO
# ============================================

def build_training_data(pipeline: Tier2DataPipeline = None, use_tier2: bool = True, 
                       synthetic_days: int = 365) -> pd.DataFrame:
    # ... (initial setup) ...
    
    df = None
    
    if use_tier2:
        try:
            if pipeline is None:
                pipeline = Tier2DataPipeline(api_key=Config.EIA_API_KEY)
            
            # Try CAISO price data first (NEW)
            logger.info("1️⃣  Attempting CAISO price data (Concurrent City Fetch)...")
            
            # --- START OF FIX: Use the existing concurrent city fetcher ---
            start_date = "20230101" # Updated to match YYYYMMDD format
            end_date = "20250731"   # Updated to match YYYYMMDD format
            
            # CAISOPriceFetcher.fetch_all_cities_prices is already concurrent and handles errors/skipping
            caiso_prices_dict = CAISOPriceFetcher.fetch_all_cities_prices(
                start_date=start_date, 
                end_date=end_date,
                market='DAM',
                max_workers=Tier2Config.MAX_WORKERS # Use your config setting
            )
            
            if caiso_prices_dict:
                # Combine all successful city dataframes into one for training
                caiso_combined = pd.concat(list(caiso_prices_dict.values()), ignore_index=True)
                
                # We need a 'timestamp' column and a 'price' column for processing
                # The columns in the returned DF from fetch_city_prices are:
                # timestamp, node, lmp, energy, congestion, loss, city_id, city_name, zone
                
                # Map 'lmp' column to 'price' for the process_caiso_price_data function
                caiso_combined.rename(columns={'lmp': 'price'}, inplace=True)
                
                # Process into training format
                df = process_caiso_price_data(caiso_combined)
                
                if df is not None and len(df) > 0:
                    logger.info(f"✓ Loaded {len(df)} CAISO records from {len(caiso_prices_dict)} cities.")
                    return df
            # --- END OF FIX ---
            
            # If we reached here, either fetch_all_cities_prices failed entirely or returned empty
            logger.error("❌ CAISO data not available or empty. Failing fast.")
            raise ValueError("CAISO data fetch failed. Unable to proceed without valid data.")
                        
        except Exception as e:
            logger.warning(f"Tier 2 data fetch failed: {e}")
            raise ValueError("Tier 2 data fetch failed. Unable to proceed without valid data.")
    
    return df

# ============================================
# Model Training
# ============================================

def train_model(df, epochs=None, batch_size=None):
    """
    Train the prediction model
    
    Args:
        df: Training DataFrame
        epochs: Number of training epochs
        batch_size: Batch size
    
    Returns:
        Tuple of (predictor, metrics)
    """
    
    epochs = epochs or TrainingConfig.EPOCHS
    batch_size = batch_size or TrainingConfig.BATCH_SIZE
    
    logger.info("=" * 70)
    logger.info("TRAINING MODEL")
    logger.info("=" * 70)
    logger.info("")
    
    # Verify no datetime columns before lag features
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    if len(datetime_cols) > 0:
        logger.warning(f"Dropping datetime columns: {datetime_cols.tolist()}")
        df = df.drop(columns=datetime_cols)
    
    # Feature engineering
    logger.info("1. Adding lag features...")
    df = add_lag_features(df)
    
    # Define feature columns (everything except price)
    feature_cols = [col for col in df.columns if col != 'price']
    logger.info(f"✓ {len(feature_cols)} features: {feature_cols[:10]}...")
    logger.info("")
    
    # Verify all columns are numeric
    non_numeric = df[feature_cols].select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        logger.error(f"Non-numeric columns found: {non_numeric.tolist()}")
        logger.error(f"Column types: {df[non_numeric].dtypes}")
        raise ValueError(f"Non-numeric features detected: {non_numeric.tolist()}")
    
    # Prepare data
    logger.info("2. Preparing data...")
    X = df[feature_cols].values
    y = df['price'].values
    
    logger.info(f"✓ X shape: {X.shape}, dtype: {X.dtype}")
    logger.info(f"✓ y shape: {y.shape}, dtype: {y.dtype}")
    logger.info(f"✓ Price range: ${y.min():.2f} - ${y.max():.2f}/MWh")
    logger.info(f"✓ Price mean: ${y.mean():.2f}/MWh")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TrainingConfig.TEST_SIZE, 
        random_state=TrainingConfig.RANDOM_STATE, shuffle=False
    )
    logger.info(f"✓ Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")
    logger.info("")
    
    # Build model
    logger.info("3. Building neural network...")
    predictor = PricePredictor(X_train.shape[1])
    model = predictor.build_model()
    logger.info("")
    
    # Train
    logger.info(f"4. Training ({epochs} epochs)...")
    history = predictor.train(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2  # One line per epoch
    )
    logger.info("✓ Training complete")
    logger.info("")
    
    # Evaluate
    logger.info("5. Evaluating model...")
    metrics = predictor.evaluate(X_test, y_test)
    logger.info(f"✓ Test Loss (MSE): {metrics['loss']:.2f}")
    logger.info(f"✓ Test MAE: {metrics['mae']:.2f} $/MWh")
    logger.info(f"✓ Test MAPE: {metrics['mape']:.2f}%")
    logger.info("")
    
    # Save
    logger.info("6. Saving model...")
    predictor.feature_names = feature_cols
    predictor.save(
        os.path.join(Config.MODEL_DIR, 'price_model.h5'),
        os.path.join(Config.MODEL_DIR, 'scaler.pkl'),
        os.path.join(Config.MODEL_DIR, 'features.json')
    )
    logger.info(f"✓ Model saved to {Config.MODEL_DIR}/")
    logger.info("")
    
    # Save training metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'training_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'features': len(feature_cols),
        'feature_names': feature_cols,
        'epochs': epochs,
        'batch_size': batch_size,
        'price_range': {
            'min': float(y.min()),
            'max': float(y.max()),
            'mean': float(y.mean())
        },
        'metrics': {
            'loss': float(metrics['loss']),
            'mae': float(metrics['mae']),
            'mape': float(metrics['mape'])
        }
    }
    
    with open(os.path.join(Config.MODEL_DIR, 'training_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("✓ Metadata saved")
    logger.info("")
    
    return predictor, metrics

# ============================================
# Main Training Pipeline
# ============================================

def main():
    """Main training pipeline"""
    
    parser = argparse.ArgumentParser(description='Retrain electricity price prediction model')
    parser.add_argument('--epochs', type=int, default=TrainingConfig.EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=TrainingConfig.BATCH_SIZE, help='Batch size')
    parser.add_argument('--synthetic-days', type=int, default=365, help='Synthetic data days if Tier 2 fails')
    parser.add_argument('--no-tier2', action='store_true', help='Skip Tier 2, use synthetic only')
    
    args = parser.parse_args()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SMART GRID ML - MODEL RETRAINING (CAISO-ENABLED)")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        # Initialize
        Config.init()
        
        # Build data (now tries CAISO first, then EIA, then synthetic)
        df = build_training_data(
            use_tier2=not args.no_tier2,
            synthetic_days=args.synthetic_days
        )
        
        if df is None or len(df) < TrainingConfig.MIN_SAMPLES:
            logger.error(f"Insufficient training data: {len(df) if df is not None else 0} samples")
            logger.error(f"Minimum required: {TrainingConfig.MIN_SAMPLES}")
            return 1
        
        # Train
        predictor, metrics = train_model(
            df,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Summary
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE ✓")
        logger.info("=" * 70)
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Test MAE: {metrics['mae']:.2f} $/MWh")
        logger.info(f"Test MAPE: {metrics['mape']:.2f}%")
        logger.info(f"Model saved to: {Config.MODEL_DIR}/")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Restart API: docker-compose restart app")
        logger.info("  2. Test prediction: curl http://localhost:8000/predict")
        logger.info("")
        
        return 0
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())