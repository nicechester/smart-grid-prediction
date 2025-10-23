#!/usr/bin/env python3
"""
train.py - Standalone model retraining script
Fetches fresh Tier 2 data and retrains the prediction model
Can be run daily/weekly via cron or scheduler

Usage:
    python train.py                 # Train with default settings
    python train.py --epochs 100    # Custom epochs
    python train.py --days 30       # Use last 30 days of data
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
from tier2_pipeline import Tier2DataPipeline

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
    
    # Fill NaN values
    return df.fillna(method='bfill').fillna(method='ffill')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
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
# Data Building
# ============================================

def build_training_data(use_tier2=True, synthetic_days=365):
    """
    Build training dataset from Tier 2 sources or synthetic fallback
    
    Args:
        use_tier2: Try to fetch real Tier 2 data first
        synthetic_days: Number of synthetic days if Tier 2 fails
    
    Returns:
        DataFrame with training data
    """
    
    logger.info("=" * 70)
    logger.info("BUILDING TRAINING DATA")
    logger.info("=" * 70)
    logger.info("")
    
    df = None
    
    # Try Tier 2 first
    if use_tier2:
        logger.info("1. Attempting to load Tier 2 data...")
        Config.init()
        tier2 = Tier2DataPipeline(api_key=Config.EIA_API_KEY)
        tier2_data = tier2.build_complete_dataset()
        
        if tier2_data.get('prices') is not None and len(tier2_data['prices']) > TrainingConfig.MIN_SAMPLES:
            df = tier2_data['prices'].copy()
            logger.info(f"✓ Loaded {len(df)} real EIA records")
            
            # Process EIA data
            try:
                # Convert period to datetime and extract features FIRST
                df['period'] = pd.to_datetime(df['period'])
                df['hour'] = df['period'].dt.hour
                df['day'] = df['period'].dt.day
                df['month'] = df['period'].dt.month
                df['is_weekend'] = (df['period'].dt.dayofweek >= 5).astype(int)
                
                # Set price from value column
                if 'value' in df.columns:
                    df['price'] = df['value']
                
                # Drop the period column NOW (before adding other features)
                df = df.drop(columns=['period'], errors='ignore')
                
                # Add weather features (mock for now - use real NOAA in production)
                df['temperature'] = 20 + 5 * np.sin(2 * np.pi * df['hour'] / 24) + np.random.normal(0, 2, len(df))
                df['cloud_cover'] = np.clip(0.4 + np.random.normal(0, 0.2, len(df)), 0, 1)
                df['wind_speed'] = np.clip(5 + np.random.normal(0, 2, len(df)), 0, 15)
                df['solar_mw'] = np.maximum(0, 2000 * (1 - df['cloud_cover']) * np.sin(np.pi * (df['hour'] - 6) / 12))
                df['wind_mw'] = np.maximum(0, 800 * (df['wind_speed'] / 10) + np.random.normal(0, 50, len(df)))
                df['total_demand'] = df['price'] * 100 + np.random.normal(0, 100, len(df))
                df['renewable_pct'] = np.clip((df['solar_mw'] + df['wind_mw']) / (df['total_demand'] + 1), 0, 1)
                df['imbalance'] = df['total_demand'] / (df['solar_mw'] + df['wind_mw'] + 5000)
                df['grid_stress'] = np.clip(df['total_demand'] / 35000, 0, 1)
                df['wildfire_risk'] = 0.0
                
                # Drop any remaining non-numeric columns
                non_numeric_cols = df.select_dtypes(include=['object', 'datetime64[ns]']).columns
                if len(non_numeric_cols) > 0:
                    logger.info(f"Dropping non-numeric columns: {non_numeric_cols.tolist()}")
                    df = df.drop(columns=non_numeric_cols)
                
                logger.info(f"✓ Processed EIA data: {df.shape}")
                logger.info(f"✓ Columns: {df.columns.tolist()}")
            except Exception as e:
                logger.warning(f"Error processing EIA data: {e}")
                df = None
    
    # Fallback to synthetic
    if df is None or len(df) < TrainingConfig.MIN_SAMPLES:
        logger.info(f"2. Falling back to synthetic data ({synthetic_days} days)...")
        df = generate_training_data(days=synthetic_days)
        logger.info(f"✓ Generated {len(df)} synthetic samples")
    
    logger.info(f"✓ Total training samples: {len(df)}")
    logger.info("")
    
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
    logger.info(model.summary())
    logger.info("")
    
    # Train
    logger.info(f"4. Training ({epochs} epochs)...")
    history = predictor.train(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size
    )
    logger.info("✓ Training complete")
    logger.info("")
    
    # Evaluate
    logger.info("5. Evaluating model...")
    metrics = predictor.evaluate(X_test, y_test)
    logger.info(f"✓ Test Loss (MSE): {metrics['loss']:.2f}")
    logger.info(f"✓ Test MAE: {metrics['mae']:.2f} €/MWh")
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
    logger.info("SMART GRID ML - MODEL RETRAINING")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        # Initialize
        Config.init()
        
        # Build data
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
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Test MAE: {metrics['mae']:.2f} €/MWh")
        logger.info(f"Test MAPE: {metrics['mape']:.2f}%")
        logger.info(f"Model saved to: {Config.MODEL_DIR}/")
        logger.info("")
        
        return 0
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())