#!/usr/bin/env python3
"""
train_geo.py - Train Geolocation-Based Price Prediction Model

Trains a neural network to predict electricity prices based on
geographic coordinates and associated features.
"""

import os
import sys
import json
import pickle
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Setup logging with file output
LOG_DIR = '/app/data/training' if os.path.exists('/app') else os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'train_geo.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================
# MODEL CONFIGURATION
# ============================================

class GeoModelConfig:
    """Configuration for geolocation model"""
    
    # Training
    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    
    # Model architecture
    HIDDEN_LAYERS = [128, 64, 32]
    DROPOUT_RATE = 0.2
    
    # Data
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MIN_SAMPLES = 100


# ============================================
# GEO PRICE PREDICTOR
# ============================================

class GeoPricePredictor:
    """Neural network for geolocation-based price prediction"""
    
    def __init__(self, n_features: int, config: GeoModelConfig = None):
        self.n_features = n_features
        self.config = config or GeoModelConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False
    
    def build_model(self) -> tf.keras.Model:
        """Build the neural network architecture"""
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=(self.n_features,)))
        
        # Hidden layers
        for i, units in enumerate(self.config.HIDDEN_LAYERS):
            model.add(tf.keras.layers.Dense(units, activation='relu', name=f'dense_{i}'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(self.config.DROPOUT_RATE))
        
        # Output layer
        model.add(tf.keras.layers.Dense(1, name='output'))
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"Built model with {self.n_features} input features")
        logger.info(f"Architecture: {self.config.HIDDEN_LAYERS}")
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray,
              validation_data: Tuple[np.ndarray, np.ndarray] = None,
              verbose: int = 1) -> tf.keras.callbacks.History:
        """Train the model"""
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        
        # Scale validation data if provided
        if validation_data:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train
        history = self.model.fit(
            X_scaled, y,
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            validation_data=validation_data,
            validation_split=self.config.VALIDATION_SPLIT if validation_data is None else 0,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()
    
    def predict_single(self, features: Dict[str, float]) -> float:
        """Predict for a single observation from feature dict"""
        if not self.feature_names:
            raise ValueError("Feature names not set")
        
        # Build feature array in correct order
        X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        
        return float(self.predict(X)[0])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        y_pred = self.model.predict(X_scaled, verbose=0).flatten()
        
        # Calculate metrics
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(mse)
        
        # MAPE (avoid division by zero)
        mask = np.abs(y) > 5  # Only for prices > $5
        if mask.sum() > 0:
            mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
        else:
            mape = 0.0
        
        # R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2)
        }
    
    def save(self, model_path: str, scaler_path: str, features_path: str):
        """Save model, scaler, and feature names"""
        # Save model
        self.model.save(model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save feature names
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        logger.info(f"Saved features to {features_path}")
    
    @classmethod
    def load(cls, model_path: str, scaler_path: str, features_path: str) -> 'GeoPricePredictor':
        """Load a saved model"""
        # Load feature names first to get n_features
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
        
        # Create instance
        predictor = cls(n_features=len(feature_names))
        predictor.feature_names = feature_names
        
        # Load model
        predictor.model = tf.keras.models.load_model(model_path)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            predictor.scaler = pickle.load(f)
        
        predictor.is_fitted = True
        
        logger.info(f"Loaded model with {len(feature_names)} features")
        return predictor


# ============================================
# TRAINING PIPELINE
# ============================================

def train_geo_model(training_df: pd.DataFrame,
                    output_dir: str,
                    config: GeoModelConfig = None) -> Tuple[GeoPricePredictor, Dict]:
    """
    Train geolocation-based price prediction model
    
    Args:
        training_df: DataFrame with features and 'price' column
        output_dir: Directory to save model artifacts
        config: Model configuration
    
    Returns:
        Tuple of (trained predictor, metrics dict)
    """
    config = config or GeoModelConfig()
    
    logger.info("=" * 70)
    logger.info("TRAINING GEOLOCATION MODEL")
    logger.info("=" * 70)
    
    # Identify feature columns (exclude non-feature columns)
    exclude_cols = ['price', 'node', 'energy', 'congestion', 'loss', 'timestamp']
    feature_cols = [c for c in training_df.columns if c not in exclude_cols]
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Sample features: {feature_cols[:10]}")
    
    # Prepare data
    X = training_df[feature_cols].values
    y = training_df['price'].values
    
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Price range: ${y.min():.2f} - ${y.max():.2f}")
    logger.info(f"Price mean: ${y.mean():.2f}")
    
    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=True
    )
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Build and train model
    predictor = GeoPricePredictor(n_features=len(feature_cols), config=config)
    predictor.feature_names = feature_cols
    predictor.build_model()
    
    logger.info("Training model...")
    history = predictor.train(
        X_train, y_train,
        validation_data=(X_test, y_test),
        verbose=2
    )
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics = predictor.evaluate(X_test, y_test)
    
    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Test MAE:  ${metrics['mae']:.2f}/MWh")
    logger.info(f"Test RMSE: ${metrics['rmse']:.2f}/MWh")
    logger.info(f"Test MAPE: {metrics['mape']:.2f}%")
    logger.info(f"Test R²:   {metrics['r2']:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'geo_model.keras')
    scaler_path = os.path.join(output_dir, 'geo_scaler.pkl')
    features_path = os.path.join(output_dir, 'geo_features.json')
    
    predictor.save(model_path, scaler_path, features_path)
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'features': len(feature_cols),
        'feature_names': feature_cols,
        'config': {
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'hidden_layers': config.HIDDEN_LAYERS,
            'dropout_rate': config.DROPOUT_RATE,
        },
        'metrics': metrics,
        'price_stats': {
            'min': float(y.min()),
            'max': float(y.max()),
            'mean': float(y.mean()),
            'std': float(y.std()),
        },
        'version': '1.0.0-geo'
    }
    
    metadata_path = os.path.join(output_dir, 'geo_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    return predictor, metrics


# ============================================
# CLI
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Train geolocation-based price model')
    
    parser.add_argument('--training-data', type=str, required=True,
                        help='Path to training data pickle')
    parser.add_argument('--output-dir', type=str,
                        default='/Users/chester.kim/workspace/tf/electricity-forecasting/tier3_poc/data/models',
                        help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=GeoModelConfig.EPOCHS,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=GeoModelConfig.BATCH_SIZE,
                        help='Batch size')
    
    args = parser.parse_args()
    
    # Configure
    config = GeoModelConfig()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("GEOLOCATION PRICE PREDICTION - MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Training data: {args.training_data}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info("")
    
    # Load training data
    training_df = pd.read_pickle(args.training_data)
    logger.info(f"Loaded {len(training_df)} training records")
    
    if len(training_df) < config.MIN_SAMPLES:
        logger.error(f"Insufficient data: {len(training_df)} < {config.MIN_SAMPLES}")
        return 1
    
    # Train
    predictor, metrics = train_geo_model(training_df, args.output_dir, config)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE ✅")
    logger.info("=" * 70)
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"MAE: ${metrics['mae']:.2f}/MWh")
    logger.info(f"R²: {metrics['r2']:.4f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

