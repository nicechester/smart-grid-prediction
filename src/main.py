#!/usr/bin/env python3
"""
Smart Grid Electricity Price Prediction System
Core ML Library - Model Definition and Configuration
Python 3.11, TensorFlow 2.14.0

This module provides:
- Configuration management
- Neural network model (PricePredictor)
- Synthetic data generation (fallback)

FIXED: Added prediction bounds and better regularization
"""

import os
import json
import pickle
from pyexpat import features
import numpy as np
import pandas as pd
from typing import Dict
from dotenv import load_dotenv
import logging

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Application configuration"""
    
    # API Keys (from .env file)
    EIA_API_KEY = None
    
    # Data paths
    DATA_DIR = "/app/data"
    MODEL_DIR = "/app/data/models"
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Training parameters (defaults)
    EPOCHS = 50
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    
    # Prediction bounds (reasonable California electricity prices)
    PRICE_MIN = -50.0   # Negative prices can happen with oversupply
    PRICE_MAX = 250.0   # Extreme grid stress scenarios
    PRICE_TYPICAL_MIN = 10.0
    PRICE_TYPICAL_MAX = 150.0
    
    @classmethod
    def init(cls):
        """Initialize configuration"""
        load_dotenv()
        cls.EIA_API_KEY = os.getenv('EIA_API_KEY', 'demo_key')
        
        # Create directories
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        
        logger.info(f"Config initialized - EIA API Key: {'Set' if cls.EIA_API_KEY != 'demo_key' else 'demo_key'}")

# ============================================
# NEURAL NETWORK MODEL
# ============================================

class PricePredictor:
    """
    Deep Neural Network for electricity price prediction
    
    Architecture:
    - Input layer: feature_dim neurons
    - Hidden layers: 128 -> 64 -> 32 -> 16 neurons
    - Output layer: 1 neuron (price)
    - Dropout and BatchNorm for regularization
    
    FIXED: Added output clipping and better regularization
    """
    
    def __init__(self, feature_dim: int):
        """
        Initialize predictor
        
        Args:
            feature_dim: Number of input features
        """
        self.feature_dim = feature_dim
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.training_price_stats = None  # Store training price statistics
    
    def build_model(self) -> keras.Model:
        """
        Build neural network architecture
        
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building neural network with {self.feature_dim} input features...")
        
        self.model = Sequential([
            # Layer 1: 128 neurons with L2 regularization
            layers.Dense(128, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(0.001),
                        input_shape=(self.feature_dim,)),
            layers.Dropout(0.3),
            
            # Layer 2: 64 neurons with batch normalization
            layers.Dense(64, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Layer 3: 32 neurons
            layers.Dense(32, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            
            # Layer 4: 16 neurons
            layers.Dense(16, activation='relu'),
            
            # Output layer: 1 neuron (price)
            layers.Dense(1)
        ])
        
        # Use Adam optimizer with gradient clipping
        optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        
        self.model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae']
        )
        
        logger.info("✓ Model built successfully")
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32, 
              verbose: int = 1) -> Dict:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Training verbosity (0=silent, 1=progress bar, 2=one line per epoch)
        
        Returns:
            Training history dictionary
        """
        logger.info(f"Training model for {epochs} epochs...")
        
        # Store training price statistics for later clipping
        self.training_price_stats = {
            'min': float(np.min(y_train)),
            'max': float(np.max(y_train)),
            'mean': float(np.mean(y_train)),
            'std': float(np.std(y_train)),
            'median': float(np.median(y_train)),
            'q25': float(np.percentile(y_train, 25)),
            'q75': float(np.percentile(y_train, 75))
        }
        
        logger.info(f"Training price range: ${self.training_price_stats['min']:.2f} - ${self.training_price_stats['max']:.2f}")
        logger.info(f"Training price mean/median: ${self.training_price_stats['mean']:.2f} / ${self.training_price_stats['median']:.2f}")
        
        # Normalize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=0.2 if validation_data is None else None,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        logger.info("✓ Training complete")
        return history.history
    
    def predict(self, X: np.ndarray, clip_to_training: bool = True) -> np.ndarray:
        """
        Predict prices with optional clipping
        
        Args:
            X: Input features
            clip_to_training: If True, clip predictions to reasonable range based on training data
        
        Returns:
            Predicted prices
        """
        X_scaled = self.scaler.transform(X)
        logger.info(f"Scaled features: {X_scaled}")
        
        predictions = self.model.predict(X_scaled, verbose=0).flatten()
        
        if clip_to_training and self.training_price_stats is not None:
            # Clip to training range + some margin
            stats = self.training_price_stats
            # Allow 20% margin beyond training range
            margin = 0.2
            lower_bound = max(Config.PRICE_MIN, stats['min'] - margin * stats['std'])
            upper_bound = min(Config.PRICE_MAX, stats['max'] + margin * stats['std'])
            
            original_predictions = predictions.copy()
            predictions = np.clip(predictions, lower_bound, upper_bound)
            
            # Log if significant clipping occurred
            clipped = np.abs(original_predictions - predictions) > 1.0
            if np.any(clipped):
                logger.warning(f"Clipped {np.sum(clipped)} predictions: {original_predictions[clipped][:3]} -> {predictions[clipped][:3]}")
                logger.info(f"Clipping bounds: ${lower_bound:.2f} - ${upper_bound:.2f}")
        
        return predictions.reshape(-1, 1)
    
    def predict_for_location(self, features: Dict, location_data: Dict = None) -> Dict:
        """
        Predict price for a specific location with additional context
        
        Args:
            features: Dictionary of feature values
            location_data: Optional location-specific data (weather, demand profile, etc.)
        
        Returns:
            Dictionary with prediction and metadata
        """
        # Adjust features based on location profile if provided
        if location_data and 'demand_profile' in location_data:
            from locations import DEMAND_PROFILES
            profile = DEMAND_PROFILES.get(location_data['demand_profile'], {})
            
            # Apply location-specific adjustments
            if 'ac_sensitivity' in profile:
                # Adjust for AC demand sensitivity
                temp = features.get('temperature', 20)
                if temp > 25:
                    ac_factor = profile['ac_sensitivity']
                    features['total_demand'] = features.get('total_demand', 20000) * (1 + ac_factor * 0.1)
            
            if 'seasonal_factor' in profile:
                # Apply seasonal multiplier
                features['grid_stress'] = features.get('grid_stress', 0.5) * profile['seasonal_factor']
        
        logger.info(f"Adjusted total_demand: {features['total_demand']}")
        logger.info(f"Adjusted grid_stress: {features['grid_stress']}")

        # Build feature array matching training order
        if self.feature_names is None:
            raise ValueError("Model not trained or loaded properly - feature_names is None")
        
        X = np.array([[features.get(fname, 0) for fname in self.feature_names]])
        
        # Predict with clipping
        price = self.predict(X, clip_to_training=True)[0][0]
        
        # Additional sanity check
        price = float(np.clip(price, Config.PRICE_TYPICAL_MIN, Config.PRICE_TYPICAL_MAX))
        
        # Build response
        result = {
            'predicted_price': price,
            'features_used': features,
            'location': location_data.get('name', 'Unknown') if location_data else 'Unknown',
            'training_range': self.training_price_stats if self.training_price_stats else None
        }
        
        return result
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Dictionary with metrics (loss, mae, mape, smape, r2)
        """
        X_test_scaled = self.scaler.transform(X_test)
        loss, mae = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        
        y_pred = self.predict(X_test, clip_to_training=False)  # Don't clip during evaluation
        y_pred_flat = y_pred.flatten()
        
        # Standard MAPE (problematic with low values)
        # Add small epsilon to avoid division by zero, but filter out very low prices
        mask = y_test > 5.0  # Only calculate MAPE for prices > $5
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_test[mask] - y_pred_flat[mask]) / y_test[mask])) * 100
        else:
            mape = 100.0
        
        # Symmetric MAPE (better for low values)
        smape = np.mean(2 * np.abs(y_pred_flat - y_test) / (np.abs(y_test) + np.abs(y_pred_flat) + 1e-8)) * 100
        
        # Calculate R² score
        ss_res = np.sum((y_test - y_pred_flat) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_test - y_pred_flat) ** 2))
        
        # Analyze prediction distribution
        logger.info(f"Prediction analysis:")
        logger.info(f"  Actual prices: min=${np.min(y_test):.2f}, max=${np.max(y_test):.2f}, mean=${np.mean(y_test):.2f}")
        logger.info(f"  Predicted prices: min=${np.min(y_pred_flat):.2f}, max=${np.max(y_pred_flat):.2f}, mean=${np.mean(y_pred_flat):.2f}")
        logger.info(f"  Errors: mean=${np.mean(y_pred_flat - y_test):.2f}, std=${np.std(y_pred_flat - y_test):.2f}")
        
        # Check for systematic bias in low-price predictions
        low_price_mask = y_test < 30
        if np.sum(low_price_mask) > 10:
            low_mae = np.mean(np.abs(y_test[low_price_mask] - y_pred_flat[low_price_mask]))
            logger.info(f"  Low price (<$30) MAE: ${low_mae:.2f} (n={np.sum(low_price_mask)})")
        
        high_price_mask = y_test >= 50
        if np.sum(high_price_mask) > 10:
            high_mae = np.mean(np.abs(y_test[high_price_mask] - y_pred_flat[high_price_mask]))
            logger.info(f"  High price (≥$50) MAE: ${high_mae:.2f} (n={np.sum(high_price_mask)})")
        
        return {
            'loss': loss,
            'mae': mae,
            'mape': mape,
            'smape': smape,
            'rmse': rmse,
            'r2': r2
        }
    
    def save(self, model_path: str, scaler_path: str, features_path: str):
        """
        Save model and preprocessing objects
        
        Args:
            model_path: Path to save Keras model (.h5)
            scaler_path: Path to save scaler (.pkl)
            features_path: Path to save feature names (.json)
        """
        logger.info(f"Saving model to {model_path}...")
        self.model.save(model_path)
        
        # Save scaler and training stats together
        scaler_data = {
            'scaler': self.scaler,
            'training_price_stats': self.training_price_stats
        }
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_data, f)
        
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        logger.info("✓ Model saved successfully")
    
    @classmethod
    def load(cls, model_path: str, scaler_path: str, features_path: str):
        """
        Load model and preprocessing objects
        
        Args:
            model_path: Path to Keras model (.keras or .h5)
            scaler_path: Path to scaler (.pkl)
            features_path: Path to feature names (.json)
        
        Returns:
            Loaded PricePredictor instance
        """
        logger.info(f"Loading model from {model_path}...")
        
        # Try .keras first (new format), fall back to .h5
        if not os.path.exists(model_path):
            # Try alternative extension
            if model_path.endswith('.h5'):
                alt_path = model_path.replace('.h5', '.keras')
            else:
                alt_path = model_path.replace('.keras', '.h5')
            
            if os.path.exists(alt_path):
                logger.info(f"Model not found at {model_path}, using {alt_path}")
                model_path = alt_path
        
        model = keras.models.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            
        # Handle both old and new pickle formats
        if isinstance(scaler_data, dict):
            scaler = scaler_data['scaler']
            training_price_stats = scaler_data.get('training_price_stats')
        else:
            scaler = scaler_data  # Old format (just the scaler)
            training_price_stats = None
            logger.warning("Loaded old scaler format without training price stats")
        
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
        
        predictor = cls(len(feature_names))
        predictor.model = model
        predictor.scaler = scaler
        predictor.feature_names = feature_names
        predictor.training_price_stats = training_price_stats
        
        if training_price_stats:
            logger.info(f"✓ Model loaded successfully ({len(feature_names)} features)")
            logger.info(f"  Training price range: ${training_price_stats['min']:.2f} - ${training_price_stats['max']:.2f}")
        else:
            logger.info(f"✓ Model loaded successfully ({len(feature_names)} features) - no training stats")
        
        return predictor

# ============================================
# MODULE INFO
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("Smart Grid ML - Core Library v2.4.0")
    print("=" * 70)
    print()
    print("This module provides core ML functionality:")
    print("  • Config: Configuration management")
    print("  • PricePredictor: Neural network model with output clipping")
    print()
    print("Improvements:")
    print("  • Better regularization (L2, higher dropout)")
    print("  • Gradient clipping in optimizer")
    print("  • Early stopping callback")
    print("  • Prediction clipping based on training range")
    print()
    print("Usage:")
    print("  • Training: python train.py")
    print("  • Web Service: python app.py")
    print("  • Docker: docker-compose up")
    print()
    print("=" * 70)