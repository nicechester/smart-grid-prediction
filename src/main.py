#!/usr/bin/env python3
"""
Smart Grid Electricity Price Prediction System
Core ML Library - Model Definition and Configuration
Python 3.11, TensorFlow 2.14.0

This module provides:
- Configuration management
- Neural network model (PricePredictor)
- Synthetic data generation (fallback)
"""

import os
import json
import pickle
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
# SYNTHETIC DATA GENERATION (FALLBACK)
# ============================================

def generate_training_data(days: int = 365) -> pd.DataFrame:
    """
    Generate realistic synthetic training data
    Used as fallback when Tier 2 data is unavailable
    
    Args:
        days: Number of days to generate (default 365)
    
    Returns:
        DataFrame with synthetic electricity grid data
    """
    np.random.seed(Config.RANDOM_STATE)
    
    logger.info(f"Generating {days} days of synthetic training data...")
    
    data = []
    for day in range(days):
        for hour in range(24):
            # Time features
            month = (day % 365) // 30 + 1
            is_weekend = day % 7 >= 5
            
            # Weather patterns (seasonal + daily cycles)
            base_temp = 20 + 10 * np.sin(2 * np.pi * day / 365)  # Yearly cycle
            daily_swing = 8 * np.sin(2 * np.pi * hour / 24 - np.pi/2)  # Daily cycle
            temperature = base_temp + daily_swing + np.random.normal(0, 1)
            
            cloud_cover = np.clip(0.4 + 0.3 * np.sin(2 * np.pi * day / 30) + np.random.normal(0, 0.1), 0, 1)
            wind_speed = np.clip(5 + 3 * np.sin(2 * np.pi * day / 20) + np.random.normal(0, 1), 0, 15)
            
            # Renewable generation
            if 6 <= hour <= 18:
                solar_mw = 2000 * (1 - cloud_cover) * np.sin(np.pi * (hour - 6) / 12)
            else:
                solar_mw = 0
            
            wind_mw = max(0, 800 * (wind_speed / 10) + np.random.normal(0, 50))
            
            # Demand patterns
            base_demand = 20000
            ac_demand = 0
            if temperature > 25:
                ac_demand = 500 * ((temperature - 25) ** 1.5)
            
            # Time-of-day multipliers
            if hour in [6, 7, 8]:
                time_multiplier = 1.3  # Morning peak
            elif hour in [17, 18, 19, 20]:
                time_multiplier = 1.4  # Evening peak
            elif 23 <= hour or hour <= 5:
                time_multiplier = 0.8  # Night
            else:
                time_multiplier = 1.0
            
            weekend_factor = 0.9 if is_weekend else 1.0
            
            total_demand = (base_demand + ac_demand) * time_multiplier * weekend_factor + np.random.normal(0, 200)
            
            # Price calculation
            total_renewable = solar_mw + wind_mw
            renewable_pct = total_renewable / total_demand if total_demand > 0 else 0
            imbalance = total_demand / (total_renewable + 5000)
            grid_stress = total_demand / 35000
            
            base_price = 40
            price = base_price * imbalance * (1 + grid_stress * 0.5)
            
            # Wildfire risk (occasional events)
            wildfire_risk = np.random.choice([0, 0, 0, 0, 0, 0.3, 0.5, 0.7, 1.0])
            if wildfire_risk > 0.5:
                price *= 1.5
            
            data.append({
                'temperature': temperature,
                'cloud_cover': cloud_cover,
                'wind_speed': wind_speed,
                'solar_mw': solar_mw,
                'wind_mw': wind_mw,
                'total_demand': total_demand,
                'renewable_pct': np.clip(renewable_pct, 0, 1),
                'imbalance': imbalance,
                'grid_stress': np.clip(grid_stress, 0, 1),
                'wildfire_risk': wildfire_risk,
                'hour': hour,
                'month': month,
                'is_weekend': int(is_weekend),
                'price': price,
            })
    
    df = pd.DataFrame(data)
    logger.info(f"✓ Generated {len(df)} samples with {len(df.columns)} features")
    return df

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
    
    def build_model(self) -> keras.Model:
        """
        Build neural network architecture
        
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building neural network with {self.feature_dim} input features...")
        
        self.model = Sequential([
            # Layer 1: 128 neurons
            layers.Dense(128, activation='relu', input_shape=(self.feature_dim,)),
            layers.Dropout(0.2),
            
            # Layer 2: 64 neurons with batch normalization
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Layer 3: 32 neurons
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            
            # Layer 4: 16 neurons
            layers.Dense(16, activation='relu'),
            
            # Output layer: 1 neuron (price)
            layers.Dense(1)
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
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
        
        # Normalize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # Train
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=0.2 if validation_data is None else None,
            verbose=verbose
        )
        
        logger.info("✓ Training complete")
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict prices
        
        Args:
            X: Input features
        
        Returns:
            Predicted prices
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0)
    
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
        
        # Build feature array matching training order
        if self.feature_names is None:
            raise ValueError("Model not trained or loaded properly - feature_names is None")
        
        X = np.array([[features.get(fname, 0) for fname in self.feature_names]])
        
        # Predict
        price = self.predict(X)[0][0]
        
        # Build response
        result = {
            'predicted_price': float(price),
            'features_used': features,
            'location': location_data.get('name', 'Unknown') if location_data else 'Unknown'
        }
        
        return result
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Dictionary with metrics (loss, mae, mape)
        """
        X_test_scaled = self.scaler.transform(X_test)
        loss, mae = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        
        y_pred = self.predict(X_test)
        mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
        
        return {
            'loss': loss,
            'mae': mae,
            'mape': mape,
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
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        logger.info("✓ Model saved successfully")
    
    @classmethod
    def load(cls, model_path: str, scaler_path: str, features_path: str):
        """
        Load model and preprocessing objects
        
        Args:
            model_path: Path to Keras model (.h5)
            scaler_path: Path to scaler (.pkl)
            features_path: Path to feature names (.json)
        
        Returns:
            Loaded PricePredictor instance
        """
        logger.info(f"Loading model from {model_path}...")
        
        model = keras.models.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
        
        predictor = cls(len(feature_names))
        predictor.model = model
        predictor.scaler = scaler
        predictor.feature_names = feature_names
        
        logger.info(f"✓ Model loaded successfully ({len(feature_names)} features)")
        return predictor

# ============================================
# MODULE INFO
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("Smart Grid ML - Core Library")
    print("=" * 70)
    print()
    print("This module provides core ML functionality:")
    print("  • Config: Configuration management")
    print("  • PricePredictor: Neural network model")
    print("  • generate_training_data(): Synthetic data generator")
    print()
    print("Usage:")
    print("  • Training: python train.py")
    print("  • Web Service: python app.py")
    print("  • Docker: docker-compose up")
    print()
    print("=" * 70)