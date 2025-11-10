#!/usr/bin/env python3
"""
train.py - Standalone model retraining script
FIXED: Match prediction feature engineering exactly
"""

import os
import sys
import json
import pickle
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# TensorFlow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import from main modules
from main import Config, PricePredictor
from locations import DEMAND_PROFILES
from data_loader import load_downloaded_data

# Setup logging - create directory first
os.makedirs('/app/data/training', exist_ok=True)
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
# EXPLICIT FEATURE DEFINITION
# ============================================

# Define EXACT features the model should use
TRAINING_FEATURES = [
    # Base weather and time features
    'temperature',
    'wind_speed',
    'hour',
    'month',
    'day_of_year',
    'day_of_week',
    'is_weekend',
    
    # Cyclical encodings
    'hour_sin',
    'hour_cos',
    'month_sin',
    'month_cos',
    'doy_sin',
    'doy_cos',
    'dow_sin',
    'dow_cos',
    
    # Grid features
    'cloud_cover',
    'solar_mw',
    'wind_mw',
    'total_demand',
    'renewable_pct',
    'imbalance',
    'grid_stress',
    'wildfire_risk',
    
    # Lag features (4 variables × 4 lags = 16)
    'price_lag_1',
    'price_lag_3',
    'price_lag_6',
    'price_lag_12',
    'temperature_lag_1',
    'temperature_lag_3',
    'temperature_lag_6',
    'temperature_lag_12',
    'solar_mw_lag_1',
    'solar_mw_lag_3',
    'solar_mw_lag_6',
    'solar_mw_lag_12',
    'wind_mw_lag_1',
    'wind_mw_lag_3',
    'wind_mw_lag_6',
    'wind_mw_lag_12',
    
    # Power plant features
    'total_plant_capacity',
    'avg_plant_capacity',
    'plant_count',
    
    # Earthquake features
    'recent_quake_count',
    'avg_quake_magnitude',
    'max_quake_magnitude',
    
    # CAISO price components
    'energy',
    'congestion',
    'loss',
    
    # Weather aggregates (mapped NOAA datatypes)
    'temp_avg',
    'temp_max',
    'temp_min',
    'precipitation',
    'snowfall',
]

# Unmapped NOAA datatypes (optional)
NOAA_UNMAPPED_DATATYPES = [
    'ADPT', 'ASLP', 'ASTP', 'AWBT', 'DAPR', 'MDPR', 'PGTM',
    'RHAV', 'RHMN', 'RHMX', 'SNWD', 'WDF2', 'WDF5', 'WESD',
    'WESF', 'WSF2', 'WSF5', 'WT01', 'WT02', 'WT03', 'WT07', 'WT08'
]

# ============================================
# Training Configuration
# ============================================

class TrainingConfig:
    """Retraining configuration"""
    
    EPOCHS = 50
    BATCH_SIZE = 32
    TEST_SIZE = 0.2
    MIN_SAMPLES = 500
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42
    DATA_DIR = '/app/data/downloads'  # Where downloaded data is stored
    MAX_DATE_RANGE_MONTHS = 6  # Limit to avoid data quality issues

# ============================================
# Data Loading Functions
# ============================================

def load_downloaded_data() -> dict:
    """Load all previously downloaded data from pickle files"""
    logger.info("=" * 70)
    logger.info("LOADING DOWNLOADED DATA")
    logger.info("=" * 70)
    
    data = {}
    
    # Check if data directory exists
    if not os.path.exists(TrainingConfig.DATA_DIR):
        raise ValueError(f"Data directory not found: {TrainingConfig.DATA_DIR}")
    
    # Load weather data
    weather_path = os.path.join(TrainingConfig.DATA_DIR, 'weather_data.pkl')
    if os.path.exists(weather_path):
        data['weather'] = pd.read_pickle(weather_path)
        logger.info(f"✅ Loaded {len(data['weather'])} weather records")
    else:
        raise ValueError(f"Weather data not found: {weather_path}")
    
    # Load CAISO prices
    caiso_path = os.path.join(TrainingConfig.DATA_DIR, 'caiso_prices.pkl')
    if os.path.exists(caiso_path):
        data['caiso'] = pd.read_pickle(caiso_path)
        logger.info(f"✅ Loaded {len(data['caiso'])} CAISO price records")
    else:
        raise ValueError(f"CAISO data not found: {caiso_path}")
    
    # Load power plants
    plants_path = os.path.join(TrainingConfig.DATA_DIR, 'power_plants.pkl')
    if os.path.exists(plants_path):
        data['plants'] = pd.read_pickle(plants_path)
        logger.info(f"✅ Loaded {len(data['plants'])} power plants")
    else:
        logger.warning(f"Power plants data not found: {plants_path}")
        data['plants'] = pd.DataFrame()
    
    # Load earthquakes
    quakes_path = os.path.join(TrainingConfig.DATA_DIR, 'earthquakes.pkl')
    if os.path.exists(quakes_path):
        data['earthquakes'] = pd.read_pickle(quakes_path)
        logger.info(f"✅ Loaded {len(data['earthquakes'])} earthquake records")
    else:
        logger.warning(f"Earthquake data not found: {quakes_path}")
        data['earthquakes'] = pd.DataFrame()
    
    logger.info("=" * 70)
    logger.info("")
    
    return data

# ============================================
# Feature Engineering
# ============================================

def add_lag_features(df: pd.DataFrame, lags: list = [1, 3, 6, 12]) -> pd.DataFrame:
    """Add historical lag features - only for numeric columns"""
    numeric_cols = ['temperature', 'solar_mw', 'wind_mw', 'price']
    
    for col in numeric_cols:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Fill NaN values with forward/backward fill
    return df.bfill().ffill()

def calculate_demand_with_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate demand using SAME logic as prediction (app.py)
    Includes AC demand, seasonal factors, and demand profiles
    """
    logger.info("Calculating demand with profiles (matching prediction logic)...")
    
    # Get default demand profile (urban_tech)
    default_profile = DEMAND_PROFILES.get('urban_tech', {})
    ac_sensitivity = default_profile.get('ac_sensitivity', 0.5)
    peak_hours = default_profile.get('peak_hours', [9, 18])
    seasonal_factor = default_profile.get('seasonal_factor', 1.0)
    
    base_demand = 20000
    
    # AC demand based on temperature (SAME as prediction)
    df['ac_demand'] = 0.0
    hot_mask = df['temperature'] > 25
    df.loc[hot_mask, 'ac_demand'] = 500 * ((df.loc[hot_mask, 'temperature'] - 25) ** 1.5) * ac_sensitivity
    
    # Time multiplier (SAME as prediction)
    df['time_multiplier'] = 1.0
    morning_peak = df['hour'].isin([6, 7, 8])
    evening_peak = df['hour'].isin([17, 18, 19, 20])
    in_peak_hours = df['hour'].isin(peak_hours)
    night = (df['hour'] >= 23) | (df['hour'] <= 5)
    
    df.loc[morning_peak, 'time_multiplier'] = 1.3
    df.loc[evening_peak, 'time_multiplier'] = 1.4
    df.loc[in_peak_hours, 'time_multiplier'] = 1.4
    df.loc[night, 'time_multiplier'] = 0.8
    
    # Weekend factor
    df['weekend_factor'] = df['is_weekend'].apply(lambda x: 0.9 if x else 1.0)
    
    # Calculate total demand (SAME as prediction)
    df['total_demand'] = (base_demand + df['ac_demand']) * df['time_multiplier'] * df['weekend_factor'] * seasonal_factor
    
    # Clean up temporary columns
    df.drop(['ac_demand', 'time_multiplier', 'weekend_factor'], axis=1, inplace=True)
    
    logger.info(f"Demand range: {df['total_demand'].min():.0f} - {df['total_demand'].max():.0f} MW")
    logger.info(f"Demand mean: {df['total_demand'].mean():.0f} MW")
    
    return df

# ============================================
# Data Building from Files
# ============================================

def build_training_data_from_files() -> pd.DataFrame:
    """
    Build training data from pre-downloaded files.
    Returns DataFrame with EXPLICIT feature columns only.
    FIXED: Now matches prediction feature engineering exactly
    """
    
    # Load all data
    data = load_downloaded_data()
    
    weather_df = data['weather']
    caiso_df = data['caiso']
    plants_df = data['plants']
    earthquakes_df = data['earthquakes']
    
    # Process weather data
    logger.info("Processing weather data...")
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], errors='coerce')
    weather_df = weather_df.dropna(subset=['timestamp'])
    weather_df['city_id'] = weather_df['city_id'].astype(str)
    
    # Pivot weather data
    logger.info("Pivoting weather data from long to wide format...")
    weather_pivot = weather_df.pivot_table(
        index=['timestamp', 'city_id'],
        columns='datatype',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    weather_pivot.columns.name = None
    
    # Rename datatype codes
    weather_column_mapping = {
        'PRCP': 'precipitation',
        'SNOW': 'snowfall',
        'TMAX': 'temp_max',
        'TMIN': 'temp_min',
        'TOBS': 'temperature',
        'AWND': 'wind_speed',
        'TAVG': 'temp_avg'
    }
    weather_pivot = weather_pivot.rename(columns=weather_column_mapping)
    
    # CRITICAL: NOAA GHCND returns values in TENTHS
    # Convert to actual units to match defaults and expectations
    temp_columns = ['temperature', 'temp_max', 'temp_min', 'temp_avg']
    for col in temp_columns:
        if col in weather_pivot.columns:
            weather_pivot[col] = weather_pivot[col] / 10.0
    
    if 'precipitation' in weather_pivot.columns:
        weather_pivot['precipitation'] = weather_pivot['precipitation'] / 10.0
    
    if 'snowfall' in weather_pivot.columns:
        weather_pivot['snowfall'] = weather_pivot['snowfall'] / 10.0
    
    if 'wind_speed' in weather_pivot.columns:
        weather_pivot['wind_speed'] = weather_pivot['wind_speed'] / 10.0
    
    # Convert unmapped NOAA datatypes (except WT* boolean flags)
    for col in weather_pivot.columns:
        if col not in ['timestamp', 'city_id'] + list(weather_column_mapping.values()):
            if not col.startswith('WT'):
                weather_pivot[col] = weather_pivot[col] / 10.0
    
    logger.info("✅ Converted NOAA values from tenths to actual units")
    
    # Fill missing weather values (now in actual units)
    weather_defaults = {
        'temperature': 20.0,
        'temp_max': 25.0,
        'temp_min': 15.0,
        'precipitation': 0.0,
        'snowfall': 0.0,
        'wind_speed': 5.0,
        'temp_avg': 20.0
    }
    for col, default in weather_defaults.items():
        if col in weather_pivot.columns:
            weather_pivot[col].fillna(default, inplace=True)
    
    logger.info(f"✅ Pivoted weather data: {len(weather_pivot)} records")
    
    # Process CAISO data
    logger.info("Processing CAISO price data...")
    caiso_df.rename(columns={'lmp': 'price'}, inplace=True)
    caiso_df['city_id'] = caiso_df['city_id'].astype(str)
    caiso_df.dropna(subset=['price'], inplace=True)
    
    # Merge CAISO + weather
    logger.info("Merging CAISO and weather data...")
    caiso_df['timestamp'] = pd.to_datetime(caiso_df['timestamp']).dt.tz_localize(None)
    weather_pivot['timestamp'] = pd.to_datetime(weather_pivot['timestamp']).dt.tz_localize(None)
    
    df = pd.merge_asof(
        caiso_df.sort_values('timestamp'),
        weather_pivot.sort_values('timestamp'),
        on='timestamp',
        by='city_id',
        direction='nearest'
    )
    
    if df is None or len(df) == 0:
        raise ValueError("Merging failed.")
    
    logger.info(f"✅ Merged dataset: {len(df)} records")
    
    # Drop metadata columns
    metadata_cols = ['node', 'city_id', 'city_name', 'zone']
    cols_to_drop = [col for col in metadata_cols if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # Add time features
    logger.info("Adding time and derived features...")
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    df['day_of_year'] = pd.to_datetime(df['timestamp']).dt.dayofyear
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Generate derived grid features
    df['cloud_cover'] = np.clip(0.4 + 0.1 * np.sin(2 * np.pi * df['day_of_year'] / 365), 0, 1)
    
    # Solar generation (SAME as prediction)
    df['solar_mw'] = 0.0
    daytime = (df['hour'] >= 6) & (df['hour'] <= 18)
    df.loc[daytime, 'solar_mw'] = (
        2000 * (1 - df.loc[daytime, 'cloud_cover']) * 
        np.sin(np.pi * (df.loc[daytime, 'hour'] - 6) / 12)
    )
    
    # Wind generation (SAME as prediction)
    df['wind_mw'] = np.maximum(0, 800 * (df['wind_speed'] / 10))
    
    # FIXED: Use the SAME demand calculation as prediction
    df = calculate_demand_with_profiles(df)
    
    # Grid metrics (SAME as prediction)
    total_renewable = df['solar_mw'] + df['wind_mw']
    df['renewable_pct'] = np.clip(total_renewable / df['total_demand'], 0, 1)
    df['imbalance'] = df['total_demand'] / (total_renewable + 5000)
    df['grid_stress'] = np.clip(df['total_demand'] / 35000, 0, 1)
    df['wildfire_risk'] = 0.0
    
    # Power plant data
    logger.info("Adding power plant features...")
    if len(plants_df) > 0:
        df['total_plant_capacity'] = plants_df['capacity_mw'].sum()
        df['avg_plant_capacity'] = plants_df['capacity_mw'].mean()
        df['plant_count'] = len(plants_df)
    else:
        df['total_plant_capacity'] = 0
        df['avg_plant_capacity'] = 0
        df['plant_count'] = 0
    
    # Earthquake data
    logger.info("Adding earthquake features...")
    if len(earthquakes_df) > 0:
        df['recent_quake_count'] = len(earthquakes_df)
        df['avg_quake_magnitude'] = earthquakes_df['magnitude'].mean()
        df['max_quake_magnitude'] = earthquakes_df['magnitude'].max()
    else:
        df['recent_quake_count'] = 0
        df['avg_quake_magnitude'] = 0.0
        df['max_quake_magnitude'] = 0.0
    
    # Ensure CAISO component columns exist
    if 'energy' not in df.columns:
        df['energy'] = 0.0
    if 'congestion' not in df.columns:
        df['congestion'] = 0.0
    if 'loss' not in df.columns:
        df['loss'] = 0.0
    
    logger.info(f"✅ Final dataset: {len(df)} records with {len(df.columns)} columns")
    
    return df

# ============================================
# Model Training with Statistics Saving
# ============================================

def train_model(df, epochs=None, batch_size=None):
    """
    Train the prediction model and save training statistics
    """
    
    epochs = epochs or TrainingConfig.EPOCHS
    batch_size = batch_size or TrainingConfig.BATCH_SIZE
    
    logger.info("=" * 70)
    logger.info("TRAINING MODEL")
    logger.info("=" * 70)
    
    # Drop timestamp before processing
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    if len(datetime_cols) > 0:
        logger.info(f"Dropping datetime columns: {datetime_cols.tolist()}")
        df = df.drop(columns=datetime_cols)
    
    # Add lag features
    logger.info("1. Adding lag features...")
    df = add_lag_features(df)
    
    # CRITICAL: Ensure ALL training features exist
    logger.info("2. Ensuring all required features exist...")
    all_features = TRAINING_FEATURES.copy()
    
    # Optionally include unmapped NOAA datatypes
    all_features.extend(NOAA_UNMAPPED_DATATYPES)
    
    for feature in all_features:
        if feature not in df.columns:
            logger.warning(f"Feature '{feature}' missing. Adding with default value 0.0")
            df[feature] = 0.0
        elif df[feature].isna().all():
            # CRITICAL: If feature has ALL NaN values, fill with 0
            logger.warning(f"Feature '{feature}' has all NaN values. Filling with 0.0")
            df[feature] = 0.0
        elif df[feature].isna().any():
            # If feature has SOME NaN values, fill with median
            median_val = df[feature].median()
            if pd.isna(median_val):
                median_val = 0.0
            logger.warning(f"Feature '{feature}' has {df[feature].isna().sum()} NaN values. Filling with median: {median_val:.2f}")
            df[feature].fillna(median_val, inplace=True)
    
    # Select ONLY the features we want
    feature_cols = all_features
    logger.info(f"✅ Using {len(feature_cols)} features")
    logger.info(f"  First 10: {feature_cols[:10]}")
    
    # SAVE TRAINING STATISTICS for each feature
    logger.info("3. Computing training statistics...")
    feature_stats = {}
    for col in feature_cols:
        if col in df.columns:
            col_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(col_data) == 0:
                logger.warning(f"Feature '{col}' has no valid data. Using default stats.")
                feature_stats[col] = {
                    'mean': 0.0,
                    'std': 1.0,
                    'min': 0.0,
                    'max': 0.0,
                    'median': 0.0
                }
            else:
                feature_stats[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std() if col_data.std() > 0 else 1.0),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'median': float(col_data.median())
                }
    
    # Log key statistics for debugging
    logger.info("Key feature statistics:")
    for key_feat in ['temperature', 'total_demand', 'price', 'solar_mw', 'wind_mw']:
        if key_feat in feature_stats:
            stats = feature_stats[key_feat]
            logger.info(f"  {key_feat}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    # CRITICAL: Check for any remaining NaN or inf values
    logger.info("4. Validating data integrity...")
    nan_cols = df[feature_cols].columns[df[feature_cols].isna().any()].tolist()
    if nan_cols:
        logger.error(f"Features still have NaN values: {nan_cols}")
        for col in nan_cols:
            df[col].fillna(0.0, inplace=True)
        logger.warning("Filled remaining NaN values with 0.0")
    
    inf_cols = df[feature_cols].columns[np.isinf(df[feature_cols]).any()].tolist()
    if inf_cols:
        logger.error(f"Features have infinite values: {inf_cols}")
        for col in inf_cols:
            df[col].replace([np.inf, -np.inf], 0.0, inplace=True)
        logger.warning("Replaced infinite values with 0.0")
    
    # Save feature statistics
    stats_path = os.path.join(Config.MODEL_DIR, 'feature_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(feature_stats, f, indent=2)
    logger.info(f"✅ Saved feature statistics to {stats_path}")
    
    # Verify all columns are numeric
    non_numeric = df[feature_cols].select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        logger.error(f"Non-numeric columns: {non_numeric.tolist()}")
        raise ValueError(f"Non-numeric features: {non_numeric.tolist()}")
    
    # Prepare data
    logger.info("5. Preparing data...")
    X = df[feature_cols].values
    y = df['price'].values
    
    # Final validation
    if np.isnan(X).any():
        logger.error(f"X still contains {np.isnan(X).sum()} NaN values!")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        logger.warning("Replaced NaN/inf in X with 0.0")
    
    if np.isnan(y).any():
        logger.error(f"y still contains {np.isnan(y).sum()} NaN values!")
        # Remove rows where price is NaN
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        logger.warning(f"Removed {(~valid_mask).sum()} rows with NaN prices")
    
    logger.info(f"✅ X shape: {X.shape}")
    logger.info(f"✅ y range: ${y.min():.2f} - ${y.max():.2f}/MWh")
    logger.info(f"✅ y mean: ${y.mean():.2f}/MWh")
    logger.info(f"✅ y median: ${np.median(y):.2f}/MWh")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TrainingConfig.TEST_SIZE, 
        random_state=TrainingConfig.RANDOM_STATE, shuffle=False
    )
    logger.info(f"✅ Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")
    
    # Build model
    logger.info("6. Building neural network...")
    predictor = PricePredictor(X_train.shape[1])
    model = predictor.build_model()
    
    # Train
    logger.info(f"7. Training ({epochs} epochs)...")
    history = predictor.train(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    logger.info("✅ Training complete")
    
    # Evaluate
    logger.info("8. Evaluating model...")
    metrics = predictor.evaluate(X_test, y_test)
    logger.info(f"✅ Test MAE: ${metrics['mae']:.2f}/MWh")
    logger.info(f"✅ Test RMSE: ${metrics['rmse']:.2f}/MWh")
    logger.info(f"✅ Test MAPE: {metrics['mape']:.2f}% (prices > $5 only)")
    logger.info(f"✅ Test SMAPE: {metrics['smape']:.2f}% (symmetric, better for low prices)")
    logger.info(f"✅ Test R²: {metrics['r2']:.4f}")
    
    # Save model
    logger.info("9. Saving model...")
    predictor.feature_names = feature_cols
    predictor.save(
        os.path.join(Config.MODEL_DIR, 'price_model.keras'),
        os.path.join(Config.MODEL_DIR, 'scaler.pkl'),
        os.path.join(Config.MODEL_DIR, 'features.json')
    )
    logger.info(f"✅ Model saved to {Config.MODEL_DIR}/")
    
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
            'mean': float(y.mean()),
            'median': float(np.median(y))
        },
        'metrics': {
            'loss': float(metrics['loss']),
            'mae': float(metrics['mae']),
            'mape': float(metrics['mape']),
            'smape': float(metrics['smape']),
            'rmse': float(metrics['rmse']),
            'r2': float(metrics['r2'])
        },
        'version': '2.4.0-demand-profile-fix'
    }
    
    with open(os.path.join(Config.MODEL_DIR, 'training_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("✅ Metadata saved")
    
    return predictor, metrics

# ============================================
# Main Training Pipeline
# ============================================

def main():
    """Main training pipeline"""
    
    parser = argparse.ArgumentParser(description='Retrain electricity price prediction model')
    parser.add_argument('--epochs', type=int, default=TrainingConfig.EPOCHS)
    parser.add_argument('--batch-size', type=int, default=TrainingConfig.BATCH_SIZE)
    parser.add_argument('--data-dir', type=str, default=TrainingConfig.DATA_DIR,
                        help='Directory containing downloaded data files')
    
    args = parser.parse_args()
    TrainingConfig.DATA_DIR = args.data_dir

    logger.info("")
    logger.info("=" * 70)
    logger.info("SMART GRID ML - MODEL RETRAINING (DEMAND PROFILE FIX)")
    logger.info("=" * 70)
    logger.info(f"Data directory: {TrainingConfig.DATA_DIR}")
    logger.info(f"Using {len(TRAINING_FEATURES)} core features")
    logger.info(f"Plus {len(NOAA_UNMAPPED_DATATYPES)} NOAA datatypes")
    logger.info(f"Total: {len(TRAINING_FEATURES) + len(NOAA_UNMAPPED_DATATYPES)} features")
    logger.info("FIX: Now using SAME demand calculation as prediction (with profiles)")
    logger.info("")
    
    try:
        Config.init()
        
        # Build training data from files
        df = build_training_data_from_files()
        
        if df is None or len(df) < TrainingConfig.MIN_SAMPLES:
            logger.error(f"Insufficient data: {len(df) if df is not None else 0}")
            return 1
        
        # Train model
        predictor, metrics = train_model(
            df,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE ✅")
        logger.info("=" * 70)
        logger.info(f"Test MAE: ${metrics['mae']:.2f}/MWh")
        logger.info(f"Test RMSE: ${metrics['rmse']:.2f}/MWh")
        logger.info(f"Test MAPE: {metrics['mape']:.2f}% (prices > $5)")
        logger.info(f"Test SMAPE: {metrics['smape']:.2f}% (symmetric)")
        logger.info(f"Test R²: {metrics['r2']:.4f}")
        logger.info(f"Feature statistics saved for {len(TRAINING_FEATURES) + len(NOAA_UNMAPPED_DATATYPES)} features")
        logger.info("")
        
        return 0
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())