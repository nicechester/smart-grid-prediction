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
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# TensorFlow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import from main modules
from main import Config, PricePredictor, generate_training_data
from tier2_pipeline import Tier2Config, Tier2DataPipeline, CAISOPriceFetcher, NOAAWeather
from tier2_pipeline import PowerPlantDB, DisasterRisk

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
    DATE_RANGE_MONTHS = 36  # Date range delta in months for data fetching

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
    """
    Build training data by combining CAISO price data and real weather data.
    """
    df = None

    if use_tier2:
        try:
            if pipeline is None:
                pipeline = Tier2DataPipeline(api_key=Config.EIA_API_KEY)
            
            # Create an instance of NOAAWeather
            noaa_weather = NOAAWeather()

            # Fetch weather data
            logger.info("1️⃣  Fetching weather data...")
            end_date = datetime.today()
            start_date = end_date - timedelta(days=TrainingConfig.DATE_RANGE_MONTHS * 30)
            
            weather_data = noaa_weather.get_all_california_weather(
                max_workers=5,
                start_date=start_date,
                end_date=end_date
            )

            if not weather_data:
                raise ValueError("Weather data fetch failed. No data available.")

            # Extract results from NOAA weather data grouped by city_id
            weather_records = []
            for city_id, response in weather_data.items():
                logger.info(f"Processing weather data for city_id: {city_id}")
                results = response.get('results', [])
                if not results:
                    logger.warning(f"No results found for city_id: {city_id}")
                    continue
                
                for record in results:
                    if 'date' not in record or 'value' not in record:
                        logger.warning(f"Skipping incomplete record for city_id: {city_id}")
                        continue
                    
                    record['city_id'] = city_id
                    record['timestamp'] = record.pop('date')
                    weather_records.append(record)
                
                logger.info(f"✓ Processed {len([r for r in weather_records if r['city_id'] == city_id])} weather records for city_id: {city_id}")

            if not weather_records:
                raise ValueError("No valid weather records found after filtering.")

            # Convert weather data to DataFrame
            weather_df = pd.DataFrame(weather_records)
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], errors='coerce')
            weather_df = weather_df.dropna(subset=['timestamp'])
            weather_df['city_id'] = weather_df['city_id'].astype(str)

            # PIVOT weather data from long to wide format
            logger.info("Pivoting weather data from long to wide format...")
            weather_pivot = weather_df.pivot_table(
                index=['timestamp', 'city_id'],
                columns='datatype',
                values='value',
                aggfunc='first'
            ).reset_index()
            
            weather_pivot.columns.name = None
            
            # Rename datatype codes to meaningful names
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
            
            # Fill missing weather values
            weather_defaults = {
                'temperature': 20.0,
                'temp_max': 25.0,
                'temp_min': 15.0,
                'precipitation': 0.0,
                'snowfall': 0.0,
                'wind_speed': 5.0
            }
            for col, default in weather_defaults.items():
                if col in weather_pivot.columns:
                    weather_pivot[col].fillna(default, inplace=True)
            
            logger.info(f"✓ Pivoted weather data: {len(weather_pivot)} records, {len(weather_pivot.columns)} columns")

            # Fetch CAISO price data
            logger.info("2️⃣  Fetching CAISO price data...")
            caiso_prices_dict = CAISOPriceFetcher.fetch_all_cities_prices(
                start_date=start_date.strftime('%Y%m%d'), 
                end_date=end_date.strftime('%Y%m%d'),
                market='DAM',
                max_workers=Tier2Config.MAX_WORKERS
            )

            if not caiso_prices_dict:
                raise ValueError("CAISO data fetch failed. No data available.")

            # Combine CAISO data into a single DataFrame
            caiso_combined = pd.concat(list(caiso_prices_dict.values()), ignore_index=True)
            caiso_combined.rename(columns={'lmp': 'price'}, inplace=True)
            caiso_combined['city_id'] = caiso_combined['city_id'].astype(str)
            caiso_combined.dropna(subset=['price'], inplace=True)

            # Merge CAISO data with pivoted weather data
            logger.info("3️⃣  Merging CAISO and weather data...")
            caiso_combined['timestamp'] = pd.to_datetime(caiso_combined['timestamp']).dt.tz_localize(None)
            weather_pivot['timestamp'] = pd.to_datetime(weather_pivot['timestamp']).dt.tz_localize(None)
            
            df = pd.merge_asof(
                caiso_combined.sort_values('timestamp'),
                weather_pivot.sort_values('timestamp'),
                on='timestamp',
                by='city_id',
                direction='nearest'
            )

            if df is None or len(df) == 0:
                raise ValueError("Merging CAISO and weather data failed. No data available.")
            
            logger.info(f"✓ Merged dataset contains {len(df)} records.")
            
            # Drop metadata columns before training
            metadata_cols = ['node', 'city_id', 'city_name', 'zone']
            cols_to_drop = [col for col in metadata_cols if col in df.columns]
            if cols_to_drop:
                logger.info(f"Dropping metadata columns: {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)
            
            # Add time features for seasonality
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
            
            # Generate derived grid features (matching app.py)
            # Cloud cover estimation
            df['cloud_cover'] = np.clip(0.4 + 0.1 * np.sin(2 * np.pi * df['day_of_year'] / 365), 0, 1)
            
            # Solar generation (zero at night, peak at noon)
            df['solar_mw'] = 0.0
            daytime = (df['hour'] >= 6) & (df['hour'] <= 18)
            df.loc[daytime, 'solar_mw'] = (
                2000 * (1 - df.loc[daytime, 'cloud_cover']) * 
                np.sin(np.pi * (df.loc[daytime, 'hour'] - 6) / 12)
            )
            
            # Wind generation (from wind_speed)
            df['wind_mw'] = np.maximum(0, 800 * (df['wind_speed'] / 10))
            
            # Demand estimation
            time_multiplier = np.ones(len(df))
            morning_peak = df['hour'].isin([6, 7, 8])
            evening_peak = df['hour'].isin([17, 18, 19, 20])
            night = (df['hour'] >= 23) | (df['hour'] <= 5)
            
            time_multiplier[morning_peak] = 1.3
            time_multiplier[evening_peak] = 1.4
            time_multiplier[night] = 0.8
            
            base_demand = 20000
            weekend_factor = np.where(df['is_weekend'], 0.9, 1.0)
            
            # df['total_demand'] = base_demand * time_multiplier * weekend_factor

            # --- CORRECTED LOGIC for train.py ---
            # 1. Convert to Celsius (this is a Series)
            temp_c = (df['temperature'] - 32) * 5 / 9

            # 2. Define threshold and sensitivity
            ac_sensitivity_threshold_c = 25.0
            ac_sensitivity = 0.5 # Default for training

            # 3. Calculate temperature delta
            temp_delta_c = temp_c - ac_sensitivity_threshold_c

            # 4. Use np.where to calculate ac_demand
            # if delta > 0, calculate demand, otherwise set to 0.0
            df['ac_demand'] = np.where(
                temp_delta_c > 0,  # Condition
                500 * (temp_delta_c ** 1.5) * ac_sensitivity,  # Value if True
                0.0  # Value if False
            )

            # 5. Now calculate total_demand
            # (Note: make sure 'base_demand', 'time_multiplier', and 'weekend_factor' are defined
            # as per the surrounding code in train.py)
            df['total_demand'] = (base_demand + df['ac_demand']) * time_multiplier * weekend_factor

            # --- End of corrected logic ---

            # Grid metrics
            total_renewable = df['solar_mw'] + df['wind_mw']
            df['renewable_pct'] = np.clip(total_renewable / df['total_demand'], 0, 1)
            df['imbalance'] = df['total_demand'] / (total_renewable + 5000)
            df['grid_stress'] = np.clip(df['total_demand'] / 35000, 0, 1)
            df['wildfire_risk'] = 0.0
            
            logger.info(f"✓ Added grid features (solar_mw, wind_mw, total_demand, etc.)")
            
            # Fetch power plant data (static aggregated features)
            logger.info("4️⃣  Fetching power plant data...")
            power_plants_df = PowerPlantDB.download_plants()

            if power_plants_df is not None:
                total_capacity = power_plants_df['capacity_mw'].sum()
                avg_capacity = power_plants_df['capacity_mw'].mean()
                plant_count = len(power_plants_df)

                df['total_plant_capacity'] = total_capacity
                df['avg_plant_capacity'] = avg_capacity
                df['plant_count'] = plant_count
                logger.info(f"✓ Added power plant features: Total Capacity={total_capacity:.2f}, Count={plant_count}")
            else:
                logger.warning("No power plant data available.")
                df['total_plant_capacity'] = 0
                df['avg_plant_capacity'] = 0
                df['plant_count'] = 0
            
            # Fetch earthquake data
            logger.info("5️⃣  Fetching earthquake data...")
            earthquakes_df = DisasterRisk.get_recent_earthquakes(
                start_date=start_date,
                end_date=end_date,
                min_magnitude=2.0
            )

            if earthquakes_df is not None and len(earthquakes_df) > 0:
                quake_count = len(earthquakes_df)
                avg_magnitude = earthquakes_df['magnitude'].mean()
                max_magnitude = earthquakes_df['magnitude'].max()

                df['recent_quake_count'] = quake_count
                df['avg_quake_magnitude'] = avg_magnitude
                df['max_quake_magnitude'] = max_magnitude
                logger.info(f"✓ Added earthquake features: Count={quake_count}, Avg Mag={avg_magnitude:.2f}, Max Mag={max_magnitude:.2f}")
            else:
                logger.warning("No earthquake data available.")
                df['recent_quake_count'] = 0
                df['avg_quake_magnitude'] = 0.0
                df['max_quake_magnitude'] = 0.0
        
        except Exception as e:
            logger.error(f"Data fetch or merge failed: {e}", exc_info=True)
            raise ValueError("Data preparation failed.")
    
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
    parser.add_argument('--date-range-months', type=int, default=TrainingConfig.DATE_RANGE_MONTHS, help='Override date range in months for data fetching')
    
    args = parser.parse_args()

    # Override TrainingConfig.DATE_RANGE_MONTHS if provided
    TrainingConfig.DATE_RANGE_MONTHS = args.date_range_months

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