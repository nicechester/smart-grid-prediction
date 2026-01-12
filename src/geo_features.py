#!/usr/bin/env python3
"""
geo_features.py - Feature Engineering for Geolocation-Based Price Prediction

Builds training features using geographic coordinates as primary input,
combined with weather interpolation, grid proximity, and temporal features.
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geo_utils import (
    haversine_distance, haversine_vectorized, 
    interpolate_value, interpolate_multiple,
    points_within_radius, find_nearest_points,
    distance_to_coast, is_in_california
)
from caiso_nodes import get_california_nodes

logger = logging.getLogger(__name__)


# ============================================
# FEATURE DEFINITIONS
# ============================================

# Core geolocation features (always included)
GEO_FEATURES = [
    'latitude',
    'longitude',
]

# Node characteristics
NODE_FEATURES = [
    'is_load_node',     # 1 if LOAD, 0 if GEN
    'is_gen_node',      # 1 if GEN, 0 if LOAD
]

# Area one-hot encoding (top areas)
AREA_FEATURES = [
    'area_CA', 'area_NV', 'area_BANC', 'area_LADWP', 
    'area_TIDC', 'area_OTHER'
]

# Derived geographic features
DERIVED_GEO_FEATURES = [
    'distance_to_coast_km',
    'latitude_normalized',   # Scaled to 0-1
    'longitude_normalized',  # Scaled to 0-1
]

# Weather features (interpolated from nearest stations)
WEATHER_FEATURES = [
    'temperature',
    'wind_speed', 
    'precipitation',
    'temp_max',
    'temp_min',
]

# Grid proximity features
GRID_FEATURES = [
    'distance_to_nearest_plant_km',
    'nearest_plant_capacity_mw',
    'plants_within_50km',
    'total_capacity_within_50km',
]

# Time features
TIME_FEATURES = [
    'hour',
    'month',
    'day_of_year',
    'day_of_week',
    'is_weekend',
    'hour_sin', 'hour_cos',
    'month_sin', 'month_cos',
    'doy_sin', 'doy_cos',
    'dow_sin', 'dow_cos',
]

# Lag features
LAG_FEATURES = [
    'price_lag_1',
    'price_lag_3', 
    'price_lag_6',
    'price_lag_12',
    'price_lag_24',
]

# Complete feature list
ALL_FEATURES = (
    GEO_FEATURES + 
    NODE_FEATURES + 
    AREA_FEATURES + 
    DERIVED_GEO_FEATURES + 
    WEATHER_FEATURES + 
    GRID_FEATURES + 
    TIME_FEATURES + 
    LAG_FEATURES
)


# ============================================
# FEATURE BUILDERS
# ============================================

class GeoFeatureBuilder:
    """Builds features for geolocation-based price prediction"""
    
    def __init__(self, 
                 stations_df: pd.DataFrame = None,
                 plants_df: pd.DataFrame = None):
        """
        Initialize feature builder
        
        Args:
            stations_df: DataFrame of weather stations with lat/lon
            plants_df: DataFrame of power plants with lat/lon and capacity
        """
        self.stations_df = stations_df
        self.plants_df = plants_df
        self.ca_nodes = get_california_nodes()
        
        # California bounds for normalization
        self.lat_min, self.lat_max = 32.5, 42.0
        self.lon_min, self.lon_max = -124.5, -114.0
    
    def load_stations(self, path: str = None):
        """Load weather stations from JSON"""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), 'stations.json')
        
        with open(path, 'r') as f:
            stations = json.load(f)
        
        self.stations_df = pd.DataFrame(stations)
        logger.info(f"Loaded {len(self.stations_df)} weather stations")
    
    def load_plants(self, path: str = None):
        """Load power plants from pickle"""
        if path is None:
            base = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(base, 'data', 'downloads', 'power_plants.pkl')
        
        if os.path.exists(path):
            self.plants_df = pd.read_pickle(path)
            logger.info(f"Loaded {len(self.plants_df)} power plants")
        else:
            logger.warning(f"Power plants file not found: {path}")
            self.plants_df = pd.DataFrame()
    
    def build_geo_features(self, lat: float, lon: float) -> Dict[str, float]:
        """Build geographic features for a single location"""
        features = {
            'latitude': lat,
            'longitude': lon,
            'latitude_normalized': (lat - self.lat_min) / (self.lat_max - self.lat_min),
            'longitude_normalized': (lon - self.lon_min) / (self.lon_max - self.lon_min),
            'distance_to_coast_km': distance_to_coast(lat, lon),
        }
        return features
    
    def build_node_features(self, node_type: str, area: str) -> Dict[str, float]:
        """Build node characteristic features"""
        features = {
            'is_load_node': 1.0 if node_type == 'LOAD' else 0.0,
            'is_gen_node': 1.0 if node_type == 'GEN' else 0.0,
        }
        
        # Area one-hot encoding
        top_areas = ['CA', 'NV', 'BANC', 'LADWP', 'TIDC']
        for a in top_areas:
            features[f'area_{a}'] = 1.0 if area == a else 0.0
        
        # Other areas
        features['area_OTHER'] = 1.0 if area not in top_areas else 0.0
        
        return features
    
    def build_weather_features(self, lat: float, lon: float,
                               weather_df: pd.DataFrame = None) -> Dict[str, float]:
        """Build interpolated weather features"""
        if weather_df is None or len(weather_df) == 0:
            # Return defaults
            return {
                'temperature': 20.0,
                'wind_speed': 5.0,
                'precipitation': 0.0,
                'temp_max': 25.0,
                'temp_min': 15.0,
            }
        
        # Interpolate from nearest stations
        weather_cols = ['temperature', 'wind_speed', 'precipitation', 'temp_max', 'temp_min']
        available = [c for c in weather_cols if c in weather_df.columns]
        
        if not available:
            return {c: 0.0 for c in weather_cols}
        
        interpolated = interpolate_multiple(
            lat, lon, weather_df, available,
            lat_column='latitude', lon_column='longitude',
            n_nearest=5
        )
        
        # Fill missing with defaults
        defaults = {'temperature': 20.0, 'wind_speed': 5.0, 'precipitation': 0.0,
                   'temp_max': 25.0, 'temp_min': 15.0}
        for col in weather_cols:
            if col not in interpolated:
                interpolated[col] = defaults.get(col, 0.0)
        
        return interpolated
    
    def build_grid_features(self, lat: float, lon: float) -> Dict[str, float]:
        """Build grid proximity features from power plant locations"""
        if self.plants_df is None or len(self.plants_df) == 0:
            return {
                'distance_to_nearest_plant_km': 100.0,
                'nearest_plant_capacity_mw': 0.0,
                'plants_within_50km': 0,
                'total_capacity_within_50km': 0.0,
            }
        
        # Find nearest plant
        nearest = find_nearest_points(lat, lon, self.plants_df, n=1)
        
        if len(nearest) > 0:
            nearest_distance = nearest.iloc[0]['distance_km']
            nearest_capacity = nearest.iloc[0].get('capacity_mw', 0)
        else:
            nearest_distance = 100.0
            nearest_capacity = 0.0
        
        # Plants within 50km
        nearby = points_within_radius(lat, lon, self.plants_df, 50.0)
        plants_count = len(nearby)
        total_capacity = nearby['capacity_mw'].sum() if len(nearby) > 0 else 0.0
        
        return {
            'distance_to_nearest_plant_km': nearest_distance,
            'nearest_plant_capacity_mw': nearest_capacity,
            'plants_within_50km': plants_count,
            'total_capacity_within_50km': total_capacity,
        }
    
    def build_time_features(self, timestamp: datetime) -> Dict[str, float]:
        """Build temporal features"""
        hour = timestamp.hour
        month = timestamp.month
        day_of_year = timestamp.timetuple().tm_yday
        day_of_week = timestamp.weekday()
        
        return {
            'hour': hour,
            'month': month,
            'day_of_year': day_of_year,
            'day_of_week': day_of_week,
            'is_weekend': 1.0 if day_of_week >= 5 else 0.0,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            'doy_sin': np.sin(2 * np.pi * day_of_year / 365),
            'doy_cos': np.cos(2 * np.pi * day_of_year / 365),
            'dow_sin': np.sin(2 * np.pi * day_of_week / 7),
            'dow_cos': np.cos(2 * np.pi * day_of_week / 7),
        }
    
    def build_lag_features(self, prices_series: pd.Series, 
                           current_idx: int) -> Dict[str, float]:
        """Build price lag features from a time series"""
        lags = [1, 3, 6, 12, 24]
        features = {}
        
        for lag in lags:
            lag_idx = current_idx - lag
            if lag_idx >= 0:
                features[f'price_lag_{lag}'] = float(prices_series.iloc[lag_idx])
            else:
                features[f'price_lag_{lag}'] = float(prices_series.iloc[current_idx])  # Use current as fallback
        
        return features
    
    def build_all_features(self, lat: float, lon: float, 
                           timestamp: datetime,
                           node_type: str = 'LOAD',
                           area: str = 'CA',
                           weather_df: pd.DataFrame = None,
                           price_history: pd.Series = None,
                           current_idx: int = 0) -> Dict[str, float]:
        """Build complete feature set for a single observation"""
        features = {}
        
        # Geographic features
        features.update(self.build_geo_features(lat, lon))
        
        # Node features
        features.update(self.build_node_features(node_type, area))
        
        # Weather features
        features.update(self.build_weather_features(lat, lon, weather_df))
        
        # Grid features
        features.update(self.build_grid_features(lat, lon))
        
        # Time features
        features.update(self.build_time_features(timestamp))
        
        # Lag features
        if price_history is not None:
            features.update(self.build_lag_features(price_history, current_idx))
        else:
            for lag in [1, 3, 6, 12, 24]:
                features[f'price_lag_{lag}'] = 0.0
        
        return features


# ============================================
# TRAINING DATA BUILDER
# ============================================

def build_training_dataset(prices_df: pd.DataFrame,
                           weather_df: pd.DataFrame = None,
                           plants_df: pd.DataFrame = None,
                           output_path: str = None) -> pd.DataFrame:
    """
    Build complete training dataset from downloaded geo prices
    
    Args:
        prices_df: DataFrame with columns: timestamp, node, lmp, latitude, longitude, area, node_type
        weather_df: Optional DataFrame with weather data
        plants_df: Optional DataFrame with power plant data
        output_path: Optional path to save the dataset
    
    Returns:
        DataFrame with all features and target (price)
    """
    logger.info("Building training dataset...")
    logger.info(f"Input: {len(prices_df)} price records")
    
    # Initialize feature builder
    builder = GeoFeatureBuilder(plants_df=plants_df)
    
    # Prepare data by node for lag calculations
    prices_df = prices_df.sort_values(['node', 'timestamp']).reset_index(drop=True)
    
    all_records = []
    nodes = prices_df['node'].unique()
    
    logger.info(f"Processing {len(nodes)} nodes...")
    
    # Track statistics
    nodes_processed = 0
    nodes_skipped_insufficient = 0
    
    for i, node in enumerate(nodes):
        if (i + 1) % 50 == 0:
            logger.info(f"  Processing node {i+1}/{len(nodes)}")
        
        node_df = prices_df[prices_df['node'] == node].copy()
        
        if len(node_df) < 25:  # Need at least 25 records for lag_24
            nodes_skipped_insufficient += 1
            continue
        
        nodes_processed += 1
        
        # Get node info
        lat = node_df.iloc[0]['latitude']
        lon = node_df.iloc[0]['longitude']
        area = node_df.iloc[0].get('area', 'CA')
        node_type = node_df.iloc[0].get('node_type', 'LOAD')
        
        # Build features for each timestamp
        for idx in range(24, len(node_df)):  # Start at 24 for lag features
            row = node_df.iloc[idx]
            
            features = builder.build_all_features(
                lat=lat,
                lon=lon,
                timestamp=pd.to_datetime(row['timestamp']),
                node_type=node_type,
                area=area,
                weather_df=weather_df,
                price_history=node_df['lmp'],
                current_idx=idx
            )
            
            # Add target
            features['price'] = float(row['lmp'])
            features['node'] = node
            
            # Add price components if available
            for col in ['energy', 'congestion', 'loss']:
                if col in row:
                    features[col] = float(row[col]) if pd.notna(row[col]) else 0.0
            
            all_records.append(features)
    
    # Create DataFrame
    training_df = pd.DataFrame(all_records)
    
    # Log statistics
    logger.info("")
    logger.info("=" * 50)
    logger.info("FEATURE ENGINEERING STATISTICS")
    logger.info("=" * 50)
    logger.info(f"  Total nodes in data:       {len(nodes)}")
    logger.info(f"  Nodes processed:           {nodes_processed}")
    logger.info(f"  Nodes skipped (<25 rows):  {nodes_skipped_insufficient}")
    logger.info(f"  (Skipping is normal - some nodes have limited historical data)")
    logger.info("")
    logger.info(f"✅ Built training dataset: {len(training_df)} records, {len(training_df.columns)} columns")
    
    # Save if output path provided
    if output_path:
        training_df.to_pickle(output_path)
        logger.info(f"✅ Saved to {output_path}")
        
        # Save feature list
        feature_list = [c for c in training_df.columns if c not in ['price', 'node', 'energy', 'congestion', 'loss']]
        features_path = output_path.replace('.pkl', '_features.json')
        with open(features_path, 'w') as f:
            json.dump(feature_list, f, indent=2)
        logger.info(f"✅ Saved feature list to {features_path}")
    
    return training_df


# ============================================
# CLI
# ============================================

if __name__ == '__main__':
    import argparse
    
    # Setup logging with file output
    LOG_DIR = '/app/data/training' if os.path.exists('/app') else os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training')
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, 'geo_features.log')),
            logging.StreamHandler()
        ]
    )
    
    parser = argparse.ArgumentParser(description='Build geolocation training features')
    parser.add_argument('--prices', type=str, required=True,
                        help='Path to geo_prices.pkl')
    parser.add_argument('--plants', type=str, default=None,
                        help='Path to power_plants.pkl')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for training dataset')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("GEOLOCATION FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    # Load prices
    prices_df = pd.read_pickle(args.prices)
    logger.info(f"Loaded {len(prices_df)} price records")
    
    # Load plants if available
    plants_df = None
    if args.plants and os.path.exists(args.plants):
        plants_df = pd.read_pickle(args.plants)
        logger.info(f"Loaded {len(plants_df)} power plants")
    
    # Build training dataset
    training_df = build_training_dataset(
        prices_df, 
        plants_df=plants_df,
        output_path=args.output
    )
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPLETE ✅")
    logger.info("=" * 70)
    logger.info(f"Training records: {len(training_df)}")
    logger.info(f"Features: {len([c for c in training_df.columns if c != 'price'])}")
    logger.info(f"Price range: ${training_df['price'].min():.2f} - ${training_df['price'].max():.2f}")

