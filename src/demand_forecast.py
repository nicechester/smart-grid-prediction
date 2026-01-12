#!/usr/bin/env python3
"""
demand_forecast.py - Electricity Demand Forecasting based on Weather

Provides:
1. Weather forecast fetching from Open-Meteo (free, 16-day hourly)
2. Demand forecasting model (Weather + Time → Demand)
3. Pattern extraction from historical demand data
4. 16-day hourly demand forecast

The demand forecast can then be used as input to the price prediction model.
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
import requests

# ML imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ============================================
# WEATHER FORECAST (Open-Meteo - Free API)
# ============================================

class WeatherForecast:
    """Fetch hourly weather forecasts from Open-Meteo (free, no API key required)"""
    
    OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
    
    @staticmethod
    def get_forecast(lat: float, lon: float, days: int = 16) -> pd.DataFrame:
        """
        Get hourly weather forecast for up to 16 days
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Number of days (max 16)
        
        Returns:
            DataFrame with hourly weather forecast indexed by timestamp
        """
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': ','.join([
                'temperature_2m',
                'relative_humidity_2m',
                'wind_speed_10m',
                'wind_direction_10m',
                'precipitation_probability',
                'cloud_cover',
                'is_day'
            ]),
            'forecast_days': min(days, 16),
            'timezone': 'America/Los_Angeles'
        }
        
        try:
            resp = requests.get(WeatherForecast.OPENMETEO_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            hourly = data['hourly']
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(hourly['time']),
                'temperature_c': hourly['temperature_2m'],
                'humidity': hourly['relative_humidity_2m'],
                'wind_speed_mps': [w * 0.27778 for w in hourly['wind_speed_10m']],  # km/h to m/s
                'wind_direction': hourly['wind_direction_10m'],
                'precipitation_prob': hourly['precipitation_probability'],
                'cloud_cover': hourly['cloud_cover'],
                'is_daytime': [bool(d) for d in hourly['is_day']]
            })
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} hours of weather forecast for ({lat}, {lon})")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch weather forecast: {e}")
            raise
    
    @staticmethod
    def get_california_forecast(days: int = 16) -> Dict[str, pd.DataFrame]:
        """
        Get weather forecasts for key California locations
        
        Returns dict of location -> weather DataFrame
        """
        # Key California locations for grid-wide demand
        locations = {
            'north': (38.58, -121.49),    # Sacramento
            'bay_area': (37.77, -122.42), # San Francisco
            'central': (36.75, -119.77),  # Fresno
            'la': (34.05, -118.24),       # Los Angeles
            'san_diego': (32.72, -117.16) # San Diego
        }
        
        forecasts = {}
        for name, (lat, lon) in locations.items():
            try:
                forecasts[name] = WeatherForecast.get_forecast(lat, lon, days)
            except Exception as e:
                logger.warning(f"Failed to get forecast for {name}: {e}")
        
        return forecasts


# ============================================
# DEMAND PATTERNS (from historical data)
# ============================================

class DemandPatterns:
    """Extract and apply demand patterns from historical data"""
    
    def __init__(self):
        self.hourly_pattern: Dict[int, float] = {}
        self.monthly_pattern: Dict[int, float] = {}
        self.dow_pattern: Dict[int, float] = {}
        self.base_demand: float = 25000.0  # MW default for California
    
    def extract_from_data(self, demand_df: pd.DataFrame, demand_col: str = 'demand_mw'):
        """
        Extract patterns from historical demand data
        
        Args:
            demand_df: DataFrame with datetime index and demand column
            demand_col: Name of demand column
        """
        if demand_col not in demand_df.columns:
            raise ValueError(f"Column {demand_col} not found in DataFrame")
        
        # Ensure datetime index
        if not isinstance(demand_df.index, pd.DatetimeIndex):
            demand_df.index = pd.to_datetime(demand_df.index)
        
        # Base demand (overall mean)
        self.base_demand = demand_df[demand_col].mean()
        
        # Hourly pattern (normalized to 1.0)
        hourly = demand_df.groupby(demand_df.index.hour)[demand_col].mean()
        self.hourly_pattern = (hourly / hourly.mean()).to_dict()
        
        # Monthly pattern
        monthly = demand_df.groupby(demand_df.index.month)[demand_col].mean()
        self.monthly_pattern = (monthly / monthly.mean()).to_dict()
        
        # Day of week pattern (0=Monday, 6=Sunday)
        dow = demand_df.groupby(demand_df.index.dayofweek)[demand_col].mean()
        self.dow_pattern = (dow / dow.mean()).to_dict()
        
        logger.info(f"Extracted patterns from {len(demand_df)} records")
        logger.info(f"Base demand: {self.base_demand:.0f} MW")
        logger.info(f"Peak hour: {max(self.hourly_pattern, key=self.hourly_pattern.get)} "
                   f"({self.hourly_pattern[max(self.hourly_pattern, key=self.hourly_pattern.get)]:.2f}x)")
        logger.info(f"Peak month: {max(self.monthly_pattern, key=self.monthly_pattern.get)} "
                   f"({self.monthly_pattern[max(self.monthly_pattern, key=self.monthly_pattern.get)]:.2f}x)")
    
    def get_typical_demand(self, timestamp: datetime) -> float:
        """Get typical demand for a given timestamp based on patterns"""
        hour_factor = self.hourly_pattern.get(timestamp.hour, 1.0)
        month_factor = self.monthly_pattern.get(timestamp.month, 1.0)
        dow_factor = self.dow_pattern.get(timestamp.weekday(), 1.0)
        
        return self.base_demand * hour_factor * month_factor * dow_factor
    
    def save(self, path: str):
        """Save patterns to JSON"""
        data = {
            'base_demand': self.base_demand,
            'hourly_pattern': self.hourly_pattern,
            'monthly_pattern': self.monthly_pattern,
            'dow_pattern': self.dow_pattern
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved demand patterns to {path}")
    
    def load(self, path: str):
        """Load patterns from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.base_demand = data['base_demand']
        # Convert string keys back to int
        self.hourly_pattern = {int(k): v for k, v in data['hourly_pattern'].items()}
        self.monthly_pattern = {int(k): v for k, v in data['monthly_pattern'].items()}
        self.dow_pattern = {int(k): v for k, v in data['dow_pattern'].items()}
        logger.info(f"Loaded demand patterns from {path}")
    
    @staticmethod
    def default_california_patterns() -> 'DemandPatterns':
        """Return default California demand patterns (when no historical data available)"""
        patterns = DemandPatterns()
        patterns.base_demand = 28000.0  # MW average for California
        
        # Typical California hourly pattern (duck curve)
        patterns.hourly_pattern = {
            0: 0.78, 1: 0.74, 2: 0.72, 3: 0.70, 4: 0.70, 5: 0.72,
            6: 0.78, 7: 0.88, 8: 0.95, 9: 1.00, 10: 1.02, 11: 1.03,
            12: 1.02, 13: 1.00, 14: 0.98, 15: 0.98, 16: 1.02, 17: 1.10,
            18: 1.18, 19: 1.22, 20: 1.18, 21: 1.10, 22: 0.98, 23: 0.88
        }
        
        # California seasonal pattern (summer peak)
        patterns.monthly_pattern = {
            1: 0.85, 2: 0.82, 3: 0.80, 4: 0.82, 5: 0.88, 6: 1.05,
            7: 1.20, 8: 1.25, 9: 1.15, 10: 0.95, 11: 0.88, 12: 0.90
        }
        
        # Day of week pattern
        patterns.dow_pattern = {
            0: 1.02, 1: 1.03, 2: 1.03, 3: 1.02, 4: 1.00,  # Mon-Fri
            5: 0.92, 6: 0.88  # Sat, Sun
        }
        
        return patterns


# ============================================
# DEMAND FORECASTER (ML Model)
# ============================================

class DemandForecaster:
    """
    Machine learning model to forecast electricity demand based on weather and time
    
    Features:
    - Temperature (linear and quadratic for AC/heating effect)
    - Humidity, wind speed, cloud cover
    - Hour of day (cyclical encoding)
    - Month (cyclical encoding)
    - Day of week
    - Weekend indicator
    - Cooling/heating degree indicators
    """
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.patterns = DemandPatterns.default_california_patterns()
    
    def build_features(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build features from weather data for demand prediction
        
        Args:
            weather_df: DataFrame with weather data (must have datetime index)
        
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=weather_df.index)
        
        # Temperature features - handle None/NaN values
        if 'temperature_c' in weather_df.columns:
            temp = weather_df['temperature_c'].fillna(20.0)
        else:
            temp = pd.Series(20.0, index=weather_df.index)
        features['temperature'] = temp
        features['temp_squared'] = temp ** 2  # Non-linear effect
        features['discomfort'] = abs(temp - 20)  # Distance from comfort zone
        
        # Cooling/Heating degree indicators
        features['cooling_degrees'] = np.maximum(0, temp - 18)
        features['heating_degrees'] = np.maximum(0, 18 - temp)
        
        # Other weather - handle None/NaN values
        features['humidity'] = weather_df['humidity'].fillna(50) if 'humidity' in weather_df.columns else 50
        features['wind_speed'] = weather_df['wind_speed_mps'].fillna(0) if 'wind_speed_mps' in weather_df.columns else 0
        features['cloud_cover'] = weather_df['cloud_cover'].fillna(50) if 'cloud_cover' in weather_df.columns else 50
        
        # Time features from index
        features['hour'] = weather_df.index.hour
        features['month'] = weather_df.index.month
        features['day_of_week'] = weather_df.index.dayofweek
        features['is_weekend'] = (weather_df.index.dayofweek >= 5).astype(int)
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # Peak hour indicators
        features['is_morning_peak'] = ((features['hour'] >= 7) & (features['hour'] <= 9)).astype(int)
        features['is_evening_peak'] = ((features['hour'] >= 17) & (features['hour'] <= 21)).astype(int)
        features['is_night'] = ((features['hour'] >= 23) | (features['hour'] <= 5)).astype(int)
        
        # Store feature names
        self.feature_names = list(features.columns)
        
        return features
    
    def train(self, weather_df: pd.DataFrame, demand_series: pd.Series,
              test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the demand forecasting model
        
        Args:
            weather_df: Weather data with datetime index
            demand_series: Actual demand values (MW)
            test_size: Fraction for test split
        
        Returns:
            Dictionary with training metrics
        """
        # Build features
        X = self.build_features(weather_df)
        y = demand_series
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        logger.info(f"Training on {len(X)} samples with {len(self.feature_names)} features")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_r2': r2_score(y_test, y_pred_test),
            'test_mape': np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        }
        
        logger.info(f"Training complete:")
        logger.info(f"  Test MAE: {metrics['test_mae']:.0f} MW")
        logger.info(f"  Test RMSE: {metrics['test_rmse']:.0f} MW")
        logger.info(f"  Test R²: {metrics['test_r2']:.3f}")
        logger.info(f"  Test MAPE: {metrics['test_mape']:.1f}%")
        
        # Feature importance
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        logger.info("Top 5 features:")
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"  {feat}: {imp:.3f}")
        
        return metrics
    
    def forecast(self, weather_forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Forecast demand from weather forecast
        
        Args:
            weather_forecast_df: Future weather data with datetime index
        
        Returns:
            DataFrame with forecasted demand
        """
        if not self.is_fitted:
            # Use pattern-based estimation if model not trained
            logger.warning("Model not trained, using pattern-based estimation")
            return self._pattern_based_forecast(weather_forecast_df)
        
        X = self.build_features(weather_forecast_df)
        predictions = self.model.predict(X)
        
        result = pd.DataFrame({
            'timestamp': weather_forecast_df.index,
            'demand_mw': predictions,
            'temperature_c': weather_forecast_df['temperature_c'].fillna(20) if 'temperature_c' in weather_forecast_df.columns else 20,
            'is_daytime': weather_forecast_df['is_daytime'].fillna(True) if 'is_daytime' in weather_forecast_df.columns else True
        })
        result.set_index('timestamp', inplace=True)
        
        return result
    
    def _pattern_based_forecast(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: forecast using patterns + weather adjustment"""
        demands = []
        
        for ts in weather_df.index:
            base = self.patterns.get_typical_demand(ts)
            
            # Weather adjustment - safely get temperature with fallback
            try:
                temp = weather_df.loc[ts, 'temperature_c']
                if temp is None or pd.isna(temp):
                    temp = 20.0
                temp = float(temp)
            except (KeyError, TypeError):
                temp = 20.0
            
            if temp > 25:
                # AC effect
                ac_factor = 1 + 0.02 * (temp - 25) ** 1.5
                base *= min(ac_factor, 1.5)  # Cap at 50% increase
            elif temp < 10:
                # Heating effect
                heat_factor = 1 + 0.01 * (10 - temp) ** 1.3
                base *= min(heat_factor, 1.3)
            
            demands.append(base)
        
        result = pd.DataFrame({
            'timestamp': weather_df.index,
            'demand_mw': demands,
            'temperature_c': weather_df['temperature_c'].fillna(20) if 'temperature_c' in weather_df.columns else 20,
            'is_daytime': weather_df['is_daytime'].fillna(True) if 'is_daytime' in weather_df.columns else True
        })
        result.set_index('timestamp', inplace=True)
        
        return result
    
    def save(self, model_path: str, patterns_path: str = None):
        """Save model and patterns"""
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }, f)
        logger.info(f"Saved demand model to {model_path}")
        
        if patterns_path:
            self.patterns.save(patterns_path)
    
    def load(self, model_path: str, patterns_path: str = None):
        """Load model and patterns"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        logger.info(f"Loaded demand model from {model_path}")
        
        if patterns_path and os.path.exists(patterns_path):
            self.patterns.load(patterns_path)


# ============================================
# FULL FORECAST PIPELINE
# ============================================

def forecast_demand_16day(lat: float, lon: float, 
                          model_path: str = None,
                          patterns_path: str = None) -> pd.DataFrame:
    """
    Full pipeline: Get 16-day demand forecast for a location
    
    Args:
        lat, lon: Location coordinates
        model_path: Path to trained demand model (optional)
        patterns_path: Path to demand patterns (optional)
    
    Returns:
        DataFrame with hourly demand forecast
    """
    # Get weather forecast
    weather_df = WeatherForecast.get_forecast(lat, lon, days=16)
    
    # Initialize forecaster
    forecaster = DemandForecaster()
    
    # Load model if available
    if model_path and os.path.exists(model_path):
        forecaster.load(model_path, patterns_path)
    
    # Forecast
    demand_forecast = forecaster.forecast(weather_df)
    
    # Add metadata
    demand_forecast['weather_temp'] = weather_df['temperature_c']
    demand_forecast['weather_humidity'] = weather_df['humidity']
    demand_forecast['weather_wind'] = weather_df['wind_speed_mps']
    
    return demand_forecast


# ============================================
# CLI
# ============================================

if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='Demand Forecasting')
    parser.add_argument('--lat', type=float, default=34.05, help='Latitude')
    parser.add_argument('--lon', type=float, default=-118.24, help='Longitude')
    parser.add_argument('--days', type=int, default=7, help='Days to forecast')
    parser.add_argument('--model', type=str, help='Path to trained model')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"DEMAND FORECAST FOR ({args.lat}, {args.lon})")
    print(f"{'='*60}\n")
    
    # Get forecast
    weather_df = WeatherForecast.get_forecast(args.lat, args.lon, days=args.days)
    
    forecaster = DemandForecaster()
    if args.model and os.path.exists(args.model):
        forecaster.load(args.model)
    
    forecast = forecaster.forecast(weather_df)
    
    # Print daily summary
    print("\nDaily Summary:")
    print("-" * 50)
    for day in range(min(args.days, 7)):
        day_data = forecast.iloc[day*24:(day+1)*24]
        if len(day_data) > 0:
            date = day_data.index[0].strftime('%Y-%m-%d')
            avg = day_data['demand_mw'].mean()
            peak = day_data['demand_mw'].max()
            peak_hour = day_data['demand_mw'].idxmax().hour
            print(f"{date}: Avg {avg:,.0f} MW | Peak {peak:,.0f} MW @ {peak_hour}:00")
    
    print(f"\n✅ Forecast complete: {len(forecast)} hours")

