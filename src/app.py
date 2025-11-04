#!/usr/bin/env python3
"""
Flask Web Service - Smart Grid Electricity Price Prediction
FIXED: Consistent unit handling between training and prediction
"""

import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd

from main import Config, PricePredictor
from tier2_pipeline import Tier2DataPipeline, PowerPlantDB, NOAAWeather, CAISOPriceFetcher
from locations import (
    CALIFORNIA_CITIES, CALIFORNIA_COUNTIES, CAISO_REGIONS,
    get_city, get_county, get_region,
    list_all_cities, list_all_counties, list_all_regions,
    DEMAND_PROFILES
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="/app/templates")
CORS(app)

# Global state
predictor = None
tier2_pipeline = None
tier2_data_cache = None
feature_stats = None
expected_features = None

# ============================================
# INITIALIZATION
# ============================================

def initialize_app():
    """Initialize application on startup"""
    global predictor, tier2_pipeline, feature_stats, expected_features
    
    logger.info("=" * 70)
    logger.info("SMART GRID ML - WEB SERVICE STARTING (UNIT FIX)")
    logger.info("=" * 70)
    
    Config.init()
    
    # Load model
    try:
        model_path = os.path.join(Config.MODEL_DIR, 'price_model.keras')
        scaler_path = os.path.join(Config.MODEL_DIR, 'scaler.pkl')
        features_path = os.path.join(Config.MODEL_DIR, 'features.json')
        
        if all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
            predictor = PricePredictor.load(model_path, scaler_path, features_path)
            expected_features = predictor.feature_names
            logger.info(f"✅ Model loaded: {len(expected_features)} features")
            
            # Load feature statistics
            stats_path = os.path.join(Config.MODEL_DIR, 'feature_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    feature_stats = json.load(f)
                logger.info(f"✅ Feature statistics loaded for {len(feature_stats)} features")
                
                # Log sample stats for debugging
                if 'temperature' in feature_stats:
                    temp_stats = feature_stats['temperature']
                    logger.info(f"Training temperature range: {temp_stats['min']:.1f} - {temp_stats['max']:.1f} (mean: {temp_stats['mean']:.1f})")
            else:
                logger.warning("⚠️ feature_stats.json not found. Using defaults.")
                feature_stats = {}
        else:
            logger.warning("⚠️ Model not found. Run 'python train.py' first.")
            predictor = None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None
    
    tier2_pipeline = Tier2DataPipeline(api_key=Config.EIA_API_KEY)
    
    logger.info("✅ Application initialized")
    logger.info("=" * 70)

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_recent_caiso_prices(location_id: str, days: int = 7) -> pd.DataFrame:
    """Fetch recent CAISO prices for lag features"""
    try:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        df = CAISOPriceFetcher.fetch_city_prices(location_id, start_date, end_date)
        
        if df is not None and len(df) > 0:
            df = df.sort_values('timestamp', ascending=False)
            logger.info(f"✅ Fetched {len(df)} recent CAISO prices for {location_id}")
            logger.info(f"   Sample: {df[['timestamp', 'lmp']].head(3).to_dict(orient='records')}")
            return df
        
        return None
    
    except Exception as e:
        logger.warning(f"Failed to fetch CAISO prices for {location_id}: {e}")
        return None

def get_feature_default(feature_name: str) -> float:
    """
    Get default value for a feature using training statistics.
    Falls back to median/mean from training, or sensible defaults.
    """
    if feature_stats and feature_name in feature_stats:
        # Use median as default (more robust than mean)
        return feature_stats[feature_name].get('median', 
               feature_stats[feature_name].get('mean', 0.0))
    
    # Fallback defaults - MATCH TRAINING SCALE (actual Celsius, not tenths)
    defaults = {
        'temperature': 20.0,   # 20°C (actual)
        'wind_speed': 5.0,     # 5 m/s (actual)
        'temp_avg': 20.0,
        'temp_max': 25.0,
        'temp_min': 15.0,
        'precipitation': 0.0,
        'snowfall': 0.0,
        'cloud_cover': 0.5,
        'solar_mw': 500.0,
        'wind_mw': 400.0,
        'total_demand': 20000.0,
        'renewable_pct': 0.3,
        'imbalance': 1.0,
        'grid_stress': 0.5,
        'wildfire_risk': 0.0,
        'total_plant_capacity': 99678.1,
        'avg_plant_capacity': 63.49,
        'plant_count': 1570,
        'recent_quake_count': 0,
        'avg_quake_magnitude': 0.0,
        'max_quake_magnitude': 0.0,
        'energy': 0.0,
        'congestion': 0.0,
        'loss': 0.0,
    }
    
    # For lag features, use training mean
    if 'price_lag' in feature_name and feature_stats and 'price_lag_1' in feature_stats:
        return feature_stats['price_lag_1'].get('mean', 50.0)
    
    return defaults.get(feature_name, 0.0)

def get_current_noaa_weather(city_id: str) -> dict:
    """
    Fetch current NOAA weather data in EXACT same format as training.
    Returns dict of datatype -> RAW VALUE (no conversion).
    """
    noaa = NOAAWeather()
    
    # Get last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    try:
        city_name = CALIFORNIA_CITIES[city_id]['name']
        response = noaa.get_historic_weather(
            city_name,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if not response or 'results' not in response:
            return {}
        
        # Convert to datatype -> RAW value dict (most recent value)
        # CRITICAL: Keep raw values from NOAA API (no conversion)
        records = {}
        for record in response['results']:
            datatype = record.get('datatype')
            value = record.get('value')
            date = record.get('date')
            
            if datatype and value is not None:
                # Keep most recent value for each datatype
                if datatype not in records or date > records[datatype]['date']:
                    records[datatype] = {'value': value, 'date': date}
        
        # Extract just the RAW values (no unit conversion)
        result = {k: v['value'] for k, v in records.items()}
        
        logger.info(f"✅ Fetched {len(result)} raw NOAA datatypes for {city_id}")
        if 'TOBS' in result:
            logger.info(f"   Sample: TOBS={result['TOBS']} (raw value from API)")
        
        return result
        
    except Exception as e:
        logger.warning(f"Failed to fetch NOAA data for {city_id}: {e}")
        return {}

def build_features_for_location(location_id: str, location_type: str = 'city') -> dict:
    """
    Build feature dictionary using EXACT features expected by model.
    Uses RAW NOAA values to match training data format.
    """
    features = {}
    
    # Get location info
    if location_type == 'city':
        location = get_city(location_id)
    elif location_type == 'county':
        location = get_county(location_id)
    else:
        location = get_region(location_id)
    
    if not location:
        raise ValueError(f"Location not found: {location_id}")
    
    logger.info(f"Building features for: {location['name']}")
    
    # Get REAL NOAA weather data (RAW VALUES - same as training)
    noaa_data = get_current_noaa_weather(location_id) if location_type == 'city' else {}
    
    # Map NOAA datatypes to features
    # NOAA GHCND API returns ALL values in tenths - must convert to actual units
    # This matches the training data conversion
    if noaa_data:
        # Get raw values from NOAA API (in tenths)
        temp_raw = noaa_data.get('TOBS', get_feature_default('temperature') * 10)
        temp_max_raw = noaa_data.get('TMAX', get_feature_default('temp_max') * 10)
        temp_min_raw = noaa_data.get('TMIN', get_feature_default('temp_min') * 10)
        temp_avg_raw = noaa_data.get('TAVG', temp_raw)
        prcp_raw = noaa_data.get('PRCP', 0.0)
        snow_raw = noaa_data.get('SNOW', 0.0)
        wind_raw = noaa_data.get('AWND', get_feature_default('wind_speed') * 10)
        
        # Convert from tenths to actual units (to match training data)
        features['temperature'] = temp_raw / 10.0
        features['temp_max'] = temp_max_raw / 10.0
        features['temp_min'] = temp_min_raw / 10.0  
        features['temp_avg'] = temp_avg_raw / 10.0
        features['precipitation'] = prcp_raw / 10.0
        features['snowfall'] = snow_raw / 10.0
        features['wind_speed'] = wind_raw / 10.0
        
        # Log for debugging
        logger.info(f"✅ NOAA raw values: temp={temp_raw}, temp_max={temp_max_raw}, temp_min={temp_min_raw}, wind={wind_raw}")
        logger.info(f"✅ Converted to actual units: temp={features['temperature']:.1f}°C, temp_max={features['temp_max']:.1f}°C, temp_min={features['temp_min']:.1f}°C, wind={features['wind_speed']:.1f}m/s")    # Map NOAA datatypes to features
    else:
        # Fallback to defaults from training stats (already in correct scale)
        features['temperature'] = get_feature_default('temperature')
        features['temp_max'] = get_feature_default('temp_max')
        features['temp_min'] = get_feature_default('temp_min')
        features['precipitation'] = 0.0
        features['snowfall'] = 0.0
        features['wind_speed'] = get_feature_default('wind_speed')
        features['temp_avg'] = features['temperature']
        
        logger.warning(f"Using training-scale default values for {location_id}")
    
    # Time features
    now = datetime.now()
    hour = now.hour
    month = now.month
    day_of_year = now.timetuple().tm_yday
    day_of_week = now.weekday()
    
    features['hour'] = hour
    features['month'] = month
    features['day_of_year'] = day_of_year
    features['day_of_week'] = day_of_week
    features['is_weekend'] = 1 if day_of_week >= 5 else 0
    
    # Cyclical encodings
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)
    features['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365)
    features['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365)
    features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    # Grid features - use actual Celsius/m/s for calculations
    temp_celsius = features['temperature']  # Already in °C
    wind_mps = features['wind_speed']  # Already in m/s
    
    features['cloud_cover'] = np.clip(0.4 + 0.1 * np.sin(2 * np.pi * day_of_year / 365), 0, 1)
    
    # Solar generation
    if 6 <= hour <= 18:
        features['solar_mw'] = 2000 * (1 - features['cloud_cover']) * np.sin(np.pi * (hour - 6) / 12)
    else:
        features['solar_mw'] = 0
    
    # Wind generation (use actual m/s for calculation)
    features['wind_mw'] = max(0, 800 * (wind_mps / 10))
    
    # Demand estimation (use actual Celsius)
    base_demand = 20000
    demand_profile = DEMAND_PROFILES.get(location.get('demand_profile', 'urban_tech'), {})
    
    ac_demand = 0
    if temp_celsius > 25:
        ac_sensitivity = demand_profile.get('ac_sensitivity', 0.5)
        ac_demand = 500 * ((temp_celsius - 25) ** 1.5) * ac_sensitivity
    
    peak_hours = demand_profile.get('peak_hours', [9, 18])
    if hour in peak_hours:
        time_multiplier = 1.4
    elif 23 <= hour or hour <= 5:
        time_multiplier = 0.8
    else:
        time_multiplier = 1.0
    
    weekend_factor = 0.9 if features['is_weekend'] else 1.0
    seasonal_factor = demand_profile.get('seasonal_factor', 1.0)
    
    features['total_demand'] = (base_demand + ac_demand) * time_multiplier * weekend_factor * seasonal_factor
    
    # Grid metrics
    total_renewable = features['solar_mw'] + features['wind_mw']
    features['renewable_pct'] = np.clip(total_renewable / features['total_demand'], 0, 1)
    features['imbalance'] = features['total_demand'] / (total_renewable + 5000)
    features['grid_stress'] = np.clip(features['total_demand'] / 35000, 0, 1)
    features['wildfire_risk'] = 0.0
    
    # Fetch recent CAISO prices for lag features
    if location_type == 'city':
        recent_prices = get_recent_caiso_prices(location_id, days=7)
        
        if recent_prices is not None and len(recent_prices) > 0:
            for lag in [1, 3, 6, 12]:
                if lag < len(recent_prices):
                    features[f'price_lag_{lag}'] = float(recent_prices.iloc[lag]['lmp'])
                else:
                    features[f'price_lag_{lag}'] = get_feature_default(f'price_lag_{lag}')
            logger.info(f"✅ Using real CAISO price lags")
        else:
            for lag in [1, 3, 6, 12]:
                features[f'price_lag_{lag}'] = get_feature_default(f'price_lag_{lag}')
            logger.info(f"Using training mean for price lags")
    else:
        for lag in [1, 3, 6, 12]:
            features[f'price_lag_{lag}'] = get_feature_default(f'price_lag_{lag}')
    
    # Temperature lags (use current as approximation)
    for lag in [1, 3, 6, 12]:
        features[f'temperature_lag_{lag}'] = features['temperature']
    
    # Solar/wind lags (use current as approximation)
    for lag in [1, 3, 6, 12]:
        features[f'solar_mw_lag_{lag}'] = features['solar_mw']
        features[f'wind_mw_lag_{lag}'] = features['wind_mw']
    
    # Static features
    features['total_plant_capacity'] = get_feature_default('total_plant_capacity')
    features['avg_plant_capacity'] = get_feature_default('avg_plant_capacity')
    features['plant_count'] = get_feature_default('plant_count')
    features['recent_quake_count'] = 0
    features['avg_quake_magnitude'] = 0.0
    features['max_quake_magnitude'] = 0.0
    
    # CAISO components
    features['energy'] = 0.0
    features['congestion'] = 0.0
    features['loss'] = 0.0
    
    # Add unmapped NOAA datatypes with RAW values from NOAA API
    noaa_unmapped = [
        'ADPT', 'ASLP', 'ASTP', 'AWBT', 'DAPR', 'MDPR', 'PGTM',
        'RHAV', 'RHMN', 'RHMX', 'SNWD', 'WDF2', 'WDF5', 'WESD',
        'WESF', 'WSF2', 'WSF5', 'WT01', 'WT02', 'WT03', 'WT07', 'WT08'
    ]
    
    for datatype in noaa_unmapped:
        if noaa_data and datatype in noaa_data:
            raw_value = noaa_data[datatype]
            # NOAA returns most values in tenths - ALWAYS convert
            # Only boolean flags (WT* codes) should stay as-is (0 or 1)
            if datatype.startswith('WT'):
                features[datatype] = raw_value  # Weather type flags (0/1)
            else:
                features[datatype] = raw_value / 10.0  # Convert from tenths
        else:
            features[datatype] = get_feature_default(datatype)
    
    # CRITICAL: Only return features expected by the model
    if expected_features:
        final_features = {}
        for feat in expected_features:
            if feat in features:
                final_features[feat] = features[feat]
            else:
                logger.warning(f"Feature '{feat}' missing. Using default: {get_feature_default(feat)}")
                final_features[feat] = get_feature_default(feat)
        
        # Check for extra features
        extra = set(features.keys()) - set(expected_features)
        if extra:
            logger.warning(f"Ignoring {len(extra)} extra features: {list(extra)[:5]}...")
        
        logger.info(f"✅ Generated {len(final_features)} features (model expects {len(expected_features)})")
        logger.info(f"Sample features: {list(final_features.items())[:5]}")
        return final_features
    else:
        logger.info(f"Generated {len(features)} features")
        logger.info(f"Sample features: {list(features.items())[:5]}")
        return features

def classify_price_level(price: float) -> tuple:
    """Classify price into levels"""
    if price < 30:
        return "LOW", "Excellent time to use electricity"
    elif price < 50:
        return "MEDIUM", "Normal grid conditions"
    elif price < 80:
        return "HIGH", "Peak demand period"
    else:
        return "CRITICAL", "Grid stress - conserve energy"

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Serve main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return jsonify({
            'message': 'Smart Grid ML API',
            'status': 'running',
            'endpoints': {
                'health': '/health',
                'predict': '/predict?location_id=los_angeles&location_type=city',
                'locations': '/locations'
            }
        })

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'feature_stats_loaded': feature_stats is not None,
        'expected_features': len(expected_features) if expected_features else 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Predict electricity price"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded. Run training first.'}), 503
    
    try:
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = request.args
        
        location_id = data.get('location_id')
        if not location_id:
            return jsonify({'error': 'location_id required'}), 400
        
        location_type = data.get('location_type', 'city')
        
        # Get location
        if location_type == 'city':
            location = get_city(location_id)
        elif location_type == 'county':
            location = get_county(location_id)
        else:
            location = get_region(location_id)
        
        if not location:
            return jsonify({'error': f'Location not found: {location_id}'}), 404
        
        location_id = location["id"]
        
        # Build features
        features = build_features_for_location(location_id, location_type)
        
        # Validate feature count
        if expected_features and len(features) != len(expected_features):
            logger.error(f"Feature count mismatch! Got {len(features)}, expected {len(expected_features)}")
            return jsonify({'error': 'Feature count mismatch'}), 500
        
        # Predict
        result = predictor.predict_for_location(features, location)
        price = result['predicted_price']
        
        logger.info(f"Raw prediction for {location_id}: ${price:.2f}/MWh")
        
        # Clip to reasonable range
        price = np.clip(price, 10, 200)
        
        # Classify
        level, description = classify_price_level(price)
        
        # Convert raw feature values for display (already in correct units)
        response = {
            'predicted_price': float(price),
            'price_level': level,
            'description': description,
            'confidence': 'High' if feature_stats else 'Medium',
            'timestamp': datetime.now().isoformat(),
            'location': {
                'id': location_id,
                'type': location_type,
                'name': location['name'],
                'region': location.get('region', 'unknown')
            },
            'features': {
                'temperature': features['temperature'],  # Already in °C
                'wind_speed': features['wind_speed'],     # Already in m/s
                'solar_generation': features['solar_mw'],
                'total_demand': features['total_demand'],
                'renewable_pct': features['renewable_pct']
            },
            'model_version': '2.3.0-unit-smart-fix'
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Location endpoints
@app.route('/locations')
def get_locations():
    return jsonify({
        'cities': [{'id': cid, 'name': c['name'], 'region': c['region']} 
                   for cid, c in CALIFORNIA_CITIES.items()],
        'counties': [{'id': cid, 'name': c['name'], 'region': c['region']} 
                     for cid, c in CALIFORNIA_COUNTIES.items()],
        'regions': [{'id': rid, 'name': r['name']} 
                    for rid, r in CAISO_REGIONS.items()]
    })

@app.route('/model-info')
def get_model_info():
    """Get model information"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        metadata_path = os.path.join(Config.MODEL_DIR, 'training_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        return jsonify({
            'version': '2.3.0-unit-smart-fix',
            'status': 'loaded',
            'features': len(expected_features) if expected_features else 0,
            'feature_names': expected_features[:10] if expected_features else [],
            'feature_stats_available': feature_stats is not None,
            'stats_count': len(feature_stats) if feature_stats else 0,
            'training_metadata': metadata,
            'unit_handling': 'Smart conversion: auto-detect if NOAA values need /10 conversion'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    initialize_app()
    
    port = int(os.getenv('PORT', 8000))
    logger.info(f"Starting Flask server on http://0.0.0.0:{port}")
    logger.info(f"Model expects {len(expected_features) if expected_features else 'unknown'} features")
    logger.info(f"Feature statistics: {'loaded' if feature_stats else 'not available'}")
    logger.info(f"Unit handling: Smart auto-detection (converts tenths to actual units)")
    
    app.run(host='0.0.0.0', port=port, debug=False)