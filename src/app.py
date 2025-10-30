#!/usr/bin/env python3
"""
Flask Web Service - Smart Grid Electricity Price Prediction
With California-wide location support and CAISO integration
"""

import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd

# Import core modules
from main import Config, PricePredictor
from tier2_pipeline import Tier2DataPipeline, PowerPlantDB, NOAAWeather, CAISOPriceFetcher
from locations import (
    CALIFORNIA_CITIES, CALIFORNIA_COUNTIES, CAISO_REGIONS,
    get_city, get_county, get_region,
    list_all_cities, list_all_counties, list_all_regions,
    DEMAND_PROFILES
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder="/app/templates")
CORS(app)

# Global state
predictor = None
tier2_pipeline = None
tier2_data_cache = None

# ============================================
# INITIALIZATION
# ============================================

def initialize_app():
    """Initialize application on startup"""
    global predictor, tier2_pipeline
    
    logger.info("=" * 70)
    logger.info("SMART GRID ML - WEB SERVICE STARTING")
    logger.info("=" * 70)
    
    # Initialize config
    Config.init()
    
    # Load model
    try:
        model_path = os.path.join(Config.MODEL_DIR, 'price_model.h5')
        scaler_path = os.path.join(Config.MODEL_DIR, 'scaler.pkl')
        features_path = os.path.join(Config.MODEL_DIR, 'features.json')
        
        if all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
            predictor = PricePredictor.load(model_path, scaler_path, features_path)
            logger.info("✓ Model loaded successfully")
        else:
            logger.warning("⚠️  Model not found. Run 'python train.py' first.")
            predictor = None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None
    
    # Initialize Tier 2 pipeline
    tier2_pipeline = Tier2DataPipeline(api_key=Config.EIA_API_KEY)
    
    logger.info("✓ Application initialized")
    logger.info("=" * 70)

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_recent_caiso_prices(location_id: str, days: int = 7) -> pd.DataFrame:
    """
    Fetch recent CAISO prices for lag features
    
    Args:
        location_id: City ID
        days: Number of days of history to fetch
    
    Returns:
        DataFrame with recent prices or None
    """
    try:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        df = CAISOPriceFetcher.fetch_city_prices(location_id, start_date, end_date)
        
        if df is not None and len(df) > 0:
            # Sort by timestamp descending (most recent first)
            df = df.sort_values('timestamp', ascending=False)
            return df
        
        return None
    
    except Exception as e:
        logger.warning(f"Failed to fetch recent CAISO prices for {location_id}: {e}")
        return None

def build_features_for_location(location_id: str, location_type: str = 'city') -> dict:
    """
    Build feature dictionary for a location using Tier 2 data
    
    Args:
        location_id: Location identifier
        location_type: 'city', 'county', or 'region'
    
    Returns:
        Dictionary of features
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
    
    # Get weather data for location
    try:
        if location_type == 'city':
            weather = NOAAWeather._fetch_city_weather(location_id, location)[1]
        else:
            # Use center coordinates for county/region
            lat, lon = location.get('center', (34.0, -118.0))
            weather = {'temperature': 20, 'wind_speed': 5}  # Mock for non-city
        
        if weather:
            features['temperature'] = weather.get('temperature', 20)
            features['wind_speed'] = weather.get('wind_speed', 5)
        else:
            features['temperature'] = 20
            features['wind_speed'] = 5
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}")
        features['temperature'] = 20
        features['wind_speed'] = 5
    
    # Current time features
    now = datetime.now()
    features['hour'] = now.hour
    features['month'] = now.month
    features['is_weekend'] = 1 if now.weekday() >= 5 else 0
    
    # Weather-derived features
    temp = features['temperature']
    hour = features['hour']
    
    # Cloud cover estimation (based on time of day and season)
    features['cloud_cover'] = np.clip(0.4 + 0.1 * np.sin(2 * np.pi * now.timetuple().tm_yday / 365), 0, 1)
    
    # Solar generation
    if 6 <= hour <= 18:
        features['solar_mw'] = 2000 * (1 - features['cloud_cover']) * np.sin(np.pi * (hour - 6) / 12)
    else:
        features['solar_mw'] = 0
    
    # Wind generation
    features['wind_mw'] = max(0, 800 * (features['wind_speed'] / 10))
    
    # Demand estimation based on location profile
    base_demand = 20000
    demand_profile = DEMAND_PROFILES.get(location.get('demand_profile', 'urban_tech'), {})
    
    # AC demand
    ac_demand = 0
    if temp > 25:
        ac_sensitivity = demand_profile.get('ac_sensitivity', 0.5)
        ac_demand = 500 * ((temp - 25) ** 1.5) * ac_sensitivity
    
    # Time-of-day multiplier based on profile
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
    features['wildfire_risk'] = 0.0  # TODO: Integrate real wildfire data
    
    # Fetch recent CAISO prices for lag features (only for cities)
    if location_type == 'city':
        try:
            recent_prices = get_recent_caiso_prices(location_id, days=7)
            
            if recent_prices is not None and len(recent_prices) > 0:
                # Get prices at specific lags (1, 3, 6, 12 hours ago)
                for lag in [1, 3, 6, 12]:
                    if lag < len(recent_prices):
                        features[f'price_lag_{lag}'] = float(recent_prices.iloc[lag]['lmp'])
                    else:
                        features[f'price_lag_{lag}'] = 50.0  # Default if not enough history
                
                logger.debug(f"Added CAISO lag features for {location_id}")
            else:
                # No recent prices - use defaults
                for lag in [1, 3, 6, 12]:
                    features[f'price_lag_{lag}'] = 50.0
                logger.debug(f"Using default lag features for {location_id}")
        
        except Exception as e:
            logger.warning(f"Failed to get CAISO lag features: {e}")
            # Use defaults
            for lag in [1, 3, 6, 12]:
                features[f'price_lag_{lag}'] = 50.0
    else:
        # Non-city locations: use defaults
        for lag in [1, 3, 6, 12]:
            features[f'price_lag_{lag}'] = 50.0
    
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
# ROUTES - CORE PREDICTION
# ============================================

@app.route('/')
def index():
    """Serve the main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return jsonify({
            'message': 'Smart Grid ML API',
            'status': 'running',
            'error': str(e),
            'endpoints': {
                'health': '/health',
                'predict': '/predict?location_id=los_angeles&location_type=city',
                'locations': '/locations',
                'cities': '/locations/cities',
            }
        }), 200

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat(),
        'locations_available': len(CALIFORNIA_CITIES)
    })

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Predict electricity price
    
    Query params (GET) or JSON body (POST):
        - location_id: City/county ID (optional, default: los_angeles)
        - location_type: 'city', 'county', or 'region' (optional, default: city)
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded. Run training first.'}), 503
    
    try:
        # Get location from request
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = request.args
        
        logger.info(f"Request args: {request.args}")

        location_id = data.get('location_id')
        if not location_id:
            return jsonify({'error': 'location_id is required and was not provided.'}), 400
        location_type = data.get('location_type', 'city')
        
        # Get location info
        if location_type == 'city':
            location = get_city(location_id)
        elif location_type == 'county':
            location = get_county(location_id)
        else:
            location = get_region(location_id)
        
        if not location:
            return jsonify({'error': f'Location not found: {location_id}'}), 404
        location_id = location["id"]  # Use canonical ID
        # Build features
        features = build_features_for_location(location_id, location_type)

        # Log the input features for debugging
        logger.info(f"Input features for prediction: {list(features.keys())}")

        # Predict
        result = predictor.predict_for_location(features, location)
        price = result['predicted_price']
        
        # Log the original predicted price before clipping
        logger.info(f"Original predicted price for {location_id}: {price:.2f}")

        # Clip to reasonable range
        # Typical electricity prices: $20-200/MWh
        price = np.clip(price, 10, 200)
        
        # Classify
        level, description = classify_price_level(price)
        
        # Build response
        response = {
            'predicted_price': float(price),
            'price_level': level,
            'description': description,
            'confidence': 'High' if predictor else 'Low',
            'timestamp': datetime.now().isoformat(),
            'location': {
                'id': location_id,
                'type': location_type,
                'name': location['name'],
                'region': location.get('region', 'unknown')
            },
            'features': {
                'temperature': features['temperature'],
                'wind_speed': features['wind_speed'],
                'solar_generation': features['solar_mw'],
                'total_demand': features['total_demand'],
                'renewable_pct': features['renewable_pct']
            },
            'model_version': '2.0.0'
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# ============================================
# ROUTES - LOCATION DATA
# ============================================

@app.route('/locations')
def get_locations():
    """Get all available locations"""
    return jsonify({
        'cities': [{'id': cid, 'name': c['name'], 'region': c['region']} 
                   for cid, c in CALIFORNIA_CITIES.items()],
        'counties': [{'id': cid, 'name': c['name'], 'region': c['region']} 
                     for cid, c in CALIFORNIA_COUNTIES.items()],
        'regions': [{'id': rid, 'name': r['name']} 
                    for rid, r in CAISO_REGIONS.items()]
    })

@app.route('/locations/cities')
def get_cities():
    """Get all cities"""
    return jsonify({
        'cities': [
            {
                'id': city_id,
                'name': city['name'],
                'region': city['region'],
                'county': city['county'],
                'population': city['population'],
                'demand_profile': city['demand_profile']
            }
            for city_id, city in CALIFORNIA_CITIES.items()
        ]
    })

@app.route('/locations/counties')
def get_counties():
    """Get all counties"""
    return jsonify({
        'counties': [
            {
                'id': county_id,
                'name': county['name'],
                'region': county['region'],
                'population': county['population']
            }
            for county_id, county in CALIFORNIA_COUNTIES.items()
        ]
    })

@app.route('/locations/regions')
def get_regions():
    """Get all CAISO regions"""
    return jsonify({
        'regions': [
            {
                'id': region_id,
                'name': region['name'],
                'description': region['description']
            }
            for region_id, region in CAISO_REGIONS.items()
        ]
    })

# ============================================
# ROUTES - TIER 2 DATA
# ============================================

@app.route('/tier2-data')
def get_tier2_data():
    """Get Tier 2 data status"""
    global tier2_data_cache
    
    try:
        # Build dataset (with caching to avoid repeated API calls)
        if tier2_data_cache is None:
            logger.info("Building Tier 2 dataset (first request)...")
            # Skip slow transmission line fetching for API responses
            tier2_data_cache = tier2_pipeline.build_complete_dataset(use_concurrent=False)
        
        data = tier2_data_cache
        
        response = {
            'power_plants': {
                'count': len(data['power_plants']) if data.get('power_plants') is not None else 0,
                'available': data.get('power_plants') is not None
            },
            'weather_by_city': {
                'count': len(data.get('weather_by_city', {})),
                'available': len(data.get('weather_by_city', {})) > 0,
                'cities': list(data.get('weather_by_city', {}).keys())[:5]  # Sample
            },
            'caiso_prices_by_city': {
                'count': len(data.get('caiso_prices_by_city', {})),
                'available': len(data.get('caiso_prices_by_city', {})) > 0,
                'cities': list(data.get('caiso_prices_by_city', {}).keys())[:5]  # Sample
            },
            'earthquakes': {
                'count': len(data['earthquakes']) if data.get('earthquakes') is not None else 0,
                'available': data.get('earthquakes') is not None
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Tier 2 data error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/tier2-data/refresh', methods=['POST'])
def refresh_tier2_data():
    """Force refresh Tier 2 data cache"""
    global tier2_data_cache
    tier2_data_cache = None
    return jsonify({'status': 'cache cleared', 'message': 'Next request will fetch fresh data'})

# ============================================
# ROUTES - MODEL INFO
# ============================================

@app.route('/model-info')
def get_model_info():
    """Get model information"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        # Load training metadata
        metadata_path = os.path.join(Config.MODEL_DIR, 'training_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        return jsonify({
            'version': '2.0.0',
            'status': 'loaded',
            'features': len(predictor.feature_names) if predictor.feature_names else 0,
            'feature_names': predictor.feature_names[:10] if predictor.feature_names else [],
            'training_metadata': metadata,
            'locations_supported': len(CALIFORNIA_CITIES)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    # Initialize
    initialize_app()
    
    # Run server
    port = int(os.getenv('PORT', 8000))
    logger.info(f"Starting Flask server on http://0.0.0.0:{port}")
    logger.info(f"Available locations: {len(CALIFORNIA_CITIES)} cities, {len(CALIFORNIA_COUNTIES)} counties")
    
    app.run(host='0.0.0.0', port=port, debug=False)