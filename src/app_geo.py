#!/usr/bin/env python3
"""
app_geo.py - Geolocation-Based Price Prediction API

Flask endpoints for predicting electricity prices based on
geographic coordinates (latitude/longitude).
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from flask import Blueprint, jsonify, request, session, redirect, url_for
from functools import wraps
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_geo import GeoPricePredictor
from geo_features import GeoFeatureBuilder
from caiso_nodes import CAISONodes, get_california_nodes, find_nearest_node
from geo_utils import is_in_california, haversine_distance

logger = logging.getLogger(__name__)

# Create Blueprint for geo endpoints
geo_bp = Blueprint('geo', __name__)

# Global state
geo_predictor: Optional[GeoPricePredictor] = None
feature_builder: Optional[GeoFeatureBuilder] = None
ca_nodes: Dict = {}


# ============================================
# INITIALIZATION
# ============================================

def initialize_geo_predictor(model_dir: str = None):
    """Initialize the geolocation predictor"""
    global geo_predictor, feature_builder, ca_nodes
    
    if model_dir is None:
        # Works in Docker (/app) and locally
        if os.path.exists('/app/data/models'):
            model_dir = '/app/data/models'
        else:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')
    
    model_path = os.path.join(model_dir, 'geo_model.keras')
    scaler_path = os.path.join(model_dir, 'geo_scaler.pkl')
    features_path = os.path.join(model_dir, 'geo_features.json')
    
    # Check if model exists
    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        logger.warning("Geolocation model not found. Train with train_geo.py first.")
        return False
    
    try:
        geo_predictor = GeoPricePredictor.load(model_path, scaler_path, features_path)
        logger.info(f"✅ Loaded geo model with {len(geo_predictor.feature_names)} features")
        
        # Initialize feature builder
        feature_builder = GeoFeatureBuilder()
        
        # Try to load power plants
        plants_path = os.path.join(os.path.dirname(model_dir), 'downloads', 'power_plants.pkl')
        if os.path.exists(plants_path):
            feature_builder.plants_df = pd.read_pickle(plants_path)
            logger.info(f"✅ Loaded {len(feature_builder.plants_df)} power plants")
        
        # Load California nodes
        ca_nodes = get_california_nodes()
        logger.info(f"✅ Loaded {len(ca_nodes)} California nodes")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize geo predictor: {e}")
        return False


# ============================================
# HELPER FUNCTIONS
# ============================================

def build_prediction_features(lat: float, lon: float) -> Dict[str, float]:
    """Build features for prediction at a given location"""
    
    # Find nearest node to get area and type
    node_id, node_info, distance = find_nearest_node(lat, lon)
    
    node_type = node_info.get('node_type', 'LOAD') if node_info else 'LOAD'
    area = node_info.get('area', 'CA') if node_info else 'CA'
    
    # Build all features
    features = feature_builder.build_all_features(
        lat=lat,
        lon=lon,
        timestamp=datetime.now(),
        node_type=node_type,
        area=area,
        weather_df=None,  # Will use defaults
        price_history=None  # Will use defaults
    )
    
    return features


def classify_price_level(price: float) -> Tuple[str, str]:
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
# API ENDPOINTS
# ============================================

@geo_bp.route('/predict/geo', methods=['GET', 'POST'])
def predict_geo():
    """
    Predict electricity price for any California location
    
    Query params or JSON body:
        - latitude: float (required)
        - longitude: float (required)
    
    Returns:
        JSON with predicted price and location info
    """
    if geo_predictor is None:
        return jsonify({
            'error': 'Geolocation model not loaded. Train with train_geo.py first.'
        }), 503
    
    try:
        # Get parameters
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = request.args
        
        lat = data.get('latitude') or data.get('lat')
        lon = data.get('longitude') or data.get('lon')
        
        if lat is None or lon is None:
            return jsonify({
                'error': 'latitude and longitude are required'
            }), 400
        
        lat = float(lat)
        lon = float(lon)
        
        # Validate California bounds
        if not is_in_california(lat, lon):
            return jsonify({
                'error': 'Location must be in California',
                'bounds': {
                    'lat_min': 32.5, 'lat_max': 42.0,
                    'lon_min': -124.5, 'lon_max': -114.0
                }
            }), 400
        
        # Find nearest CAISO node
        node_id, node_info, node_distance = find_nearest_node(lat, lon)
        
        # Build features
        features = build_prediction_features(lat, lon)
        
        # Ensure all expected features are present
        for feat in geo_predictor.feature_names:
            if feat not in features:
                features[feat] = 0.0
        
        # Predict
        price = geo_predictor.predict_single(features)
        
        # Clip to reasonable range
        price = float(np.clip(price, -10, 500))
        
        # Classify
        level, description = classify_price_level(price)
        
        return jsonify({
            'predicted_price': price,
            'price_level': level,
            'description': description,
            'location': {
                'latitude': lat,
                'longitude': lon,
                'in_california': True
            },
            'nearest_node': {
                'node_id': node_id,
                'distance_km': round(node_distance, 2),
                'area': node_info.get('area', '') if node_info else '',
                'node_type': node_info.get('node_type', '') if node_info else ''
            },
            'features_used': len(geo_predictor.feature_names),
            'timestamp': datetime.now().isoformat(),
            'model_version': 'geo-1.0.0'
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@geo_bp.route('/predict/address', methods=['GET', 'POST'])
def predict_address():
    """
    Predict electricity price from an address (requires Google Geocoding API)
    
    Query params or JSON body:
        - address: str (required)
    
    Returns:
        JSON with predicted price and geocoded location
    """
    if geo_predictor is None:
        return jsonify({
            'error': 'Geolocation model not loaded'
        }), 503
    
    try:
        # Get address
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = request.args
        
        address = data.get('address')
        
        if not address:
            return jsonify({'error': 'address is required'}), 400
        
        # Try to geocode using Google Maps API
        google_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        
        if not google_api_key:
            return jsonify({
                'error': 'Google Maps API key not configured. Use /predict/geo with lat/lon instead.',
                'hint': 'Set GOOGLE_MAPS_API_KEY environment variable'
            }), 501
        
        import requests as http_requests
        
        geocode_url = 'https://maps.googleapis.com/maps/api/geocode/json'
        params = {
            'address': address,
            'key': google_api_key
        }
        
        response = http_requests.get(geocode_url, params=params, timeout=10)
        response.raise_for_status()
        
        geocode_data = response.json()
        
        if geocode_data.get('status') != 'OK' or not geocode_data.get('results'):
            return jsonify({
                'error': f'Could not geocode address: {geocode_data.get("status")}',
                'address': address
            }), 400
        
        location = geocode_data['results'][0]['geometry']['location']
        lat = location['lat']
        lon = location['lng']
        formatted_address = geocode_data['results'][0].get('formatted_address', address)
        
        # Check if in California
        if not is_in_california(lat, lon):
            return jsonify({
                'error': 'Address is not in California',
                'geocoded_location': {'latitude': lat, 'longitude': lon},
                'formatted_address': formatted_address
            }), 400
        
        # Get prediction using coordinates
        features = build_prediction_features(lat, lon)
        
        for feat in geo_predictor.feature_names:
            if feat not in features:
                features[feat] = 0.0
        
        price = geo_predictor.predict_single(features)
        price = float(np.clip(price, -10, 500))
        
        level, description = classify_price_level(price)
        node_id, node_info, node_distance = find_nearest_node(lat, lon)
        
        return jsonify({
            'predicted_price': price,
            'price_level': level,
            'description': description,
            'address': {
                'input': address,
                'formatted': formatted_address
            },
            'location': {
                'latitude': lat,
                'longitude': lon
            },
            'nearest_node': {
                'node_id': node_id,
                'distance_km': round(node_distance, 2),
                'area': node_info.get('area', '') if node_info else ''
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Address prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@geo_bp.route('/nodes/nearby', methods=['GET'])
def get_nearby_nodes():
    """
    Find CAISO nodes near a location
    
    Query params:
        - latitude: float (required)
        - longitude: float (required)
        - radius_km: float (optional, default 25)
        - limit: int (optional, default 10)
    """
    try:
        lat = float(request.args.get('latitude') or request.args.get('lat'))
        lon = float(request.args.get('longitude') or request.args.get('lon'))
        radius_km = float(request.args.get('radius_km', 25))
        limit = int(request.args.get('limit', 10))
        
        if not is_in_california(lat, lon):
            return jsonify({'error': 'Location must be in California'}), 400
        
        # Find nearby nodes
        from caiso_nodes import get_caiso_nodes
        nodes_manager = get_caiso_nodes()
        nearby = nodes_manager.find_nodes_within_radius(lat, lon, radius_km)
        
        # Limit results
        nearby = nearby[:limit]
        
        result = []
        for node_id, node_info, distance in nearby:
            result.append({
                'node_id': node_id,
                'distance_km': round(distance, 2),
                'latitude': node_info['latitude'],
                'longitude': node_info['longitude'],
                'area': node_info['area'],
                'node_type': node_info['node_type'],
                'snapshot_price': node_info.get('day_ahead_price', 0)
            })
        
        return jsonify({
            'location': {'latitude': lat, 'longitude': lon},
            'radius_km': radius_km,
            'nodes_found': len(result),
            'nodes': result
        })
        
    except Exception as e:
        logger.error(f"Nearby nodes error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@geo_bp.route('/nodes/info/<node_id>', methods=['GET'])
def get_node_info(node_id: str):
    """Get detailed information about a specific node"""
    try:
        from caiso_nodes import get_caiso_nodes
        nodes_manager = get_caiso_nodes()
        
        node_info = nodes_manager.get_node(node_id)
        
        if not node_info:
            return jsonify({'error': f'Node not found: {node_id}'}), 404
        
        return jsonify({
            'node_id': node_id,
            'latitude': node_info['latitude'],
            'longitude': node_info['longitude'],
            'area': node_info['area'],
            'node_type': node_info['node_type'],
            'prices': {
                'day_ahead': node_info.get('day_ahead_price', 0),
                'day_ahead_congestion': node_info.get('day_ahead_congestion', 0),
                'day_ahead_loss': node_info.get('day_ahead_loss', 0),
                'rt_15min': node_info.get('rt_15min_price', 0),
                'rt_5min': node_info.get('rt_5min_price', 0)
            }
        })
        
    except Exception as e:
        logger.error(f"Node info error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@geo_bp.route('/geo/health', methods=['GET'])
def geo_health():
    """Health check for geo prediction service"""
    return jsonify({
        'status': 'healthy' if geo_predictor else 'model_not_loaded',
        'model_loaded': geo_predictor is not None,
        'features': len(geo_predictor.feature_names) if geo_predictor else 0,
        'nodes_loaded': len(ca_nodes),
        'timestamp': datetime.now().isoformat()
    })


@geo_bp.route('/geo/model-info', methods=['GET'])
def geo_model_info():
    """Get information about the loaded model"""
    if geo_predictor is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    # Works in Docker (/app) and locally
    if os.path.exists('/app/data/models'):
        model_dir = '/app/data/models'
    else:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')
    metadata_path = os.path.join(model_dir, 'geo_metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return jsonify({
        'status': 'loaded',
        'features': len(geo_predictor.feature_names),
        'feature_names': geo_predictor.feature_names[:20],  # First 20
        'training_metadata': metadata,
        'nodes_available': len(ca_nodes)
    })


# ============================================
# APP FACTORY & MODULE-LEVEL APP
# ============================================

# Fixed credentials (can be moved to env vars later)
AUTH_USERNAME = os.getenv('AUTH_USERNAME', 'user')
AUTH_PASSWORD = os.getenv('AUTH_PASSWORD', '2026')


def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def create_geo_app(model_dir: str = None):
    """Create Flask app for geo prediction"""
    from flask import Flask, render_template
    from flask_cors import CORS
    
    # Template folder - works in Docker (/app) and locally
    if os.path.exists('/app/templates'):
        template_dir = '/app/templates'
    else:
        template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
    
    app = Flask(__name__, template_folder=template_dir)
    app.secret_key = os.getenv('SECRET_KEY', 'smart-grid-secret-key-change-in-production')
    CORS(app)
    
    # Register blueprint
    app.register_blueprint(geo_bp)
    
    # Login page
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        error = None
        if request.method == 'POST':
            username = request.form.get('username', '')
            password = request.form.get('password', '')
            
            if username == AUTH_USERNAME and password == AUTH_PASSWORD:
                session['logged_in'] = True
                session['username'] = username
                logger.info(f"User '{username}' logged in")
                return redirect(url_for('index'))
            else:
                error = 'Invalid credentials'
                logger.warning(f"Failed login attempt for '{username}'")
        
        return render_template('login.html', error=error)
    
    # Logout
    @app.route('/logout')
    def logout():
        username = session.get('username', 'unknown')
        session.clear()
        logger.info(f"User '{username}' logged out")
        return redirect(url_for('login'))
    
    # Serve the main UI (protected)
    @app.route('/')
    @login_required
    def index():
        google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY', '')
        return render_template('index.html', google_maps_api_key=google_maps_api_key)
    
    # API info endpoint
    @app.route('/api')
    def api_info():
        return jsonify({
            'service': 'Smart Grid Geo Price Prediction API',
            'status': 'running',
            'endpoints': {
                'predict': '/predict/geo?latitude=34.05&longitude=-118.24',
                'address': '/predict/address?address=Los+Angeles,+CA',
                'nearby_nodes': '/nodes/nearby?latitude=34.05&longitude=-118.24',
                'health': '/geo/health'
            }
        })
    
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'healthy' if geo_predictor else 'model_not_loaded',
            'model_loaded': geo_predictor is not None
        })
    
    # Initialize predictor
    with app.app_context():
        initialize_geo_predictor(model_dir)
    
    return app


# Setup logging with file output - works in Docker and locally
LOG_DIR = '/app/data/prediction' if os.path.exists('/app') else os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'prediction')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'app_geo.log')),
        logging.StreamHandler()
    ]
)

# Create module-level app for Gunicorn (Cloud Run)
# Gunicorn imports this as: gunicorn src.app_geo:app
app = create_geo_app()


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8001))
    logger.info(f"Starting Geo Prediction API on http://0.0.0.0:{port}")
    
    app.run(host='0.0.0.0', port=port, debug=True)

