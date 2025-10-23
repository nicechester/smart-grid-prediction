#!/usr/bin/env python3
"""
Flask web service for electricity price prediction
Provides REST API for real-time predictions
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Tuple

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

from main import Config, PricePredictor
from tier2_pipeline import Tier2DataPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# Flask App Configuration
# ============================================

app = Flask(__name__)
CORS(app)

# Global predictor (loaded once)
predictor = None
tier2 = None

# ============================================
# Initialization
# ============================================

def init_app():
    """Initialize predictor model and Tier 2 pipeline"""
    global predictor, tier2
    
    logger.info("Initializing Flask app...")
    
    # Initialize config
    Config.init()
    
    # Load model
    model_path = os.path.join(Config.MODEL_DIR, 'price_model.h5')
    scaler_path = os.path.join(Config.MODEL_DIR, 'scaler.pkl')
    features_path = os.path.join(Config.MODEL_DIR, 'features.json')
    
    if os.path.exists(model_path):
        try:
            predictor = PricePredictor.load(model_path, scaler_path, features_path)
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            predictor = None
    else:
        logger.warning(f"Model not found at {model_path}")
        logger.warning("Run 'python train.py' to train the model first")
        predictor = None
    
    # Initialize Tier 2 pipeline
    tier2 = Tier2DataPipeline(api_key=Config.EIA_API_KEY)
    logger.info("✓ Tier 2 pipeline initialized")

# Initialize on startup
init_app()

# ============================================
# Health Check Endpoint
# ============================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if predictor else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': predictor is not None,
        'message': 'Ready for predictions' if predictor else 'Model not loaded'
    }
    return jsonify(status)

# ============================================
# Prediction Endpoint
# ============================================

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Real-time electricity price prediction
    
    GET: Uses current Tier 2 data
    POST: Accepts custom features as JSON
    
    Example GET:
        curl http://localhost:8000/predict
    
    Example POST:
        curl -X POST http://localhost:8000/predict \
          -H "Content-Type: application/json" \
          -d '{
            "temperature": 28.5,
            "cloud_cover": 0.3,
            "wind_speed": 8.5,
            "solar_mw": 2500,
            "wind_mw": 1200,
            "total_demand": 28000,
            "hour": 14
          }'
    """
    
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Run python train.py to train the model'
        }), 503
    
    try:
        if request.method == 'POST':
            # Use provided features
            data = request.get_json()
            features = prepare_features_from_dict(data)
        else:
            # Fetch current Tier 2 data
            features = get_current_features()
        
        if features is None:
            return jsonify({'error': 'Could not prepare features'}), 400
        
        # Make prediction
        prediction = predictor.predict(features)
        price = float(prediction[0][0])
        
        # Classify price level
        if price < 60:
            level = "LOW"
            description = "Good price for consumers"
        elif price < 120:
            level = "MEDIUM"
            description = "Normal conditions"
        elif price < 200:
            level = "HIGH"
            description = "Tight supply"
        else:
            level = "CRITICAL"
            description = "Emergency pricing"
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'predicted_price': price,
            'price_level': level,
            'description': description,
            'currency': 'EUR/MWh',
            'confidence': 'High' if 0.5 < price < 500 else 'Medium'
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# ============================================
# Data Endpoints
# ============================================

@app.route('/tier2-data', methods=['GET'])
def tier2_data():
    """Get latest Tier 2 data"""
    try:
        logger.info("Fetching Tier 2 data...")
        
        tier2_data_dict = tier2.build_complete_dataset()
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'power_plants': {
                'count': len(tier2_data_dict['power_plants']) if tier2_data_dict.get('power_plants') is not None else 0,
                'available': tier2_data_dict.get('power_plants') is not None
            },
            'transmission_lines': {
                'count': len(tier2_data_dict['transmission_lines']) if tier2_data_dict.get('transmission_lines') is not None else 0,
                'available': tier2_data_dict.get('transmission_lines') is not None
            },
            'weather': tier2_data_dict.get('weather'),
            'prices': {
                'count': len(tier2_data_dict['prices']) if tier2_data_dict.get('prices') is not None else 0,
                'available': tier2_data_dict.get('prices') is not None
            },
            'earthquakes': {
                'count': len(tier2_data_dict['earthquakes']) if tier2_data_dict.get('earthquakes') is not None else 0,
                'available': tier2_data_dict.get('earthquakes') is not None
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Tier 2 data error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        metadata_path = os.path.join(Config.MODEL_DIR, 'training_metadata.json')
        
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        info = {
            'model': 'SmartGrid Price Predictor',
            'version': '1.0',
            'features': len(predictor.feature_names) if predictor.feature_names else 0,
            'feature_names': predictor.feature_names[:10] if predictor.feature_names else [],
            'training_metadata': metadata,
            'status': 'ready'
        }
        
        return jsonify(info)
    
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================
# Web UI
# ============================================

@app.route('/')
def index():
    """Web dashboard"""
    return render_template('index.html')

# ============================================
# Helper Functions
# ============================================

def get_current_features():
    """Fetch current features from Tier 2 data"""
    try:
        logger.info("Fetching current features from Tier 2...")
        
        tier2_data_dict = tier2.build_complete_dataset()
        
        # Get weather
        weather = tier2_data_dict.get('weather', {})
        
        # Get latest price data
        prices_df = tier2_data_dict.get('prices')
        if prices_df is not None and len(prices_df) > 0:
            latest = prices_df.iloc[0]
        else:
            latest = {}
        
        # Build feature dict
        features = {
            'temperature': weather.get('temperature', 20),
            'cloud_cover': 0.4,
            'wind_speed': weather.get('wind_speed', 5),
            'solar_mw': 1000,
            'wind_mw': 600,
            'total_demand': float(latest.get('value', 25000)) if 'value' in latest else 25000,
            'renewable_pct': 0.25,
            'imbalance': 2.0,
            'grid_stress': 0.7,
            'wildfire_risk': 0.0,
            'hour': datetime.now().hour,
            'month': datetime.now().month,
            'is_weekend': int(datetime.now().weekday() >= 5),
        }
        
        # Add lag features (would be real in production)
        for col in ['temperature', 'solar_mw', 'wind_mw', 'price']:
            for lag in [1, 3, 6, 12]:
                features[f'{col}_lag_{lag}'] = features.get(col, 50)
        
        return prepare_features_from_dict(features)
    
    except Exception as e:
        logger.error(f"Error getting current features: {e}")
        return None

def prepare_features_from_dict(data: Dict) -> Tuple:
    """
    Prepare feature vector from dictionary
    
    Args:
        data: Dictionary with feature values
    
    Returns:
        Feature vector in correct order for model
    """
    try:
        import numpy as np
        
        if predictor is None or predictor.feature_names is None:
            raise ValueError("Model or feature names not loaded")
        
        # Create vector in correct order
        feature_vector = []
        for feature_name in predictor.feature_names:
            value = data.get(feature_name, 0)
            feature_vector.append(float(value))
        
        return np.array(feature_vector).reshape(1, -1)
    
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return None

# ============================================
# Error Handlers
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================
# Startup/Shutdown
# ============================================

@app.before_request
def before_request():
    """Log incoming requests"""
    logger.debug(f"{request.method} {request.path}")

# ============================================
# Main
# ============================================

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    logger.info("Available endpoints:")
    logger.info("  GET  /health              - Health check")
    logger.info("  GET  /predict             - Real-time prediction")
    logger.info("  POST /predict             - Prediction with custom data")
    logger.info("  GET  /tier2-data          - Tier 2 data status")
    logger.info("  GET  /model-info          - Model information")
    logger.info("  GET  /                    - Web dashboard")
    logger.info("")
    
    app.run(host='0.0.0.0', port=8000, debug=False)