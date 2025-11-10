import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_downloaded_data(data_dir: str) -> dict:
    """Load all previously downloaded data from pickle files"""
    logger.info("=" * 70)
    logger.info("LOADING DOWNLOADED DATA")
    logger.info("=" * 70)
    
    data = {}
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Load weather data
    weather_path = os.path.join(data_dir, 'weather_data.pkl')
    if os.path.exists(weather_path):
        data['weather'] = pd.read_pickle(weather_path)
        logger.info(f"✅ Loaded {len(data['weather'])} weather records")
    else:
        raise ValueError(f"Weather data not found: {weather_path}")
    
    # Load CAISO prices
    caiso_path = os.path.join(data_dir, 'caiso_prices.pkl')
    if os.path.exists(caiso_path):
        data['caiso'] = pd.read_pickle(caiso_path)
        logger.info(f"✅ Loaded {len(data['caiso'])} CAISO price records")
    else:
        raise ValueError(f"CAISO data not found: {caiso_path}")
    
    # Load power plants
    plants_path = os.path.join(data_dir, 'power_plants.pkl')
    if os.path.exists(plants_path):
        data['plants'] = pd.read_pickle(plants_path)
        logger.info(f"✅ Loaded {len(data['plants'])} power plants")
    else:
        logger.warning(f"Power plants data not found: {plants_path}")
        data['plants'] = pd.DataFrame()
    
    # Load earthquakes
    quakes_path = os.path.join(data_dir, 'earthquakes.pkl')
    if os.path.exists(quakes_path):
        data['earthquakes'] = pd.read_pickle(quakes_path)
        logger.info(f"✅ Loaded {len(data['earthquakes'])} earthquake records")
    else:
        logger.warning(f"Earthquake data not found: {quakes_path}")
        data['earthquakes'] = pd.DataFrame()
    
    logger.info("=" * 70)
    logger.info("")
    
    return data