#!/usr/bin/env python3
"""
Tier 2 Data Pipeline - Historical/Reference Data
COMPLETE UPDATED IMPLEMENTATION based on all working APIs
Integrates: CartoDB Power Plants, NOAA Weather, EIA Prices, USGS Earthquakes, osmnx Transmission Lines
"""

import os
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from dotenv import load_dotenv

# Load .env file explicitly
load_dotenv()

logger = logging.getLogger(__name__)

# Try to import geospatial libraries
try:
    import osmnx as ox
    import geopandas as gpd
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    logger.warning("osmnx/geopandas not installed. Transmission lines unavailable.")

# ============================================
# 1. CARTODB - Power Plant Database (WORKING)
# ============================================

class PowerPlantDB:
    """
    Global Power Plant Database via WRI/CartoDB
    API: CartoDB SQL API
    Data: powerwatch_data_20180102 table
    ~30,000 power plants worldwide
    """
    
    @staticmethod
    def download_plants() -> Optional[pd.DataFrame]:
        """Download power plant database from CartoDB SQL API"""
        try:
            logger.info("Fetching Global Power Plant Database from CartoDB...")
            
            carto_account = "wri-rw"
            table_name = "powerwatch_data_20180102"
            
            # SQL query
            sql_query = f"SELECT name, primary_fuel, capacity_mw, latitude, longitude, country_long FROM {table_name} LIMIT 50000"
            
            # CartoDB SQL API endpoint
            url = f"https://{carto_account}.carto.com/api/v2/sql"
            params = {
                'q': sql_query,
                'format': 'json'
            }
            
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if 'rows' not in data or len(data['rows']) == 0:
                logger.warning("No data returned from CartoDB")
                return PowerPlantDB._get_mock_california_plants()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['rows'])
            
            # Clean data
            df['capacity_mw'] = pd.to_numeric(df['capacity_mw'], errors='coerce')
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            
            # Remove nulls
            df = df.dropna(subset=['capacity_mw', 'latitude', 'longitude'])
            
            logger.info(f"✓ Downloaded {len(df)} power plants from CartoDB")
            return df
        
        except Exception as e:
            logger.warning(f"CartoDB download failed: {e}. Using mock data.")
            return PowerPlantDB._get_mock_california_plants()
    
    @staticmethod
    def _get_mock_california_plants() -> pd.DataFrame:
        """Mock California power plant data for testing"""
        logger.info("Using mock California power plant data")
        
        mock_plants = {
            'name': [
                'Diablo Canyon', 'Palo Verde', 'San Onofre',
                'Solar Farm 1', 'Wind Farm 1',
                'Natural Gas Plant A', 'Natural Gas Plant B',
                'Hoover Dam', 'Glen Canyon Dam'
            ],
            'primary_fuel': [
                'Nuclear', 'Nuclear', 'Nuclear',
                'Solar', 'Wind',
                'Gas', 'Gas',
                'Hydro', 'Hydro'
            ],
            'capacity_mw': [2240, 3937, 2150, 500, 300, 800, 600, 2080, 1320],
            'latitude': [35.2, 32.4, 33.2, 34.0, 35.5, 34.5, 37.0, 36.0, 37.3],
            'longitude': [-120.9, -113.4, -117.6, -118.5, -120.0, -118.0, -122.0, -114.7, -111.5],
            'country_long': ['United States'] * 9
        }
        
        return pd.DataFrame(mock_plants)
    
    @staticmethod
    def get_california_plants(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Filter for California plants only"""
        if df is None or len(df) == 0:
            return PowerPlantDB._get_mock_california_plants()
        
        try:
            # California bounding box
            ca_bounds = (
                (df['latitude'] >= 32.5) & (df['latitude'] <= 42) &
                (df['longitude'] >= -124.5) & (df['longitude'] <= -114)
            )
            
            ca_plants = df[ca_bounds].copy()
            
            logger.info(f"✓ Found {len(ca_plants)} California plants")
            
            # Standardize columns
            result = pd.DataFrame()
            result['name'] = ca_plants.get('name', 'Unknown')
            result['primary_fuel'] = ca_plants.get('primary_fuel', 'Unknown')
            result['capacity_mw'] = ca_plants['capacity_mw']
            result['latitude'] = ca_plants['latitude']
            result['longitude'] = ca_plants['longitude']
            
            return result if len(result) > 0 else PowerPlantDB._get_mock_california_plants()
        
        except Exception as e:
            logger.warning(f"Error filtering California plants: {e}")
            return PowerPlantDB._get_mock_california_plants()

# ============================================
# 2. OSMNX - Transmission Lines (WORKING)
# ============================================

class GridTopology:
    """Get transmission lines from OpenStreetMap via osmnx"""
    
    @staticmethod
    def get_transmission_lines(place: str) -> Optional[gpd.GeoDataFrame]:
        """
        Get transmission lines in place
        
        Note: Large areas may fail. This is optional data.
        """
        if not GEOSPATIAL_AVAILABLE:
            logger.warning("osmnx not available. Skipping transmission lines.")
            return None
        
        try:
            # south, west, north, east = bbox
            # logger.info("Fetching transmission lines from OpenStreetMap...")    
            # tags = {"power": "line"}  # Tag for transmission lines
            # gdf = ox.geometries.geometries_from_bbox(*bbox, tags=tags)
            # logger.info(f"✓ Found {len(gdf)} transmission lines")
            # return gdf
            
            tags = {"power": "line"}
            gdf = ox.features.features_from_place(place, tags)
            logger.info(f"✓ Found {len(gdf)} transmission lines")
            return gdf
        
        except Exception as e:
            logger.warning(f"Transmission lines fetch failed: {e} (this is optional data)")
            return None

# ============================================
# 3. NOAA - Weather Data (FROM main.py)
# ============================================

class NOAAWeather:
    """NOAA weather forecast data"""
    
    @staticmethod
    def get_weather_forecast(lat: float = 34.0522, lon: float = -118.2437) -> Optional[Dict]:
        """Get NOAA weather forecast"""
        try:
            session = requests.Session()
            session.headers.update({'User-Agent': 'SmartGridML/1.0'})
            
            # Get grid point metadata
            points_url = f"https://api.weather.gov/points/{lat},{lon}"
            points_resp = session.get(points_url, timeout=10)
            points_resp.raise_for_status()
            points_data = points_resp.json()
            
            # Get forecast URL
            forecast_url = points_data['properties']['forecast']
            
            # Get hourly forecast
            forecast_resp = session.get(forecast_url, timeout=10)
            forecast_resp.raise_for_status()
            forecast_data = forecast_resp.json()
            
            periods = forecast_data['properties']['periods'][:24]
            
            temps = []
            winds = []
            
            for period in periods:
                temp_f = period['temperature']
                temp_c = (temp_f - 32) * 5/9
                temps.append(temp_c)
                
                wind_str = period['windSpeed']
                wind_mph = float(wind_str.split()[0])
                wind_mps = wind_mph * 0.44704
                winds.append(wind_mps)
            
            return {
                'temperature': np.mean(temps),
                'max_temperature': max(temps),
                'wind_speed': np.mean(winds),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.warning(f"NOAA fetch failed: {e}")
            return None

# ============================================
# 4. EIA - Electricity Prices (FROM main.py)
# ============================================

class PriceHistory:
    """Historical electricity prices from EIA"""
    
    @staticmethod
    def get_wholesale_prices(api_key: str, region: str = 'CISO') -> Optional[pd.DataFrame]:
        """Get historical electricity demand data (hourly) - proxy for price"""
        try:
            if api_key == 'demo_key' or not api_key:
                logger.warning("No valid EIA API key. Skipping historical data.")
                return None
            
            logger.info("Fetching hourly demand data from EIA...")
            
            # Use Demand (D) and Net Generation (NG) - these have data for CISO
            data_types = ['D', 'NG']  # Demand, Net Generation
            
            for data_type in data_types:
                try:
                    url = (
                        f"https://api.eia.gov/v2/electricity/rto/region-data/data?"
                        f"api_key={api_key}&frequency=hourly&data%5B0%5D=value&"
                        f"facets%5Brespondent%5D%5B%5D={region}&"
                        f"facets%5Btype%5D%5B%5D={data_type}&"
                        f"sort%5B0%5D%5Bcolumn%5D=period&"
                        f"sort%5B0%5D%5Bdirection%5D=desc&"
                        f"length=5000"
                    )
                    
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'response' in data and 'data' in data['response'] and len(data['response']['data']) > 0:
                        records = []
                        for record in data['response']['data']:
                            try:
                                records.append({
                                    'period': record['period'],
                                    'value': float(record['value']),
                                    'type': data_type,
                                    'type_name': record.get('type-name', data_type)
                                })
                            except (ValueError, KeyError, TypeError):
                                continue
                        
                        if records:
                            df = pd.DataFrame(records)
                            logger.info(f"✓ Retrieved {len(df)} hourly {data_type} records")
                            return df
                
                except Exception as e:
                    logger.debug(f"Data type {data_type} fetch failed: {e}")
                    continue
            
            logger.warning("No data found from EIA")
            return None
        
        except Exception as e:
            logger.warning(f"EIA data fetch failed: {e}")
            return None

# ============================================
# 5. USGS - Earthquake Data (FROM main.py)
# ============================================

class DisasterRisk:
    """Earthquake hazards from USGS"""
    
    @staticmethod
    def get_recent_earthquakes(days: int = 7) -> Optional[pd.DataFrame]:
        """Get earthquakes in past N days"""
        try:
            url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            quakes = []
            for feature in data['features']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                
                quakes.append({
                    'latitude': coords[1],
                    'longitude': coords[0],
                    'magnitude': props.get('mag', 0),
                    'time': datetime.fromtimestamp(props['time'] / 1000),
                    'depth_km': coords[2],
                })
            
            df = pd.DataFrame(quakes)
            logger.info(f"✓ Retrieved {len(df)} recent earthquakes")
            return df if len(df) > 0 else None
        
        except Exception as e:
            logger.warning(f"Earthquake data fetch failed: {e}")
            return None

# ============================================
# 6. COMPLETE TIER 2 PIPELINE
# ============================================

class Tier2DataPipeline:
    """
    Complete Tier 2 integration
    Combines: Power Plants, Weather, Prices, Earthquakes, Transmission Lines
    """
    
    def __init__(self, api_key: str = None):
        # Try to get API key from parameter, then environment, then default
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('EIA_API_KEY') or 'demo_key'
        
        logger.info(f"EIA API Key: {'Set' if self.api_key != 'demo_key' else 'Not set (using demo)'}")
        
        self.cache_dir = 'tier2_data'
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def build_complete_dataset(self) -> Dict:
        """Build complete Tier 2 dataset"""
        
        logger.info("=" * 70)
        logger.info("TIER 2 DATA PIPELINE")
        logger.info("=" * 70)
        logger.info("")
        
        data = {}
        
        # 1. Power Plants
        logger.info("1️⃣ Fetching power plants...")
        plants_df = PowerPlantDB.download_plants()
        ca_plants = PowerPlantDB.get_california_plants(plants_df)
        data['power_plants'] = ca_plants
        
        if ca_plants is not None:
            logger.info(f"  ✓ {len(ca_plants)} CA plants")
            logger.info(f"  ✓ Fuel mix: {ca_plants['primary_fuel'].value_counts().to_dict()}")
            logger.info(f"  ✓ Total capacity: {ca_plants['capacity_mw'].sum():.0f} MW")
        logger.info("")
        
        # 2. Transmission Lines
        logger.info("2️⃣ Fetching transmission topology...")
        trans_lines = GridTopology.get_transmission_lines("Los Angeles, California, USA")
        data['transmission_lines'] = trans_lines
        
        if trans_lines is not None:
            logger.info(f"  ✓ {len(trans_lines)} transmission lines")
        logger.info("")
        
        # 3. Weather
        logger.info("3️⃣ Fetching weather forecast...")
        weather = NOAAWeather.get_weather_forecast()
        data['weather'] = weather
        
        if weather:
            logger.info(f"  ✓ Temp: {weather['temperature']:.1f}°C")
            logger.info(f"  ✓ Wind: {weather['wind_speed']:.1f} m/s")
        logger.info("")
        
        # 4. Historical Prices
        logger.info("4️⃣ Fetching historical prices...")
        prices = PriceHistory.get_wholesale_prices(self.api_key)
        data['prices'] = prices
        
        if prices is not None:
            logger.info(f"  ✓ {len(prices)} price records")
        logger.info("")
        
        # 5. Earthquakes
        logger.info("5️⃣ Fetching seismic data...")
        earthquakes = DisasterRisk.get_recent_earthquakes()
        data['earthquakes'] = earthquakes
        
        if earthquakes is not None:
            logger.info(f"  ✓ {len(earthquakes)} recent earthquakes")
        logger.info("")
        
        logger.info("✅ Tier 2 pipeline complete!")
        logger.info("")
        
        return data

# ============================================
# 7. USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize
    api_key = os.getenv('EIA_API_KEY', 'demo_key')
    tier2 = Tier2DataPipeline(api_key=api_key)
    
    # Build dataset
    data = tier2.build_complete_dataset()
    
    # Summary
    print("\n" + "=" * 70)
    print("TIER 2 DATA SUMMARY")
    print("=" * 70)
    
    for key, value in data.items():
        if value is None:
            print(f"\n{key.upper()}: Not available")
        elif isinstance(value, pd.DataFrame):
            print(f"\n{key.upper()}:")
            print(f"  Records: {len(value)}")
            print(f"  Columns: {value.columns.tolist()}")
        elif isinstance(value, dict):
            print(f"\n{key.upper()}: {value}")
        else:
            print(f"\n{key.upper()}: {type(value)}")