#!/usr/bin/env python3
"""
Tier 2 Data Pipeline - California-Wide Coverage with CAISO OASIS
FINAL FIX: 
- resultformat=6 for CSV
- Price data is in MW column, filtered by XML_DATA_ITEM
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
import zipfile
import io

# Import locations
from locations import (
    CALIFORNIA_CITIES, CALIFORNIA_COUNTIES, CAISO_REGIONS,
    get_city, get_county, list_all_cities
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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
# CONFIGURATION
# ============================================

class Tier2Config:
    """Configuration for Tier 2 data fetching"""
    
    MAX_WORKERS = 1
    REQUEST_TIMEOUT = 30
    CACHE_DIR = 'tier2_data'
    CACHE_DURATION = 3600
    OSMX_TIMEOUT = 180
    OSMX_MAX_RETRIES = 2
    RATE_LIMIT_DELAY = 5.0

# ============================================
# CAISO NODE MAPPING
# ============================================

CAISO_PRICE_NODES = {
    'san_francisco': ('BAYSHOR2_1_N001', 'Bayshore', 'PG&E'),
    'oakland': ('BAYSHOR2_1_N001', 'Bayshore', 'PG&E'),
    'san_jose': ('SLAP_PGE-APND', 'San Jose Area', 'PG&E'),
    'sacramento': ('CAPTJCK_1_N003', 'Captain Jack', 'PG&E'),
    'fresno': ('GATES_6_N003', 'Gates', 'PG&E'),
    'bakersfield': ('MIDWAY_2_N001', 'Midway', 'PG&E'),
    'visalia': ('GATES_6_N003', 'Gates', 'PG&E'),
    'los_angeles': ('0096WD_7_N001', 'Willow Pass', 'SCE'),
    'san_diego': ('TL20B_7_N001', 'San Diego', 'SDG&E'),
    'riverside': ('VILLA_6_N001', 'Villa Park', 'SCE'),
    'santa_ana': ('0096WD_7_N001', 'Willow Pass', 'SCE'),
    'anaheim': ('0096WD_7_N001', 'Willow Pass', 'SCE'),
}

CAISO_REGIONAL_NODES = {
    'north': ['BAYSHOR2_1_N001', 'CAPTJCK_1_N003', 'SLAP_PGE-APND'],
    'central': ['GATES_6_N003', 'MIDWAY_2_N001'],
    'south': ['0096WD_7_N001', 'TL20B_7_N001', 'VILLA_6_N001']
}

# ============================================
# CAISO OASIS - LMP Price Fetcher
# ============================================

class CAISOPriceFetcher:
    """Fetch LMP prices from CAISO OASIS API"""
    
    BASE_URL = "https://oasis.caiso.com/oasisapi/SingleZip"
    
    @staticmethod
    def _get_date_chunks(start_date_str: str, end_date_str: str, chunk_days: int = 29) -> List[Tuple[str, str]]:
        """Splits date range into 29-day chunks (under 31-day API limit)"""
        try:
            start_date = datetime.strptime(start_date_str, '%Y%m%d')
            end_date = datetime.strptime(end_date_str, '%Y%m%d')
        except ValueError:
            logger.error("Invalid date format. Use YYYYMMDD.")
            return []

        chunks = []
        current_start = start_date
        
        while current_start <= end_date:
            current_end = current_start + timedelta(days=chunk_days)
            if current_end > end_date:
                current_end = end_date
                
            chunks.append((
                current_start.strftime('%Y%m%d'),
                current_end.strftime('%Y%m%d')
            ))
            
            current_start = current_end + timedelta(days=1)
        
        return chunks
    
    @staticmethod
    def fetch_lmp_data(node_id: str, start_date_full: str, end_date_full: str, 
                       market: str = 'DAM') -> Optional[pd.DataFrame]:
        """
        Fetch LMP data for a node with automatic chunking
        
        CAISO CSV Structure:
        - resultformat=6 returns CSV with columns:
          INTERVALSTARTTIME_GMT, NODE_ID, XML_DATA_ITEM, MW, etc.
        - XML_DATA_ITEM values: LMP_PRC, LMP_ENE_PRC, LMP_CONG_PRC, LMP_LOSS_PRC
        - Actual price values are in the MW column
        
        Args:
            node_id: CAISO node ID (e.g., 'BAYSHOR2_1_N001')
            start_date_full: Start date 'YYYYMMDD'
            end_date_full: End date 'YYYYMMDD'
            market: 'DAM' (Day-Ahead) or 'RTM' (Real-Time)
        
        Returns:
            DataFrame with LMP data
        """
        all_dfs = []
        
        # Generate 29-day chunks
        date_chunks = CAISOPriceFetcher._get_date_chunks(start_date_full, end_date_full, chunk_days=29)
        
        logger.info(f"Fetching CAISO LMP for node {node_id} ({market})... {len(date_chunks)} chunks")
        
        for chunk_start_date, chunk_end_date in date_chunks:
            
            # Format dates for API
            start_date_api = f"{chunk_start_date}T07:00-0000"
            end_date_api = f"{chunk_end_date}T07:00-0000"
            
            # Build query parameters
            params = {
                'resultformat': '6',  # CSV format
                'queryname': 'PRC_LMP',
                'version': '12',
                'startdatetime': start_date_api,
                'enddatetime': end_date_api,
                'market_run_id': market,
                'node': node_id
            }
            
            try:
                response = requests.get(
                    CAISOPriceFetcher.BASE_URL,
                    params=params,
                    timeout=60
                )
                response.raise_for_status()
                
                # Check for XML error
                if response.content.startswith(b'<?xml'):
                    logger.warning(f"API returned XML error for {chunk_start_date}-{chunk_end_date}")
                    time.sleep(Tier2Config.RATE_LIMIT_DELAY)
                    continue

                # Open zip file
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
                    
                    if not csv_files:
                        logger.warning(f"Empty zip for {chunk_start_date}-{chunk_end_date}")
                        time.sleep(Tier2Config.RATE_LIMIT_DELAY)
                        continue
                    
                    csv_filename = csv_files[0]
                    
                    with zf.open(csv_filename) as f:
                        df = pd.read_csv(f, sep=',', engine='python')

                if df.empty:
                    logger.warning(f"No data for {node_id} chunk {chunk_start_date}")
                    time.sleep(Tier2Config.RATE_LIMIT_DELAY)
                    continue

                # Parse timestamp
                df['timestamp'] = pd.to_datetime(df['INTERVALSTARTTIME_GMT'])
                
                # CRITICAL FIX: Price data is in MW column, filtered by XML_DATA_ITEM
                # XML_DATA_ITEM values indicate the price type:
                # - LMP_PRC: Total locational marginal price
                # - LMP_ENE_PRC: Energy component
                # - LMP_CONG_PRC: Congestion component  
                # - LMP_LOSS_PRC: Loss component
                
                df_lmp = df[df['XML_DATA_ITEM'] == 'LMP_PRC'].copy()
                
                if df_lmp.empty:
                    logger.warning(f"No LMP_PRC rows for {chunk_start_date}")
                    time.sleep(Tier2Config.RATE_LIMIT_DELAY)
                    continue
                
                # Build result using MW column for prices
                result = pd.DataFrame({
                    'timestamp': df_lmp['timestamp'],
                    'node': df_lmp['NODE_ID'],
                    'lmp': pd.to_numeric(df_lmp['MW'], errors='coerce')
                }).reset_index(drop=True)
                
                # Optionally add component prices
                df_energy = df[df['XML_DATA_ITEM'] == 'LMP_ENE_PRC'].copy()
                df_cong = df[df['XML_DATA_ITEM'] == 'LMP_CONG_PRC'].copy()
                df_loss = df[df['XML_DATA_ITEM'] == 'LMP_LOSS_PRC'].copy()
                
                if len(df_energy) > 0:
                    energy_prices = pd.to_numeric(df_energy['MW'], errors='coerce').reset_index(drop=True)
                    if len(energy_prices) == len(result):
                        result['energy'] = energy_prices
                
                if len(df_cong) > 0:
                    cong_prices = pd.to_numeric(df_cong['MW'], errors='coerce').reset_index(drop=True)
                    if len(cong_prices) == len(result):
                        result['congestion'] = cong_prices
                
                if len(df_loss) > 0:
                    loss_prices = pd.to_numeric(df_loss['MW'], errors='coerce').reset_index(drop=True)
                    if len(loss_prices) == len(result):
                        result['loss'] = loss_prices

                # Remove NaN prices
                result = result.dropna(subset=['lmp'])
                
                if len(result) > 0:
                    all_dfs.append(result)
                    logger.info(f"  ✓ {len(result)} records for {chunk_start_date}-{chunk_end_date}")
                else:
                    logger.warning(f"No valid prices for {chunk_start_date}")

            except Exception as e:
                logger.error(f"Failed to fetch {node_id} chunk {chunk_start_date}: {e}")
            
            time.sleep(Tier2Config.RATE_LIMIT_DELAY)
            
        if not all_dfs:
            logger.warning(f"No data retrieved for {node_id}")
            return None
            
        # Combine chunks
        final_df = pd.concat(all_dfs, ignore_index=True).sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        
        logger.info(f"✓ Final LMP data for {node_id}: {len(final_df)} records")
        logger.info(f"  Price range: ${final_df['lmp'].min():.2f} - ${final_df['lmp'].max():.2f}/MWh")
        logger.info(f"  Mean price: ${final_df['lmp'].mean():.2f}/MWh")
        
        return final_df
        
    @staticmethod
    def fetch_city_prices(city_id: str, start_date: str, end_date: str, 
                          market: str = 'DAM') -> Optional[pd.DataFrame]:
        """Fetch LMP data for a California city"""
        if city_id not in CAISO_PRICE_NODES:
            logger.warning(f"No CAISO node mapping for city: {city_id}")
            return None
        
        node_id, node_name, zone = CAISO_PRICE_NODES[city_id]
        city = get_city(city_id)
        
        logger.info(f"Fetching prices for {city['name']} (node: {node_name})")
        
        df = CAISOPriceFetcher.fetch_lmp_data(node_id, start_date, end_date, market)
        
        if df is not None:
            df['city_id'] = city_id
            df['city_name'] = city['name']
            df['zone'] = zone
        
        return df
    
    @staticmethod
    def fetch_all_cities_prices(start_date: str, end_date: str, 
                                market: str = 'DAM', 
                                max_workers: int = 5) -> Dict[str, pd.DataFrame]:
        """Fetch LMP data for all cities concurrently"""
        logger.info("=" * 70)
        logger.info("FETCHING CAISO PRICES FOR ALL CALIFORNIA CITIES")
        logger.info("=" * 70)
        
        results = {}
        future_to_city = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for city_id in CAISO_PRICE_NODES.keys():
                future = executor.submit(
                    CAISOPriceFetcher.fetch_city_prices, 
                    city_id, start_date, end_date, market
                )
                future_to_city[future] = city_id
                time.sleep(Tier2Config.RATE_LIMIT_DELAY)

            for future in as_completed(future_to_city):
                city_id = future_to_city[future]
                try:
                    df = future.result()
                    if df is not None:
                        results[city_id] = df
                except Exception as e:
                    logger.error(f"Error fetching prices for {city_id}: {e}")
        
        logger.info(f"✓ Successfully fetched prices for {len(results)}/{len(CAISO_PRICE_NODES)} cities")
        return results

# ============================================
# OTHER DATA SOURCES (UNCHANGED)
# ============================================

class PowerPlantDB:
    """Global Power Plant Database via WRI/CartoDB"""
    
    @staticmethod
    @lru_cache(maxsize=1)
    def download_plants() -> Optional[pd.DataFrame]:
        """Download power plant database from CartoDB"""
        try:
            logger.info("Fetching Global Power Plant Database from CartoDB...")
            
            carto_account = "wri-rw"
            table_name = "powerwatch_data_20180102"
            
            sql_query = f"""
                SELECT name, primary_fuel, capacity_mw, latitude, longitude, country_long 
                FROM {table_name} 
                WHERE latitude >= 32.5 AND latitude <= 42 
                AND longitude >= -124.5 AND longitude <= -114
                LIMIT 10000
            """
            
            url = f"https://{carto_account}.carto.com/api/v2/sql"
            params = {'q': sql_query, 'format': 'json'}
            
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if 'rows' not in data or len(data['rows']) == 0:
                raise ValueError("No data returned from CartoDB")
            
            df = pd.DataFrame(data['rows'])
            df['capacity_mw'] = pd.to_numeric(df['capacity_mw'], errors='coerce')
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            df = df.dropna(subset=['capacity_mw', 'latitude', 'longitude'])
            
            logger.info(f"✓ Downloaded {len(df)} California power plants")
            return df
        
        except Exception as e:
            logger.warning(f"CartoDB download failed: {e}")
            raise ValueError("Failed to download power plant data")


class NOAAWeather:
    """NOAA weather forecast data"""
    
    @staticmethod
    def _fetch_city_weather(city_id: str, city_data: Dict) -> Tuple[str, Optional[Dict]]:
        """Fetch weather for a single city"""
        try:
            lat = city_data['lat']
            lon = city_data['lon']
            
            session = requests.Session()
            session.headers.update({'User-Agent': 'SmartGridML/1.0'})
            
            points_url = f"https://api.weather.gov/points/{lat},{lon}"
            points_resp = session.get(points_url, timeout=10)
            points_resp.raise_for_status()
            points_data = points_resp.json()
            
            forecast_url = points_data['properties']['forecast']
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
            
            weather = {
                'city_id': city_id,
                'city_name': city_data['name'],
                'temperature': np.mean(temps),
                'max_temperature': max(temps),
                'min_temperature': min(temps),
                'wind_speed': np.mean(winds),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✓ Weather for {city_data['name']}: {weather['temperature']:.1f}°C")
            return city_id, weather
            
        except Exception as e:
            logger.warning(f"✗ Weather fetch failed for {city_data['name']}: {e}")
            return city_id, None
    
    @staticmethod
    def get_all_california_weather(max_workers: int = 5) -> Dict[str, Dict]:
        """Fetch weather for all cities concurrently"""
        logger.info("=" * 70)
        logger.info("FETCHING WEATHER FOR ALL CALIFORNIA CITIES")
        logger.info("=" * 70)
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_city = {
                executor.submit(NOAAWeather._fetch_city_weather, city_id, city_data): city_id
                for city_id, city_data in CALIFORNIA_CITIES.items()
            }
            
            for future in as_completed(future_to_city):
                city_id = future_to_city[future]
                try:
                    city_id, weather = future.result()
                    if weather is not None:
                        results[city_id] = weather
                    time.sleep(Tier2Config.RATE_LIMIT_DELAY)
                except Exception as e:
                    logger.error(f"Error processing {city_id}: {e}")
        
        logger.info(f"✓ Successfully fetched weather for {len(results)}/{len(CALIFORNIA_CITIES)} cities")
        return results


class DisasterRisk:
    """Earthquake hazards from USGS"""
    
    @staticmethod
    def get_recent_earthquakes(days: int = 7, min_magnitude: float = 2.0) -> Optional[pd.DataFrame]:
        """Get earthquakes in California"""
        try:
            url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            quakes = []
            for feature in data['features']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                
                lat, lon = coords[1], coords[0]
                if 32.5 <= lat <= 42 and -124.5 <= lon <= -114:
                    mag = props.get('mag', 0)
                    if mag >= min_magnitude:
                        quakes.append({
                            'latitude': lat,
                            'longitude': lon,
                            'magnitude': mag,
                            'time': datetime.fromtimestamp(props['time'] / 1000),
                            'depth_km': coords[2],
                            'place': props.get('place', 'Unknown')
                        })
            
            df = pd.DataFrame(quakes)
            logger.info(f"✓ Retrieved {len(df)} California earthquakes (M≥{min_magnitude})")
            return df if len(df) > 0 else None
        
        except Exception as e:
            logger.warning(f"Earthquake data fetch failed: {e}")
            return None


class Tier2DataPipeline:
    """Complete Tier 2 integration pipeline"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('EIA_API_KEY') or 'demo_key'
        self.cache_dir = Tier2Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self._plants_cache = None
        self._weather_cache = {}
        self._prices_cache = {}
    
    def build_complete_dataset(self, use_concurrent: bool = True, 
                              start_date: str = None, 
                              end_date: str = None) -> Dict:
        """Build complete Tier 2 dataset with CAISO prices"""
        logger.info("=" * 70)
        logger.info("TIER 2 DATA PIPELINE - CAISO OASIS ENABLED")
        logger.info("=" * 70)
        logger.info("")
        
        # Default dates: last 7 days
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        data = {}
        
        # 1. CAISO Prices
        logger.info("1️⃣  Fetching CAISO LMP prices (all cities)...")
        if use_concurrent:
            caiso_prices = CAISOPriceFetcher.fetch_all_cities_prices(
                start_date, end_date, market='DAM', max_workers=5
            )
            self._prices_cache = caiso_prices
            data['caiso_prices_by_city'] = caiso_prices
            
            if caiso_prices:
                total_records = sum(len(df) for df in caiso_prices.values())
                logger.info(f"  ✓ {len(caiso_prices)} cities, {total_records} total records")
        else:
            logger.info("  ⊘ Skipping CAISO (use_concurrent=False)")
            data['caiso_prices_by_city'] = {}
        logger.info("")
        
        # 2. Power Plants
        logger.info("2️⃣  Fetching power plants...")
        plants_df = PowerPlantDB.download_plants()
        self._plants_cache = plants_df
        data['power_plants'] = plants_df
        
        if plants_df is not None:
            logger.info(f"  ✓ {len(plants_df)} CA plants")
        logger.info("")
        
        # 3. Weather
        logger.info("3️⃣  Fetching weather...")
        if use_concurrent:
            weather_data = NOAAWeather.get_all_california_weather(max_workers=5)
            self._weather_cache = weather_data
            data['weather_by_city'] = weather_data
            logger.info(f"  ✓ {len(weather_data)} cities")
        else:
            data['weather_by_city'] = {}
        logger.info("")
        
        # 4. Earthquakes
        logger.info("4️⃣  Fetching seismic data...")
        earthquakes = DisasterRisk.get_recent_earthquakes()
        data['earthquakes'] = earthquakes
        
        if earthquakes is not None:
            logger.info(f"  ✓ {len(earthquakes)} recent earthquakes")
        logger.info("")
        
        logger.info("✅ Tier 2 pipeline complete!")
        logger.info("")
        
        return data


if __name__ == "__main__":
    print("\nTesting CAISO API with MW column fix...")
    
    today = datetime.now().strftime('%Y%m%d')
    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
    
    print(f"\nFetching LMP for Los Angeles ({week_ago} to {today})...")
    la_prices = CAISOPriceFetcher.fetch_city_prices('los_angeles', week_ago, today)
    
    if la_prices is not None:
        print(f"\n✓ Retrieved {len(la_prices)} records")
        print(f"Price stats: ${la_prices['lmp'].min():.2f} - ${la_prices['lmp'].max():.2f}/MWh")
        print(f"Mean: ${la_prices['lmp'].mean():.2f}/MWh")
        print("\nSample data:")
        print(la_prices.head())
    else:
        print("\n❌ Failed to fetch data")
    
    # Test full pipeline
    print("\n" + "=" * 70)
    print("Testing full pipeline...")
    print("=" * 70)
    
    pipeline = Tier2DataPipeline()
    data = pipeline.build_complete_dataset(
        use_concurrent=True,
        start_date=week_ago,
        end_date=today
    )
    
    print("\nResults:")
    print(f"  CAISO cities: {len(data.get('caiso_prices_by_city', {}))}")
    print(f"  Power plants: {len(data.get('power_plants', [])) if data.get('power_plants') is not None else 0}")
    print(f"  Weather cities: {len(data.get('weather_by_city', {}))}")
    print(f"  Earthquakes: {len(data.get('earthquakes', [])) if data.get('earthquakes') is not None else 0}")