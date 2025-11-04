#!/usr/bin/env python3
"""
download_data.py - Standalone data download script
Downloads and saves all Tier 2 data to files for reuse
"""

import os
import sys
import json
import pickle
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd

from tier2_pipeline import (
    Tier2Config, Tier2DataPipeline, CAISOPriceFetcher, 
    NOAAWeather, PowerPlantDB, DisasterRisk
)

# Setup logging - create directory first
os.makedirs('/app/data/downloads', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/data/downloads/download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================

class DownloadConfig:
    """Data download configuration"""
    DATA_DIR = '/app/data/downloads'
    DATE_RANGE_MONTHS = 3  # Default: 3 months of data

# ============================================
# Data Download Functions
# ============================================

def download_weather_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Download and save weather data"""
    logger.info("=" * 70)
    logger.info("DOWNLOADING WEATHER DATA")
    logger.info("=" * 70)
    
    noaa_weather = NOAAWeather()
    
    weather_data = noaa_weather.get_all_california_weather(
        max_workers=5,
        start_date=start_date,
        end_date=end_date
    )
    
    if not weather_data:
        raise ValueError("Weather data fetch failed.")
    
    # Process weather data into DataFrame
    weather_records = []
    for city_id, response in weather_data.items():
        results = response.get('results', [])
        if not results:
            logger.warning(f"No results for city_id: {city_id}")
            continue
        
        for record in results:
            if 'date' not in record or 'value' not in record:
                continue
            
            record['city_id'] = city_id
            record['timestamp'] = record.pop('date')
            weather_records.append(record)
    
    if not weather_records:
        raise ValueError("No valid weather records found.")
    
    weather_df = pd.DataFrame(weather_records)
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], errors='coerce')
    weather_df = weather_df.dropna(subset=['timestamp'])
    weather_df['city_id'] = weather_df['city_id'].astype(str)
    
    # Save to pickle
    output_path = os.path.join(DownloadConfig.DATA_DIR, 'weather_data.pkl')
    weather_df.to_pickle(output_path)
    logger.info(f"✅ Saved {len(weather_df)} weather records to {output_path}")
    
    # Also save metadata
    metadata = {
        'download_date': datetime.now().isoformat(),
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'records': len(weather_df),
        'cities': len(weather_df['city_id'].unique()),
        'datatypes': weather_df['datatype'].unique().tolist() if 'datatype' in weather_df.columns else []
    }
    
    with open(os.path.join(DownloadConfig.DATA_DIR, 'weather_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return weather_df


def download_caiso_prices(start_date: datetime, end_date: datetime) -> dict:
    """Download and save CAISO price data"""
    logger.info("=" * 70)
    logger.info("DOWNLOADING CAISO PRICE DATA")
    logger.info("=" * 70)
    
    caiso_prices_dict = CAISOPriceFetcher.fetch_all_cities_prices(
        start_date=start_date.strftime('%Y%m%d'),
        end_date=end_date.strftime('%Y%m%d'),
        market='DAM',
        max_workers=Tier2Config.MAX_WORKERS
    )
    
    if not caiso_prices_dict:
        raise ValueError("CAISO data fetch failed.")
    
    # Combine all cities into one DataFrame
    caiso_combined = pd.concat(list(caiso_prices_dict.values()), ignore_index=True)
    
    # Save to pickle
    output_path = os.path.join(DownloadConfig.DATA_DIR, 'caiso_prices.pkl')
    caiso_combined.to_pickle(output_path)
    logger.info(f"✅ Saved {len(caiso_combined)} price records to {output_path}")
    
    # Save metadata
    metadata = {
        'download_date': datetime.now().isoformat(),
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'records': len(caiso_combined),
        'cities': len(caiso_prices_dict),
        'city_ids': list(caiso_prices_dict.keys()),
        'price_range': {
            'min': float(caiso_combined['lmp'].min()),
            'max': float(caiso_combined['lmp'].max()),
            'mean': float(caiso_combined['lmp'].mean())
        }
    }
    
    with open(os.path.join(DownloadConfig.DATA_DIR, 'caiso_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return caiso_prices_dict


def download_power_plants() -> pd.DataFrame:
    """Download and save power plant data"""
    logger.info("=" * 70)
    logger.info("DOWNLOADING POWER PLANT DATA")
    logger.info("=" * 70)
    
    plants_df = PowerPlantDB.download_plants()
    
    if plants_df is None:
        raise ValueError("Power plant data fetch failed.")
    
    # Save to pickle
    output_path = os.path.join(DownloadConfig.DATA_DIR, 'power_plants.pkl')
    plants_df.to_pickle(output_path)
    logger.info(f"✅ Saved {len(plants_df)} power plants to {output_path}")
    
    # Save metadata
    metadata = {
        'download_date': datetime.now().isoformat(),
        'plants': len(plants_df),
        'total_capacity_mw': float(plants_df['capacity_mw'].sum()),
        'avg_capacity_mw': float(plants_df['capacity_mw'].mean()),
        'fuel_types': plants_df['primary_fuel'].unique().tolist() if 'primary_fuel' in plants_df.columns else []
    }
    
    with open(os.path.join(DownloadConfig.DATA_DIR, 'power_plants_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return plants_df


def download_earthquakes(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Download and save earthquake data"""
    logger.info("=" * 70)
    logger.info("DOWNLOADING EARTHQUAKE DATA")
    logger.info("=" * 70)
    
    earthquakes_df = DisasterRisk.get_recent_earthquakes(
        start_date=start_date,
        end_date=end_date,
        min_magnitude=2.0
    )
    
    if earthquakes_df is None or len(earthquakes_df) == 0:
        logger.warning("No earthquake data available")
        earthquakes_df = pd.DataFrame()
    
    # Save to pickle
    output_path = os.path.join(DownloadConfig.DATA_DIR, 'earthquakes.pkl')
    earthquakes_df.to_pickle(output_path)
    logger.info(f"✅ Saved {len(earthquakes_df)} earthquake records to {output_path}")
    
    # Save metadata
    metadata = {
        'download_date': datetime.now().isoformat(),
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'records': len(earthquakes_df)
    }
    
    if len(earthquakes_df) > 0:
        metadata.update({
            'magnitude_range': {
                'min': float(earthquakes_df['magnitude'].min()),
                'max': float(earthquakes_df['magnitude'].max()),
                'mean': float(earthquakes_df['magnitude'].mean())
            }
        })
    
    with open(os.path.join(DownloadConfig.DATA_DIR, 'earthquakes_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return earthquakes_df


# ============================================
# Main Download Pipeline
# ============================================

def main():
    """Main data download pipeline"""
    
    parser = argparse.ArgumentParser(description='Download all training data')
    parser.add_argument('--date-range-months', type=int, default=DownloadConfig.DATE_RANGE_MONTHS,
                        help='Number of months of historical data to download')
    parser.add_argument('--output-dir', type=str, default=DownloadConfig.DATA_DIR,
                        help='Directory to save downloaded data')
    
    args = parser.parse_args()
    DownloadConfig.DATA_DIR = args.output_dir
    DownloadConfig.DATE_RANGE_MONTHS = args.date_range_months
    
    # Create output directory
    os.makedirs(DownloadConfig.DATA_DIR, exist_ok=True)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SMART GRID ML - DATA DOWNLOADER")
    logger.info("=" * 70)
    logger.info(f"Output directory: {DownloadConfig.DATA_DIR}")
    logger.info(f"Date range: {args.date_range_months} months")
    logger.info("")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.date_range_months * 30)
    
    logger.info(f"Downloading data from {start_date.date()} to {end_date.date()}")
    logger.info("")
    
    try:
        # Download all data
        weather_df = download_weather_data(start_date, end_date)
        caiso_prices = download_caiso_prices(start_date, end_date)
        plants_df = download_power_plants()
        earthquakes_df = download_earthquakes(start_date, end_date)
        
        # Create summary
        summary = {
            'download_completed': datetime.now().isoformat(),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'months': args.date_range_months
            },
            'datasets': {
                'weather': {
                    'records': len(weather_df),
                    'file': 'weather_data.pkl'
                },
                'caiso_prices': {
                    'records': len(pd.concat(list(caiso_prices.values()))) if caiso_prices else 0,
                    'file': 'caiso_prices.pkl'
                },
                'power_plants': {
                    'records': len(plants_df),
                    'file': 'power_plants.pkl'
                },
                'earthquakes': {
                    'records': len(earthquakes_df),
                    'file': 'earthquakes.pkl'
                }
            }
        }
        
        with open(os.path.join(DownloadConfig.DATA_DIR, 'download_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("DOWNLOAD COMPLETE ✅")
        logger.info("=" * 70)
        logger.info(f"Weather records: {len(weather_df)}")
        logger.info(f"CAISO price records: {summary['datasets']['caiso_prices']['records']}")
        logger.info(f"Power plants: {len(plants_df)}")
        logger.info(f"Earthquakes: {len(earthquakes_df)}")
        logger.info(f"\nAll data saved to: {DownloadConfig.DATA_DIR}")
        logger.info("")
        
        return 0
    
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())