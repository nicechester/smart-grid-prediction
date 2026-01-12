#!/usr/bin/env python3
"""
download_geo.py - Download historical prices for California CAISO nodes

Downloads LMP prices for all (or sampled) California price nodes with their
geographic coordinates for training the geolocation-based prediction model.

Features:
- Sample N nodes for faster initial training
- Checkpoint/resume support
- Rate limiting to respect CAISO API
- Progress tracking
"""

import os
import sys
import json
import pickle
import argparse
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caiso_nodes import CAISONodes, get_california_nodes
from tier2_pipeline import CAISOPriceFetcher, Tier2Config, PowerPlantDB, DisasterRisk, NOAAWeather

# Setup logging - works in Docker (/app) and locally
LOG_DIR = '/app/data/downloads' if os.path.exists('/app') else os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'downloads')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'download_geo.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GeoDataDownloader:
    """Download price data for California nodes with geographic information"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or '/Users/chester.kim/workspace/tf/electricity-forecasting/tier3_poc/data/downloads'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load California nodes
        self.ca_nodes = get_california_nodes()
        logger.info(f"Loaded {len(self.ca_nodes)} California nodes")
        
        # Track progress
        self.checkpoint_file = os.path.join(self.output_dir, 'geo_download_checkpoint.json')
        self.completed_nodes: Set[str] = set()
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load download checkpoint if exists"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                self.completed_nodes = set(data.get('completed_nodes', []))
            logger.info(f"Loaded checkpoint: {len(self.completed_nodes)} nodes already completed")
    
    def _save_checkpoint(self):
        """Save download checkpoint"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'completed_nodes': list(self.completed_nodes),
                'last_updated': datetime.now().isoformat()
            }, f)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def sample_nodes(self, n: int, strategy: str = 'stratified') -> Dict[str, dict]:
        """
        Sample N nodes for faster training
        
        Args:
            n: Number of nodes to sample
            strategy: 'random', 'stratified' (by area), or 'geographic' (spatial spread)
        
        Returns:
            Dictionary of sampled nodes
        """
        if n >= len(self.ca_nodes):
            return self.ca_nodes
        
        if strategy == 'random':
            node_ids = random.sample(list(self.ca_nodes.keys()), n)
            return {nid: self.ca_nodes[nid] for nid in node_ids}
        
        elif strategy == 'stratified':
            # Sample proportionally from each area
            by_area = {}
            for node_id, info in self.ca_nodes.items():
                area = info['area']
                if area not in by_area:
                    by_area[area] = []
                by_area[area].append(node_id)
            
            sampled = {}
            total = len(self.ca_nodes)
            
            for area, node_ids in by_area.items():
                # Proportion of this area
                area_n = max(1, int(n * len(node_ids) / total))
                area_sample = random.sample(node_ids, min(area_n, len(node_ids)))
                for nid in area_sample:
                    sampled[nid] = self.ca_nodes[nid]
            
            # Trim to exact n if needed
            if len(sampled) > n:
                keys = random.sample(list(sampled.keys()), n)
                sampled = {k: sampled[k] for k in keys}
            
            return sampled
        
        elif strategy == 'geographic':
            # Sample to maximize geographic spread using a grid
            import numpy as np
            
            # Create grid cells
            lat_min, lat_max = 32.5, 42.0
            lon_min, lon_max = -124.5, -114.0
            
            grid_size = int(np.sqrt(n))
            lat_step = (lat_max - lat_min) / grid_size
            lon_step = (lon_max - lon_min) / grid_size
            
            sampled = {}
            for node_id, info in self.ca_nodes.items():
                lat_cell = int((info['latitude'] - lat_min) / lat_step)
                lon_cell = int((info['longitude'] - lon_min) / lon_step)
                cell_key = (lat_cell, lon_cell)
                
                # Keep one node per cell (randomly replace)
                if cell_key not in sampled or random.random() < 0.3:
                    sampled[cell_key] = (node_id, info)
            
            # Convert back to dict
            result = {nid: info for nid, info in sampled.values()}
            
            # Trim to exact n
            if len(result) > n:
                keys = random.sample(list(result.keys()), n)
                result = {k: result[k] for k in keys}
            
            return result
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def download_node_prices(self, node_id: str, start_date: str, end_date: str,
                            market: str = 'DAM') -> Optional[pd.DataFrame]:
        """Download prices for a single node"""
        try:
            df = CAISOPriceFetcher.fetch_lmp_data(node_id, start_date, end_date, market)
            
            if df is not None and len(df) > 0:
                # Add node info
                node_info = self.ca_nodes.get(node_id, {})
                df['latitude'] = node_info.get('latitude', 0)
                df['longitude'] = node_info.get('longitude', 0)
                df['area'] = node_info.get('area', '')
                df['node_type'] = node_info.get('node_type', '')
                
                return df
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to download {node_id}: {e}")
            return None
    
    def download_all_prices(self, nodes: Dict[str, dict], 
                           start_date: str, end_date: str,
                           market: str = 'DAM',
                           rate_limit_delay: float = 2.0,
                           checkpoint_interval: int = 10) -> pd.DataFrame:
        """
        Download prices for multiple nodes
        
        Args:
            nodes: Dictionary of node_id -> node_info
            start_date: Start date 'YYYYMMDD'
            end_date: End date 'YYYYMMDD'
            market: 'DAM' or 'RTM'
            rate_limit_delay: Seconds between requests
            checkpoint_interval: Save checkpoint every N nodes
        
        Returns:
            Combined DataFrame with all prices
        """
        all_data = []
        total = len(nodes)
        remaining = [nid for nid in nodes.keys() if nid not in self.completed_nodes]
        
        logger.info(f"Downloading prices for {len(remaining)}/{total} nodes")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Market: {market}")
        
        # Track statistics
        nodes_with_data = 0
        nodes_no_data = 0
        nodes_failed = 0
        
        # Track timing
        download_start_time = time.time()
        node_times = []  # List of times for each node download
        
        for i, node_id in enumerate(remaining):
            node_start_time = time.time()
            
            # Calculate timing info
            elapsed_total = time.time() - download_start_time
            if node_times:
                avg_time_per_node = sum(node_times) / len(node_times)
                remaining_nodes = len(remaining) - i
                eta_seconds = avg_time_per_node * remaining_nodes
                eta_str = self._format_duration(eta_seconds)
                elapsed_str = self._format_duration(elapsed_total)
                timing_info = f"[Elapsed: {elapsed_str} | ETA: {eta_str} | Avg: {avg_time_per_node:.1f}s/node]"
            else:
                timing_info = "[Starting...]"
            
            logger.info(f"{timing_info}")
            logger.info(f"[{i+1}/{len(remaining)}] Downloading {node_id}...")
            
            try:
                df = self.download_node_prices(node_id, start_date, end_date, market)
                
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    nodes_with_data += 1
                    node_elapsed = time.time() - node_start_time
                    logger.info(f"  ✓ {len(df)} records ({node_elapsed:.1f}s)")
                else:
                    nodes_no_data += 1
                    node_elapsed = time.time() - node_start_time
                    logger.warning(f"  ✗ No data for {node_id} ({node_elapsed:.1f}s) - node may not have data for this period")
            except Exception as e:
                nodes_failed += 1
                node_elapsed = time.time() - node_start_time
                logger.error(f"  ✗ Failed {node_id} ({node_elapsed:.1f}s): {e}")
            
            # Track node time (excluding rate limit delay)
            node_times.append(time.time() - node_start_time)
            
            self.completed_nodes.add(node_id)
            
            # Checkpoint
            if (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint()
                logger.info(f"  Checkpoint saved ({len(self.completed_nodes)} nodes)")
            
            # Rate limiting
            time.sleep(rate_limit_delay)
        
        # Final checkpoint
        self._save_checkpoint()
        
        # Calculate total time
        total_download_time = time.time() - download_start_time
        avg_time = sum(node_times) / len(node_times) if node_times else 0
        
        # Log statistics
        logger.info("")
        logger.info("=" * 60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Total time:         {self._format_duration(total_download_time)}")
        logger.info(f"  Average per node:   {avg_time:.1f}s")
        logger.info(f"  Nodes with data:    {nodes_with_data}")
        logger.info(f"  Nodes without data: {nodes_no_data} (normal - some nodes have limited history)")
        logger.info(f"  Nodes failed:       {nodes_failed}")
        logger.info(f"  Success rate:       {nodes_with_data}/{len(remaining)} ({100*nodes_with_data/len(remaining):.1f}%)")
        logger.info("=" * 60)
        logger.info("")
        
        if not all_data:
            logger.error("No data downloaded!")
            return pd.DataFrame()
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"✅ Downloaded {len(combined)} total records from {nodes_with_data} nodes")
        
        return combined
    
    def save_geo_prices(self, df: pd.DataFrame, filename: str = 'geo_prices.pkl'):
        """Save downloaded prices to pickle file"""
        output_path = os.path.join(self.output_dir, filename)
        df.to_pickle(output_path)
        logger.info(f"✅ Saved {len(df)} records to {output_path}")
        
        # Also save metadata
        metadata = {
            'download_date': datetime.now().isoformat(),
            'records': len(df),
            'nodes': df['node'].nunique() if 'node' in df.columns else 0,
            'date_range': {
                'min': str(df['timestamp'].min()) if 'timestamp' in df.columns else '',
                'max': str(df['timestamp'].max()) if 'timestamp' in df.columns else '',
            },
            'price_stats': {
                'min': float(df['lmp'].min()) if 'lmp' in df.columns else 0,
                'max': float(df['lmp'].max()) if 'lmp' in df.columns else 0,
                'mean': float(df['lmp'].mean()) if 'lmp' in df.columns else 0,
            },
            'areas': df['area'].unique().tolist() if 'area' in df.columns else [],
        }
        
        metadata_path = os.path.join(self.output_dir, 'geo_prices_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_path


def download_supporting_data(output_dir: str, start_date: datetime, end_date: datetime):
    """Download power plants and earthquake data (unchanged from before)"""
    
    # Power plants
    logger.info("Downloading power plant data...")
    try:
        plants_df = PowerPlantDB.download_plants()
        if plants_df is not None:
            plants_path = os.path.join(output_dir, 'power_plants.pkl')
            plants_df.to_pickle(plants_path)
            logger.info(f"✅ Saved {len(plants_df)} power plants")
    except Exception as e:
        logger.error(f"Failed to download power plants: {e}")
    
    # Earthquakes
    logger.info("Downloading earthquake data...")
    try:
        quakes_df = DisasterRisk.get_recent_earthquakes(start_date, end_date)
        if quakes_df is not None:
            quakes_path = os.path.join(output_dir, 'earthquakes.pkl')
            quakes_df.to_pickle(quakes_path)
            logger.info(f"✅ Saved {len(quakes_df)} earthquakes")
    except Exception as e:
        logger.error(f"Failed to download earthquakes: {e}")


def main():
    parser = argparse.ArgumentParser(description='Download geolocation-based CAISO price data')
    
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample N nodes (default: all nodes)')
    parser.add_argument('--strategy', type=str, default='stratified',
                        choices=['random', 'stratified', 'geographic'],
                        help='Sampling strategy')
    parser.add_argument('--months', type=int, default=3,
                        help='Months of historical data (default: 3)')
    parser.add_argument('--market', type=str, default='DAM',
                        choices=['DAM', 'RTM'],
                        help='Market type (default: DAM)')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Rate limit delay in seconds (default: 2.0)')
    parser.add_argument('--output-dir', type=str, 
                        default='/Users/chester.kim/workspace/tf/electricity-forecasting/tier3_poc/data/downloads',
                        help='Output directory')
    parser.add_argument('--skip-supporting', action='store_true',
                        help='Skip downloading power plants and earthquakes')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--list-areas', action='store_true',
                        help='List areas and node counts, then exit')
    
    args = parser.parse_args()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("GEOLOCATION-BASED DATA DOWNLOADER")
    logger.info("=" * 70)
    
    downloader = GeoDataDownloader(args.output_dir)
    
    # List areas mode
    if args.list_areas:
        from caiso_nodes import CAISONodes
        nodes = CAISONodes()
        stats = nodes.get_area_stats()
        print("\nAreas and node counts:")
        for area, s in sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"  {area:8s}: {s['count']:4d} nodes ({s['load_nodes']} LOAD, {s['gen_nodes']} GEN)")
        return 0
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.months * 30)
    
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    logger.info(f"Date range: {start_str} to {end_str}")
    logger.info(f"Market: {args.market}")
    
    # Select nodes
    if args.sample:
        logger.info(f"Sampling {args.sample} nodes using {args.strategy} strategy")
        nodes = downloader.sample_nodes(args.sample, args.strategy)
    else:
        nodes = downloader.ca_nodes
    
    logger.info(f"Will download data for {len(nodes)} nodes")
    
    # Clear checkpoint if not resuming
    if not args.resume:
        downloader.completed_nodes = set()
    
    # Download prices
    prices_df = downloader.download_all_prices(
        nodes, start_str, end_str,
        market=args.market,
        rate_limit_delay=args.delay
    )
    
    if len(prices_df) > 0:
        downloader.save_geo_prices(prices_df)
    
    # Download supporting data
    if not args.skip_supporting:
        download_supporting_data(args.output_dir, start_date, end_date)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("DOWNLOAD COMPLETE ✅")
    logger.info("=" * 70)
    logger.info(f"Price records: {len(prices_df)}")
    logger.info(f"Output: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

