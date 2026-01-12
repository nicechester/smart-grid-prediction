#!/usr/bin/env python3
"""
caiso_nodes.py - CAISO Price Node Management for Geolocation-Based Prediction

Extracts and manages California price nodes with geographic coordinates
from the CAISO price map data.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)

# California geographic bounds
CA_BOUNDS = {
    'lat_min': 32.5,
    'lat_max': 42.0,
    'lon_min': -124.5,
    'lon_max': -114.0
}

# California-related CAISO areas
CA_AREAS = {
    'CA',      # Generic California
    'LADWP',   # Los Angeles DWP
    'BANC',    # Balancing Authority of Northern California
    'IID',     # Imperial Irrigation District
    'TIDC',    # Turlock Irrigation District
    'SMUD',    # Sacramento Municipal Utility District
    'NCPA',    # Northern California Power Agency
    'MID',     # Modesto Irrigation District
    'TID',     # Turlock Irrigation District
    'WAPA',    # Western Area Power Administration (partial CA)
}


class CAISONodes:
    """Manages CAISO price nodes with geographic coordinates"""
    
    def __init__(self, price_map_path: str = None):
        """
        Initialize with path to caiso-price-map.json
        
        Args:
            price_map_path: Path to the CAISO price map JSON file
        """
        if price_map_path is None:
            # Default path relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            price_map_path = os.path.join(base_dir, 'data', 'caiso-price-map.json')
        
        self.price_map_path = price_map_path
        self._nodes_cache = None
        self._ca_nodes_cache = None
    
    def load_all_nodes(self) -> Dict[str, dict]:
        """Load all nodes from the price map file"""
        if self._nodes_cache is not None:
            return self._nodes_cache
        
        logger.info(f"Loading nodes from {self.price_map_path}...")
        
        with open(self.price_map_path, 'r') as f:
            data = json.load(f)
        
        nodes = {}
        
        # Extract nodes from all layers
        for layer in data.get('l', []):
            for marker in layer.get('m', []):
                if marker.get('t') != 'Node':
                    continue
                
                node_id = marker.get('n')
                coords = marker.get('c', [])
                
                if not node_id or len(coords) != 2:
                    continue
                
                lat, lon = coords
                
                nodes[node_id] = {
                    'node_id': node_id,
                    'latitude': lat,
                    'longitude': lon,
                    'node_type': marker.get('p'),  # LOAD or GEN
                    'area': marker.get('a'),
                    # Price data (snapshot from file)
                    'day_ahead_price': self._safe_float(marker.get('dp')),
                    'day_ahead_congestion': self._safe_float(marker.get('dc')),
                    'day_ahead_loss': self._safe_float(marker.get('dl')),
                    'rt_15min_price': self._safe_float(marker.get('qp')),
                    'rt_5min_price': self._safe_float(marker.get('fp')),
                }
        
        self._nodes_cache = nodes
        logger.info(f"✅ Loaded {len(nodes)} total nodes")
        return nodes
    
    def get_california_nodes(self, include_by_coords: bool = True, 
                              include_by_area: bool = True) -> Dict[str, dict]:
        """
        Get only California-related nodes
        
        Args:
            include_by_coords: Include nodes within CA geographic bounds
            include_by_area: Include nodes with CA-related area codes
        
        Returns:
            Dictionary of node_id -> node_info for California nodes
        """
        if self._ca_nodes_cache is not None:
            return self._ca_nodes_cache
        
        all_nodes = self.load_all_nodes()
        ca_nodes = {}
        
        for node_id, node_info in all_nodes.items():
            lat = node_info['latitude']
            lon = node_info['longitude']
            area = node_info['area']
            
            # Check if in California bounds
            in_bounds = (
                include_by_coords and
                CA_BOUNDS['lat_min'] <= lat <= CA_BOUNDS['lat_max'] and
                CA_BOUNDS['lon_min'] <= lon <= CA_BOUNDS['lon_max']
            )
            
            # Check if California area
            in_ca_area = include_by_area and area in CA_AREAS
            
            if in_bounds or in_ca_area:
                ca_nodes[node_id] = node_info
        
        self._ca_nodes_cache = ca_nodes
        logger.info(f"✅ Found {len(ca_nodes)} California nodes")
        return ca_nodes
    
    def find_nearest_node(self, lat: float, lon: float, 
                          node_type: str = None) -> Tuple[str, dict, float]:
        """
        Find the nearest CAISO node to a given coordinate
        
        Args:
            lat: Latitude
            lon: Longitude
            node_type: Optional filter by 'LOAD' or 'GEN'
        
        Returns:
            Tuple of (node_id, node_info, distance_km)
        """
        from geo_utils import haversine_distance
        
        ca_nodes = self.get_california_nodes()
        
        nearest_id = None
        nearest_info = None
        min_distance = float('inf')
        
        for node_id, node_info in ca_nodes.items():
            if node_type and node_info['node_type'] != node_type:
                continue
            
            distance = haversine_distance(
                lat, lon,
                node_info['latitude'], node_info['longitude']
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_id = node_id
                nearest_info = node_info
        
        return nearest_id, nearest_info, min_distance
    
    def find_nodes_within_radius(self, lat: float, lon: float, 
                                  radius_km: float,
                                  node_type: str = None) -> List[Tuple[str, dict, float]]:
        """
        Find all nodes within a radius of a coordinate
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers
            node_type: Optional filter by 'LOAD' or 'GEN'
        
        Returns:
            List of (node_id, node_info, distance_km) tuples, sorted by distance
        """
        from geo_utils import haversine_distance
        
        ca_nodes = self.get_california_nodes()
        results = []
        
        for node_id, node_info in ca_nodes.items():
            if node_type and node_info['node_type'] != node_type:
                continue
            
            distance = haversine_distance(
                lat, lon,
                node_info['latitude'], node_info['longitude']
            )
            
            if distance <= radius_km:
                results.append((node_id, node_info, distance))
        
        # Sort by distance
        results.sort(key=lambda x: x[2])
        return results
    
    def get_nodes_by_area(self, area: str) -> Dict[str, dict]:
        """Get all nodes in a specific area"""
        ca_nodes = self.get_california_nodes()
        return {
            node_id: info 
            for node_id, info in ca_nodes.items() 
            if info['area'] == area
        }
    
    def get_node(self, node_id: str) -> Optional[dict]:
        """Get info for a specific node"""
        all_nodes = self.load_all_nodes()
        return all_nodes.get(node_id)
    
    def export_california_nodes(self, output_path: str):
        """Export California nodes to a JSON file for faster loading"""
        ca_nodes = self.get_california_nodes()
        
        with open(output_path, 'w') as f:
            json.dump(ca_nodes, f, indent=2)
        
        logger.info(f"✅ Exported {len(ca_nodes)} California nodes to {output_path}")
    
    def get_area_stats(self) -> Dict[str, dict]:
        """Get statistics by area"""
        ca_nodes = self.get_california_nodes()
        
        stats = {}
        for node_info in ca_nodes.values():
            area = node_info['area']
            if area not in stats:
                stats[area] = {
                    'count': 0,
                    'load_nodes': 0,
                    'gen_nodes': 0,
                    'lat_min': float('inf'),
                    'lat_max': float('-inf'),
                    'lon_min': float('inf'),
                    'lon_max': float('-inf'),
                }
            
            stats[area]['count'] += 1
            if node_info['node_type'] == 'LOAD':
                stats[area]['load_nodes'] += 1
            elif node_info['node_type'] == 'GEN':
                stats[area]['gen_nodes'] += 1
            
            stats[area]['lat_min'] = min(stats[area]['lat_min'], node_info['latitude'])
            stats[area]['lat_max'] = max(stats[area]['lat_max'], node_info['latitude'])
            stats[area]['lon_min'] = min(stats[area]['lon_min'], node_info['longitude'])
            stats[area]['lon_max'] = max(stats[area]['lon_max'], node_info['longitude'])
        
        return stats
    
    @staticmethod
    def _safe_float(value) -> float:
        """Safely convert value to float"""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0


# Singleton instance for easy access
_nodes_instance = None

def get_caiso_nodes() -> CAISONodes:
    """Get singleton instance of CAISONodes"""
    global _nodes_instance
    if _nodes_instance is None:
        _nodes_instance = CAISONodes()
    return _nodes_instance


def get_california_nodes() -> Dict[str, dict]:
    """Convenience function to get California nodes"""
    return get_caiso_nodes().get_california_nodes()


def find_nearest_node(lat: float, lon: float, node_type: str = None) -> Tuple[str, dict, float]:
    """Convenience function to find nearest node"""
    return get_caiso_nodes().find_nearest_node(lat, lon, node_type)


# ============================================
# CLI for testing and node extraction
# ============================================

if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    parser = argparse.ArgumentParser(description='CAISO Nodes Management')
    parser.add_argument('--export', type=str, help='Export CA nodes to JSON file')
    parser.add_argument('--stats', action='store_true', help='Show area statistics')
    parser.add_argument('--find', type=str, help='Find node by ID')
    parser.add_argument('--nearest', type=str, help='Find nearest node to lat,lon (e.g., "34.05,-118.24")')
    
    args = parser.parse_args()
    
    nodes = CAISONodes()
    
    if args.export:
        nodes.export_california_nodes(args.export)
    
    elif args.stats:
        print("\n" + "=" * 60)
        print("CAISO California Node Statistics")
        print("=" * 60)
        
        ca_nodes = nodes.get_california_nodes()
        print(f"\nTotal California nodes: {len(ca_nodes)}")
        
        stats = nodes.get_area_stats()
        print(f"\nBy Area:")
        for area, area_stats in sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"  {area:8s}: {area_stats['count']:4d} nodes "
                  f"({area_stats['load_nodes']} LOAD, {area_stats['gen_nodes']} GEN)")
    
    elif args.find:
        node = nodes.get_node(args.find)
        if node:
            print(f"\nNode: {args.find}")
            for key, value in node.items():
                print(f"  {key}: {value}")
        else:
            print(f"Node not found: {args.find}")
    
    elif args.nearest:
        lat, lon = map(float, args.nearest.split(','))
        node_id, node_info, distance = nodes.find_nearest_node(lat, lon)
        print(f"\nNearest node to ({lat}, {lon}):")
        print(f"  Node ID: {node_id}")
        print(f"  Distance: {distance:.2f} km")
        print(f"  Type: {node_info['node_type']}")
        print(f"  Area: {node_info['area']}")
        print(f"  Coordinates: ({node_info['latitude']}, {node_info['longitude']})")
    
    else:
        # Default: show summary
        print("\n" + "=" * 60)
        print("CAISO Price Nodes - Geolocation Database")
        print("=" * 60)
        
        all_nodes = nodes.load_all_nodes()
        ca_nodes = nodes.get_california_nodes()
        
        print(f"\nTotal nodes in file: {len(all_nodes)}")
        print(f"California nodes: {len(ca_nodes)}")
        
        # Count by type
        load_count = sum(1 for n in ca_nodes.values() if n['node_type'] == 'LOAD')
        gen_count = sum(1 for n in ca_nodes.values() if n['node_type'] == 'GEN')
        print(f"\nCalifornia node types:")
        print(f"  LOAD nodes: {load_count}")
        print(f"  GEN nodes: {gen_count}")
        
        print("\nUse --help for more options")

