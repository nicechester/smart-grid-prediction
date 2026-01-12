#!/usr/bin/env python3
"""
geo_utils.py - Geospatial Utilities for Location-Based Price Prediction

Provides distance calculations, coordinate operations, and spatial interpolation
for the geolocation-based electricity price prediction model.
"""

import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Earth's radius in kilometers
EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth
    using the Haversine formula.
    
    Args:
        lat1, lon1: Coordinates of first point (in degrees)
        lat2, lon2: Coordinates of second point (in degrees)
    
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return EARTH_RADIUS_KM * c


def haversine_vectorized(lat1: float, lon1: float, 
                         lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Calculate distances from one point to multiple points (vectorized)
    
    Args:
        lat1, lon1: Reference point coordinates
        lats, lons: Arrays of target coordinates
    
    Returns:
        Array of distances in kilometers
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lats)
    delta_lat = np.radians(lats - lat1)
    delta_lon = np.radians(lons - lon1)
    
    a = (np.sin(delta_lat / 2) ** 2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * 
         np.sin(delta_lon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return EARTH_RADIUS_KM * c


def inverse_distance_weights(distances: np.ndarray, power: float = 2, 
                             min_distance: float = 0.1) -> np.ndarray:
    """
    Calculate inverse distance weights for interpolation
    
    Args:
        distances: Array of distances
        power: Power parameter (higher = more weight to nearest)
        min_distance: Minimum distance to avoid division by zero
    
    Returns:
        Normalized weights array (sums to 1)
    """
    # Avoid division by zero
    safe_distances = np.maximum(distances, min_distance)
    
    # Calculate inverse distance weights
    weights = 1.0 / (safe_distances ** power)
    
    # Normalize to sum to 1
    return weights / weights.sum()


def interpolate_value(lat: float, lon: float, 
                      points_df: pd.DataFrame,
                      value_column: str,
                      lat_column: str = 'latitude',
                      lon_column: str = 'longitude',
                      n_nearest: int = 5,
                      power: float = 2) -> float:
    """
    Interpolate a value at a location using inverse distance weighting
    
    Args:
        lat, lon: Target location
        points_df: DataFrame with source points
        value_column: Column containing values to interpolate
        lat_column, lon_column: Column names for coordinates
        n_nearest: Number of nearest points to use
        power: IDW power parameter
    
    Returns:
        Interpolated value
    """
    if len(points_df) == 0:
        return 0.0
    
    # Calculate distances to all points
    distances = haversine_vectorized(
        lat, lon,
        points_df[lat_column].values,
        points_df[lon_column].values
    )
    
    # Get n nearest points
    n_nearest = min(n_nearest, len(points_df))
    nearest_indices = np.argsort(distances)[:n_nearest]
    
    nearest_distances = distances[nearest_indices]
    nearest_values = points_df.iloc[nearest_indices][value_column].values
    
    # Handle case where point is exactly at a data point
    if nearest_distances[0] < 0.001:  # Less than 1 meter
        return float(nearest_values[0])
    
    # Calculate weights
    weights = inverse_distance_weights(nearest_distances, power)
    
    # Weighted average
    return float(np.sum(weights * nearest_values))


def interpolate_multiple(lat: float, lon: float,
                         points_df: pd.DataFrame,
                         value_columns: List[str],
                         lat_column: str = 'latitude',
                         lon_column: str = 'longitude',
                         n_nearest: int = 5,
                         power: float = 2) -> Dict[str, float]:
    """
    Interpolate multiple values at once (more efficient)
    
    Args:
        lat, lon: Target location
        points_df: DataFrame with source points
        value_columns: List of columns to interpolate
        lat_column, lon_column: Column names for coordinates
        n_nearest: Number of nearest points to use
        power: IDW power parameter
    
    Returns:
        Dictionary of column_name -> interpolated_value
    """
    if len(points_df) == 0:
        return {col: 0.0 for col in value_columns}
    
    # Calculate distances to all points
    distances = haversine_vectorized(
        lat, lon,
        points_df[lat_column].values,
        points_df[lon_column].values
    )
    
    # Get n nearest points
    n_nearest = min(n_nearest, len(points_df))
    nearest_indices = np.argsort(distances)[:n_nearest]
    
    nearest_distances = distances[nearest_indices]
    
    # Handle case where point is exactly at a data point
    if nearest_distances[0] < 0.001:
        result = {}
        for col in value_columns:
            result[col] = float(points_df.iloc[nearest_indices[0]][col])
        return result
    
    # Calculate weights once
    weights = inverse_distance_weights(nearest_distances, power)
    
    # Interpolate each value column
    result = {}
    for col in value_columns:
        values = points_df.iloc[nearest_indices][col].values
        result[col] = float(np.sum(weights * values))
    
    return result


def find_nearest_points(lat: float, lon: float,
                        points_df: pd.DataFrame,
                        n: int = 5,
                        lat_column: str = 'latitude',
                        lon_column: str = 'longitude') -> pd.DataFrame:
    """
    Find the n nearest points to a location
    
    Args:
        lat, lon: Target location
        points_df: DataFrame with points
        n: Number of points to return
        lat_column, lon_column: Column names for coordinates
    
    Returns:
        DataFrame with n nearest points, with 'distance_km' column added
    """
    distances = haversine_vectorized(
        lat, lon,
        points_df[lat_column].values,
        points_df[lon_column].values
    )
    
    n = min(n, len(points_df))
    nearest_indices = np.argsort(distances)[:n]
    
    result = points_df.iloc[nearest_indices].copy()
    result['distance_km'] = distances[nearest_indices]
    
    return result


def points_within_radius(lat: float, lon: float,
                         points_df: pd.DataFrame,
                         radius_km: float,
                         lat_column: str = 'latitude',
                         lon_column: str = 'longitude') -> pd.DataFrame:
    """
    Find all points within a radius of a location
    
    Args:
        lat, lon: Target location
        points_df: DataFrame with points
        radius_km: Search radius in kilometers
        lat_column, lon_column: Column names for coordinates
    
    Returns:
        DataFrame with points within radius, with 'distance_km' column added
    """
    distances = haversine_vectorized(
        lat, lon,
        points_df[lat_column].values,
        points_df[lon_column].values
    )
    
    mask = distances <= radius_km
    result = points_df[mask].copy()
    result['distance_km'] = distances[mask]
    
    return result.sort_values('distance_km')


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing (direction) from point 1 to point 2
    
    Args:
        lat1, lon1: Starting point
        lat2, lon2: Ending point
    
    Returns:
        Bearing in degrees (0-360, where 0 is North)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
    
    bearing = math.atan2(x, y)
    return (math.degrees(bearing) + 360) % 360


def destination_point(lat: float, lon: float, 
                      bearing: float, distance_km: float) -> Tuple[float, float]:
    """
    Calculate the destination point given start, bearing, and distance
    
    Args:
        lat, lon: Starting point
        bearing: Direction in degrees
        distance_km: Distance in kilometers
    
    Returns:
        Tuple of (latitude, longitude) of destination
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing)
    
    angular_distance = distance_km / EARTH_RADIUS_KM
    
    lat2 = math.asin(
        math.sin(lat_rad) * math.cos(angular_distance) +
        math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
    )
    
    lon2 = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
        math.cos(angular_distance) - math.sin(lat_rad) * math.sin(lat2)
    )
    
    return (math.degrees(lat2), math.degrees(lon2))


def bounding_box(lat: float, lon: float, 
                 radius_km: float) -> Tuple[float, float, float, float]:
    """
    Calculate a bounding box around a point
    
    Args:
        lat, lon: Center point
        radius_km: Radius in kilometers
    
    Returns:
        Tuple of (lat_min, lat_max, lon_min, lon_max)
    """
    # Approximate degrees per km at this latitude
    lat_delta = radius_km / 111.0  # ~111 km per degree latitude
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
    
    return (
        lat - lat_delta,
        lat + lat_delta,
        lon - lon_delta,
        lon + lon_delta
    )


def is_in_california(lat: float, lon: float) -> bool:
    """
    Check if coordinates are within California bounds
    
    Args:
        lat, lon: Coordinates to check
    
    Returns:
        True if within California bounds
    """
    return (32.5 <= lat <= 42.0 and -124.5 <= lon <= -114.0)


def estimate_elevation(lat: float, lon: float) -> float:
    """
    Rough elevation estimate based on California geography
    (For accurate elevation, use an elevation API)
    
    Args:
        lat, lon: Coordinates
    
    Returns:
        Estimated elevation in meters
    """
    # Simplified model based on California geography
    # Coastal areas (west of -121) are lower
    # Sierra Nevada (around -119 to -120, lat 36-40) is higher
    # Central Valley (around -121, lat 35-40) is low
    
    base_elevation = 100  # meters
    
    # Coastal effect (lower near coast)
    if lon < -122:
        base_elevation = 50
    elif lon < -121:
        base_elevation = 100
    
    # Sierra Nevada effect
    if -120.5 < lon < -118.5 and 36 < lat < 40:
        base_elevation = 1500 + (lat - 36) * 300
    
    # Desert areas (higher)
    if lon > -117 and lat < 35:
        base_elevation = 600
    
    return base_elevation


def distance_to_coast(lat: float, lon: float) -> float:
    """
    Estimate distance to California coast
    
    Args:
        lat, lon: Coordinates
    
    Returns:
        Approximate distance to coast in kilometers
    """
    # Simplified California coastline approximation
    # Coast runs roughly from (42, -124.4) to (32.5, -117.1)
    
    # Very rough approximation using longitude
    if lat > 38:  # Northern CA
        coast_lon = -124.0
    elif lat > 35:  # Central CA  
        coast_lon = -122.5
    else:  # Southern CA
        coast_lon = -118.5 + (35 - lat) * 0.3
    
    # Calculate distance to approximate coast longitude
    coast_distance = abs(lon - coast_lon) * 111 * math.cos(math.radians(lat))
    
    return max(0, coast_distance)


# ============================================
# Convenience functions for specific use cases
# ============================================

def interpolate_weather(lat: float, lon: float, 
                        weather_df: pd.DataFrame,
                        n_stations: int = 5) -> Dict[str, float]:
    """
    Interpolate weather data from nearby stations
    
    Args:
        lat, lon: Target location
        weather_df: DataFrame with weather station data
        n_stations: Number of nearest stations to use
    
    Returns:
        Dictionary of weather variables
    """
    weather_columns = ['temperature', 'wind_speed', 'precipitation', 
                       'temp_max', 'temp_min', 'humidity']
    
    # Filter to columns that exist
    available_columns = [col for col in weather_columns if col in weather_df.columns]
    
    if not available_columns:
        return {}
    
    return interpolate_multiple(
        lat, lon, weather_df, available_columns,
        n_nearest=n_stations
    )


def aggregate_nearby_prices(lat: float, lon: float,
                            prices_df: pd.DataFrame,
                            radius_km: float = 25) -> Dict[str, float]:
    """
    Aggregate price statistics from nearby nodes
    
    Args:
        lat, lon: Target location
        prices_df: DataFrame with node prices (must have latitude, longitude, price columns)
        radius_km: Radius for aggregation
    
    Returns:
        Dictionary with price statistics
    """
    nearby = points_within_radius(lat, lon, prices_df, radius_km)
    
    if len(nearby) == 0:
        return {
            'nearby_avg_price': 0.0,
            'nearby_std_price': 0.0,
            'nearby_min_price': 0.0,
            'nearby_max_price': 0.0,
            'nearby_node_count': 0
        }
    
    prices = nearby['price'].values if 'price' in nearby.columns else nearby['lmp'].values
    
    return {
        'nearby_avg_price': float(np.mean(prices)),
        'nearby_std_price': float(np.std(prices)),
        'nearby_min_price': float(np.min(prices)),
        'nearby_max_price': float(np.max(prices)),
        'nearby_node_count': len(nearby)
    }


# ============================================
# Testing
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("Geospatial Utilities Test")
    print("=" * 60)
    
    # Test haversine distance
    # LA to SF
    la_coords = (34.0522, -118.2437)
    sf_coords = (37.7749, -122.4194)
    
    distance = haversine_distance(*la_coords, *sf_coords)
    print(f"\nLA to SF distance: {distance:.1f} km")
    
    # Test is_in_california
    print(f"\nLA in California: {is_in_california(*la_coords)}")
    print(f"NYC in California: {is_in_california(40.7128, -74.0060)}")
    
    # Test distance to coast
    print(f"\nLA distance to coast: {distance_to_coast(*la_coords):.1f} km")
    print(f"Sacramento distance to coast: {distance_to_coast(38.5816, -121.4944):.1f} km")
    print(f"Bakersfield distance to coast: {distance_to_coast(35.3733, -119.0187):.1f} km")
    
    # Test bounding box
    bbox = bounding_box(34.0522, -118.2437, 50)
    print(f"\n50km bounding box around LA:")
    print(f"  Lat: {bbox[0]:.3f} to {bbox[1]:.3f}")
    print(f"  Lon: {bbox[2]:.3f} to {bbox[3]:.3f}")
    
    # Test vectorized distance
    test_lats = np.array([34.0, 35.0, 36.0, 37.0, 38.0])
    test_lons = np.array([-118.0, -119.0, -120.0, -121.0, -122.0])
    
    distances = haversine_vectorized(34.0522, -118.2437, test_lats, test_lons)
    print(f"\nDistances from LA to test points:")
    for i, d in enumerate(distances):
        print(f"  ({test_lats[i]}, {test_lons[i]}): {d:.1f} km")

