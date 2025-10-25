#!/usr/bin/env python3
"""
California locations configuration
Defines regions, counties, and cities for location-based predictions
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================
# California Grid Regions (CAISO zones)
# ============================================

CAISO_REGIONS = {
    'north': {
        'name': 'Northern California',
        'bbox': (38.0, -124.5, 42.0, -114.0),
        'center': (39.5, -121.0),
        'eia_respondent': 'CISO',
        'description': 'Bay Area, Sacramento, Redding'
    },
    'central': {
        'name': 'Central Coast',
        'bbox': (35.0, -122.0, 38.0, -119.0),
        'center': (36.5, -120.5),
        'eia_respondent': 'CISO',
        'description': 'San Luis Obispo, Fresno, Kern'
    },
    'south': {
        'name': 'Southern California',
        'bbox': (32.5, -120.0, 35.0, -114.0),
        'center': (33.5, -117.0),
        'eia_respondent': 'CISO',
        'description': 'Los Angeles, San Diego, Inland Empire'
    }
}

# ============================================
# Major California Cities (with coordinates)
# ============================================

CALIFORNIA_CITIES = {
    # Northern California
    'san_francisco': {
        'name': 'San Francisco',
        'region': 'north',
        'county': 'San Francisco',
        'lat': 37.7749,
        'lon': -122.4194,
        'population': 873965,
        'demand_profile': 'urban_tech',
        'bbox': (37.7, -122.5, 37.85, -122.35),
    },
    'oakland': {
        'name': 'Oakland',
        'region': 'north',
        'county': 'Alameda',
        'lat': 37.8044,
        'lon': -122.2712,
        'population': 433031,
        'demand_profile': 'urban_industrial',
        'bbox': (37.75, -122.35, 37.85, -122.15),
    },
    'san_jose': {
        'name': 'San Jose',
        'region': 'north',
        'county': 'Santa Clara',
        'lat': 37.3382,
        'lon': -121.8863,
        'population': 1021795,
        'demand_profile': 'urban_tech',
        'bbox': (37.2, -122.0, 37.5, -121.7),
    },
    'sacramento': {
        'name': 'Sacramento',
        'region': 'north',
        'county': 'Sacramento',
        'lat': 38.5816,
        'lon': -121.4944,
        'population': 525491,
        'demand_profile': 'urban_gov',
        'bbox': (38.4, -121.6, 38.7, -121.3),
    },
    
    # Central California
    'fresno': {
        'name': 'Fresno',
        'region': 'central',
        'county': 'Fresno',
        'lat': 36.7469,
        'lon': -119.7726,
        'population': 530093,
        'demand_profile': 'agricultural_urban',
        'bbox': (36.6, -119.9, 36.9, -119.6),
    },
    'bakersfield': {
        'name': 'Bakersfield',
        'region': 'central',
        'county': 'Kern',
        'lat': 35.3733,
        'lon': -119.0187,
        'population': 403455,
        'demand_profile': 'industrial_oil',
        'bbox': (35.2, -119.2, 35.5, -118.8),
    },
    'visalia': {
        'name': 'Visalia',
        'region': 'central',
        'county': 'Tulare',
        'lat': 36.3302,
        'lon': -119.2944,
        'population': 141384,
        'demand_profile': 'agricultural',
        'bbox': (36.2, -119.5, 36.5, -119.1),
    },
    
    # Southern California
    'los_angeles': {
        'name': 'Los Angeles',
        'region': 'south',
        'county': 'Los Angeles',
        'lat': 34.0522,
        'lon': -118.2437,
        'population': 3849297,
        'demand_profile': 'urban_sprawl_hot',
        'bbox': (33.8, -118.5, 34.2, -117.8),
    },
    'san_diego': {
        'name': 'San Diego',
        'region': 'south',
        'county': 'San Diego',
        'lat': 32.7157,
        'lon': -117.1611,
        'population': 1386932,
        'demand_profile': 'urban_coastal',
        'bbox': (32.6, -117.3, 32.85, -117.0),
    },
    'riverside': {
        'name': 'Riverside',
        'region': 'south',
        'county': 'Riverside',
        'lat': 33.9425,
        'lon': -117.2808,
        'population': 314998,
        'demand_profile': 'suburban_hot',
        'bbox': (33.8, -117.5, 34.1, -117.0),
    },
    'santa_ana': {
        'name': 'Santa Ana',
        'region': 'south',
        'county': 'Orange',
        'lat': 33.7456,
        'lon': -117.8678,
        'population': 310127,
        'demand_profile': 'urban_industrial',
        'bbox': (33.6, -117.95, 33.85, -117.75),
    },
    'anaheim': {
        'name': 'Anaheim',
        'region': 'south',
        'county': 'Orange',
        'lat': 33.8354,
        'lon': -117.9126,
        'population': 346824,
        'demand_profile': 'urban_tourist',
        'bbox': (33.7, -118.0, 33.95, -117.8),
    },
}

# ============================================
# California Counties (major ones)
# ============================================

CALIFORNIA_COUNTIES = {
    'los_angeles': {
        'name': 'Los Angeles',
        'region': 'south',
        'population': 9863589,
        'major_cities': ['los_angeles', 'santa_ana', 'anaheim'],
        'center': (34.1, -118.2),
    },
    'san_diego': {
        'name': 'San Diego',
        'region': 'south',
        'population': 3343364,
        'major_cities': ['san_diego'],
        'center': (32.8, -117.2),
    },
    'orange': {
        'name': 'Orange',
        'region': 'south',
        'population': 3186989,
        'major_cities': ['santa_ana', 'anaheim'],
        'center': (33.7, -117.8),
    },
    'riverside': {
        'name': 'Riverside',
        'region': 'south',
        'population': 2418185,
        'major_cities': ['riverside'],
        'center': (33.9, -117.3),
    },
    'kern': {
        'name': 'Kern',
        'region': 'central',
        'population': 909235,
        'major_cities': ['bakersfield'],
        'center': (35.4, -119.0),
    },
    'fresno': {
        'name': 'Fresno',
        'region': 'central',
        'population': 1012552,
        'major_cities': ['fresno'],
        'center': (36.7, -119.8),
    },
    'santa_clara': {
        'name': 'Santa Clara',
        'region': 'north',
        'population': 1927525,
        'major_cities': ['san_jose'],
        'center': (37.3, -121.9),
    },
    'alameda': {
        'name': 'Alameda',
        'region': 'north',
        'population': 1666753,
        'major_cities': ['oakland'],
        'center': (37.8, -122.2),
    },
    'san_francisco': {
        'name': 'San Francisco',
        'region': 'north',
        'population': 883305,
        'major_cities': ['san_francisco'],
        'center': (37.77, -122.42),
    },
    'sacramento': {
        'name': 'Sacramento',
        'region': 'north',
        'population': 525514,
        'major_cities': ['sacramento'],
        'center': (38.5, -121.5),
    },
}

# ============================================
# Demand Profiles (affects predictions)
# ============================================

DEMAND_PROFILES = {
    'urban_tech': {
        'description': 'Tech hub with high peak demand',
        'peak_hours': [9, 18, 20],
        'seasonal_factor': 1.1,
        'ac_sensitivity': 0.5,  # Medium AC sensitivity
    },
    'urban_sprawl_hot': {
        'description': 'Large sprawling city, hot summers',
        'peak_hours': [15, 18, 21],
        'seasonal_factor': 1.3,
        'ac_sensitivity': 0.9,  # High AC sensitivity
    },
    'urban_coastal': {
        'description': 'Coastal city, mild weather',
        'peak_hours': [9, 18],
        'seasonal_factor': 0.9,
        'ac_sensitivity': 0.3,  # Low AC sensitivity
    },
    'agricultural': {
        'description': 'Agricultural area',
        'peak_hours': [6, 18],
        'seasonal_factor': 1.2,
        'ac_sensitivity': 0.7,
    },
    'agricultural_urban': {
        'description': 'Mixed agricultural and urban',
        'peak_hours': [9, 17],
        'seasonal_factor': 1.15,
        'ac_sensitivity': 0.8,
    },
    'industrial_oil': {
        'description': 'Industrial with oil/gas operations',
        'peak_hours': [7, 14],
        'seasonal_factor': 1.0,
        'ac_sensitivity': 0.4,
    },
    'urban_industrial': {
        'description': 'Urban with industrial',
        'peak_hours': [7, 18],
        'seasonal_factor': 1.05,
        'ac_sensitivity': 0.6,
    },
    'urban_gov': {
        'description': 'Government/administrative center',
        'peak_hours': [9, 17],
        'seasonal_factor': 1.0,
        'ac_sensitivity': 0.5,
    },
    'suburban_hot': {
        'description': 'Suburban area, hot summers',
        'peak_hours': [16, 20],
        'seasonal_factor': 1.25,
        'ac_sensitivity': 0.85,
    },
    'urban_tourist': {
        'description': 'Tourist destination',
        'peak_hours': [10, 18, 22],
        'seasonal_factor': 1.2,
        'ac_sensitivity': 0.7,
    },
}

# ============================================
# Location Helper Functions
# ============================================

def get_region(region_id: str) -> Optional[Dict]:
    """Get region by ID"""
    return CAISO_REGIONS.get(region_id)

def get_city(city_id: str) -> Optional[Dict]:
    """Get city by ID"""
    return CALIFORNIA_CITIES.get(city_id)

def get_county(county_id: str) -> Optional[Dict]:
    """Get county by ID"""
    return CALIFORNIA_COUNTIES.get(county_id)

def get_cities_in_region(region_id: str) -> List[Tuple[str, Dict]]:
    """Get all cities in a region"""
    return [(cid, city) for cid, city in CALIFORNIA_CITIES.items() 
            if city['region'] == region_id]

def get_cities_in_county(county_id: str) -> List[Tuple[str, Dict]]:
    """Get all cities in a county"""
    county = get_county(county_id)
    if not county:
        return []
    city_ids = county.get('major_cities', [])
    return [(cid, get_city(cid)) for cid in city_ids]

def get_nearby_cities(lat: float, lon: float, radius_km: float = 50) -> List[Tuple[str, Dict, float]]:
    """Find cities near a coordinate"""
    from math import radians, cos, sin, asin, sqrt
    
    def haversine(lon1, lat1, lon2, lat2):
        """Calculate distance in km"""
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return 6371 * c
    
    nearby = []
    for city_id, city in CALIFORNIA_CITIES.items():
        distance = haversine(lon, lat, city['lon'], city['lat'])
        if distance <= radius_km:
            nearby.append((city_id, city, distance))
    
    return sorted(nearby, key=lambda x: x[2])

def list_all_regions() -> List[Tuple[str, str]]:
    """List all available regions"""
    return [(rid, r['name']) for rid, r in CAISO_REGIONS.items()]

def list_all_cities() -> List[Tuple[str, str]]:
    """List all available cities"""
    return [(cid, c['name']) for cid, c in CALIFORNIA_CITIES.items()]

def list_all_counties() -> List[Tuple[str, str]]:
    """List all available counties"""
    return [(cid, c['name']) for cid, c in CALIFORNIA_COUNTIES.items()]

# ============================================
# Usage Example
# ============================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("California Regions:", list_all_regions())
    print("\nLos Angeles city:", get_city('los_angeles'))
    print("\nCities in Southern California:", 
          [(cid, c['name']) for cid, c in get_cities_in_region('south')])
    print("\nCities in LA County:", 
          [(cid, c['name']) for cid, c in get_cities_in_county('los_angeles')])
    print("\nNearby cities to (34.05, -118.24):", 
          [(cid, c['name'], f"{dist:.1f}km") for cid, c, dist in get_nearby_cities(34.05, -118.24, 50)])