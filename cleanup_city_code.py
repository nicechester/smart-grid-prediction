#!/usr/bin/env python3
"""
cleanup_city_code.py - Remove unused city-based code and data

Run this script AFTER the geolocation-based system is working correctly.
This will remove old city-based files and simplify the codebase.

Usage:
    python cleanup_city_code.py --dry-run    # Show what would be deleted
    python cleanup_city_code.py --execute    # Actually delete files
"""

import os
import shutil
import argparse

# Base directory
BASE_DIR = '/Users/chester.kim/workspace/tf/electricity-forecasting/tier3_poc'

# Files to DELETE completely
FILES_TO_DELETE = [
    'src/city_price_nodes.py',           # City to node mapping
    'src/locations.py',                   # City/county definitions
    'data/downloads/caiso_prices.pkl',    # Old city-based prices
    'data/downloads/caiso_metadata.json', # Old metadata
    'data/models/price_model.keras',      # Old city-based model
    'data/models/price_model.h5',         # Old model (if exists)
    'data/models/scaler.pkl',             # Old scaler (replaced by geo_scaler.pkl)
    'data/models/features.json',          # Old features (replaced by geo_features.json)
    'data/models/feature_stats.json',     # Old feature stats
    'data/models/training_metadata.json', # Old training metadata
    'data/downloads/download_summary.json', # Old download summary
    'data/downloads/geo_download_checkpoint.json',  # Checkpoint (after training complete)
]

# Directories to clean (delete contents, keep dir)
DIRS_TO_CLEAN = [
    'src/cache',  # API response cache
]

# Files that are REPLACED (keep backup before deleting)
FILES_TO_BACKUP_AND_REPLACE = [
    # These will be replaced by new geo versions
    # 'src/train.py',        # Keep for reference
    # 'src/download_data.py', # Keep for reference
]

# Code sections to REMOVE from files
# Format: (file_path, start_marker, end_marker)
CODE_SECTIONS_TO_REMOVE = [
    # tier2_pipeline.py - Remove city mappings
    ('src/tier2_pipeline.py', 'CAISO_PRICE_NODES = {', '}'),
    ('src/tier2_pipeline.py', 'CAISO_REGIONAL_NODES = {', '}'),
]


def main():
    parser = argparse.ArgumentParser(description='Cleanup city-based code')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be deleted without actually deleting')
    parser.add_argument('--execute', action='store_true',
                        help='Actually perform the cleanup')
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.execute:
        print("Usage: python cleanup_city_code.py --dry-run OR --execute")
        print("\nRun with --dry-run first to see what will be deleted.")
        return 1
    
    print("=" * 70)
    print("CLEANUP CITY-BASED CODE")
    print("=" * 70)
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No files will be deleted\n")
    else:
        print("\n⚠️  EXECUTING - Files will be deleted!\n")
    
    # 1. Delete files
    print("FILES TO DELETE:")
    print("-" * 50)
    for rel_path in FILES_TO_DELETE:
        full_path = os.path.join(BASE_DIR, rel_path)
        exists = os.path.exists(full_path)
        status = "✓ exists" if exists else "✗ not found"
        print(f"  {rel_path:<50} [{status}]")
        
        if args.execute and exists:
            try:
                os.remove(full_path)
                print(f"    → Deleted")
            except Exception as e:
                print(f"    → Error: {e}")
    
    # 2. Clean directories
    print("\n\nDIRECTORIES TO CLEAN:")
    print("-" * 50)
    for rel_path in DIRS_TO_CLEAN:
        full_path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(full_path):
            files = os.listdir(full_path)
            print(f"  {rel_path:<50} [{len(files)} files]")
            
            if args.execute:
                for f in files:
                    try:
                        fp = os.path.join(full_path, f)
                        if os.path.isfile(fp):
                            os.remove(fp)
                    except Exception as e:
                        print(f"    → Error deleting {f}: {e}")
                print(f"    → Cleaned")
        else:
            print(f"  {rel_path:<50} [not found]")
    
    # 3. Summary of what to manually update
    print("\n\nMANUAL UPDATES NEEDED:")
    print("-" * 50)
    print("""
1. UPDATE app.py:
   - Remove imports from locations.py
   - Remove city-based /predict endpoint (keep only /predict/geo)
   - Remove /locations endpoint
   - Integrate geo_bp blueprint from app_geo.py

2. UPDATE tier2_pipeline.py:
   - Remove CAISO_PRICE_NODES dict (~15 lines)
   - Remove CAISO_REGIONAL_NODES dict (~5 lines)
   - Remove fetch_city_prices() method (~20 lines)
   - Remove fetch_all_cities_prices() method (~30 lines)
   - Remove imports from locations.py

3. UPDATE Dockerfile:
   - Ensure new geo files are included
   - Update entrypoint if needed

4. UPDATE docker-compose.yml:
   - Update any environment variables
   - Ensure correct model paths

5. UPDATE README.md:
   - Document new /predict/geo endpoint
   - Remove city-based endpoint documentation
""")
    
    # 4. New files created
    print("\nNEW FILES CREATED (keep these):")
    print("-" * 50)
    new_files = [
        'src/caiso_nodes.py',
        'src/geo_utils.py', 
        'src/geo_features.py',
        'src/train_geo.py',
        'src/download_geo.py',
        'src/app_geo.py',
        'data/caiso-price-map.json',
        'data/caiso_nodes_california.json',
        'data/downloads/geo_prices.pkl',
        'data/downloads/geo_training.pkl',
        'data/models/geo_model.keras',
        'data/models/geo_scaler.pkl',
        'data/models/geo_features.json',
        'data/models/geo_metadata.json',
    ]
    
    for rel_path in new_files:
        full_path = os.path.join(BASE_DIR, rel_path)
        exists = os.path.exists(full_path)
        status = "✓ exists" if exists else "✗ not found"
        print(f"  {rel_path:<50} [{status}]")
    
    print("\n" + "=" * 70)
    
    if args.dry_run:
        print("Dry run complete. Run with --execute to perform cleanup.")
    else:
        print("Cleanup complete!")
    
    return 0


if __name__ == '__main__':
    exit(main())

