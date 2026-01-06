#!/usr/bin/env python3
"""
Weather Cache Management Utility

Command-line tool for managing the METRIC ET weather cache.
Provides functionality to view cache statistics, clear cache, and perform maintenance.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import metric_et modules
sys.path.insert(0, str(Path(__file__).parent))

from metric_et.io.weather_cache import WeatherCache


def format_size(bytes_size: float) -> str:
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return ".1f"
        bytes_size /= 1024.0
    return ".1f"


def print_cache_stats(cache: WeatherCache):
    """Print cache statistics."""
    stats = cache.get_cache_stats()

    print("Weather Cache Statistics")
    print("=" * 40)
    print(f"Total scenes cached: {stats.get('total_scenes', 0)}")
    print(f"Total cache size: {format_size(stats.get('total_size_bytes', 0))}")
    print(f"Total access count: {stats.get('total_accesses', 0)}")
    print(f"Average access count: {stats.get('avg_access_count', 0):.1f}")
    print(f"Max access count: {stats.get('max_access_count', 0)}")
    print(f"Recent scenes (30 days): {stats.get('recent_scenes_30d', 0)}")
    print(f"Cache hit ratio: {stats.get('cache_hit_ratio', 0):.2f}")


def list_cached_scenes(cache: WeatherCache):
    """List all cached scenes."""
    scenes = cache.list_cached_scenes()

    if not scenes:
        print("No scenes cached.")
        return

    print("Cached Scenes")
    print("=" * 80)
    print("<12")
    print("-" * 80)

    for scene in scenes:
        bbox = scene.get('bbox', [])
        bbox_str = ".3f" if len(bbox) == 4 else "N/A"
        size_str = format_size(scene.get('data_size_bytes', 0))

        print("<12")


def clear_cache(cache: WeatherCache, older_than_days: int = None):
    """Clear cache entries."""
    if older_than_days:
        confirm_msg = f"Clear cache entries older than {older_than_days} days?"
    else:
        confirm_msg = "Clear ALL cache entries?"

    response = input(f"{confirm_msg} (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        return

    cleared = cache.clear_cache(older_than_days)
    print(f"Cleared {cleared} cache entries.")


def cleanup_cache(cache: WeatherCache, max_size_mb: float, max_age_days: int):
    """Perform cache cleanup."""
    print(f"Performing cache cleanup (max size: {max_size_mb} MB, max age: {max_age_days} days)...")

    stats = cache.cleanup_cache(max_size_mb, max_age_days)
    print(f"Cleanup completed: {stats['cleared_by_age']} old entries, {stats['cleared_by_size']} by size")


def main():
    parser = argparse.ArgumentParser(description="METRIC ET Weather Cache Manager")
    parser.add_argument('--cache-dir', default='cache',
                       help='Cache directory (default: cache)')
    parser.add_argument('--stats', action='store_true',
                       help='Show cache statistics')
    parser.add_argument('--list', action='store_true',
                       help='List all cached scenes')
    parser.add_argument('--clear', action='store_true',
                       help='Clear all cache entries')
    parser.add_argument('--clear-old', type=int, metavar='DAYS',
                       help='Clear cache entries older than DAYS')
    parser.add_argument('--cleanup', action='store_true',
                       help='Perform automatic cache cleanup')
    parser.add_argument('--max-size', type=float, default=1000.0,
                       help='Maximum cache size in MB for cleanup (default: 1000)')
    parser.add_argument('--max-age', type=int, default=365,
                       help='Maximum cache age in days for cleanup (default: 365)')

    args = parser.parse_args()

    # Initialize cache
    try:
        cache = WeatherCache(args.cache_dir)
    except Exception as e:
        print(f"Error initializing cache: {e}")
        return 1

    # Execute requested operation
    try:
        if args.stats:
            print_cache_stats(cache)
        elif args.list:
            list_cached_scenes(cache)
        elif args.clear:
            clear_cache(cache)
        elif args.clear_old:
            clear_cache(cache, args.clear_old)
        elif args.cleanup:
            cleanup_cache(cache, args.max_size, args.max_age)
        else:
            parser.print_help()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())