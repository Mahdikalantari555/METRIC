"""
Weather data caching system for METRIC ET processing.

Provides scene-based caching of meteorological data to avoid redundant
Open-Meteo API calls. Uses SQLite database with compression for efficient storage.
"""

import sqlite3
import json
import gzip
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class WeatherCache:
    """SQLite-based cache for weather data with scene-based organization."""

    def __init__(self, cache_dir: str = "cache", db_name: str = "weather_cache.db"):
        """
        Initialize weather cache.

        Args:
            cache_dir: Directory to store cache database
            db_name: Name of the SQLite database file
        """
        self.cache_dir = cache_dir
        self.db_path = os.path.join(cache_dir, db_name)
        self._ensure_cache_dir()
        self._init_database()

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _init_database(self) -> None:
        """Initialize database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scene_weather_cache (
                    scene_id TEXT PRIMARY KEY,
                    scene_date TEXT,
                    bbox TEXT,
                    grid_spacing REAL,
                    weather_data BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scene_date ON scene_weather_cache(scene_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_accessed_at ON scene_weather_cache(accessed_at)")

            # Set schema version
            conn.execute("""
                INSERT OR REPLACE INTO cache_metadata (key, value)
                VALUES ('schema_version', '1.0')
            """)

            conn.commit()

    def _compress_data(self, data: Dict) -> bytes:
        """Compress weather data using gzip."""
        json_str = json.dumps(data, separators=(',', ':'))
        return gzip.compress(json_str.encode('utf-8'))

    def _decompress_data(self, compressed_data: bytes) -> Dict:
        """Decompress weather data from gzip."""
        json_str = gzip.decompress(compressed_data).decode('utf-8')
        return json.loads(json_str)

    def _generate_fallback_scene_id(self, bbox: List[float], scene_date: str, grid_spacing: float) -> str:
        """Generate fallback scene ID when proper scene ID is not available."""
        bbox_str = f"{bbox[0]:.6f}_{bbox[1]:.6f}_{bbox[2]:.6f}_{bbox[3]:.6f}"
        bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()[:8]
        return f"bbox_{bbox_hash}_{scene_date}_{grid_spacing:.1f}"

    def save_scene_weather(self, scene_id: str, scene_date: str, weather_data: Dict,
                          bbox: List[float], grid_spacing: float) -> bool:
        """
        Save weather data for a scene to cache.

        Args:
            scene_id: Scene identifier
            scene_date: Scene acquisition date (YYYY-MM-DD)
            weather_data: Weather data dictionary with grid_points and weather_variables
            bbox: Scene bounding box [min_lon, min_lat, max_lon, max_lat]
            grid_spacing: Grid spacing used for weather points

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            compressed_data = self._compress_data(weather_data)
            bbox_json = json.dumps(bbox)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO scene_weather_cache
                    (scene_id, scene_date, bbox, grid_spacing, weather_data, accessed_at, access_count)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 0)
                """, (scene_id, scene_date, bbox_json, grid_spacing, compressed_data))

                conn.commit()

            logger.debug(f"Cached weather data for scene {scene_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to cache weather data for scene {scene_id}: {e}")
            return False

    def load_scene_weather(self, scene_id: str) -> Optional[Dict]:
        """
        Load weather data for a scene from cache.

        Args:
            scene_id: Scene identifier

        Returns:
            Weather data dictionary or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("""
                    SELECT weather_data, access_count FROM scene_weather_cache
                    WHERE scene_id = ?
                """, (scene_id,)).fetchone()

                if row:
                    compressed_data, access_count = row
                    weather_data = self._decompress_data(compressed_data)

                    # Update access statistics
                    conn.execute("""
                        UPDATE scene_weather_cache
                        SET accessed_at = CURRENT_TIMESTAMP, access_count = ?
                        WHERE scene_id = ?
                    """, (access_count + 1, scene_id))

                    conn.commit()

                    logger.debug(f"Cache hit for scene {scene_id} (accessed {access_count + 1} times)")
                    return weather_data

        except Exception as e:
            logger.warning(f"Failed to load cached weather data for scene {scene_id}: {e}")

        return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total scenes cached
                total_scenes = conn.execute("SELECT COUNT(*) FROM scene_weather_cache").fetchone()[0]

                # Total size
                total_size = conn.execute("SELECT SUM(LENGTH(weather_data)) FROM scene_weather_cache").fetchone()[0] or 0

                # Access statistics
                access_stats = conn.execute("""
                    SELECT
                        COUNT(*) as total_accesses,
                        AVG(access_count) as avg_access_count,
                        MAX(access_count) as max_access_count
                    FROM scene_weather_cache
                """).fetchone()

                # Recent activity (last 30 days)
                recent_scenes = conn.execute("""
                    SELECT COUNT(*) FROM scene_weather_cache
                    WHERE accessed_at > datetime('now', '-30 days')
                """).fetchone()[0]

                return {
                    "total_scenes": total_scenes,
                    "total_size_bytes": total_size,
                    "total_size_mb": total_size / (1024 * 1024),
                    "total_accesses": access_stats[0] or 0,
                    "avg_access_count": access_stats[1] or 0,
                    "max_access_count": access_stats[2] or 0,
                    "recent_scenes_30d": recent_scenes,
                    "cache_hit_ratio": access_stats[1] if access_stats[1] else 0
                }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear cache entries.

        Args:
            older_than_days: If specified, only clear entries older than this many days

        Returns:
            Number of entries cleared
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if older_than_days:
                    cursor = conn.execute("""
                        DELETE FROM scene_weather_cache
                        WHERE accessed_at < datetime('now', '-{} days')
                    """.format(older_than_days))
                else:
                    cursor = conn.execute("DELETE FROM scene_weather_cache")

                cleared_count = cursor.rowcount
                conn.commit()

                logger.info(f"Cleared {cleared_count} cache entries")
                return cleared_count

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0

    def cleanup_cache(self, max_size_mb: float = 1000, max_age_days: int = 365) -> Dict[str, int]:
        """
        Clean up cache based on size and age limits.

        Args:
            max_size_mb: Maximum cache size in MB
            max_age_days: Maximum age of cache entries in days

        Returns:
            Dictionary with cleanup statistics
        """
        stats = {"cleared_by_age": 0, "cleared_by_size": 0}

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clear old entries
                cursor = conn.execute("""
                    DELETE FROM scene_weather_cache
                    WHERE accessed_at < datetime('now', '-{} days')
                """.format(max_age_days))
                stats["cleared_by_age"] = cursor.rowcount

                # Check size and clear least recently used if needed
                current_size = conn.execute("SELECT SUM(LENGTH(weather_data)) FROM scene_weather_cache").fetchone()[0] or 0
                max_size_bytes = max_size_mb * 1024 * 1024

                if current_size > max_size_bytes:
                    # Clear oldest entries until under size limit
                    size_to_clear = current_size - max_size_bytes
                    cleared_size = 0

                    # Get entries ordered by access time (oldest first)
                    rows = conn.execute("""
                        SELECT scene_id, LENGTH(weather_data) as size
                        FROM scene_weather_cache
                        ORDER BY accessed_at ASC
                    """).fetchall()

                    scenes_to_delete = []
                    for scene_id, size in rows:
                        cleared_size += size
                        scenes_to_delete.append(scene_id)
                        if cleared_size >= size_to_clear:
                            break

                    if scenes_to_delete:
                        placeholders = ','.join('?' * len(scenes_to_delete))
                        conn.execute(f"""
                            DELETE FROM scene_weather_cache
                            WHERE scene_id IN ({placeholders})
                        """, scenes_to_delete)
                        stats["cleared_by_size"] = len(scenes_to_delete)

                conn.commit()

                logger.info(f"Cache cleanup: cleared {stats['cleared_by_age']} old entries, {stats['cleared_by_size']} by size")

        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")

        return stats

    def list_cached_scenes(self) -> List[Dict[str, Any]]:
        """List all cached scenes with metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("""
                    SELECT scene_id, scene_date, bbox, grid_spacing,
                           LENGTH(weather_data) as data_size,
                           created_at, accessed_at, access_count
                    FROM scene_weather_cache
                    ORDER BY accessed_at DESC
                """).fetchall()

                scenes = []
                for row in rows:
                    scene_id, scene_date, bbox_json, grid_spacing, data_size, created_at, accessed_at, access_count = row
                    bbox = json.loads(bbox_json) if bbox_json else None

                    scenes.append({
                        "scene_id": scene_id,
                        "scene_date": scene_date,
                        "bbox": bbox,
                        "grid_spacing": grid_spacing,
                        "data_size_bytes": data_size,
                        "created_at": created_at,
                        "accessed_at": accessed_at,
                        "access_count": access_count
                    })

                return scenes

        except Exception as e:
            logger.error(f"Failed to list cached scenes: {e}")
            return []