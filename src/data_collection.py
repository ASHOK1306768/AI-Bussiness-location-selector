"""
Data Collection Module for AI-Based Business Location Selector.

Handles geospatial data collection from OpenStreetMap (Overpass API)
with disk caching, source metadata verification, and fallback handling.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RAW_OSM_DIR = os.path.join("data", "raw", "osm")
METADATA_DIR = os.path.join("data", "metadata")


class OSMDataCollector:
    """Collector for OpenStreetMap infrastructure, points of interest, and transit data."""

    def __init__(self, endpoint_url: str = DEFAULT_OVERPASS_URL, cache_dir: str = RAW_OSM_DIR):
        self.endpoint_url = os.getenv("OVERPASS_API_URL", endpoint_url)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, cache_key: str) -> str:
        """Returns the file path for a cached query response."""
        safe_key = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in cache_key)
        return os.path.join(self.cache_dir, f"{safe_key}.json")

    def query_overpass(self, query: str, cache_key: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Executes an Overpass QL query with disk caching and rate limit handling.

        Args:
            query: Valid Overpass QL string.
            cache_key: Unique identifier for caching the query result.
            force_refresh: If True, bypasses local disk cache.

        Returns:
            Parsed JSON dictionary from Overpass API.
        """
        cache_path = self._get_cache_path(cache_key)

        if not force_refresh and os.path.exists(cache_path):
            logger.info("Loading cached Overpass data from %s", cache_path)
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        logger.info("Querying live Overpass API for [%s]...", cache_key)
        headers = {"User-Agent": "AIBusinessLocationSelector/1.0 (BTech Final Year Research Project)"}

        try:
            response = requests.post(self.endpoint_url, data={"data": query}, headers=headers, timeout=25)
            response.raise_for_status()
            data = response.json()

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Polite rate-limiting between successive queries
            time.sleep(1.0)
            return data
        except requests.exceptions.RequestException as e:
            logger.warning("Overpass API request failed (%s). Checking for fallback cache...", str(e))
            if os.path.exists(cache_path):
                logger.info("Serving stale cached data for [%s] due to network error", cache_key)
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {"elements": [], "error": str(e), "status": "failed"}

    def fetch_pois_around_point(
        self, lat: float, lon: float, radius_meters: int = 1000, force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Fetches nearby amenities, shops, healthcare, and educational facilities within radius.
        """
        cache_key = f"pois_{lat:.4f}_{lon:.4f}_{radius_meters}m"
        query = f"""
        [out:json][timeout:25];
        (
          node["amenity"](around:{radius_meters},{lat},{lon});
          node["shop"](around:{radius_meters},{lat},{lon});
          node["office"](around:{radius_meters},{lat},{lon});
          node["public_transport"](around:{radius_meters},{lat},{lon});
        );
        out body;
        """
        result = self.query_overpass(query, cache_key, force_refresh)
        return result.get("elements", [])

    def fetch_city_transit_stops(
        self, city_name: str, bbox: tuple[float, float, float, float], force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Fetches bus stops, metro stations, and transport hubs within a bounding box (south, west, north, east).
        """
        cache_key = f"transit_{city_name.lower().replace(' ', '_')}"
        s, w, n, e = bbox
        query = f"""
        [out:json][timeout:25];
        (
          node["highway"="bus_stop"]({s},{w},{n},{e});
          node["railway"="station"]({s},{w},{n},{e});
          node["railway"="subway_entrance"]({s},{w},{n},{e});
        );
        out body;
        """
        result = self.query_overpass(query, cache_key, force_refresh)
        return result.get("elements", [])


def load_data_sources_register(filepath: Optional[str] = None) -> pd.DataFrame:
    """Loads and validates the data sources audit registry."""
    path = filepath or os.path.join(METADATA_DIR, "data_sources.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data sources register not found at {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %d registered data sources.", len(df))
    return df


if __name__ == "__main__":
    collector = OSMDataCollector()
    print("OSM Data Collector initialized successfully.")
    sources = load_data_sources_register()
    print("Registered Data Sources:")
    print(sources[["source_id", "dataset_name", "licence"]])
