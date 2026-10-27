"""
Validation Utilities for AI-Based Business Location Selector.

Enforces geographic boundaries for India, monetary and area ranges,
weight normalization, and business category constraints.
"""

from typing import Dict, List, Tuple
import numpy as np

# Geographic bounding box for India (approximate mainland coordinates)
INDIA_LAT_MIN = 8.0
INDIA_LAT_MAX = 37.5
INDIA_LON_MIN = 68.0
INDIA_LON_MAX = 97.5

VALID_LOCATION_LEVELS = {"locality", "street", "property"}


def validate_coordinates(lat: float, lon: float) -> Tuple[bool, str]:
    """
    Validates whether coordinates fall within valid mainland India boundaries.

    Returns:
        (is_valid, error_message)
    """
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return False, "Coordinates must be numeric."

    if np.isnan(lat) or np.isnan(lon):
        return False, "Coordinates cannot be NaN."

    if not (INDIA_LAT_MIN <= lat <= INDIA_LAT_MAX):
        return False, f"Latitude {lat:.4f} is outside India's range [{INDIA_LAT_MIN}, {INDIA_LAT_MAX}]."

    if not (INDIA_LON_MIN <= lon <= INDIA_LON_MAX):
        return False, f"Longitude {lon:.4f} is outside India's range [{INDIA_LON_MIN}, {INDIA_LON_MAX}]."

    return True, "Valid coordinates."


def validate_budget_and_area(budget: float, area_sqft: float) -> Tuple[bool, str]:
    """
    Validates positive budget and area constraints.
    """
    if budget <= 0:
        return False, "Budget must be greater than zero."
    if area_sqft <= 0:
        return False, "Area in sqft must be greater than zero."
    if budget > 100_000_000:
        return False, "Unusually high monthly budget (> ₹10 Crore)."
    return True, "Valid budget and area."


def validate_scoring_weights(weights: Dict[str, float], tolerance: float = 1e-3) -> Tuple[bool, str]:
    """
    Ensures component weights are non-negative and sum to 1.0.
    """
    for factor, w in weights.items():
        if w < 0:
            return False, f"Weight for '{factor}' cannot be negative."

    total_weight = sum(weights.values())
    if not (1.0 - tolerance <= total_weight <= 1.0 + tolerance):
        return False, f"Weights must sum to 1.0 (current sum: {total_weight:.4f})."

    return True, "Weights are valid."


def validate_location_level(level: str) -> Tuple[bool, str]:
    """
    Ensures location_level is one of 'locality', 'street', or 'property'.
    """
    if level.lower() not in VALID_LOCATION_LEVELS:
        return False, f"Invalid location_level '{level}'. Must be one of {VALID_LOCATION_LEVELS}."
    return True, "Valid location level."
