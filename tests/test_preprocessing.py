"""
Unit Tests for Data Preprocessing and Validation.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.validation import (
    validate_coordinates,
    validate_budget_and_area,
    validate_scoring_weights,
    validate_location_level,
)
from src.preprocessing import LocationDataPreprocessor


def test_validate_coordinates():
    # Valid Indian coordinates (Bhopal, Pune)
    assert validate_coordinates(23.25, 77.41)[0] is True
    assert validate_coordinates(18.52, 73.85)[0] is True

    # Invalid coordinates (outside India)
    assert validate_coordinates(51.5074, -0.1278)[0] is False  # London
    assert validate_coordinates(-33.8688, 151.2093)[0] is False  # Sydney
    assert validate_coordinates(np.nan, 77.0)[0] is False


def test_validate_budget_and_area():
    assert validate_budget_and_area(50000, 500)[0] is True
    assert validate_budget_and_area(-1000, 500)[0] is False
    assert validate_budget_and_area(50000, -10)[0] is False
    assert validate_budget_and_area(0, 500)[0] is False


def test_validate_location_level():
    assert validate_location_level("locality")[0] is True
    assert validate_location_level("street")[0] is True
    assert validate_location_level("property")[0] is True
    assert validate_location_level("country")[0] is False


def test_preprocessing_loads_and_cleans():
    prep = LocationDataPreprocessor()
    df = prep.get_consolidated_dataset(save_output=False)

    assert not df.empty
    assert "location_id" in df.columns
    assert "avg_monthly_rent" in df.columns
    assert "data_quality_score" in df.columns

    # Verify no nulls in core coordinate fields
    assert df["latitude"].isnull().sum() == 0
    assert df["longitude"].isnull().sum() == 0

    # Verify data quality scores are within [0, 100]
    assert (df["data_quality_score"] >= 0).all()
    assert (df["data_quality_score"] <= 100).all()
