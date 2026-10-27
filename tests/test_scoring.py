"""
Unit Tests for Feature Engineering and Multi-Criteria Suitability Scoring.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.validation import validate_scoring_weights
from src.feature_engineering import LocationFeatureEngineer
from src.suitability_scoring import LocationSuitabilityScorer
from src.preprocessing import LocationDataPreprocessor


def test_validate_scoring_weights():
    valid_weights = {
        "demand": 0.25,
        "accessibility": 0.20,
        "footfall": 0.15,
        "parking_transport": 0.15,
        "competition": 0.10,
        "affordability": 0.15,
    }
    assert validate_scoring_weights(valid_weights)[0] is True

    # Invalid: does not sum to 1.0
    invalid_weights = {"demand": 0.5, "accessibility": 0.1}
    assert validate_scoring_weights(invalid_weights)[0] is False

    # Invalid: negative weight
    negative_weight = {"demand": -0.1, "accessibility": 1.1}
    assert validate_scoring_weights(negative_weight)[0] is False


def test_rent_affordability_scoring():
    fe = LocationFeatureEngineer()

    # Rent well below budget -> high score (near 100)
    score_low_rent = fe.calculate_rent_affordability_score(monthly_rent=20000, user_budget=50000)
    assert score_low_rent == 100.0

    # Rent equal to budget -> 70.0
    score_exact_budget = fe.calculate_rent_affordability_score(monthly_rent=50000, user_budget=50000)
    assert score_exact_budget == 70.0

    # Rent way over budget -> 0.0
    score_over_budget = fe.calculate_rent_affordability_score(monthly_rent=90000, user_budget=50000)
    assert score_over_budget == 0.0


def test_end_to_end_suitability_score_bounds():
    prep = LocationDataPreprocessor()
    df_raw = prep.get_consolidated_dataset(save_output=False)

    fe = LocationFeatureEngineer(df_raw)
    components = fe.calculate_normalized_components(df_raw, user_budget=50000, business_type="Cafe")

    # Verify all normalized components are strictly in [0, 100]
    for col in [
        "score_demand",
        "score_accessibility",
        "score_footfall",
        "score_parking_transport",
        "score_competition",
        "score_affordability",
    ]:
        assert (components[col] >= 0.0).all(), f"{col} has negative values"
        assert (components[col] <= 100.0).all(), f"{col} has values > 100"

    scorer = LocationSuitabilityScorer()
    scored = scorer.score_locations(components, business_type="Cafe", user_budget=50000)

    # Suitability score must be in [0, 100]
    assert (scored["suitability_score"] >= 0.0).all()
    assert (scored["suitability_score"] <= 100.0).all()
