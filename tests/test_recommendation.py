"""
Unit Tests for Location Recommendation Engine.
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.location_recommendation import LocationRecommendationEngine


def test_recommendation_engine_top_n():
    engine = LocationRecommendationEngine()

    result = engine.recommend(
        business_type="Pharmacy / Medical Store",
        city="Indore",
        monthly_budget=50000,
        required_area_sqft=400,
        recommendation_level="all",
        top_n=5,
    )

    assert "results" in result
    assert "top_candidates" in result
    assert len(result["top_candidates"]) == 5

    df = result["results"]
    assert not df.empty
    # Must be filtered to requested city
    assert (df["city"] == "Indore").all()

    # Scores must be sorted in descending order
    scores = df["suitability_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_recommendation_filtering_by_level():
    engine = LocationRecommendationEngine()

    result = engine.recommend(
        business_type="Cafe",
        city="Pune",
        monthly_budget=90000,
        required_area_sqft=800,
        recommendation_level="property",
        top_n=3,
    )

    df = result["results"]
    assert not df.empty
    # All candidates must have level == 'property'
    assert (df["location_level"] == "property").all()


def test_pan_india_metro_recommendations():
    engine = LocationRecommendationEngine()

    # Test Bengaluru
    res_blr = engine.recommend(
        business_type="Cafe",
        city="Bengaluru",
        monthly_budget=150000,
        required_area_sqft=1000,
        top_n=3,
    )
    assert not res_blr["results"].empty
    assert (res_blr["results"]["city"] == "Bengaluru").all()

    # Test Delhi
    res_del = engine.recommend(
        business_type="Retail Shop",
        city="Delhi",
        monthly_budget=120000,
        required_area_sqft=800,
        top_n=3,
    )
    assert not res_del["results"].empty
    assert (res_del["results"]["city"] == "Delhi").all()

