"""
Feature Engineering Module for AI-Based Business Location Selector.

Calculates geodesic distance matrices, catchment radii features, competitor density,
rent affordability index, and normalized component scores for decision analysis.
"""

import logging
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geopy.distance import geodesic
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_geodesic_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes accurate geodesic surface distance in kilometers between two coordinates.
    """
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers


class LocationFeatureEngineer:
    """Computes geospatial, catchment, and financial features for candidate business locations."""

    def __init__(self, df_consolidated: Optional[pd.DataFrame] = None):
        self.df = df_consolidated

    def compute_distance_matrix(self, df_candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Computes pairwise geodesic distance matrix (in km) among candidate locations.
        """
        n = len(df_candidates)
        loc_ids = df_candidates["location_id"].tolist()
        matrix = np.zeros((n, n))

        coords = list(zip(df_candidates["latitude"], df_candidates["longitude"]))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 0.0
                else:
                    dist = calculate_geodesic_distance_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                    matrix[i][j] = dist
                    matrix[j][i] = dist

        return pd.DataFrame(matrix, index=loc_ids, columns=loc_ids)

    def calculate_rent_affordability_score(self, monthly_rent: float, user_budget: float) -> float:
        """
        Calculates a transparent Rent Affordability Score on a 0-100 scale.

        Rules:
        - If rent is well below budget (rent <= 0.6 * budget), score is near 100.
        - If rent is at budget (rent == budget), score is 70 (viable but tight).
        - If rent exceeds budget (rent > budget), score degrades rapidly towards 0.
        """
        if user_budget <= 0:
            return 0.0

        ratio = monthly_rent / user_budget

        if ratio <= 0.5:
            return 100.0
        elif ratio <= 1.0:
            # Linear scaling from 100 down to 70 as rent approaches 100% of budget
            return round(100.0 - 30.0 * ((ratio - 0.5) / 0.5), 2)
        elif ratio <= 1.3:
            # Exceeds budget by up to 30%: steep penalty from 70 down to 10
            return round(max(0.0, 70.0 - 60.0 * ((ratio - 1.0) / 0.3)), 2)
        else:
            # Over 30% over budget: severe penalty
            return 0.0

    def calculate_competition_opportunity_score(
        self, competitor_density: float, business_type: str, max_density: float = 35.0
    ) -> float:
        """
        Calculates Competition Opportunity Score (0-100).

        Context-aware:
        - For cluster-friendly businesses (e.g. Coaching, Jewellers, Auto parts, Food St),
          some existing competitors prove destination demand.
        - For dispersion businesses (e.g. Supermarket, Pharmacy, Salon),
          lower competitor density provides higher market opportunity.
        """
        cluster_friendly = [
            "Coaching & Tuition Centre",
            "Skill Training Institute",
            "Clothing Boutique",
            "Electronics & Mobile Store",
            "Fast Food / QSR",
            "Cafe",
        ]

        # Normalized density ratio [0, 1]
        norm_density = min(1.0, max(0.0, competitor_density / max(max_density, 1.0)))

        if business_type in cluster_friendly:
            # Sweet spot is moderate clustering (0.2 to 0.5 of max density)
            if norm_density < 0.15:
                # Too few competitors: untested market
                score = 60.0 + (norm_density / 0.15) * 35.0
            elif norm_density <= 0.55:
                # Prime cluster zone
                score = 95.0 - ((norm_density - 0.15) / 0.40) * 15.0
            else:
                # Saturated cluster
                score = max(20.0, 80.0 - ((norm_density - 0.55) / 0.45) * 60.0)
        else:
            # Dispersion businesses prefer lower competitor density
            score = max(10.0, 100.0 - (norm_density * 85.0))

        return round(float(score), 2)

    def calculate_normalized_components(
        self, df: pd.DataFrame, user_budget: float, business_type: str
    ) -> pd.DataFrame:
        """
        Transforms raw feature metrics into normalized 0-100 component scores:
        - D: Customer Demand Indicator (Population & Facility catchment)
        - A: Accessibility Score (Road & Corridor Connectivity)
        - F: Footfall Indicator (Pedestrian & Shopper Flow)
        - P: Parking & Public Transport Readiness
        - C: Competition Opportunity Score
        - R: Rent Affordability Score
        """
        res = df.copy()

        # D: Demand Indicator (combining population & nearby facilities)
        pop_max = max(res["population"].max(), 100000)
        fac_max = max(res["nearby_facilities_count"].max(), 120)
        norm_pop = (res["population"] / pop_max).clip(0, 1)
        norm_fac = (res["nearby_facilities_count"] / fac_max).clip(0, 1)
        res["score_demand"] = (0.55 * norm_pop + 0.45 * norm_fac) * 100.0
        res["score_demand"] = res["score_demand"].round(2)

        # A: Accessibility Score (scale 1-10 -> 0-100)
        res["score_accessibility"] = (res["accessibility_score"] / 10.0 * 100.0).clip(0, 100).round(2)

        # F: Footfall Indicator (scale 0-35,000 -> 0-100)
        footfall_max = max(res["footfall_estimate"].max(), 30000)
        res["score_footfall"] = (res["footfall_estimate"] / footfall_max * 100.0).clip(0, 100).round(2)

        # P: Parking and Public Transport Readiness (combined 50/50, scale 1-10 -> 0-100)
        res["score_parking_transport"] = (
            (0.5 * (res["parking_score"] / 10.0) + 0.5 * (res["public_transport_score"] / 10.0)) * 100.0
        ).clip(0, 100).round(2)

        # C: Competition Opportunity Score
        res["score_competition"] = res["competitor_density"].apply(
            lambda cd: self.calculate_competition_opportunity_score(cd, business_type)
        )

        # R: Rent Affordability Score
        res["score_affordability"] = res["avg_monthly_rent"].apply(
            lambda rent: self.calculate_rent_affordability_score(rent, user_budget)
        )

        return res


if __name__ == "__main__":
    from src.preprocessing import LocationDataPreprocessor

    prep = LocationDataPreprocessor()
    df_data = prep.get_consolidated_dataset(save_output=False)
    fe = LocationFeatureEngineer(df_data)

    test_scored = fe.calculate_normalized_components(df_data, user_budget=60000, business_type="Cafe")
    print("Normalized Components for Sample Cafe Analysis (Budget INR 60,000):")
    sample_cols = [
        "location_id",
        "city",
        "locality",
        "score_demand",
        "score_accessibility",
        "score_footfall",
        "score_competition",
        "score_affordability",
    ]
    print(test_scored[sample_cols].head())
