"""
Multi-Criteria Suitability Scoring Module for AI-Based Business Location Selector.

Implements configurable weighted Multi-Criteria Decision Analysis (MCDA),
budget constraints, explainability breakdown, and data quality weighting.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from src.feature_engineering import LocationFeatureEngineer
from utils.validation import validate_scoring_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LocationSuitabilityScorer:
    """Computes transparent multi-criteria suitability scores and qualitative explanations."""

    def __init__(self, categories_path: str = os.path.join("data", "processed", "business_categories.csv")):
        self.categories_path = categories_path
        self.df_categories = pd.read_csv(categories_path) if os.path.exists(categories_path) else None

    def get_category_weights(self, business_type: str) -> Dict[str, float]:
        """
        Retrieves baseline scoring weights for a specific business category.
        """
        default_weights = {
            "demand": 0.25,
            "accessibility": 0.20,
            "footfall": 0.15,
            "competition": 0.10,
            "affordability": 0.15,
            "parking_transport": 0.15,
        }

        if self.df_categories is None:
            return default_weights

        cat_row = self.df_categories[self.df_categories["business_type"] == business_type]
        if cat_row.empty:
            logger.info("Category '%s' not found. Using default weights.", business_type)
            return default_weights

        row = cat_row.iloc[0]
        # Combine parking and transport into one combined criterion for the scoring equation
        parking_transport_wt = float(row.get("parking_weight", 0.08)) + float(row.get("transport_weight", 0.07))

        weights = {
            "demand": float(row.get("demand_weight", 0.25)),
            "accessibility": float(row.get("accessibility_weight", 0.20)),
            "footfall": float(row.get("footfall_weight", 0.15)),
            "competition": float(row.get("competition_weight", 0.10)),
            "affordability": float(row.get("affordability_weight", 0.15)),
            "parking_transport": parking_transport_wt,
        }

        # Normalize to ensure sum is exactly 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 4) for k, v in weights.items()}

        return weights

    def score_locations(
        self,
        df_scored_components: pd.DataFrame,
        business_type: str,
        custom_weights: Optional[Dict[str, float]] = None,
        strict_budget: bool = True,
        user_budget: float = 50000.0,
    ) -> pd.DataFrame:
        """
        Calculates overall suitability score S in [0, 100] with factor breakdowns.

        Formula:
            S = w_D * D + w_A * A + w_F * F + w_P * P + w_C * C + w_R * R
        """
        df = df_scored_components.copy()
        weights = custom_weights or self.get_category_weights(business_type)

        # Validate weights
        is_valid, msg = validate_scoring_weights(weights)
        if not is_valid:
            logger.warning("Invalid weights: %s. Normalizing.", msg)
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        w_D = weights.get("demand", 0.25)
        w_A = weights.get("accessibility", 0.20)
        w_F = weights.get("footfall", 0.15)
        w_P = weights.get("parking_transport", 0.15)
        w_C = weights.get("competition", 0.10)
        w_R = weights.get("affordability", 0.15)

        # Calculate individual weighted contributions
        df["contrib_demand"] = (w_D * df["score_demand"]).round(2)
        df["contrib_accessibility"] = (w_A * df["score_accessibility"]).round(2)
        df["contrib_footfall"] = (w_F * df["score_footfall"]).round(2)
        df["contrib_parking_transport"] = (w_P * df["score_parking_transport"]).round(2)
        df["contrib_competition"] = (w_C * df["score_competition"]).round(2)
        df["contrib_affordability"] = (w_R * df["score_affordability"]).round(2)

        # Raw overall score
        df["raw_suitability_score"] = (
            df["contrib_demand"]
            + df["contrib_accessibility"]
            + df["contrib_footfall"]
            + df["contrib_parking_transport"]
            + df["contrib_competition"]
            + df["contrib_affordability"]
        ).round(2)

        # Data quality adjustment: locations with verified field data get full weight;
        # lower quality or heavily imputed locations face slight confidence dampening
        quality_factor = (df["data_quality_score"] / 100.0).clip(0.70, 1.0)
        df["suitability_score"] = (df["raw_suitability_score"] * quality_factor).clip(0.0, 100.0).round(2)

        # Hard budget constraint check
        if strict_budget and user_budget > 0:
            over_budget_mask = df["avg_monthly_rent"] > (user_budget * 1.15)
            # Apply severe penalty for candidates exceeding budget by over 15%
            df.loc[over_budget_mask, "suitability_score"] = (
                df.loc[over_budget_mask, "suitability_score"] * 0.4
            ).round(2)
            df["budget_status"] = np.where(
                df["avg_monthly_rent"] <= user_budget,
                "Within Budget",
                np.where(df["avg_monthly_rent"] <= user_budget * 1.15, "Slight Stretch", "Exceeds Budget"),
            )
        else:
            df["budget_status"] = "N/A"

        # Generate qualitative trade-offs and explanations
        df["advantages"] = df.apply(self._generate_advantages, axis=1)
        df["disadvantages"] = df.apply(self._generate_disadvantages, axis=1)

        return df

    def _generate_advantages(self, row: pd.Series) -> List[str]:
        """Identifies key competitive advantages for a candidate location."""
        advs = []
        if row.get("score_footfall", 0) >= 80:
            advs.append("Exceptionally high pedestrian and shopper footfall")
        if row.get("score_accessibility", 0) >= 85:
            advs.append("Superb road connectivity and arterial accessibility")
        if row.get("score_parking_transport", 0) >= 80:
            advs.append("Convenient parking and rapid public transit proximity")
        if row.get("score_demand", 0) >= 80:
            advs.append("Dense customer catchment and extensive nearby institutional facilities")
        if row.get("score_affordability", 0) >= 85:
            advs.append("Attractive rental value offering strong cost buffer")
        if row.get("score_competition", 0) >= 85:
            advs.append("Optimal competition balance and high commercial synergy")

        if not advs:
            advs.append("Balanced baseline across operational factors")
        return advs

    def _generate_disadvantages(self, row: pd.Series) -> List[str]:
        """Identifies potential risks and trade-offs for a candidate location."""
        disadvs = []
        if row.get("budget_status") == "Exceeds Budget":
            disadvs.append("Average monthly rent exceeds specified budget ceiling")
        if row.get("score_affordability", 0) < 50:
            disadvs.append("Premium rental rates requiring higher breakeven turnover")
        if row.get("score_footfall", 0) < 50:
            disadvs.append("Relatively modest pedestrian footfall; may require marketing push")
        if row.get("parking_score", 0) < 5.0:
            disadvs.append("Constrained parking availability during peak business hours")
        if row.get("competitor_density", 0) > 25.0:
            disadvs.append("High competitor saturation in immediate catchment")
        if row.get("data_quality_score", 100) < 85:
            disadvs.append("Partial data coverage; on-site reconnaissance recommended")

        if not disadvs:
            disadvs.append("No major operational flags detected")
        return disadvs


if __name__ == "__main__":
    from src.feature_engineering import LocationFeatureEngineer
    from src.preprocessing import LocationDataPreprocessor

    prep = LocationDataPreprocessor()
    df_raw = prep.get_consolidated_dataset(save_output=False)

    fe = LocationFeatureEngineer(df_raw)
    df_comp = fe.calculate_normalized_components(df_raw, user_budget=55000, business_type="Cafe")

    scorer = LocationSuitabilityScorer()
    ranked = scorer.score_locations(df_comp, business_type="Cafe", user_budget=55000)
    ranked = ranked.sort_values(by="suitability_score", ascending=False)

    print("Top 5 Recommended Locations for Cafe (Budget INR 55,000):")
    cols = [
        "location_id",
        "city",
        "locality",
        "location_level",
        "suitability_score",
        "avg_monthly_rent",
        "budget_status",
    ]
    print(ranked[cols].head(5))
