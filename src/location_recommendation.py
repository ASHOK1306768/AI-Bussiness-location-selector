"""
Location Recommendation Engine for AI-Based Business Location Selector.

Integrates Multi-Criteria Decision Analysis (MCDA), K-Means clustering,
supervised demand estimation, budget constraints, and transparent explainability.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from src.clustering import LocationClusterer
from src.feature_engineering import LocationFeatureEngineer
from src.prediction import LocationDemandPredictor
from src.preprocessing import LocationDataPreprocessor
from src.suitability_scoring import LocationSuitabilityScorer
from utils.validation import validate_budget_and_area, validate_location_level

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LocationRecommendationEngine:
    """Orchestrates end-to-end multi-city, multi-level business location recommendations."""

    def __init__(self):
        self.preprocessor = LocationDataPreprocessor()
        self.scorer = LocationSuitabilityScorer()
        self.df_consolidated = self.preprocessor.get_consolidated_dataset(save_output=False)
        self.feature_engineer = LocationFeatureEngineer(self.df_consolidated)

        # Load or fit clustering model
        try:
            self.clusterer = LocationClusterer.load_model()
        except Exception:
            logger.info("Initializing and fitting new clusterer...")
            self.clusterer = LocationClusterer(n_clusters=4)
            self.clusterer.fit_predict(self.df_consolidated)

        # Load or initialize predictor
        try:
            self.predictor = LocationDemandPredictor.load_model()
        except Exception:
            self.predictor = None

    def get_available_cities(self) -> Dict[str, List[str]]:
        """Returns available states and their corresponding cities in the dataset."""
        res = {}
        for state, group in self.df_consolidated.groupby("state"):
            res[state] = sorted(group["city"].unique().tolist())
        return res

    def get_supported_categories(self) -> List[str]:
        """Returns list of supported business categories."""
        if self.scorer.df_categories is not None:
            return sorted(self.scorer.df_categories["business_type"].unique().tolist())
        return ["Cafe", "Grocery Supermarket", "Pharmacy / Medical Store", "Coaching & Tuition Centre"]

    def recommend(
        self,
        business_type: str,
        city: str,
        monthly_budget: float,
        required_area_sqft: float,
        state: Optional[str] = None,
        recommendation_level: str = "all",
        custom_weights: Optional[Dict[str, float]] = None,
        strict_budget: bool = True,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Executes candidate filtering, MCDA scoring, ML enrichment, and ranking.

        Returns:
            Dictionary containing:
            - 'results': Ranked candidate DataFrame
            - 'top_candidates': Top N candidates list of dicts
            - 'summary': High-level recommendation metadata
        """
        # Validate inputs
        valid_b, msg_b = validate_budget_and_area(monthly_budget, required_area_sqft)
        if not valid_b:
            raise ValueError(msg_b)

        # 1. Filter by geographic scope
        candidates = self.df_consolidated.copy()

        if city and city.lower() != "all":
            candidates = candidates[candidates["city"].str.lower() == city.lower()]

        if state and state.lower() != "all":
            candidates = candidates[candidates["state"].str.lower() == state.lower()]

        # 2. Filter by recommendation level (locality, street, property)
        if recommendation_level and recommendation_level.lower() != "all":
            candidates = candidates[candidates["location_level"].str.lower() == recommendation_level.lower()]

        if candidates.empty:
            return {
                "results": pd.DataFrame(),
                "top_candidates": [],
                "message": f"No candidate locations found matching City='{city}', Level='{recommendation_level}'.",
                "summary": {},
            }

        # 3. Calculate normalized component scores (D, A, F, P, C, R)
        scored_components = self.feature_engineer.calculate_normalized_components(
            candidates, user_budget=monthly_budget, business_type=business_type
        )

        # 4. Multi-Criteria Suitability Scoring
        scored_df = self.scorer.score_locations(
            df_scored_components=scored_components,
            business_type=business_type,
            custom_weights=custom_weights,
            strict_budget=strict_budget,
            user_budget=monthly_budget,
        )

        # 5. Enrich with Clustering Profile
        if self.clusterer and self.clusterer.model:
            try:
                X_feat = scored_df[self.clusterer.FEATURE_COLS].fillna(
                    scored_df[self.clusterer.FEATURE_COLS].median()
                )
                X_scaled = self.clusterer.scaler.transform(X_feat)
                scored_df["cluster_id"] = self.clusterer.model.predict(X_scaled)
                scored_df["cluster_label"] = scored_df["cluster_id"].map(self.clusterer.cluster_profiles)
            except Exception as e:
                logger.warning("Clustering prediction skipped: %s", str(e))
                scored_df["cluster_label"] = "Standard Commercial Zone"

        # 6. Enrich with Supervised Demand Prediction
        if self.predictor and self.predictor.best_model:
            try:
                scored_df["predicted_daily_customers"] = self.predictor.predict(scored_df)
            except Exception as e:
                logger.warning("Supervised prediction skipped: %s", str(e))
                scored_df["predicted_daily_customers"] = np.nan
        else:
            scored_df["predicted_daily_customers"] = np.nan

        # 7. Sort and extract Top N
        ranked_df = scored_df.sort_values(by="suitability_score", ascending=False).reset_index(drop=True)
        ranked_df["rank"] = ranked_df.index + 1

        top_candidates = ranked_df.head(top_n).to_dict(orient="records")

        summary = {
            "business_type": business_type,
            "city": city,
            "user_budget": monthly_budget,
            "required_area_sqft": required_area_sqft,
            "recommendation_level": recommendation_level,
            "total_candidates_analyzed": len(candidates),
            "top_candidates_returned": len(top_candidates),
        }

        return {
            "results": ranked_df,
            "top_candidates": top_candidates,
            "summary": summary,
        }


if __name__ == "__main__":
    recommender = LocationRecommendationEngine()
    print("Available Cities:", recommender.get_available_cities())

    response = recommender.recommend(
        business_type="Cafe",
        city="Bhopal",
        monthly_budget=50000,
        required_area_sqft=600,
        recommendation_level="all",
        top_n=5,
    )

    print(f"\nRecommendation Results for Cafe in Bhopal (Budget INR 50,000):")
    cols = [
        "rank",
        "locality",
        "location_level",
        "suitability_score",
        "avg_monthly_rent",
        "budget_status",
        "cluster_label",
    ]
    print(response["results"][cols].head(5))
