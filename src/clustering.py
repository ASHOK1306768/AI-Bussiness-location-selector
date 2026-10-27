"""
Location Clustering Module for AI-Based Business Location Selector.

Applies K-Means clustering with feature standardization, Elbow method,
Silhouette evaluation, automated cluster profiling, and Joblib model persistence.
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


class LocationClusterer:
    """Standardizes location indicators, trains K-Means, and profiles commercial zones."""

    FEATURE_COLS = [
        "population",
        "competitor_density",
        "footfall_estimate",
        "accessibility_score",
        "parking_score",
        "public_transport_score",
        "avg_monthly_rent",
    ]

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: Optional[KMeans] = None
        self.cluster_profiles: Dict[int, str] = {}

    def evaluate_cluster_range(
        self, df: pd.DataFrame, k_range: range = range(2, 8)
    ) -> Dict[str, List[float]]:
        """
        Computes inertia (Elbow method) and silhouette scores across candidate k values.
        """
        X = df[self.FEATURE_COLS].copy().fillna(df[self.FEATURE_COLS].median())
        X_scaled = self.scaler.fit_transform(X)

        inertias = []
        silhouettes = []
        k_values = list(k_range)

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_scaled, labels))

        return {"k": k_values, "inertia": inertias, "silhouette": silhouettes}

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits K-Means on the standardized location feature vectors and assigns cluster labels.
        """
        res = df.copy()
        X = res[self.FEATURE_COLS].copy().fillna(res[self.FEATURE_COLS].median())
        X_scaled = self.scaler.fit_transform(X)

        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        res["cluster_id"] = self.model.fit_predict(X_scaled)

        # Profile and label each cluster based on centroid attributes
        self._generate_cluster_profiles(res)
        res["cluster_label"] = res["cluster_id"].map(self.cluster_profiles)

        sil = silhouette_score(X_scaled, res["cluster_id"])
        logger.info(
            "Trained K-Means (k=%d). Silhouette Score: %.3f, Inertia: %.2f",
            self.n_clusters,
            sil,
            self.model.inertia_,
        )
        return res

    def _generate_cluster_profiles(self, df_with_clusters: pd.DataFrame):
        """
        Derives meaningful domain characterizations for each cluster based on average values.
        """
        summary = df_with_clusters.groupby("cluster_id")[self.FEATURE_COLS].mean()

        for c_id, row in summary.iterrows():
            footfall = row["footfall_estimate"]
            rent = row["avg_monthly_rent"]
            parking = row["parking_score"]
            density = row["competitor_density"]

            if footfall >= 20000 and rent >= 70000:
                label = "Prime High-Footfall Commercial Hub"
            elif footfall >= 16000 and rent < 60000:
                label = "High-Traffic Value District (Student / High Density)"
            elif rent < 45000 and parking >= 7.0:
                label = "Suburban Growth & Logistics Corridor"
            elif density <= 8.0 and rent <= 50000:
                label = "Low Competition Emerging Market"
            else:
                label = "Established Mixed Commercial Area"

            self.cluster_profiles[c_id] = f"Cluster {c_id}: {label}"

    def save_model(self, filepath: Optional[str] = None) -> str:
        """Saves the fitted scaler, KMeans model, and profiles using Joblib."""
        path = filepath or os.path.join(MODELS_DIR, "kmeans_model.joblib")
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self.FEATURE_COLS,
            "cluster_profiles": self.cluster_profiles,
            "n_clusters": self.n_clusters,
        }
        joblib.dump(payload, path)
        logger.info("Saved KMeans clusterer to %s", path)
        return path

    @classmethod
    def load_model(cls, filepath: Optional[str] = None) -> "LocationClusterer":
        """Loads a persisted cluster model from disk."""
        path = filepath or os.path.join(MODELS_DIR, "kmeans_model.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        payload = joblib.load(path)
        instance = cls(n_clusters=payload["n_clusters"])
        instance.model = payload["model"]
        instance.scaler = payload["scaler"]
        instance.cluster_profiles = payload["cluster_profiles"]
        return instance


if __name__ == "__main__":
    from src.preprocessing import LocationDataPreprocessor

    prep = LocationDataPreprocessor()
    df_locs = prep.get_consolidated_dataset(save_output=False)

    clusterer = LocationClusterer(n_clusters=4)
    evaluation = clusterer.evaluate_cluster_range(df_locs)
    print("Cluster Evaluation (k=2 to 7):")
    for k, iner, sil in zip(evaluation["k"], evaluation["inertia"], evaluation["silhouette"]):
        print(f"k={k} -> Inertia: {iner:.2f}, Silhouette Score: {sil:.3f}")

    clustered_df = clusterer.fit_predict(df_locs)
    model_path = clusterer.save_model()

    print("\nCluster Profiles Identified:")
    for cid, name in clusterer.cluster_profiles.items():
        count = (clustered_df["cluster_id"] == cid).sum()
        print(f"  {name} ({count} locations)")
