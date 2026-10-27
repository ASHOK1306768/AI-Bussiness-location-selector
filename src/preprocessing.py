"""
Data Preprocessing Module for AI-Based Business Location Selector.

Handles dataset loading, cleaning, coordinate boundary checks, deduplication,
missing value imputation, data quality calculation, and safe relational joins.
"""

import logging
import os
import sys
from typing import Dict, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from utils.validation import validate_coordinates, validate_location_level


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join("data", "processed")


class LocationDataPreprocessor:
    """Preprocesses and merges geospatial, demographic, and commercial rental data."""

    def __init__(self, data_dir: str = PROCESSED_DIR):
        self.data_dir = data_dir

    def load_all_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads the four primary domain datasets."""
        loc_path = os.path.join(self.data_dir, "locations.csv")
        feat_path = os.path.join(self.data_dir, "location_features.csv")
        rent_path = os.path.join(self.data_dir, "commercial_rents.csv")
        cat_path = os.path.join(self.data_dir, "business_categories.csv")

        for p in [loc_path, feat_path, rent_path, cat_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required dataset file not found: {p}")

        df_locations = pd.read_csv(loc_path)
        df_features = pd.read_csv(feat_path)
        df_rents = pd.read_csv(rent_path)
        df_categories = pd.read_csv(cat_path)

        logger.info(
            "Loaded datasets: %d locations, %d feature rows, %d rental records, %d categories.",
            len(df_locations),
            len(df_features),
            len(df_rents),
            len(df_categories),
        )
        return df_locations, df_features, df_rents, df_categories

    def clean_locations(self, df_locations: pd.DataFrame) -> pd.DataFrame:
        """Cleans, normalizes, and validates the master locations dataframe."""
        df = df_locations.copy()

        # String stripping and casing
        for col in ["state", "city", "locality", "street_name", "location_level"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Remove duplicate location IDs
        duplicates = df.duplicated(subset=["location_id"])
        if duplicates.any():
            logger.warning("Found %d duplicate location_ids. Dropping duplicates.", duplicates.sum())
            df = df.drop_duplicates(subset=["location_id"], keep="first")

        # Validate coordinate boundaries for all rows
        valid_mask = []
        for _, row in df.iterrows():
            is_valid, msg = validate_coordinates(row["latitude"], row["longitude"])
            if not is_valid:
                logger.warning("Invalid coordinate for %s: %s", row["location_id"], msg)
            valid_mask.append(is_valid)

        df = df[valid_mask].reset_index(drop=True)

        # Validate location level
        level_mask = df["location_level"].apply(lambda lvl: validate_location_level(lvl)[0])
        df = df[level_mask].reset_index(drop=True)

        return df

    def clean_features(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Cleans and imputes missing values in feature indicators."""
        df = df_features.copy()

        numerical_cols = [
            "population",
            "competitor_count",
            "competitor_density",
            "footfall_estimate",
            "accessibility_score",
            "parking_score",
            "public_transport_score",
            "nearby_facilities_count",
        ]

        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Impute missing values with column median and flag
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[f"{col}_imputed"] = df[col].isnull().astype(int)
                    df[col] = df[col].fillna(median_val)
                    logger.info("Imputed missing values in feature '%s' with median %.2f", col, median_val)
                else:
                    df[f"{col}_imputed"] = 0

        # Bound scores to logical 1-10 range
        for score_col in ["accessibility_score", "parking_score", "public_transport_score"]:
            if score_col in df.columns:
                df[score_col] = df[score_col].clip(1.0, 10.0)

        return df

    def clean_rents(self, df_rents: pd.DataFrame) -> pd.DataFrame:
        """Cleans rental prices, property areas, and status flags."""
        df = df_rents.copy()

        df["monthly_rent"] = pd.to_numeric(df["monthly_rent"], errors="coerce")
        df["area_sqft"] = pd.to_numeric(df["area_sqft"], errors="coerce")
        df["deposit"] = pd.to_numeric(df["deposit"], errors="coerce")

        # Drop impossible non-positive rents or areas
        valid_rents = (df["monthly_rent"] > 0) & (df["area_sqft"] > 0)
        df = df[valid_rents].copy()

        # Derive rate per sqft for standardized comparison
        df["rent_per_sqft"] = (df["monthly_rent"] / df["area_sqft"]).round(2)
        df["availability_status"] = df["availability_status"].str.strip().str.title()

        return df

    def get_consolidated_dataset(self, save_output: bool = True) -> pd.DataFrame:
        """
        Merges locations, features, and aggregated rental statistics into a unified dataset.
        """
        df_locations, df_features, df_rents, _ = self.load_all_datasets()

        df_loc_clean = self.clean_locations(df_locations)
        df_feat_clean = self.clean_features(df_features)
        df_rents_clean = self.clean_rents(df_rents)

        # Compute rental benchmarks per location (mean rent, mean sqft, count of available properties)
        rent_summary = (
            df_rents_clean.groupby("location_id")
            .agg(
                avg_monthly_rent=("monthly_rent", "mean"),
                min_monthly_rent=("monthly_rent", "min"),
                max_monthly_rent=("monthly_rent", "max"),
                avg_area_sqft=("area_sqft", "mean"),
                avg_rent_per_sqft=("rent_per_sqft", "mean"),
                available_property_count=(
                    "availability_status",
                    lambda s: (s.str.lower() == "available").sum(),
                ),
            )
            .reset_index()
        )

        # Merge locations with features
        merged = pd.merge(df_loc_clean, df_feat_clean, on="location_id", how="inner")

        # Merge with rental summary
        consolidated = pd.merge(merged, rent_summary, on="location_id", how="left")

        # Fill missing rental stats for locations without explicit property listings with city-level averages
        city_rents = consolidated.groupby("city")["avg_monthly_rent"].transform("median")
        consolidated["avg_monthly_rent"] = consolidated["avg_monthly_rent"].fillna(city_rents)
        consolidated["avg_rent_per_sqft"] = consolidated["avg_rent_per_sqft"].fillna(
            consolidated.groupby("city")["avg_rent_per_sqft"].transform("median")
        )
        consolidated["available_property_count"] = consolidated["available_property_count"].fillna(0).astype(int)

        if save_output:
            out_path = os.path.join(self.data_dir, "consolidated_locations.csv")
            consolidated.to_csv(out_path, index=False)
            logger.info("Saved consolidated dataset with %d rows to %s", len(consolidated), out_path)

        return consolidated


if __name__ == "__main__":
    preprocessor = LocationDataPreprocessor()
    df_cons = preprocessor.get_consolidated_dataset()
    print("Consolidated Dataset Preview:")
    print(df_cons[["location_id", "city", "locality", "location_level", "avg_monthly_rent", "data_quality_score"]].head())
