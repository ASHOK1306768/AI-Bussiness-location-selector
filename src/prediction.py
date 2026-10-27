"""
Predictive Machine Learning Module for AI-Based Business Location Selector.

Implements Linear Regression (Baseline), Decision Tree, and Random Forest Regressors
to predict observed business demand, footfall, or revenue outcomes.
Evaluates models with MAE, RMSE, and R2, and saves model pipelines using Joblib.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


class LocationDemandPredictor:
    """Supervised regression engine comparing Linear Regression, Decision Tree, and Random Forest."""

    FEATURE_COLS = [
        "population",
        "competitor_density",
        "footfall_estimate",
        "accessibility_score",
        "parking_score",
        "public_transport_score",
        "avg_monthly_rent",
    ]

    TARGET_COL = "daily_customers"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {
            "Linear Regression (Baseline)": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=random_state),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=random_state),
        }
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        self.evaluation_results: Dict[str, Dict[str, float]] = {}
        self.feature_importances: Dict[str, float] = {}

    def prepare_dataset(
        self,
        df_consolidated: pd.DataFrame,
        df_outcomes: pd.DataFrame,
        target_col: str = "daily_customers",
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Merges consolidated location features with observed outcomes.
        """
        self.TARGET_COL = target_col
        merged = pd.merge(df_outcomes, df_consolidated, on="location_id", how="inner")

        if merged.empty:
            raise ValueError("Merged outcomes and location features dataset is empty!")

        X_df = merged[self.FEATURE_COLS].copy()
        X_df = X_df.fillna(X_df.median())
        y = merged[target_col].values

        X_scaled = self.scaler.fit_transform(X_df)
        return X_scaled, y, self.FEATURE_COLS

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.20,
    ) -> Dict[str, Dict[str, float]]:
        """
        Trains Linear Regression, Decision Tree, and Random Forest Regressors.
        Compares MAE, RMSE, R2, and 5-fold cross-validation scores.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        best_r2 = -float("inf")

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            # 5-fold Cross-validation on full dataset
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
            cv_r2_mean = float(np.mean(cv_scores))

            self.evaluation_results[name] = {
                "MAE": round(float(mae), 2),
                "RMSE": round(float(rmse), 2),
                "R2_Test": round(float(r2), 4),
                "CV_R2_Mean": round(cv_r2_mean, 4),
            }

            logger.info("%s -> MAE: %.2f | RMSE: %.2f | R2: %.4f | CV R2: %.4f", name, mae, rmse, r2, cv_r2_mean)

            if r2 > best_r2:
                best_r2 = r2
                self.best_model_name = name
                self.best_model = model

        # Extract feature importances if available
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
            self.feature_importances = {
                name: round(float(imp), 4) for name, imp in zip(feature_names, importances)
            }
        elif hasattr(self.best_model, "coef_"):
            coefs = np.abs(self.best_model.coef_)
            total = sum(coefs) or 1.0
            self.feature_importances = {
                name: round(float(c / total), 4) for name, c in zip(feature_names, coefs)
            }

        return self.evaluation_results

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        """Generates outcome predictions using the best trained model."""
        if self.best_model is None:
            raise ValueError("Model is not trained yet. Call train_and_evaluate first.")

        X_df = df_features[self.FEATURE_COLS].copy().fillna(df_features[self.FEATURE_COLS].median())
        X_scaled = self.scaler.transform(X_df)
        preds = self.best_model.predict(X_scaled)
        return np.maximum(0, preds).round(1)

    def save_model(self, filepath: Optional[str] = None) -> str:
        """Saves best model, scaler, and evaluation metrics using Joblib."""
        path = filepath or os.path.join(MODELS_DIR, "demand_prediction_model.joblib")
        payload = {
            "best_model_name": self.best_model_name,
            "model": self.best_model,
            "scaler": self.scaler,
            "feature_cols": self.FEATURE_COLS,
            "target_col": self.TARGET_COL,
            "evaluation_results": self.evaluation_results,
            "feature_importances": self.feature_importances,
        }
        joblib.dump(payload, path)
        logger.info("Saved best predictive model (%s) to %s", self.best_model_name, path)
        return path

    @classmethod
    def load_model(cls, filepath: Optional[str] = None) -> "LocationDemandPredictor":
        """Loads saved predictor from disk."""
        path = filepath or os.path.join(MODELS_DIR, "demand_prediction_model.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        payload = joblib.load(path)
        instance = cls()
        instance.best_model_name = payload["best_model_name"]
        instance.best_model = payload["model"]
        instance.scaler = payload["scaler"]
        instance.FEATURE_COLS = payload["feature_cols"]
        instance.TARGET_COL = payload["target_col"]
        instance.evaluation_results = payload["evaluation_results"]
        instance.feature_importances = payload["feature_importances"]
        return instance


if __name__ == "__main__":
    from src.preprocessing import LocationDataPreprocessor

    prep = LocationDataPreprocessor()
    df_locs = prep.get_consolidated_dataset(save_output=False)
    df_outcomes = pd.read_csv(os.path.join("data", "processed", "observed_outcomes.csv"))

    predictor = LocationDemandPredictor()
    X, y, feats = predictor.prepare_dataset(df_locs, df_outcomes, target_col="daily_customers")

    print("\nTraining and Comparing Machine Learning Models:")
    results = predictor.train_and_evaluate(X, y, feats)

    for m_name, metrics in results.items():
        print(f"  {m_name}: R2={metrics['R2_Test']}, MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, CV_R2={metrics['CV_R2_Mean']}")

    predictor.save_model()
    print(f"\nBest Model: {predictor.best_model_name}")
    print("Feature Importances:")
    for f_name, imp in predictor.feature_importances.items():
        print(f"  {f_name}: {imp * 100:.1f}%")
