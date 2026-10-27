"""
Script to generate exploratory analysis charts and save them in outputs/charts/.
"""

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import LocationDataPreprocessor
from src.clustering import LocationClusterer

CHARTS_DIR = os.path.join("outputs", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams["figure.autolayout"] = True


def generate_all_charts():
    prep = LocationDataPreprocessor()
    df = prep.get_consolidated_dataset(save_output=False)

    # 1. Rent distribution by City (Box Plot)
    plt.figure(figsize=(8, 5), dpi=150)
    sns.boxplot(data=df, x="city", y="avg_monthly_rent", palette="Blues_d")
    plt.title("Commercial Monthly Rent Distribution Across Pilot Cities", fontsize=13, weight="bold")
    plt.xlabel("City", fontsize=11)
    plt.ylabel("Average Monthly Rent (INR)", fontsize=11)
    plt.savefig(os.path.join(CHARTS_DIR, "rent_distribution_by_city.png"))
    plt.close()
    print("Saved rent_distribution_by_city.png")

    # 2. Footfall vs Rent Scatter
    plt.figure(figsize=(8, 5), dpi=150)
    sns.scatterplot(
        data=df,
        x="avg_monthly_rent",
        y="footfall_estimate",
        hue="city",
        style="location_level",
        s=80,
        palette="Set1",
    )
    plt.title("Commercial Footfall vs Average Monthly Rent", fontsize=13, weight="bold")
    plt.xlabel("Average Monthly Rent (INR)", fontsize=11)
    plt.ylabel("Estimated Daily Pedestrian Footfall", fontsize=11)
    plt.savefig(os.path.join(CHARTS_DIR, "footfall_vs_rent_scatter.png"))
    plt.close()
    print("Saved footfall_vs_rent_scatter.png")

    # 3. Correlation Heatmap
    num_cols = [
        "population",
        "competitor_density",
        "footfall_estimate",
        "accessibility_score",
        "parking_score",
        "public_transport_score",
        "avg_monthly_rent",
        "data_quality_score",
    ]
    plt.figure(figsize=(9, 7), dpi=150)
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="Blues", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix of Spatial and Commercial Indicators", fontsize=13, weight="bold")
    plt.savefig(os.path.join(CHARTS_DIR, "correlation_matrix.png"))
    plt.close()
    print("Saved correlation_matrix.png")

    # 4. Clustering Silhouette & Distribution Chart
    clusterer = LocationClusterer(n_clusters=4)
    clustered_df = clusterer.fit_predict(df)

    plt.figure(figsize=(8, 5), dpi=150)
    cluster_counts = clustered_df["cluster_label"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster Profile", "Count"]
    sns.barplot(data=cluster_counts, y="Cluster Profile", x="Count", palette="viridis")
    plt.title("Location Frequency by K-Means Commercial Archetype (k=4)", fontsize=13, weight="bold")
    plt.xlabel("Number of Locations", fontsize=11)
    plt.ylabel("Cluster Profile", fontsize=11)
    plt.savefig(os.path.join(CHARTS_DIR, "cluster_distribution.png"))
    plt.close()
    print("Saved cluster_distribution.png")

    print("All exploratory charts generated successfully.")


if __name__ == "__main__":
    generate_all_charts()
