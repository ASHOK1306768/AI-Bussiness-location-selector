"""
Builder script to generate all 7 standard Jupyter Notebooks with markdown and code cells.
"""

import os
import nbformat as nbf

NOTEBOOKS_DIR = "notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def create_nb(filename, title, cells_data):
    nb = nbf.v4.new_notebook()
    nb["cells"] = []

    # Title header
    nb["cells"].append(nbf.v4.new_markdown_cell(f"# {title}\n**AI-Based Business Location Selector Project**"))

    for cell_type, content in cells_data:
        if cell_type == "markdown":
            nb["cells"].append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "code":
            nb["cells"].append(nbf.v4.new_code_cell(content))

    path = os.path.join(NOTEBOOKS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated {path}")


def build_all_notebooks():
    # 01_data_collection.ipynb
    create_nb(
        "01_data_collection.ipynb",
        "01 - Data Source Investigation and Collection Pipeline",
        [
            ("markdown", "## 1. Objective\nInspect data sources, verify access and licenses in `data_sources.csv`, and demonstrate Overpass API caching."),
            ("code", "import os\nimport sys\nsys.path.insert(0, '..')\nimport pandas as pd\nfrom src.data_collection import load_data_sources_register, OSMDataCollector\n\nsources_df = load_data_sources_register('../data/metadata/data_sources.csv')\nsources_df"),
            ("markdown", "## 2. Test Live / Cached Overpass API Query\nFetch nearby amenities and transit stops around Bhopal commercial center (MP Nagar)."),
            ("code", "collector = OSMDataCollector(cache_dir='../data/raw/osm')\n# Coordinates for MP Nagar, Bhopal\nlat, lon = 23.2332, 77.4334\npois = collector.fetch_pois_around_point(lat, lon, radius_meters=500)\nprint(f'Retrieved {len(pois)} POIs within 500m radius of MP Nagar.')\nif pois:\n    print('Sample POI:', pois[0])"),
        ],
    )

    # 02_data_cleaning.ipynb
    create_nb(
        "02_data_cleaning.ipynb",
        "02 - Data Validation and Preprocessing Pipeline",
        [
            ("markdown", "## 1. Objective\nValidate coordinate ranges, remove duplicates, impute missing features, and produce the consolidated dataset."),
            ("code", "import os\nimport sys\nsys.path.insert(0, '..')\nimport pandas as pd\nfrom src.preprocessing import LocationDataPreprocessor\n\nprep = LocationDataPreprocessor(data_dir='../data/processed')\nlocs, feats, rents, cats = prep.load_all_datasets()\nprint(f'Raw: {len(locs)} locations, {len(rents)} rental records.')"),
            ("markdown", "## 2. Generate and Verify Consolidated Dataset\nMerges spatial coordinates with demographic features and rental statistics."),
            ("code", "df_cons = prep.get_consolidated_dataset(save_output=False)\nprint('Consolidated Shape:', df_cons.shape)\ndf_cons[['location_id', 'city', 'locality', 'location_level', 'avg_monthly_rent', 'data_quality_score']].head(8)"),
        ],
    )

    # 03_exploratory_analysis.ipynb
    create_nb(
        "03_exploratory_analysis.ipynb",
        "03 - Exploratory Data Analysis (EDA)",
        [
            ("markdown", "## 1. Objective\nExplore distributions of rents, footfall, accessibility, and competitor density across pilot cities."),
            ("code", "import os\nimport sys\nsys.path.insert(0, '..')\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom src.preprocessing import LocationDataPreprocessor\n\nprep = LocationDataPreprocessor(data_dir='../data/processed')\ndf = prep.get_consolidated_dataset(save_output=False)\ndf.describe()"),
            ("markdown", "## 2. Rental Comparison by City\nExamine variation in commercial rents between Bhopal, Indore, and Pune."),
            ("code", "plt.figure(figsize=(8, 4.5))\nsns.boxplot(data=df, x='city', y='avg_monthly_rent', palette='Blues_d')\nplt.title('Commercial Rent Distribution by City')\nplt.ylabel('Monthly Rent (INR)')\nplt.show()"),
            ("markdown", "## 3. Correlation Matrix\nAnalyze collinearity between demographic catchment, footfall, and rent."),
            ("code", "plt.figure(figsize=(8, 6))\nnum_cols = ['population', 'competitor_density', 'footfall_estimate', 'accessibility_score', 'parking_score', 'avg_monthly_rent']\nsns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')\nplt.title('Correlation Matrix')\nplt.show()"),
        ],
    )

    # 04_feature_engineering.ipynb
    create_nb(
        "04_feature_engineering.ipynb",
        "04 - Geospatial Catchment & Feature Engineering",
        [
            ("markdown", "## 1. Objective\nCompute geodesic distance matrices, rent affordability ratios, and normalized 0-100 decision components."),
            ("code", "import os\nimport sys\nsys.path.insert(0, '..')\nimport pandas as pd\nfrom src.preprocessing import LocationDataPreprocessor\nfrom src.feature_engineering import LocationFeatureEngineer, calculate_geodesic_distance_km\n\nprep = LocationDataPreprocessor(data_dir='../data/processed')\ndf = prep.get_consolidated_dataset(save_output=False)\nfe = LocationFeatureEngineer(df)\n\n# Geodesic distance between Bhopal MP Nagar and New Market\ndist_km = calculate_geodesic_distance_km(23.2332, 77.4334, 23.2376, 77.4012)\nprint(f'Distance MP Nagar to New Market: {dist_km:.2f} km')"),
            ("markdown", "## 2. Normalized Components Calculation\nTransform raw features into scale-standardized [0, 100] decision factors."),
            ("code", "df_norm = fe.calculate_normalized_components(df, user_budget=60000, business_type='Cafe')\ndf_norm[['location_id', 'locality', 'score_demand', 'score_accessibility', 'score_footfall', 'score_affordability']].head()"),
        ],
    )

    # 05_location_clustering.ipynb
    create_nb(
        "05_location_clustering.ipynb",
        "05 - Unsupervised Location Clustering (K-Means)",
        [
            ("markdown", "## 1. Objective\nCluster commercial zones using standardized features, evaluate Elbow & Silhouette curves, and profile commercial archetypes."),
            ("code", "import os\nimport sys\nsys.path.insert(0, '..')\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom src.preprocessing import LocationDataPreprocessor\nfrom src.clustering import LocationClusterer\n\nprep = LocationDataPreprocessor(data_dir='../data/processed')\ndf = prep.get_consolidated_dataset(save_output=False)\nclusterer = LocationClusterer(n_clusters=4)\neval_res = clusterer.evaluate_cluster_range(df, k_range=range(2, 7))\n\nplt.figure(figsize=(10, 4))\nplt.subplot(1, 2, 1)\nplt.plot(eval_res['k'], eval_res['inertia'], 'bo-')\nplt.title('Elbow Method (Inertia)')\nplt.xlabel('k (Clusters)')\n\nplt.subplot(1, 2, 2)\nplt.plot(eval_res['k'], eval_res['silhouette'], 'ro-')\nplt.title('Silhouette Score Analysis')\nplt.xlabel('k (Clusters)')\nplt.show()"),
            ("markdown", "## 2. Fit K-Means and Examine Identified Profiles"),
            ("code", "clustered_df = clusterer.fit_predict(df)\nfor cid, name in clusterer.cluster_profiles.items():\n    print(f'{name}: {(clustered_df[\"cluster_id\"] == cid).sum()} locations')"),
        ],
    )

    # 06_model_training.ipynb
    create_nb(
        "06_model_training.ipynb",
        "06 - Supervised Model Training and Comparison",
        [
            ("markdown", "## 1. Objective\nTrain and benchmark Linear Regression (Baseline), Decision Tree, and Random Forest Regressors on observed customer footfall."),
            ("code", "import os\nimport sys\nsys.path.insert(0, '..')\nimport pandas as pd\nfrom src.preprocessing import LocationDataPreprocessor\nfrom src.prediction import LocationDemandPredictor\n\nprep = LocationDataPreprocessor(data_dir='../data/processed')\ndf_locs = prep.get_consolidated_dataset(save_output=False)\ndf_obs = pd.read_csv('../data/processed/observed_outcomes.csv')\n\npredictor = LocationDemandPredictor()\nX, y, feats = predictor.prepare_dataset(df_locs, df_obs, target_col='daily_customers')\nresults = predictor.train_and_evaluate(X, y, feats)\npd.DataFrame(results).T"),
        ],
    )

    # 07_model_evaluation.ipynb
    create_nb(
        "07_model_evaluation.ipynb",
        "07 - Model Evaluation, Feature Importance & Limitations",
        [
            ("markdown", "## 1. Objective\nInspect feature importances, analyze evaluation metrics (MAE, RMSE, R2), and document methodology trade-offs."),
            ("code", "import os\nimport sys\nsys.path.insert(0, '..')\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom src.prediction import LocationDemandPredictor\n\npredictor = LocationDemandPredictor.load_model('../models/demand_prediction_model.joblib')\nfi_df = pd.Series(predictor.feature_importances).sort_values(ascending=True)\n\nplt.figure(figsize=(8, 4))\nfi_df.plot(kind='barh', color='teal')\nplt.title('Random Forest Feature Importances')\nplt.xlabel('Relative Weight')\nplt.show()"),
            ("markdown", "## 2. Discussion: Why MCDA is the primary decision backbone\nSupervised regression identifies correlations (e.g. competitor agglomeration and parking), but ground-truth business performance data is scarce in emerging markets. Multi-Criteria Decision Analysis (MCDA) prevents overfitting and guarantees transparent, explainable recommendations."),
        ],
    )


if __name__ == "__main__":
    build_all_notebooks()
