# 📍 AI-Based Business Location Selector Using Machine Learning, Clustering, and Multi-Criteria Decision Analysis

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![License: ODbL / Open](https://img.shields.io/badge/Data_License-ODbL%20%7C%20OGD-green.svg)](https://opendatacommons.org/licenses/odbl/)
[![Tests](https://img.shields.io/badge/pytest-10%20passed-brightgreen.svg)](tests/)

An intelligent decision-support system developed as a **BTech Final-Year Capstone Project**. It helps entrepreneurs, retail analysts, and small business owners identify optimal physical locations across Indian cities (**Delhi, Mumbai, Bengaluru, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad, Jaipur, Lucknow, Chandigarh, Bhopal, Indore, Kochi, Bhubaneswar, Guwahati, and Nagpur**) by combining real-world geospatial infrastructure, municipal demographics, commercial rental benchmarks, unsupervised K-Means clustering, and Multi-Criteria Decision Analysis (MCDA).

---

## 🌟 Key Features

1. **Multi-Category Business Support:**
   - Pre-configured profiles and custom weight matrices for **19 business categories** across Food & Beverage, Retail, Healthcare, Education, Services, Automotive, Hospitality, and Logistics.
2. **Multi-Level Recommendation:**
   - Differentiates between **Commercial Localities** (areal centroids), **High-Street Corridors** (street segments), and **Verified Commercial Properties**.
3. **Geospatial Catchment Analytics:**
   - Computes geodesic distance matrices (WGS-84 ellipsoidal via `geopy`), competitor densities, and 500m catchment buffers.
4. **Transparent Multi-Criteria Suitability Scoring (MCDA):**
   - Normalized 0–100 score:
     $$\mathbf{S} = w_D \cdot D + w_A \cdot A + w_F \cdot F + w_P \cdot P + w_C \cdot C + w_R \cdot R$$
   - Transparent contribution breakdown with automated advantages, trade-offs, and budget constraint validation.
5. **Machine Learning & Clustering:**
   - **K-Means Clustering ($k=4$):** Automatically segments locations into commercial archetypes (e.g. *Prime High-Footfall Hub*, *Value Transit District*, *Suburban Logistics Corridor*).
   - **Supervised Regression:** Compares Linear Regression (Baseline), Decision Tree, and Random Forest Regressors to predict customer turnout and assess feature importance.
6. **Interactive Streamlit Web Dashboard:**
   - 6-page interactive interface featuring dynamic KPI cards, Folium geospatial maps, Plotly radar/scatter charts, and CSV report export.

---

## 📁 Project Directory Structure

```text
AI_Business_Location_Selector/
│
├── app.py                         # Streamlit interactive web application
├── requirements.txt               # Pinned Python package dependencies
├── README.md                      # Comprehensive setup & run instructions
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules for virtualenvs, caches, logs
│
├── data/
│   ├── raw/
│   │   ├── osm/                   # Cached Overpass API queries
│   │   ├── census/                # Census ward demographics
│   │   ├── government/            # Public urban transport registries
│   │   └── property_listings/     # Commercial property listing archives
│   ├── processed/
│   │   ├── locations.csv          # Master locations registry (localities, streets, properties)
│   │   ├── location_features.csv  # Engineered spatial, footfall, and demographic features
│   │   ├── commercial_rents.csv   # Commercial rents, areas, and availability status
│   │   ├── business_categories.csv# Business weights, min areas, and typical rents
│   │   ├── observed_outcomes.csv  # Empirical merchant customer count records
│   │   └── consolidated_locations.csv # Fully joined & imputed master dataset
│   └── metadata/
│       └── data_sources.csv       # Audit trail documenting providers, URLs, and licenses
│
├── src/
│   ├── data_collection.py         # Overpass OSM API client with caching & fallback
│   ├── preprocessing.py           # Data cleaning, coordinate validation & joins
│   ├── feature_engineering.py     # Geodesic distances, catchment densities & 0-100 normalization
│   ├── suitability_scoring.py     # Multi-criteria weighted scoring engine & explainability
│   ├── clustering.py              # K-Means clustering, Elbow method & silhouette profiling
│   ├── prediction.py              # Supervised regression benchmark (Linear, Decision Tree, RF)
│   └── location_recommendation.py # End-to-end recommendation orchestrator
│
├── utils/
│   ├── maps.py                    # Folium interactive map builder with layer controls
│   └── validation.py              # Coordinate boundary, budget, and weight validators
│
├── models/
│   ├── kmeans_model.joblib        # Persisted K-Means clustering model & scaler
│   └── demand_prediction_model.joblib # Persisted Random Forest regression model
│
├── notebooks/
│   ├── 01_data_collection.ipynb   # Data source inspection & Overpass API queries
│   ├── 02_data_cleaning.ipynb     # Schema validation & preprocessing pipeline
│   ├── 03_exploratory_analysis.ipynb # Statistical EDA & rent/footfall distributions
│   ├── 04_feature_engineering.ipynb # Geospatial catchment calculation & distance matrices
│   ├── 05_location_clustering.ipynb # K-Means evaluation (Elbow, Silhouette) & archetypes
│   ├── 06_model_training.ipynb    # Supervised regression model training & cross-validation
│   └── 07_model_evaluation.ipynb  # Residual analysis, feature importance & trade-offs
│
├── outputs/
│   ├── charts/                    # Generated EDA plots (boxplots, scatter, correlation heatmaps)
│   ├── maps/                      # Map exports and geospatial snapshots
│   └── reports/
│       ├── ARCHITECTURE_ROADMAP.md # System architecture specification document
│       ├── PROJECT_REPORT.md       # Complete academic BTech project report
│       └── VIVA_QUESTIONS.md       # Comprehensive Viva Voce Q&A preparation guide
│
├── scripts/
│   ├── generate_eda_charts.py     # Reproducible script to generate figures in outputs/charts/
│   └── build_notebooks.py         # Script generating standard multi-cell Jupyter notebooks
│
└── tests/
    ├── test_preprocessing.py      # Unit tests for coordinate bounds & cleaning
    ├── test_scoring.py            # Unit tests for MCDA scoring & weight normalization
    └── test_recommendation.py     # Unit tests for recommendation ranking & constraints
```

---

## 🚀 Step-by-Step Setup Guide (Windows & VS Code)

### 1. Prerequisites
- **Operating System:** Windows 10 or 11
- **Python:** Python 3.12+ installed ([Python.org](https://www.python.org/downloads/))
- **Editor:** Visual Studio Code (VS Code)

### 2. Open Project in VS Code
1. Launch VS Code.
2. Open the folder: `c:\Users\olpul\OneDrive\Desktop\AI Business Location Selector`.
3. Open a PowerShell terminal in VS Code (`Ctrl + ` `~` or `Terminal -> New Terminal`).

### 3. Create & Activate Virtual Environment (Optional but Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Run Automated Unit Tests
To verify all preprocessing modules, scoring algorithms, and recommendation constraints:
```powershell
python -m pytest tests -v
```
*Expected Output:* `9 passed in ~3.5s`

### 6. Launch the Streamlit Web Application
Execute the following command in your terminal:
```powershell
python -m streamlit run app.py
```
*(Alternatively, you can simply double-click or run `.\run_app.bat`)*

Streamlit will automatically open your default browser at:  
👉 **`http://localhost:8501`**

---

## 🖥️ Using the Streamlit Dashboard

| Page | Functionality |
| :--- | :--- |
| **🏠 Home Dashboard** | View system overview, dynamic KPIs across pilot cities, MCDA equation, and `data_sources.csv` audit register. |
| **🎯 Find a Location** | Select your business category (e.g. *Cafe*, *Pharmacy*, *Coaching Centre*), target city, monthly budget, required area, and customize decision weights. |
| **🏆 Top Recommendations** | View the top 5 ranked locations with suitability scores, budget comparison badges, zone profiles, advantages, disadvantages, and multi-criteria comparison bar charts. |
| **🗺️ Interactive Map** | Explore Folium geospatial maps with toggleable layers for Localities, Streets, Properties, and 500m catchment circles. |
| **📊 ML & Analytics** | Analyze K-Means cluster distributions (Rent vs Footfall), Elbow curves, model performance tables (Linear vs Decision Tree vs Random Forest), and feature importance plots. |
| **📑 Reports & Export** | Download the ranked recommendation table as CSV or export the complete project dossier. |

---

## 📓 Running Jupyter Notebooks

All 7 analytical notebooks are located in `notebooks/`. You can open and run them directly in VS Code (with the Jupyter extension) or start Jupyter Lab:
```powershell
jupyter lab
```
Run notebooks sequentially:
1. `01_data_collection.ipynb`
2. `02_data_cleaning.ipynb`
3. `03_exploratory_analysis.ipynb`
4. `04_feature_engineering.ipynb`
5. `05_location_clustering.ipynb`
6. `06_model_training.ipynb`
7. `07_model_evaluation.ipynb`

---

## 🛡️ Responsible AI & Ethical Data Practices

- **Decision-Support Scope:** The application serves as an objective analytical guide. It does **not** guarantee business profitability or substitute for on-site due diligence.
- **Ground-Truth Separation:** Observed records, estimated indicators, and spatial indicators are explicitly marked with separate data provenance IDs.
- **Privacy & Licensing:** Utilizes open geospatial data under the **Open Database License (ODbL) 1.0** and **Open Government Data (OGD) License India**. No private personal consumer data is tracked or harvested.

---

## 🎓 Academic Viva Voce Preparation
A complete 20-question Viva Voce preparation guide covering machine learning concepts, K-Means clustering, the Elbow Method, Random Forest regression, overfitting, and location analysis is available at:  
👉 [`outputs/reports/VIVA_QUESTIONS.md`](outputs/reports/VIVA_QUESTIONS.md)

Detailed BTech final year project methodology and system documentation is available at:  
👉 [`outputs/reports/PROJECT_REPORT.md`](outputs/reports/PROJECT_REPORT.md)
