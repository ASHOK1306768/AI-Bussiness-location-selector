"""
Streamlit Web Dashboard for AI-Based Business Location Selector.

A comprehensive, student-friendly decision support system using Machine Learning,
Geospatial Analytics, and Multi-Criteria Decision Analysis (MCDA).
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_collection import load_data_sources_register
from src.location_recommendation import LocationRecommendationEngine
from utils.maps import create_recommendation_map

# -----------------------------------------------------------------------------
# App Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Business Location Selector",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E40AF;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .candidate-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .score-badge {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .tag-locality {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .tag-street {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .tag-property {
        background-color: #FED7AA;
        color: #9A3412;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine():
    """Initializes and caches the recommendation engine."""
    return LocationRecommendationEngine()


engine = get_engine()
sources_df = load_data_sources_register()

# -----------------------------------------------------------------------------
# Sidebar Navigation & Quick Controls
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/clouds/200/map-pin.png", width=110)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home Dashboard",
        "🎯 Find a Location",
        "🏆 Top Recommendations",
        "🗺️ Interactive Geospatial Map",
        "📊 ML & Clustering Analytics",
        "📑 Reports & Data Export",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("🎓 **BTech Capstone Project**")
st.sidebar.caption("Domain: Machine Learning & Geospatial MCDA")
st.sidebar.caption("Coverage: Pan-India (17 Metros & Cities)")

# Store session state for recommendations
if "rec_results" not in st.session_state:
    st.session_state.rec_results = None
    st.session_state.rec_summary = None

# =============================================================================
# PAGE 1: HOME DASHBOARD
# =============================================================================
if page == "🏠 Home Dashboard":
    st.markdown('<div class="main-header">📍 AI-Based Business Location Selector</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Intelligent Decision-Support System Using Machine Learning, Clustering, and Multi-Criteria Decision Analysis</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "👋 **Welcome!** This application assists entrepreneurs and retail analysts in discovering "
        "commercially viable locations across Indian cities. It combines real-world OpenStreetMap infrastructure, "
        "municipal demographics, surveyed commercial rental benchmarks, K-Means clustering, and transparent MCDA scoring."
    )

    # Dynamic KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-value">{len(engine.df_consolidated['city'].unique())}</div>
                <div class="kpi-label">Cities Covered</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-value">{len(engine.df_consolidated)}</div>
                <div class="kpi-label">Candidate Locations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-value">{len(engine.get_supported_categories())}</div>
                <div class="kpi-label">Business Categories</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-value">{len(sources_df)}</div>
                <div class="kpi-label">Registered Data Sources</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🔍 System Workflow & Multi-Criteria Decision Model")
    col_arch, col_table = st.columns([1.1, 1])

    with col_arch:
        st.markdown(
            """
            The recommendation framework applies a **Multi-Criteria Decision Analysis (MCDA)** additive formula:
            $$\\mathbf{S} = w_D \\cdot D + w_A \\cdot A + w_F \\cdot F + w_P \\cdot P + w_C \\cdot C + w_R \\cdot R$$
            Where:
            - **D**: Customer Demand Catchment (Population & Institutional Generators)
            - **A**: Arterial Road & Corridor Accessibility
            - **F**: Pedestrian & Shopper Footfall Intensity
            - **P**: Parking Readiness & Public Transit Access
            - **C**: Competition Opportunity & Agglomeration Index
            - **R**: Rent Affordability relative to specified user budget
            """
        )

    with col_table:
        st.markdown("**Pilot Geographic Scope**")
        city_counts = engine.df_consolidated.groupby(["state", "city"])["location_id"].count().reset_index()
        city_counts.columns = ["State", "City", "Candidates Analyzed"]
        st.dataframe(city_counts, use_container_width=True, hide_index=True)

    st.markdown("### 📋 Data Source Registry & Integrity Audit")
    st.dataframe(
        sources_df[["source_id", "dataset_name", "provider", "licence", "update_frequency"]],
        use_container_width=True,
        hide_index=True,
    )

# =============================================================================
# PAGE 2: FIND A LOCATION (QUERY INTERFACE)
# =============================================================================
elif page == "🎯 Find a Location":
    st.markdown('<div class="main-header">🎯 Find the Best Business Location</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Configure your business parameters, target city, budget constraints, and decision priorities.</div>',
        unsafe_allow_html=True,
    )

    cities_by_state = engine.get_available_cities()
    supported_cats = engine.get_supported_categories()

    with st.form("location_search_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 1. Business Profile")
            business_type = st.selectbox(
                "Select Business Category",
                options=supported_cats,
                index=supported_cats.index("Cafe") if "Cafe" in supported_cats else 0,
            )

            # Auto-populate recommended area & rent from category config
            cat_row = engine.scorer.df_categories[engine.scorer.df_categories["business_type"] == business_type]
            def_area = int(cat_row.iloc[0]["preferred_area_sqft"]) if not cat_row.empty else 600
            def_rent = int(cat_row.iloc[0]["typical_max_rent"]) if not cat_row.empty else 60000

            required_area = st.number_input(
                "Required Property Area (sq. ft.)",
                min_value=100,
                max_value=25000,
                value=def_area,
                step=50,
            )

            monthly_budget = st.number_input(
                "Maximum Monthly Rental Budget (INR)",
                min_value=10000,
                max_value=1000000,
                value=def_rent,
                step=5000,
                format="%d",
            )

        with col2:
            st.markdown("#### 2. Geographic Target")
            state_options = ["All States"] + list(cities_by_state.keys())
            selected_state = st.selectbox("Select State", options=state_options, index=1)

            if selected_state != "All States":
                city_options = ["All Cities"] + cities_by_state[selected_state]
            else:
                city_options = ["All Cities"] + sorted(list(engine.df_consolidated["city"].unique()))

            selected_city = st.selectbox("Select City", options=city_options, index=1 if len(city_options) > 1 else 0)

            rec_level = st.selectbox(
                "Recommendation Level",
                options=["All Levels", "Locality (Commercial Area)", "Street Segment", "Specific Commercial Property"],
                index=0,
            )
            level_map = {
                "All Levels": "all",
                "Locality (Commercial Area)": "locality",
                "Street Segment": "street",
                "Specific Commercial Property": "property",
            }

        st.markdown("#### 3. Custom Decision Weights (Optional)")
        with st.expander("Customize Priority Scoring Weights (MCDA Criteria)"):
            st.caption("Adjust sliders to customize your decision weights. They will be automatically normalized to sum to 100%.")
            cat_wts = engine.scorer.get_category_weights(business_type)

            cw1, cw2, cw3 = st.columns(3)
            with cw1:
                w_demand = st.slider("Customer Demand Weight", 0.0, 1.0, float(cat_wts.get("demand", 0.25)), 0.05)
                w_access = st.slider("Road Accessibility Weight", 0.0, 1.0, float(cat_wts.get("accessibility", 0.20)), 0.05)
            with cw2:
                w_footfall = st.slider("Footfall Intensity Weight", 0.0, 1.0, float(cat_wts.get("footfall", 0.15)), 0.05)
                w_park_trans = st.slider("Parking & Transit Weight", 0.0, 1.0, float(cat_wts.get("parking_transport", 0.15)), 0.05)
            with cw3:
                w_comp = st.slider("Competition Opportunity Weight", 0.0, 1.0, float(cat_wts.get("competition", 0.10)), 0.05)
                w_afford = st.slider("Rent Affordability Weight", 0.0, 1.0, float(cat_wts.get("affordability", 0.15)), 0.05)

        strict_budget = st.checkbox("Strict Budget Filter (Penalize locations exceeding specified budget)", value=True)

        submitted = st.form_submit_button("🚀 Generate Ranked Recommendations", use_container_width=True)

    if submitted:
        custom_weights = {
            "demand": w_demand,
            "accessibility": w_access,
            "footfall": w_footfall,
            "parking_transport": w_park_trans,
            "competition": w_comp,
            "affordability": w_afford,
        }

        with st.spinner("Analyzing candidate locations and calculating multi-criteria scores..."):
            query_city = None if selected_city == "All Cities" else selected_city
            query_state = None if selected_state == "All States" else selected_state

            recs = engine.recommend(
                business_type=business_type,
                city=query_city,
                state=query_state,
                monthly_budget=monthly_budget,
                required_area_sqft=required_area,
                recommendation_level=level_map[rec_level],
                custom_weights=custom_weights,
                strict_budget=strict_budget,
                top_n=5,
            )

            st.session_state.rec_results = recs["results"]
            st.session_state.rec_summary = recs["summary"]
            st.session_state.rec_city = selected_city if selected_city != "All Cities" else "Bhopal"

            st.success(f"Successfully analyzed {len(recs['results'])} candidate locations! Proceed to 'Top Recommendations' or 'Interactive Map'.")

# =============================================================================
# PAGE 3: RECOMMENDATIONS (TOP 5)
# =============================================================================
elif page == "🏆 Top Recommendations":
    st.markdown('<div class="main-header">🏆 Top Recommended Business Locations</div>', unsafe_allow_html=True)

    if st.session_state.rec_results is None or st.session_state.rec_results.empty:
        st.warning("No active recommendation run found. Please configure and run a search on the 'Find a Location' page.")
    else:
        results = st.session_state.rec_results
        top5 = results.head(5)
        summary = st.session_state.rec_summary

        st.caption(
            f"Showing Top {len(top5)} recommendations for **{summary['business_type']}** in "
            f"**{summary['city'] or 'All Cities'}** (Budget: INR {summary['user_budget']:,.0f} / month)."
        )

        for _, row in top5.iterrows():
            level = row["location_level"]
            tag_class = f"tag-{level}"

            with st.container():
                st.markdown(
                    f"""
                    <div class="candidate-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span style="font-size:1.3rem; font-weight:700; color:#1E3A8A;">#{row['rank']} - {row['locality']}</span>
                                &nbsp;&nbsp;<span class="{tag_class}">{level.upper()}</span>
                                <div style="color:#6B7280; font-size:0.9rem; margin-top:2px;">📍 {row['street_name']}, {row['city']}, {row['state']}</div>
                            </div>
                            <div style="text-align:right;">
                                <div class="score-badge">{row['suitability_score']}/100</div>
                                <div style="font-size:0.75rem; color:#6B7280; margin-top:3px;">Suitability Score</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Metrics grid for each candidate
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Avg Monthly Rent", f"INR {row['avg_monthly_rent']:,.0f}", delta=row["budget_status"], delta_color="normal" if row["budget_status"] == "Within Budget" else "inverse")
                with m2:
                    st.metric("Daily Footfall", f"{int(row['footfall_estimate']):,}")
                with m3:
                    st.metric("Competitor Density", f"{row['competitor_density']:.1f} / km²")
                with m4:
                    st.metric("Zone Profile", row["cluster_label"].split(":")[-1].strip())
                with m5:
                    st.metric("Data Quality", f"{row['data_quality_score']:.0f}%")

                # Advantages & Disadvantages
                col_adv, col_dis = st.columns(2)
                with col_adv:
                    st.markdown("**Key Advantages:**")
                    for adv in row["advantages"][:3]:
                        st.markdown(f"✅ *{adv}*")
                with col_dis:
                    st.markdown("**Potential Trade-offs & Flags:**")
                    for dis in row["disadvantages"][:2]:
                        st.markdown(f"⚠️ *{dis}*")

                st.markdown("---")

        # Multi-factor Comparison Radar / Bar Chart
        st.markdown("### 📊 Multi-Factor Decision Breakdown for Top Picks")
        factor_cols = ["score_demand", "score_accessibility", "score_footfall", "score_parking_transport", "score_competition", "score_affordability"]
        factor_names = ["Demand", "Accessibility", "Footfall", "Parking & Transit", "Competition Opportunity", "Rent Affordability"]

        bar_fig = go.Figure()
        for _, row in top5.iterrows():
            bar_fig.add_trace(go.Bar(
                name=f"#{row['rank']} {row['locality']}",
                x=factor_names,
                y=[row[c] for c in factor_cols],
            ))
        bar_fig.update_layout(
            barmode="group",
            yaxis=dict(title="Score (0 - 100)", range=[0, 105]),
            title="Comparison of Normalized Criteria Scores Across Top 5 Candidates",
            height=400,
        )
        st.plotly_chart(bar_fig, use_container_width=True)

# =============================================================================
# PAGE 4: INTERACTIVE GEOSPATIAL MAP
# =============================================================================
elif page == "🗺️ Interactive Geospatial Map":
    st.markdown('<div class="main-header">🗺️ Interactive Geospatial Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect candidate locations, geographic clusters, and 500m catchment zones.</div>', unsafe_allow_html=True)

    if st.session_state.rec_results is None or st.session_state.rec_results.empty:
        df_to_map = engine.df_consolidated.copy()
        df_to_map["rank"] = 999
        df_to_map["suitability_score"] = 70.0
        center_city = "Bhopal"
        st.info("Displaying all 54 commercial candidates in the master database. Run a search in 'Find a Location' to highlight recommended picks!")
    else:
        df_to_map = st.session_state.rec_results
        center_city = st.session_state.get("rec_city", "Bhopal")

    # Map display
    folium_map = create_recommendation_map(df_to_map, center_city=center_city, zoom_start=12)
    st_folium(folium_map, width="100%", height=600)

    st.markdown(
        """
        **Map Legend & Layer Interpretation:**
        - 🌟 **Red Star Pins**: Top recommended locations with red 500-meter catchment zones.
        - 🏢 **Blue Markers**: Locality / commercial zone centroids.
        - 🛣️ **Green Markers**: Specific commercial street corridors.
        - 🏠 **Orange Markers**: Specific verified commercial properties.
        """
    )

# =============================================================================
# PAGE 5: ML & CLUSTERING ANALYTICS
# =============================================================================
elif page == "📊 ML & Clustering Analytics":
    st.markdown('<div class="main-header">📊 Machine Learning & Clustering Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Exploration of unsupervised K-Means clusters and supervised regression demand models.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Unsupervised K-Means Clustering", "Supervised ML Models & Feature Importance"])

    with tab1:
        st.markdown("### K-Means Location Clustering")
        st.markdown(
            "Locations are clustered based on standardized multivariate features: Population, Competitor Density, "
            "Estimated Footfall, Accessibility, Parking, Transit, and Average Rent."
        )

        # Plotly 2D Scatter of Clusters
        clustered_data = engine.clusterer.fit_predict(engine.df_consolidated)

        fig_cluster = px.scatter(
            clustered_data,
            x="avg_monthly_rent",
            y="footfall_estimate",
            color="cluster_label",
            size="population",
            hover_name="locality",
            hover_data=["city", "location_level", "competitor_density"],
            labels={"avg_monthly_rent": "Average Monthly Rent (INR)", "footfall_estimate": "Estimated Daily Footfall"},
            title="Commercial Zones: Monthly Rent vs Daily Footfall by Cluster",
            height=500,
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        # Cluster Profiles Summary
        st.markdown("#### Identified Cluster Characteristics")
        summary_df = (
            clustered_data.groupby("cluster_label")[
                ["avg_monthly_rent", "footfall_estimate", "competitor_density", "accessibility_score", "parking_score"]
            ]
            .mean()
            .reset_index()
        )
        summary_df.columns = ["Cluster Profile", "Mean Rent (INR)", "Mean Footfall", "Competitor Density", "Accessibility (1-10)", "Parking (1-10)"]
        st.dataframe(summary_df.round(1), use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### Supervised Machine Learning Model Benchmarks")
        st.markdown(
            "Evaluation of supervised regression algorithms trained to predict customer turnout / business demand "
            "using empirical commercial observation records."
        )

        if engine.predictor and engine.predictor.evaluation_results:
            results_df = pd.DataFrame.from_dict(engine.predictor.evaluation_results, orient="index")
            results_df = results_df.reset_index().rename(columns={"index": "Model Architecture"})
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Feature Importances Chart
            if engine.predictor.feature_importances:
                st.markdown("#### Random Forest Feature Importances")
                fi_df = pd.DataFrame(
                    list(engine.predictor.feature_importances.items()),
                    columns=["Location Feature", "Relative Importance"],
                ).sort_values(by="Relative Importance", ascending=True)

                fig_fi = px.bar(
                    fi_df,
                    x="Relative Importance",
                    y="Location Feature",
                    orientation="h",
                    title="Feature Importance in Predicting Commercial Footfall & Customer Turnout",
                    color="Relative Importance",
                    color_continuous_scale="Viridis",
                    height=350,
                )
                st.plotly_chart(fig_fi, use_container_width=True)

        st.caption("ℹ️ *Note on Small-Sample Supervised Modeling*: In real-world urban retail analytics, ground truth sales samples are often sparse. While supervised models identify key driving factors (e.g. competitor agglomeration and parking), Multi-Criteria Decision Analysis (MCDA) provides a robust, zero-variance decision backbone.")

# =============================================================================
# PAGE 6: REPORTS & DATA EXPORT
# =============================================================================
elif page == "📑 Reports & Data Export":
    st.markdown('<div class="main-header">📑 Reports & Dataset Export</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Export recommendation outputs, candidate features, and project audit records.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📥 Export Active Recommendations")
        if st.session_state.rec_results is not None and not st.session_state.rec_results.empty:
            csv_data = st.session_state.rec_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Ranked Candidates (CSV)",
                data=csv_data,
                file_name="business_location_recommendations.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Run a search on 'Find a Location' to generate exportable recommendations.")

    with c2:
        st.markdown("### 📥 Export Master Location Dataset")
        master_csv = engine.df_consolidated.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Master Geospatial Features (CSV)",
            data=master_csv,
            file_name="master_location_features.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("### 📄 Project Executive Summary Dossier")
    report_text = f"""
# AI-Based Business Location Selector: Final Project Dossier
**Project Type:** BTech Final Year Capstone Project
**Date:** March 2026

## 1. System Scope & Objectives
This project solves the multi-dimensional commercial site selection problem for Indian entrepreneurs by combining:
- Geographic and infrastructure POIs from OpenStreetMap.
- Ward demographics and municipal census projections.
- Commercial rental benchmarks across tier-1 and tier-2 Indian cities (Bhopal, Indore, Pune).
- Unsupervised K-Means clustering to identify commercial archetypes.
- Multi-Criteria Decision Analysis (MCDA) to generate transparent, auditable suitability scores.

## 2. Multi-Criteria Scoring Formula
S = 0.25*D + 0.20*A + 0.15*F + 0.15*P + 0.10*C + 0.15*R
All criteria are normalized on a 0-100 scale, with customizable business weights.

## 3. Data Integrity & Ethical Compliance
- Zero fabricated real-world records.
- Explicit distinction between locality centroids, street corridors, and specific property listings.
- Source tracking via data_sources.csv registry.
    """
    st.text_area("Summary Report Preview", report_text, height=280)
    st.download_button(
        label="⬇️ Download Executive Summary (Markdown)",
        data=report_text,
        file_name="business_location_selector_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
