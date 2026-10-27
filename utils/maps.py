"""
Geospatial Mapping Utilities for AI-Based Business Location Selector.

Builds interactive Folium maps with layer controls, catchment buffers,
and distinctive visual markers for localities, streets, and commercial properties.
"""

import folium
from folium.plugins import MarkerCluster
import pandas as pd


CITY_CENTERS = {
    "delhi": [28.6139, 77.2090],
    "mumbai": [19.0760, 72.8777],
    "bengaluru": [12.9716, 77.5946],
    "hyderabad": [17.3850, 78.4867],
    "chennai": [13.0827, 80.2707],
    "kolkata": [22.5726, 88.3639],
    "ahmedabad": [23.0225, 72.5714],
    "pune": [18.5204, 73.8567],
    "jaipur": [26.9124, 75.7873],
    "lucknow": [26.8467, 80.9462],
    "chandigarh": [30.7333, 76.7794],
    "bhopal": [23.2350, 77.4100],
    "indore": [22.7300, 75.8750],
    "kochi": [9.9312, 76.2673],
    "bhubaneswar": [20.2961, 85.8245],
    "guwahati": [26.1445, 91.7362],
    "nagpur": [21.1458, 79.0882],
}



def create_recommendation_map(
    df_results: pd.DataFrame,
    center_city: str = "Bhopal",
    zoom_start: int = 12,
    highlight_top_n: int = 5,
) -> folium.Map:
    """
    Constructs an interactive Folium map representing candidate locations.

    Visual encoding:
    - Top N recommended candidates: Gold star/numbered badge with pulsating outline.
    - Locality level: Blue marker.
    - Street level: Green marker.
    - Specific commercial property: Orange/Red marker.
    """
    city_key = center_city.lower() if center_city else "bhopal"
    default_center = CITY_CENTERS.get(city_key, [22.0, 77.0])

    if not df_results.empty:
        center_lat = df_results["latitude"].mean()
        center_lon = df_results["longitude"].mean()
    else:
        center_lat, center_lon = default_center

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
    )

    # Feature groups for toggleable layers
    fg_top = folium.FeatureGroup(name="Top 5 Recommendations (Highlighted)")
    fg_localities = folium.FeatureGroup(name="Commercial Localities")
    fg_streets = folium.FeatureGroup(name="Commercial Streets")
    fg_properties = folium.FeatureGroup(name="Commercial Properties")
    fg_catchment = folium.FeatureGroup(name="500m Catchment Buffers (Top Picks)")

    for idx, row in df_results.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        rank = row.get("rank", idx + 1)
        score = row.get("suitability_score", 0.0)
        rent = row.get("avg_monthly_rent", 0.0)
        level = row.get("location_level", "locality").lower()
        locality = row.get("locality", "Unknown")
        street = row.get("street_name", "")
        cluster = row.get("cluster_label", "Standard Commercial Zone")
        budget_status = row.get("budget_status", "N/A")

        # HTML popup card
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 200px; padding: 4px;">
            <div style="font-size: 14px; font-weight: bold; color: #1E3A8A; margin-bottom: 4px;">
                #{rank} - {locality}
            </div>
            <div style="font-size: 11px; color: #555; margin-bottom: 6px;">{street}</div>
            <hr style="margin: 4px 0; border: 0; border-top: 1px solid #ddd;">
            <table style="width: 100%; font-size: 12px; line-height: 1.5;">
                <tr><td><b>Suitability:</b></td><td><span style="color:#059669; font-weight:bold;">{score}/100</span></td></tr>
                <tr><td><b>Avg Monthly Rent:</b></td><td>INR {rent:,.0f}</td></tr>
                <tr><td><b>Level:</b></td><td><span style="text-transform: capitalize;">{level}</span></td></tr>
                <tr><td><b>Budget:</b></td><td>{budget_status}</td></tr>
                <tr><td><b>Zone Profile:</b></td><td style="font-size: 10px;">{cluster}</td></tr>
            </table>
        </div>
        """
        popup = folium.Popup(popup_html, max_width=300)

        # Marker appearance
        if rank <= highlight_top_n:
            # Highlight top candidates with distinct gold/crimson badge
            folium.Marker(
                location=[lat, lon],
                popup=popup,
                tooltip=f"Rank #{rank}: {locality} (Score: {score})",
                icon=folium.Icon(color="red", icon="star", prefix="fa"),
            ).add_to(fg_top)

            # Add catchment buffer circle (500m radius)
            folium.Circle(
                location=[lat, lon],
                radius=500,
                color="#EF4444",
                weight=1.5,
                fill=True,
                fill_color="#EF4444",
                fill_opacity=0.12,
                tooltip=f"500m Catchment: {locality}",
            ).add_to(fg_catchment)

        elif level == "property":
            folium.Marker(
                location=[lat, lon],
                popup=popup,
                tooltip=f"Property: {locality}",
                icon=folium.Icon(color="orange", icon="home", prefix="fa"),
            ).add_to(fg_properties)

        elif level == "street":
            folium.Marker(
                location=[lat, lon],
                popup=popup,
                tooltip=f"Street: {locality}",
                icon=folium.Icon(color="green", icon="road", prefix="fa"),
            ).add_to(fg_streets)

        else:
            folium.Marker(
                location=[lat, lon],
                popup=popup,
                tooltip=f"Locality: {locality}",
                icon=folium.Icon(color="blue", icon="building", prefix="fa"),
            ).add_to(fg_localities)

    fg_top.add_to(m)
    fg_localities.add_to(m)
    fg_streets.add_to(m)
    fg_properties.add_to(m)
    fg_catchment.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
