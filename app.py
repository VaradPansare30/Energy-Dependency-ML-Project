import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib

# ===============================
# Load pretrained ML artifacts
# ===============================
scaler = joblib.load("energy_dependency_scaler.pkl")
kmeans = joblib.load("energy_dependency_kmeans.pkl")

# ===============================
# District coordinates (UI ONLY)
# ===============================
DISTRICT_COORDINATES = {
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
    'Pune': {'lat': 18.5204, 'lon': 73.8567},
    'Nagpur': {'lat': 21.1458, 'lon': 79.0882},
    'Nashik': {'lat': 20.0059, 'lon': 73.7907},
    'Aurangabad': {'lat': 19.8762, 'lon': 75.3433},
    'Amravati': {'lat': 20.9331, 'lon': 77.7520},
    'Akola': {'lat': 20.7039, 'lon': 77.0023},
    'Beed': {'lat': 18.9894, 'lon': 75.7564},
    'Satara': {'lat': 17.6805, 'lon': 74.0183},
    'Sangli': {'lat': 16.8524, 'lon': 74.5815},
    'Kolhapur': {'lat': 16.7050, 'lon': 74.2433},
    'Thane': {'lat': 19.2183, 'lon': 72.9781},
    'Palghar': {'lat': 19.6942, 'lon': 72.7654},
    'Raigad': {'lat': 18.5782, 'lon': 73.1207},
    'Ratnagiri': {'lat': 16.9944, 'lon': 73.3002},
    'Sindhudurg': {'lat': 16.1169, 'lon': 73.6669},
    'Dhule': {'lat': 20.9028, 'lon': 74.7773},
    'Nandurbar': {'lat': 21.3669, 'lon': 74.2409},
    'Jalgaon': {'lat': 21.0077, 'lon': 75.5626},
    'Ahmednagar': {'lat': 19.0952, 'lon': 74.7496},
    'Solapur': {'lat': 17.6599, 'lon': 75.9064},
    'Jalna': {'lat': 19.8345, 'lon': 75.8816},
    'Parbhani': {'lat': 19.2686, 'lon': 76.7709},
    'Hingoli': {'lat': 19.7155, 'lon': 77.1420},
    'Nanded': {'lat': 19.1383, 'lon': 77.3210},
    'Osmanabad': {'lat': 18.1667, 'lon': 76.0500},
    'Latur': {'lat': 18.4000, 'lon': 76.5833},
    'Buldhana': {'lat': 20.5333, 'lon': 76.1833},
    'Washim': {'lat': 20.1000, 'lon': 77.1500},
    'Yavatmal': {'lat': 20.4000, 'lon': 78.1333},
    'Wardha': {'lat': 20.7500, 'lon': 78.6167},
    'Chandrapur': {'lat': 19.9500, 'lon': 79.3000},
    'Gadchiroli': {'lat': 19.8000, 'lon': 80.2000},
    'Gondiya': {'lat': 21.4500, 'lon': 80.2000},
    'Bhandara': {'lat': 21.1667, 'lon': 79.6500},
    'Mumbai City': {'lat': 19.0760, 'lon': 72.8777},
    'Mumbai Suburban': {'lat': 19.0865, 'lon': 72.8744}
}

# ===============================
# Load base clustered dataset
# ===============================
base_df = pd.read_excel("maharashtra_energy_clustered_final.xlsx")

# Add coordinates dynamically
base_df["Latitude"] = base_df["area_name"].map(
    lambda x: DISTRICT_COORDINATES.get(x, {}).get("lat")
)
base_df["Longitude"] = base_df["area_name"].map(
    lambda x: DISTRICT_COORDINATES.get(x, {}).get("lon")
)

base_df.dropna(subset=["Latitude", "Longitude"], inplace=True)

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Maharashtra Energy Dependency",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Maharashtra Renewable vs Conventional Energy Dependency")

# ===============================
# Cluster → label mapping (MODEL DRIVEN)
# ===============================
cluster_means = kmeans.cluster_centers_.flatten()
sorted_idx = np.argsort(cluster_means)

CLUSTER_LABELS = {
    sorted_idx[0]: "Renewable Source Dependent",
    sorted_idx[1]: "Moderately Conventional Source Dependent",
    sorted_idx[2]: "Highly Conventional Source Dependent"
}

COLOR_MAP = {
    "Renewable Source Dependent": "#27ae60",
    "Moderately Conventional Source Dependent": "#f39c12",
    "Highly Conventional Source Dependent": "#e74c3c"
}

# ===============================
# Sidebar inputs
# ===============================
with st.sidebar:
    st.header("🔧 District Analysis")

    district = st.selectbox(
        "Select District",
        sorted(base_df["area_name"].unique())
    )

    solar = st.number_input("Solar (MW)", 0.0)
    wind = st.number_input("Wind (MW)", 0.0)
    biomass = st.number_input("Biomass (MW)", 0.0)
    hydro = st.number_input("Hydro (MW)", 0.0)
    grid = st.number_input("Grid Demand (MW)", min_value=1.0)

    analyze = st.button("⚡ Analyze District")

# ===============================
# Base map (ALWAYS visible, one-time)
# ===============================
base_df["point_size"] = 18

fig = px.scatter_mapbox(
    base_df,
    lat="Latitude",
    lon="Longitude",
    hover_name="area_name",
    color="dependency_category",
    color_discrete_map=COLOR_MAP,
    size="point_size",
    zoom=5,
    size_max=25
)

fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_center={"lat": 19.7515, "lon": 75.7139},
    margin={"r": 0, "t": 30, "l": 0, "b": 0}
)

# ===============================
# User prediction overlay (does NOT modify base map)
# ===============================
if analyze:
    # Capacity factors
    CF = {"solar": 0.2, "wind": 0.3, "biomass": 0.8, "hydro": 0.4}

    total_renewable = solar * CF["solar"] + wind * CF["wind"] + biomass * CF["biomass"] + hydro * CF["hydro"]
    total_usage = grid + total_renewable
    grid_ratio = grid / total_usage if total_usage > 0 else 1.0

    # Prepare input for model
    X = scaler.transform([[grid_ratio]])
    cluster = kmeans.predict(X)[0]
    category = CLUSTER_LABELS[cluster]

    st.success(f"🔍 {district} is **{category}**")

# ===============================
# Display base map (STATIC)
# ===============================
st.plotly_chart(fig, use_container_width=True)
