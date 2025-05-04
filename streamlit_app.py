import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules

# Page setup
st.set_page_config(layout="wide")
st.sidebar.title("📊 Dashboard Menu")
page = st.sidebar.radio("Select a page:", ["🏠 Home", "🌍 Global Map", "🌐 Country-Level Deep Analysis"])

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("owid-energy-data.csv")

df = load_data()

# 🏠 Home Page
if page == "🏠 Home":
    st.title("🔌 Global Energy Dashboard")
    st.markdown("This interactive dashboard visualizes global energy consumption data from [Our World in Data](https://ourworldindata.org/energy).")

    st.markdown("### 📊 Features:")
    st.markdown("- 🌍 **Global Map**: Explore per capita energy consumption by country and year.")
    st.markdown("- 🌐 **Country-Level Analysis**: Discover hidden associations between different energy types with support, confidence, and lift metrics.")
    st.markdown("- 🔥 **Heatmaps & Rules**: Visualize energy consumption correlations and strongest association rules.")

    st.markdown("### 📝 How to Use:")
    st.markdown("Select a page from the sidebar to start exploring the data.")

    st.markdown("---")
    st.info("This dashboard is developed as part of a Bachelor's Graduation Project in Computer Engineering.")

# 🌍 Page 1 - Global Map
elif page == "🌍 Global Map":
    st.title("🌍 Global Energy Consumption per Capita")
    st.markdown("Measured in kilowatt-hours per person. Source: [Our World in Data](https://ourworldindata.org/energy)")

    st.markdown("### 📅 Year Selection")
    df_map = df[["iso_code", "country", "year", "energy_per_capita"]].dropna()
    year = st.slider("Select Year", int(df_map["year"].min()), int(df_map["year"].max()), 2023)
    df_year = df_map[df_map["year"] == year]

    country_list = sorted(df_year["country"].unique())
    selected_country = st.selectbox("🌎 Select a Country to View Details", country_list)
    selected_row = df_year[df_year["country"] == selected_country].iloc[0]

    st.markdown(f"#### 📄 Details for {selected_country} ({year})")
    st.markdown(f"- 📊 Energy per Capita: **{selected_row['energy_per_capita']:.2f} kWh/person**")
    st.markdown(f"- 📆 Year: **{selected_row['year']}**")
    st.markdown("---")
    st.markdown(f"Currently showing energy use for the year **{year}**")

    fig = px.choropleth(
        df_year,
        locations="iso_code",
        color="energy_per_capita",
        hover_name="country",
        color_continuous_scale=["#1A5319", "#508D4E", "#80AF81", "#D6EFD8"],
        labels={"energy_per_capita": "kWh / person"},
        title=f"Per Capita Energy Consumption ({year})"
    )

    fig.update_geos(
        showframe=False,
        showcoastlines=False,
        projection_type="natural earth"
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="#111111",
        plot_bgcolor="#1e1e1e",
        font=dict(color="white", size=12),
        geo_bgcolor="#1e1e1e"
    )

    st.plotly_chart(fig, use_container_width=True)
