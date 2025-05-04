import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(layout="wide")
st.sidebar.title("📊 Dashboard Menu")
page = st.sidebar.radio("Select a page:", ["🌍 Global Map", "🌐 Country-Level Deep Analysis"])

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("owid-energy-data.csv")

df = load_data()

# Page 1 - World Map
if page == "🌍 Global Map":
    st.title("🌍 Global Energy Consumption per Capita")
    st.markdown("Measured in kilowatt-hours per person. Source: [Our World in Data](https://ourworldindata.org/energy)")

    df_map = df[["iso_code", "country", "year", "energy_per_capita"]].dropna()
    year = st.slider("Select Year", int(df_map["year"].min()), int(df_map["year"].max()), 2023)
    df_year = df_map[df_map["year"] == year]

    fig = px.choropleth(
        df_year,
        locations="iso_code",
        color="energy_per_capita",
        hover_name="country",
        color_continuous_scale=["#1A5319", "#508D4E", "#80AF81", "#D6EFD8"],
        labels={"energy_per_capita": "kWh / person"},
    )

    fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    st.plotly_chart(fig, use_container_width=True)

# Page 2 - Association Analysis
elif page == "🌐 Country-Level Deep Analysis":
    st.title("🔗 Energy Consumption Association Analysis")

    selected_countries = st.multiselect(
        "Select Countries",
        sorted(df["country"].dropna().unique()),
        default=["Turkey", "Germany", "United States", "France"]
    )

    threshold = st.slider("Binary Threshold (0–1 scale)", 0.1, 0.9, 0.3)
    min_support = st.slider("Minimum Support", 0.1, 1.0, 0.4)
    min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.0)

    if st.button("Run Analysis"):
        filtered_df = df[
            (df["country"].isin(selected_countries)) &
            (df["year"].between(1965, 2022))
        ].copy()

        energy_columns = [col for col in filtered_df.columns if 'consumption' in col and 'change' not in col]
        filtered_df = filtered_df[["country", "year"] + energy_columns].dropna()

        # Normalize
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(filtered_df[energy_columns])
        normalized_df = pd.DataFrame(normalized_data, columns=energy_columns)

        # Binary conversion
        binary_df = (normalized_df > threshold).astype(int)

        # Apply Apriori
        frequent_itemsets = apriori(binary_df, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
        rules_sorted = rules.sort_values(by=["lift", "confidence", "support"], ascending=False)

        if not rules_sorted.empty:
            st.success(f"{len(rules_sorted)} rules found.")
            st.dataframe(rules_sorted)
        else:
            st.warning("No rules found. Try adjusting the threshold, support, or country selection.")
