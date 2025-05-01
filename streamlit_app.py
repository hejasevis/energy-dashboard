import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.sidebar.title("📊 Dashboard Menü")
page = st.sidebar.radio("Bir sayfa seçin:", ["🌍 Dünya Haritası", "🔗 Association Kuralları"])

# Page 1: Global Map with hover & mini trend
if page == "🌍 World Map":
    st.title("🌍 Global Per Capita Energy Consumption")
    st.markdown("Source: [Our World in Data](https://ourworldindata.org/energy) – measured in kilowatt-hours (kWh) per person.")

    # Load data
    df = pd.read_csv("owid-energy-data.csv")
    df = df[["iso_code", "country", "year", "energy_per_capita"]].dropna()

    # Select year
    year = st.slider("Select Year", int(df["year"].min()), int(df["year"].max()), 2023)
    df_year = df[df["year"] == year]

    # Choropleth map
    fig = px.choropleth(
        df_year,
        locations="iso_code",
        color="energy_per_capita",
        hover_name="country",
        hover_data={"energy_per_capita": True, "iso_code": False, "year": True},
        color_continuous_scale=[
            "#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b",
            "#fd8d3c", "#f16913", "#d94801", "#a63603", "#7f2704"
        ],
        labels={"energy_per_capita": "kWh per person"}
    )

    fig.update_geos(
        showframe=False,
        showcoastlines=False,
        projection_type="natural earth"
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        height=700,
        coloraxis_colorbar=dict(
            title="Energy use<br>(kWh/person)",
            ticks="outside",
            tickvals=[0, 1000, 3000, 10000, 30000, 100000],
            ticktext=["0", "1k", "3k", "10k", "30k", "100k"]
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        geo_bgcolor='rgba(0,0,0,0)'
    )

    # Display map
    st.plotly_chart(fig, use_container_width=True)

    # Optional: mini line plot below
    st.markdown("### 📈 Country Energy Trend")
    selected_country = st.selectbox("Select a country to view historical trend", sorted(df["country"].unique()), index=0)
    country_data = df[df["country"] == selected_country]

    fig_line = px.line(
        country_data,
        x="year",
        y="energy_per_capita",
        labels={"year": "Year", "energy_per_capita": "kWh per person"},
        title=f"{selected_country} – Per Capita Energy Consumption Over Time"
    )

    fig_line.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig_line, use_container_width=True)


# Sayfa 2: Association Rules
elif page == "🔗 Association Kuralları":
    st.title("Enerji Tüketimi İlişkilendirme Kuralları")
    st.markdown("""
    Bu sayfada OWID verisiyle oluşturulmuş Apriori kurallarını filtreleyebilirsiniz.
    """)

    uploaded_file = st.file_uploader("Kurallar CSV dosyasını yükleyin (rules_sorted.csv)", type="csv")
    if uploaded_file:
        rules = pd.read_csv(uploaded_file)

        st.sidebar.subheader("🔍 Filtre Ayarları")
        min_lift = st.sidebar.slider("Min. Lift", 0.0, 5.0, 1.0, 0.1)
        min_confidence = st.sidebar.slider("Min. Confidence", 0.0, 1.0, 0.5, 0.05)
        min_support = st.sidebar.slider("Min. Support", 0.0, 1.0, 0.05, 0.01)

        filtered = rules[
            (rules["lift"] >= min_lift) &
            (rules["confidence"] >= min_confidence) &
            (rules["support"] >= min_support)
        ]

        st.subheader("Filtrelenmiş Kurallar")
        st.dataframe(filtered)
        st.success(f"{len(filtered)} kural bulundu.")
    else:
        st.warning("Lütfen 'rules_sorted.csv' dosyasını yükleyin.")
