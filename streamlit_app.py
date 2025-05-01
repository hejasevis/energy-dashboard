import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.sidebar.title("📊 Dashboard Menü")
page = st.sidebar.radio("Bir sayfa seçin:", ["🌍 Dünya Haritası", "🔗 Association Kuralları"])

# Page 1 
if page == "🌍 World Map":
    st.title("🌍 Global Energy Use")
    st.markdown("Measured in kilowatt-hours per person.")

    df = pd.read_csv("owid-energy-data.csv")
    df_map = df[["iso_code", "country", "year", "energy_per_capita"]].dropna()

    year = st.slider("Select Year", int(df_map["year"].min()), int(df_map["year"].max()), 2023)
    df_year = df_map[df_map["year"] == year]

    fig = px.choropleth(
        df_year,
        locations="iso_code",
        color="energy_per_capita",
        hover_name="country",
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={"energy_per_capita": "kWh / person"},
    )

    fig.update_geos(
        showframe=False,
        showcoastlines=False,
        projection_type="natural earth"
    )
    fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',   # graphics area transparent
    geo_bgcolor='rgba(0,0,0,0)'      # Map background is transparent
    )

    st.plotly_chart(fig, use_container_width=True)
    
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
