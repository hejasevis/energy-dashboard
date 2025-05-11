import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules
from PIL import Image
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly


# Page setup
st.set_page_config(layout="wide")
from streamlit_option_menu import option_menu

from streamlit_option_menu import option_menu

from streamlit_option_menu import option_menu

with st.sidebar:
    page = option_menu(
        menu_title="Dashboard Menu",
        options=["🏠 Home", "🌍 Global Map", "🌐 Deep Analysis", "📈 Growth Rates", "⚖️ Country vs Energy Type","🔮 Energy Consumption Forecast"],
        icons=[""] * 6,
        default_index=0,
        styles={
            "icon": {"display": "none"}
        }
    )

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("owid-energy-data.csv")

df = load_data()

 # 🏠 Home Page
if page == "🏠 Home":
    
    st.image("images/b.png", use_container_width=True)

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
        color_continuous_scale=["#76c893", "#34a0a4","#1a759f","#1e6091", "#184e77"],
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
        height= 600,
        paper_bgcolor="#111111",
        plot_bgcolor="#1e1e1e",
        font=dict(color="white", size=12),
        geo_bgcolor="#1e1e1e"
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # 🌐 Page 2 - Country-Level Deep Analysis
elif page == "🌐 Deep Analysis":
    st.title("🔗 Energy Consumption Association Analysis")

    selected_countries = st.multiselect(
        "Select Countries",
        sorted(df["country"].dropna().unique()),
        default=["Turkey", "Germany", "United States", "France"]
    )

    threshold = st.slider("Binary Threshold (0–1 scale)", 0.1, 0.9, 0.3)
    min_support = st.slider("Minimum Support", 0.1, 1.0, 0.4)
    min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.0)

    year_range = st.slider("Select Year Range", 1965, 2023, (2000, 2022))

    if st.button("Run Analysis"):
        filtered_df = df[
            (df["country"].isin(selected_countries)) &
            (df["year"].between(year_range[0], year_range[1]))
        ].copy()

        energy_columns = [col for col in filtered_df.columns if 'consumption' in col and 'change' not in col]
        filtered_df = filtered_df[["country", "year"] + energy_columns].dropna()

        # Normalize
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(filtered_df[energy_columns])
        norm_df = pd.DataFrame(normalized, columns=energy_columns)

        # Binary
        binary_df = (norm_df > threshold).astype(int)

        # Apriori
        frequent_itemsets = apriori(binary_df, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
        rules_sorted = rules.sort_values(by=["lift", "confidence", "support"], ascending=False)

        # 📋 1. Association Rules Table
        st.subheader("📋 Association Rules")
        st.markdown(f"📅 Showing rules for **{year_range[0]}–{year_range[1]}**")
        st.dataframe(rules_sorted)

        # 🔥 2. Correlation Heatmap (Plotly)
        st.subheader("🔥 Correlation Heatmap")
        import plotly.figure_factory as ff

        corr = norm_df.corr()
        z = corr.values
        x = list(corr.columns)
        y = list(corr.index)

        fig_heatmap = ff.create_annotated_heatmap(
            z=z,
            x=x,
            y=y,
            annotation_text=[[f"{val:.2f}" for val in row] for row in z],
            colorscale="YlGnBu",
            showscale=True
        )

        fig_heatmap.update_layout(
            title=dict(
                text="Correlation Between Energy Types",
                y=1.0,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=16)
            ),
            font=dict(size=12),
            height=600,
            margin=dict(l=60, r=60, t=80, b=60),
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickangle=45, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10))
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        # 📊 3. Top 10 Rules by Support (Bar Chart)
        st.subheader("📊 Top 10 Rules by Support")

        if not rules_sorted.empty:
            top_support = rules_sorted.nlargest(10, 'support')
            bar_data = top_support[['antecedents', 'consequents', 'support']].copy()

            def format_set(s):
                return ", ".join(sorted(list(s)))

            bar_data['rule'] = bar_data.apply(
                lambda row: f"{format_set(row['antecedents'])} → {format_set(row['consequents'])}",
                axis=1
            )

            fig2 = px.bar(
                bar_data,
                x='rule',
                y='support',
                title=f"Top Rules by Support ({year_range[0]}–{year_range[1]})",
                text='support',
                color='support',
                color_continuous_scale='Blues'
            )

            fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig2.update_layout(
                xaxis_tickangle=30,
                xaxis_title="Rule",
                yaxis_title="Support",
                title_font_size=20,
                font=dict(size=12),
                height=600,
                margin=dict(l=60, r=60, t=60, b=200),  # ← bu doğru olan
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )

            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No rules to visualize. Try adjusting thresholds or year range.")
           

# 📈 Energy Growth Rates 
elif page == "📈 Growth Rates":
    st.title("📈 Energy Source Growth Analysis")
    st.markdown("Visualize **annual growth/change rates** of various energy sources for the World or selected countries.")

    # 📊 Veriyi yükle
    energy_cols = [col for col in df.columns if col.endswith("_consumption")]
    df_clean = df[["country", "year"] + energy_cols].dropna()

    # 🌍 Ülke seçimi
    countries = sorted(df_clean["country"].unique())
    countries.insert(0, "World")
    selected_country = st.selectbox("Select Country (or World):", countries)

    # 📆 Yıl aralığı seçimi
    country_df = df_clean[df_clean["country"] == selected_country]
    min_year = int(country_df["year"].min())
    max_year = int(country_df["year"].max())
    year_range = st.slider("Select Year Range:", min_year, max_year, (2010, 2022))
    filtered_df = country_df[(country_df["year"] >= year_range[0]) & (country_df["year"] <= year_range[1])].copy()

    # 🔢 Yıllık % değişim oranı hesapla
    for col in energy_cols:
        filtered_df[col + "_change_%"] = filtered_df[col].pct_change() * 100

    # ⚡ Enerji türü seçimi
    selected_sources = st.multiselect("Select Energy Sources:", energy_cols, default=energy_cols[:3])

    # 📈 Plotly grafiği oluştur
    st.markdown("### 📊 Annual Growth Rates by Source")
    fig = go.Figure()

    for col in selected_sources:
        fig.add_trace(go.Scatter(
            x=filtered_df["year"],
            y=filtered_df[col + "_change_%"],
            mode='lines+markers',
            name=col.replace("_consumption", "").title()
        ))

    fig.update_layout(
        title=f"{selected_country} – Annual Energy Consumption Growth Rates",
        xaxis_title="Year",
        yaxis_title="Change Rate (%)",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    
    # ⚖️ Country vs Energy Type"
elif page == "⚖️ Country vs Energy Type":
    st.title("⚖️ Country-Specific Energy Source Breakdown")
    st.markdown("Compare energy source consumption breakdown for a selected country by year or year range.")

    # Enerji kolonları
    energy_cols = [col for col in df.columns if col.endswith("_consumption")]
    df_energy = df[["country", "year"] + energy_cols].dropna()

    # Ülke seçimi
    country_list = sorted(df_energy["country"].unique())
    selected_country = st.selectbox("Select a Country:", country_list)

    # Yıl aralığı seçimi
    min_year = int(df_energy["year"].min())
    max_year = int(df_energy["year"].max())
    year_range = st.slider("Select Year Range:", min_year, max_year, (2020, 2022))

    # Filtrelenmiş veri
    country_data = df_energy[(df_energy["country"] == selected_country) & 
                             (df_energy["year"] >= year_range[0]) & 
                             (df_energy["year"] <= year_range[1])]

    # Enerji türü seçimi
    selected_energy = st.multiselect("Select Energy Sources to Compare:", energy_cols, default=energy_cols[:5])

    # Ortalama tüketim hesapla
    avg_data = country_data[selected_energy].mean().sort_values(ascending=False)
    avg_df = avg_data.reset_index()
    avg_df.columns = ["Energy Source", "Average Consumption"]

    # 🥧 Pie Chart
    st.markdown("### 🥧 Energy Type Share (Pie Chart)")
    fig_pie = px.pie(
        avg_df,
        names="Energy Source",
        values="Average Consumption",
        title=f"{selected_country} – Energy Type Share ({year_range[0]}–{year_range[1]})",
        hole=0.3
    )
    fig_pie.update_layout(template="plotly_white")
    st.plotly_chart(fig_pie, use_container_width=True)

    # 📊 Bar Chart – SONRA
    st.markdown("### 📊 Average Energy Consumption (Bar Chart)")
    fig_bar = px.bar(
        avg_df,
        x="Energy Source",
        y="Average Consumption",
        text="Average Consumption",
        title=f"{selected_country} – Average Consumption ({year_range[0]}–{year_range[1]})",
        labels={"Average Consumption": "kWh"},
        color="Average Consumption",
        color_continuous_scale="Tealgrn"
    )
    fig_bar.update_layout(
        xaxis_tickangle=30,
        height=600,
        template="plotly_white"
    )
    fig_bar.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)
        # 📋 Otomatik Yorumlama
    st.markdown("### 🧠 Automatic Insights")

    total = avg_df["Average Consumption"].sum()
    avg_df["Percentage"] = (avg_df["Average Consumption"] / total * 100).round(2)

    top_row = avg_df.iloc[0]
    bottom_row = avg_df.iloc[-1]

    st.markdown(f"""
    - **Most used energy source:** `{top_row['Energy Source'].replace('_consumption', '').title()}` with **{top_row['Percentage']}%**
    - **Least used energy source:** `{bottom_row['Energy Source'].replace('_consumption', '').title()}` with **{bottom_row['Percentage']}%**
    - Total consumption (for selected sources and years): **{total:,.0f} kWh**
    """)

    # 👀 Detaylı oranlar listesi
    with st.expander("🔍 See Full Share Breakdown"):
        for _, row in avg_df.iterrows():
            st.markdown(f"- `{row['Energy Source'].replace('_consumption', '').title()}`: **{row['Percentage']}%**")


    # 🔮 Energy Consumption Forecast
elif page == "🔮 Energy Consumption Forecast":
    st.title("🔮 Forecasting Energy Consumption")
    st.markdown("Predict future consumption for a selected country and energy source using time series modeling (Prophet).")

    try:
        from prophet import Prophet
        from prophet.plot import plot_plotly
    except ImportError:
        st.error("❌ Prophet is not installed. Please add `prophet` to your requirements.txt file.")

    # Enerji tüketim sütunlarını al
    energy_cols = [col for col in df.columns if col.endswith("_consumption")]
    df_forecast = df[["country", "year"] + energy_cols].dropna()

    # Ülke ve enerji türü seçimi
    countries = sorted(df_forecast["country"].unique())
    selected_country = st.selectbox("🌍 Select a Country:", countries)

    selected_source = st.selectbox("⚡ Select Energy Type:", energy_cols)

    # Seçilen ülke ve kaynak için veri hazırlığı
    country_data = df_forecast[df_forecast["country"] == selected_country][["year", selected_source]].copy()
    country_data = country_data.dropna()

    if country_data.empty:
        st.warning("No data available for this selection.")
    else:
        # Prophet formatı: ds (date), y (value)
        country_data.columns = ["ds", "y"]
        country_data["ds"] = pd.to_datetime(country_data["ds"], format="%Y")

        # Prophet modeli
        model = Prophet(yearly_seasonality=True)
        model.fit(country_data)

        # Kullanıcıdan tahmin yılı sayısı
        future_years = st.slider("🗓️ Years to Predict:", 1, 20, 5)

        future = model.make_future_dataframe(periods=future_years, freq="Y")
        forecast = model.predict(future)

        # Tahmin grafiği
        st.markdown("### 📈 Forecast Plot")
        fig1 = plot_plotly(model, forecast)
        fig1.update_layout(
            height=600,
            title=f"{selected_country} – Forecast of {selected_source.replace('_consumption', '').title()} Consumption"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Tahmin tablosu
        st.markdown("### 📋 Forecasted Values")
        forecast_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(future_years)
        forecast_display.columns = ["Year", "Prediction", "Lower Bound", "Upper Bound"]
        forecast_display["Year"] = forecast_display["Year"].dt.year
        st.dataframe(forecast_display)
        
        # 🧠 Yorum
        st.markdown("### 🧠 Forecast Interpretation")

        # Yıl bazlı fark hesapla
        future_diff = forecast_display["Prediction"].diff().dropna()
        avg_growth = future_diff.mean()
        trend = "increasing" if avg_growth > 0 else "decreasing"
        direction_arrow = "📈" if avg_growth > 0 else "📉"

        # Ortalama büyüme yüzdesi
        first_val = forecast_display["Prediction"].iloc[0]
        last_val = forecast_display["Prediction"].iloc[-1]
        growth_percent = ((last_val - first_val) / first_val) * 100 if first_val != 0 else 0

        # Güven aralığı yorumu
        forecast_display["uncertainty"] = forecast_display["Upper Bound"] - forecast_display["Lower Bound"]
        avg_uncertainty = forecast_display["uncertainty"].mean()

        st.markdown(f"""
        - {direction_arrow} **The predicted trend is {trend}.**
        - The average yearly change is approximately **{avg_growth:,.0f} kWh**.
        - From {forecast_display['Year'].iloc[0]} to {forecast_display['Year'].iloc[-1]}, the predicted consumption changes by **{growth_percent:.2f}%**.
        - The average uncertainty in prediction is around **±{avg_uncertainty:,.0f} kWh**, which indicates {"high" if avg_uncertainty > first_val * 0.3 else "reasonable"} model confidence.
        """)

        st.caption("📘 This summary is generated automatically based on Prophet model outputs.")







