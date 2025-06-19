# Core Libraries
import streamlit as st
import pandas as pd
import numpy as np

# Visualization
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns

# Data Processing & Machine Learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Association Rule Mining
from mlxtend.frequent_patterns import apriori, association_rules

# Time Series Forecasting
from prophet import Prophet
from prophet.plot import plot_plotly

# Image Processing
from PIL import Image

# Streamlit Option Menu
from streamlit_option_menu import option_menu

# Page Configuration
st.set_page_config(layout="wide")

# Sidebar Navigation Menu
with st.sidebar:
    page = option_menu(
        menu_title="Menu",
        options=[
            "Home",
            "Global Energy Map",
            "Energy Relationships",
            "Growth Rate Trends",
            "Country Energy Mix",
            "Future Energy Forecast",
        ],
        icons=["house", "globe", "bar-chart-line", "graph-up-arrow", "pie-chart", "stars"],
        default_index=0,
    )

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("owid-energy-data.csv")

df = load_data()

# Home
if page == "Home":

    st.image("images/background.png", use_container_width=True)

    st.title("🔌 Global Energy Insights: Tracing the Origins and Outcomes of the World’s Energy Dependence ")
    st.markdown("""
        Welcome to the **Global Energy Insights Dashboard**, an interactive platform designed to visualize, analyze, and forecast worldwide energy consumption patterns.  
        This project leverages data from [Our World in Data](https://ourworldindata.org/energy).
    """)

    st.markdown("### 🔍 Dashboard Overview")
    st.markdown("""
    - 🗺️ **Global Map**  
      View energy consumption per capita by country over time.

    - 🌐 **Deep Analysis**  
      Uncover meaningful associations between different energy types using Apriori-based rule mining.

    - 📈 **Growth Rates**  
      Analyze year-over-year changes and long-term trends in energy use.

    - 🗺 **Country vs Energy Type**  
      Compare countries based on specific energy type consumption and production.

    - 🔮 **Forecasting Module**  
      Predict future energy demand using machine learning models (Prophet & Random Forest).
    """)

    st.markdown("### 🧭 How to Navigate")
    st.markdown("""
        Use the sidebar on the left to select different dashboard sections.  
        Each section presents a specific aspect of the dataset, allowing you to explore patterns and predictions.
    """)

    st.markdown("---")
    st.info("📌 *This dashboard is developed using Python, Streamlit, and machine learning tools as part of a final year project.*")


# Page 1 -  Global Energy Map
elif page == "Global Energy Map":

    # Page Title and Introductory Info
    st.title("🗺️ Global Map of Energy Use per Capita")

    # Helper info box to guide users
    st.info("""
    This map illustrates per capita energy consumption (in kilowatt-hours per person) 
    across countries for a selected year.  
    Use the **slider** to pick a year and the **dropdown** to explore a specific country's values.
    """)

    # Year selection section for filtering the map data
    st.markdown("### 📅 Year Selection")

    # Select relevant columns and remove rows with missing values
    df_map = df[["iso_code", "country", "year", "energy_per_capita", "population"]].dropna()

    # Slider to pick a specific year from the available dataset
    year = st.slider("Select Year", int(df_map["year"].min()), int(df_map["year"].max()), 2023)

    # Filter the data based on selected year
    df_year = df_map[df_map["year"] == year]

    # Country selection for displaying specific data
    country_list = sorted(df_year["country"].unique())
    selected_country = st.selectbox("🌎 Select a Country to View Details", country_list)

    # Get data for the selected country
    selected_row = df_year[df_year["country"] == selected_country].iloc[0]

    # Choropleth Map
    # This visualizes per capita energy consumption using a colored world map
    fig = px.choropleth(
        df_year,
        locations="iso_code",
        color="energy_per_capita",
        hover_name="country",
        color_continuous_scale=["#76c893", "#34a0a4", "#1a759f", "#1e6091", "#184e77"],
        labels={"energy_per_capita": "kWh / person"},
        title=f"Per Capita Energy Consumption ({year})"
    )

    # Map style configuration - hides borders and sets projection
    fig.update_geos(
        showframe=False,
        showcoastlines=False,
        projection_type="natural earth"
    )

    # Layout and Theme
    fig.update_layout(
    template="plotly_white",  # Light mode styling
    margin=dict(l=0, r=0, t=60, b=0),
    height=600,
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="black", size=12),
    geo_bgcolor="white"
    )


    # Render the map
    st.plotly_chart(fig, use_container_width=True)

    # Additional analysis and country-level metrics

    # Calculate global average for comparison
    global_avg = df_year["energy_per_capita"].mean()
    diff = selected_row["energy_per_capita"] - global_avg
    comparison = "above" if diff > 0 else "below"

    # Calculate total energy consumption
    population = selected_row["population"]
    total_energy = selected_row["energy_per_capita"] * population

    # Display enhanced info box
    st.info(f"""
    #### 📄 Details for {selected_country} ({year})
    - 📊 Energy per Capita: **{selected_row['energy_per_capita']:.2f} kWh/person**
    - 👥 Population: **{population:,.0f}**
    - 🔋 Estimated Total Energy Consumption: **{total_energy:,.0f} kWh**
    - 🌍 Global Average (same year): **{global_avg:.2f} kWh/person**
    - 🔎 This is **{abs(diff):.2f} kWh/person {comparison}** the global average.
    - 📆 Year: **{selected_row['year']}**
    """)

  
# Page 2 - Energy Relationships 
elif page == "Energy Relationships":

    # Page Title and Overview
    st.title("🌐 Country-Level Energy Pattern Discovery")

    # Explain the purpose of the page
    st.info("""
    This section applies **association rule mining** and **correlation analysis** to uncover hidden relationships 
    between energy consumption types in selected countries and years.  
    You can adjust thresholds, year range, and countries to explore different patterns.
    """)

    # User Inputs: countries, thresholds, year range
    selected_countries = st.multiselect(
        "Select Countries",
        sorted(df["country"].dropna().unique()),
        default=["Turkey", "Germany", "United States", "France"]
    )

    threshold = st.slider("Binary Threshold (0–1 scale)", 0.1, 0.9, 0.3)
    min_support = st.slider("Minimum Support", 0.1, 1.0, 0.4)
    min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.0)
    year_range = st.slider("Select Year Range", 1965, 2023, (2000, 2022))

    # Trigger analysis with button
    if st.button("Run Analysis"):

        # Filter the dataset based on user selection
        filtered_df = df[
            (df["country"].isin(selected_countries)) &
            (df["year"].between(year_range[0], year_range[1]))
        ].copy()

        # Select only consumption-related columns, drop others
        energy_columns = [col for col in filtered_df.columns if 'consumption' in col and 'change' not in col]
        filtered_df = filtered_df[["country", "year"] + energy_columns].dropna()

        # Normalize data between 0–1
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(filtered_df[energy_columns])
        norm_df = pd.DataFrame(normalized, columns=energy_columns)

        # Binarize based on selected threshold
        binary_df = (norm_df > threshold).astype(int)

        # Apply Apriori algorithm to discover frequent itemsets
        frequent_itemsets = apriori(binary_df, min_support=min_support, use_colnames=True)

        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
        rules_sorted = rules.sort_values(by=["lift", "confidence", "support"], ascending=False)

        # 1. Association Rules Table
        st.subheader("📋 Association Rules")
        st.markdown(f"📅 Showing rules for **{year_range[0]}–{year_range[1]}**")
        st.dataframe(rules_sorted)

        # 2. Correlation Heatmap
        st.subheader("⚡️ Correlation Heatmap")
        st.markdown("This heatmap shows normalized Pearson correlations between different energy consumption types.")

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

        # 3. Top 10 Rules by Support
        st.subheader("📊 Top 10 Rules by Support")
        st.markdown("Shows the rules with the highest support values, indicating common energy patterns.")

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
                margin=dict(l=60, r=60, t=60, b=200),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )

            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No rules to visualize. Try adjusting thresholds or year range.")

        # 4. Insights
        st.markdown("### 🔍 Insights")
        if not rules_sorted.empty:
            avg_support = rules_sorted['support'].mean()
            avg_lift = rules_sorted['lift'].mean()
            avg_confidence = rules_sorted['confidence'].mean()

            st.markdown(f"""
            - **Average Support:** {avg_support:.2f}  
            - **Average Lift:** {avg_lift:.2f}  
            - **Average Confidence:** {avg_confidence:.2f}
            """)
        else:
            st.info("No insights available due to lack of valid rules.")

          
# Page 3 - Growth Rate Trends
elif page == "Growth Rate Trends":

    # Page Title
    st.title("📈 Annual Growth Trends in Energy Consumption")

    # Info box to explain this section
    st.info("""
    This section visualizes **annual growth rates** in energy consumption for various sources.  
    You can select a specific country or view global trends, and filter by year range and energy types.
    """)

    # Select energy consumption columns (e.g. coal_consumption, solar_consumption)
    energy_cols = [col for col in df.columns if col.endswith("_consumption")]

    # Drop rows with missing values in relevant columns
    df_clean = df[["country", "year"] + energy_cols].dropna()

    # Country selection
    countries = sorted(df_clean["country"].unique())
    countries.insert(0, "World")  # Add 'World' to allow global analysis
    selected_country = st.selectbox("Select Country (or World):", countries)

    # Year range selection
    country_df = df_clean[df_clean["country"] == selected_country]
    min_year = int(country_df["year"].min())
    max_year = int(country_df["year"].max())
    year_range = st.slider("Select Year Range:", min_year, max_year, (2010, 2022))

    # Filter data for selected year range
    filtered_df = country_df[
        (country_df["year"] >= year_range[0]) &
        (country_df["year"] <= year_range[1])
    ].copy()

    # Calculate annual % change for each energy type
    for col in energy_cols:
        filtered_df[col + "_change_%"] = filtered_df[col].pct_change() * 100

    # Energy source selection
    selected_sources = st.multiselect(
        "Select Energy Sources:",
        energy_cols,
        default=energy_cols[:3]
    )

    # Plotly line chart
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

    # Insight Section
    st.markdown("### 💡 Insights")

    # Calculate average growth rate for each selected source
    insight_df = filtered_df[[col + "_change_%" for col in selected_sources]].mean().sort_values(ascending=False)
    insight_df = insight_df.reset_index()
    insight_df.columns = ["Energy Source", "Average Annual Growth (%)"]
    insight_df["Energy Source"] = insight_df["Energy Source"].str.replace("_consumption_change_%", "").str.title()

    # Top and bottom performing sources
    top = insight_df.iloc[0]
    bottom = insight_df.iloc[-1]

    st.markdown(f"""
    - **Highest average growth:** `{top['Energy Source']}` with **{top['Average Annual Growth (%)']:.2f}%**
    - **Lowest (or negative) average growth:** `{bottom['Energy Source']}` with **{bottom['Average Annual Growth (%)']:.2f}%**
    """)

    # Optional: Show full table in expander
    with st.expander("🔍 See All Source Growth Averages"):
        st.dataframe(insight_df)

    
# Page 4 - Country Energy Mix
elif page == "Country Energy Mix":

    # Title and Info Box
    st.title("🗺 Energy Mix Analysis by Country and Year")
    st.info("""
    This section compares how different energy sources contribute to total energy consumption for a selected country.  
    You can analyze energy share based on a specific year range and focus on selected types such as fossil, nuclear, or renewables.
    """)

    # Prepare column groups
    energy_cols = [col for col in df.columns if col.endswith("_consumption")]
    renewable_cols = [
        "solar_consumption", "wind_consumption", "biofuel_consumption",
        "hydro_consumption", "other_renewable_consumption"
    ]
    non_renewable_cols = [
        "coal_consumption", "oil_consumption", "gas_consumption", "nuclear_consumption"
    ]

    # Drop rows with missing energy values
    df_energy = df[["country", "year"] + energy_cols].dropna()

    # Country selection
    country_list = sorted(df_energy["country"].unique())
    selected_country = st.selectbox("Select a Country:", country_list)

    # Year range selection
    min_year = int(df_energy["year"].min())
    max_year = int(df_energy["year"].max())
    year_range = st.slider("Select Year Range:", min_year, max_year, (2020, 2022))

    # Filter dataset by country and year
    country_data = df_energy[
        (df_energy["country"] == selected_country) &
        (df_energy["year"] >= year_range[0]) &
        (df_energy["year"] <= year_range[1])
    ]

    # Energy source selection
    selected_energy = st.multiselect(
        "Select Energy Sources to Compare:",
        energy_cols,
        default=energy_cols[:5]
    )

    if not selected_energy:
        st.warning("Please select at least one energy source to display the energy mix.")
        st.stop()

    # Average consumption for selected energy types
    avg_data = country_data[selected_energy].mean().sort_values(ascending=False)
    avg_df = avg_data.reset_index()
    avg_df.columns = ["Energy Source", "Average Consumption"]

    # Pie Chart
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

    # Insight Section
    st.markdown("### ⚡ Insights")

    total = avg_df["Average Consumption"].sum()
    avg_df["Percentage"] = (avg_df["Average Consumption"] / total * 100).round(2)

    top_row = avg_df.iloc[0]
    bottom_row = avg_df.iloc[-1]

    st.markdown(f"""
    - **Most used energy source:** `{top_row['Energy Source'].replace('_consumption', '').title()}` with **{top_row['Percentage']}%**
    - **Least used energy source:** `{bottom_row['Energy Source'].replace('_consumption', '').title()}` with **{bottom_row['Percentage']}%**
    - Total consumption (for selected sources and years): **{total:,.0f} kWh**
    """)

    # Expandable full breakdown
    with st.expander("🔍 See Full Share Breakdown"):
        for _, row in avg_df.iterrows():
            st.markdown(f"- `{row['Energy Source'].replace('_consumption', '').title()}`: **{row['Percentage']}%**")

    # Additional Indicator: Renewable Ratio
    st.markdown("### Renewable Energy Share")

    # Calculate renewable and non-renewable sums
    renew_sum = country_data[renewable_cols].sum().sum() if set(renewable_cols).issubset(country_data.columns) else 0
    non_renew_sum = country_data[non_renewable_cols].sum().sum() if set(non_renewable_cols).issubset(country_data.columns) else 0

    if renew_sum + non_renew_sum > 0:
        renewable_ratio = (renew_sum / (renew_sum + non_renew_sum)) * 100
        st.success(f"🔋 **Estimated Renewable Share:** {renewable_ratio:.2f}% of total energy consumption")
    else:
        st.warning("Renewable/non-renewable data not sufficient to calculate ratio.")


# Future Energy Forecast
elif page == "Future Energy Forecast":
    st.title("🔮 Future Energy Forecast with Machine Learning")
    st.info("""
    Forecast future energy consumption for a selected country and energy type using Prophet and Random Forest models.
    This tool uses historical consumption from 1965 to 2022 and projects up to 20 years forward.
    """)

    # Select country and energy type
    energy_cols = [col for col in df.columns if col.endswith("_consumption")]
    countries = sorted(df["country"].dropna().unique())

    selected_country = st.selectbox("🌍 Select a Country:", countries)
    selected_source = st.selectbox("⚡ Select Energy Type:", energy_cols)
    future_years = st.slider("🗓️ Years to Predict:", 1, 20, 5)

    # Filter data
    country_data = df[(df["country"] == selected_country)][["year", selected_source]].dropna().sort_values("year")
    if len(country_data) < 10:
        st.warning("⚠️ Not enough valid data points for this country and energy type.")
        st.stop()

    ### --- PROPHET --- ###
    st.subheader("📈 Prophet Forecast")
    prophet_df = country_data.rename(columns={"year": "ds", selected_source: "y"})
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")

    prophet_model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
    )
    prophet_model.fit(prophet_df)

    future_df = prophet_model.make_future_dataframe(periods=future_years, freq="YS")
    forecast = prophet_model.predict(future_df)

    prophet_plot = go.Figure()
    prophet_plot.add_trace(go.Scatter(x=country_data["year"], y=country_data[selected_source], mode="lines+markers", name="Historical"))
    prophet_plot.add_trace(go.Scatter(x=forecast["ds"].dt.year.tail(future_years), y=forecast["yhat"].tail(future_years),
                                     mode="lines+markers", name="Prophet Prediction", line=dict(color="blue")))
    prophet_plot.update_layout(title="Prophet Forecast", xaxis_title="Year", yaxis_title="Predicted Consumption", template="plotly_white")
    st.plotly_chart(prophet_plot)

    ### --- RANDOM FOREST --- ###
    st.subheader("🌲 Random Forest Forecast")
    rf_df = country_data.copy().reset_index(drop=True)
    rf_df["year_scaled"] = rf_df["year"] - rf_df["year"].min()
    # create lagged features and model year-over-year change
    rf_df["lag1"] = rf_df[selected_source].shift(1)
    rf_df["lag2"] = rf_df[selected_source].shift(2)
    rf_df["lag3"] = rf_df[selected_source].shift(3)
    rf_df["diff"] = rf_df[selected_source].diff()
    rf_df = rf_df.dropna()

    X_rf = rf_df[["year_scaled", "lag1", "lag2", "lag3"]]
    y_rf = rf_df["diff"]

    rf_model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    rf_model.fit(X_rf, y_rf)

    last_year = int(rf_df["year"].max())
    future_years_rf = list(range(last_year + 1, last_year + future_years + 1))
    current_year_scaled = rf_df["year_scaled"].iloc[-1]

    history = rf_df[selected_source].tolist()[-3:]
    last_value = history[-1]
    predictions_rf = []
    # forecast year-over-year changes and accumulate
    for _ in future_years_rf:
        current_year_scaled += 1
        features = pd.DataFrame({
            "year_scaled": [current_year_scaled],
            "lag1": [history[-1]],
            "lag2": [history[-2]],
            "lag3": [history[-3]],
        })
        diff_pred = rf_model.predict(features)[0]
        next_val = last_value + diff_pred
        predictions_rf.append(next_val)
        history.append(next_val)
        last_value = next_val

    rf_plot = go.Figure()
    rf_plot.add_trace(go.Scatter(x=future_years_rf, y=predictions_rf, mode="lines+markers", name="RF Prediction", line=dict(color="green")))
    rf_plot.update_layout(title="Random Forest Forecast", xaxis_title="Year", yaxis_title="Predicted Consumption", template="plotly_white")
    st.plotly_chart(rf_plot)

    ### --- COMPARISON --- ###
    st.subheader("🔍 Prophet vs Random Forest Forecast Comparison")
    forecast_display = forecast[["ds", "yhat"]].tail(future_years).copy()
    forecast_display["Year"] = forecast_display["ds"].dt.year

    comparison_df = pd.DataFrame({
        "Year": future_years_rf,
        "Prophet_Prediction": forecast_display["yhat"].values,
        "RF_Prediction": predictions_rf
    })
    st.dataframe(comparison_df)

    ### --- BACKTESTING --- ###
    st.subheader("🧪 Backtesting: Model Accuracy")
    max_year = int(country_data["year"].max())
    split_year = st.slider(
        "📆 Select Last Training Year:",
        min_value=1965,
        max_value=2015,
        value=min(2015, max_year - future_years),
    )

    test_years = list(range(split_year + 1, split_year + future_years + 1))
    df_train = country_data[country_data["year"] <= split_year]
    df_test = country_data[country_data["year"].isin(test_years)]

    if len(df_test) < future_years:
        st.warning("⚠️ Not enough data for testing period.")
    else:
        # Prophet backtest
        prophet_bt = df_train.rename(columns={"year": "ds", selected_source: "y"})
        prophet_bt["ds"] = pd.to_datetime(prophet_bt["ds"], format="%Y")
        model_prophet = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
        )
        model_prophet.fit(prophet_bt)
        future_bt = model_prophet.make_future_dataframe(periods=future_years, freq="YS")
        forecast_bt = model_prophet.predict(future_bt)
        prophet_preds = forecast_bt[["ds", "yhat"]].tail(future_years)
        prophet_preds["year"] = prophet_preds["ds"].dt.year

        # RF backtest
        lagged = country_data.copy()
        lagged["year_scaled"] = lagged["year"] - lagged["year"].min()
        lagged["lag1"] = lagged[selected_source].shift(1)
        lagged["lag2"] = lagged[selected_source].shift(2)
        lagged["lag3"] = lagged[selected_source].shift(3)
        lagged["diff"] = lagged[selected_source].diff()
        lagged = lagged.dropna()

        train_df = lagged[lagged["year"] <= split_year]
        test_df = lagged[lagged["year"].isin(test_years)]

        rf_back = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
        rf_back.fit(train_df[["year_scaled", "lag1", "lag2", "lag3"]], train_df["diff"])

        diff_preds = rf_back.predict(test_df[["year_scaled", "lag1", "lag2", "lag3"]])
        last_known = country_data[country_data["year"] == split_year][selected_source].values[0]
        rf_preds = []
        # accumulate predicted differences to reconstruct levels
        for d in diff_preds:
            last_known += d
            rf_preds.append(last_known)

        df_compare = pd.DataFrame({
            "Year": test_years,
            "Actual": df_test[selected_source].values,
            "Prophet_Prediction": prophet_preds["yhat"].values,
            "RF_Prediction": rf_preds
        })

        rmse_prophet = np.sqrt(mean_squared_error(df_compare["Actual"], df_compare["Prophet_Prediction"]))
        rmse_rf = np.sqrt(mean_squared_error(df_compare["Actual"], df_compare["RF_Prediction"]))

        st.dataframe(df_compare)
        st.markdown(f"📉 **Prophet RMSE:** {rmse_prophet:.2f}")
        st.markdown(f"🌲 **Random Forest RMSE:** {rmse_rf:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_compare["Year"], y=df_compare["Actual"], mode="lines+markers", name="Actual"))
        fig.add_trace(go.Scatter(x=df_compare["Year"], y=df_compare["Prophet_Prediction"], mode="lines+markers", name="Prophet"))
        fig.add_trace(go.Scatter(x=df_compare["Year"], y=df_compare["RF_Prediction"], mode="lines+markers", name="Random Forest"))
        fig.update_layout(title="📊 Actual vs Predicted Energy Consumption", xaxis_title="Year", yaxis_title="Energy Consumption", template="plotly_white")
        st.plotly_chart(fig)

    # Insight Section
        st.subheader("💡 Forecasting Insights")
        stronger = "Prophet" if rmse_prophet < rmse_rf else "Random Forest"
        st.success(f"🔍 Based on RMSE, the **{stronger}** model performed better in this backtesting scenario.")
