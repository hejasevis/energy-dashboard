import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules
from PIL import Image
import streamlit as st

# Page setup
st.set_page_config(layout="wide")
st.sidebar.title("📊 Dashboard Menu")
page = st.sidebar.radio("Select a page:", ["🏠 Home", "🌍 Global Map", "🌐 Country-Level Deep Analysis"])

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("owid-energy-data.csv")

df = load_data()

st.markdown(
    """
    <style>
    .small-margin {
        margin-top: -50px;
        margin-bottom: -30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="small-margin">', unsafe_allow_html=True)
st.image("images/1.svg", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


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
