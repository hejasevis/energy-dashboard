import streamlit as st
import pandas as pd

st.title("Enerji Tüketimi Analizi ve İlişkilendirme Kuralları")
st.markdown("""
Bu dashboard, OWID enerji verisine dayalı olarak yapılan birliktelik analizini sunar.
Apriori algoritmasıyla çıkarılan kuralları filtreleyebilir ve görüntüleyebilirsiniz.
""")

uploaded_file = st.file_uploader("Kurallar CSV dosyasını yükle (rules_sorted.csv)", type="csv")
if uploaded_file:
    rules = pd.read_csv(uploaded_file)

    st.sidebar.header("Filtreler")
    min_lift = st.sidebar.slider("Min. Lift Değeri", 0.0, 5.0, 1.0, 0.1)
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
    st.warning("Lütfen CSV dosyasını yükleyin.")
