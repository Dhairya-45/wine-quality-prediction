import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "pkl", "wine_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base, "pkl", "wine_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'wine_type'
]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🍷 Wine Quality Prediction")
st.markdown("Predict whether a wine is **Good** or **Bad** using XGBoost — trained on 6,500+ red & white wines.")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Wine Properties")

    col1, col2, col3 = st.columns(3)

    with col1:
        wine_type    = st.selectbox("Wine Type", ["Red Wine", "White Wine"])
        fixed_acid   = st.number_input("Fixed Acidity",     min_value=3.0,  max_value=16.0, value=7.4,  step=0.1)
        volatile_acid= st.number_input("Volatile Acidity",  min_value=0.0,  max_value=2.0,  value=0.70, step=0.01)
        citric_acid  = st.number_input("Citric Acid",       min_value=0.0,  max_value=1.5,  value=0.00, step=0.01)

    with col2:
        residual_sugar = st.number_input("Residual Sugar",        min_value=0.0,  max_value=70.0, value=1.9,   step=0.1)
        chlorides      = st.number_input("Chlorides",             min_value=0.0,  max_value=1.0,  value=0.076, step=0.001, format="%.3f")
        free_so2       = st.number_input("Free Sulfur Dioxide",   min_value=0.0,  max_value=300.0,value=11.0,  step=1.0)
        total_so2      = st.number_input("Total Sulfur Dioxide",  min_value=0.0,  max_value=500.0,value=34.0,  step=1.0)

    with col3:
        density    = st.number_input("Density",    min_value=0.980, max_value=1.010, value=0.9978, step=0.0001, format="%.4f")
        pH         = st.number_input("pH",         min_value=2.5,   max_value=4.5,   value=3.51,   step=0.01)
        sulphates  = st.number_input("Sulphates",  min_value=0.0,   max_value=2.5,   value=0.56,   step=0.01)
        alcohol    = st.number_input("Alcohol (%)",min_value=7.0,   max_value=15.0,  value=9.4,    step=0.1)

    st.markdown("")
    if st.button("🔮 Predict Quality", use_container_width=True):
        input_data = pd.DataFrame([{
            'fixed acidity':        fixed_acid,
            'volatile acidity':     volatile_acid,
            'citric acid':          citric_acid,
            'residual sugar':       residual_sugar,
            'chlorides':            chlorides,
            'free sulfur dioxide':  free_so2,
            'total sulfur dioxide': total_so2,
            'density':              density,
            'pH':                   pH,
            'sulphates':            sulphates,
            'alcohol':              alcohol,
            'wine_type':            0 if wine_type == "Red Wine" else 1
        }])

        scaled = scaler.transform(input_data)
        pred   = model.predict(scaled)[0]
        prob   = model.predict_proba(scaled)[0]

        st.markdown("---")
        c1, c2, c3 = st.columns(3)

        with c1:
            if pred == 1:
                st.success("✅ Good Wine")
            else:
                st.error("❌ Bad Wine")

        with c2:
            quality_score = round(3 + prob[1] * 5, 1)
            st.metric("Estimated Quality Score", f"{quality_score} / 8")

        with c3:
            st.metric("Confidence (Good Wine)", f"{prob[1]*100:.1f}%")

        st.progress(float(prob[1]))
        st.caption(f"Good Wine probability: {prob[1]*100:.1f}%  |  Bad Wine probability: {prob[0]*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Upload CSV for Batch Prediction")

    st.markdown("""
    **Required columns:**
    `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`,
    `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`,
    `pH`, `sulphates`, `alcohol`, `wine_type` *(0 = Red, 1 = White)*
    """)

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"**Uploaded:** {len(df)} rows")
        st.dataframe(df.head())

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            scaled = scaler.transform(df[FEATURES])
            preds  = model.predict(scaled)
            probs  = model.predict_proba(scaled)[:, 1]

            df['Prediction']       = ['Good Wine 🍷' if p == 1 else 'Bad Wine' for p in preds]
            df['Good Probability'] = (probs * 100).round(1).astype(str) + '%'
            df['Quality Score']    = (3 + probs * 5).round(1)

            st.markdown("### Results")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="wine_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

            good = (preds == 1).sum()
            bad  = (preds == 0).sum()
            st.markdown(f"**Summary:** {good} Good Wines ✅ | {bad} Bad Wines ❌")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with XGBoost & Streamlit | Dataset: UCI Wine Quality (Red + White)")
