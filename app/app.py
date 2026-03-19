import streamlit as st
import joblib, pandas as pd
import numpy as np
from pathlib import Path

# Compute the absolute root of the project to reliably find assets
BASE_DIR = Path(__file__).resolve().parent.parent

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data" / "processed" / "cleaned_data_final.csv")

# Load data to get real categories
df = load_data()

model_path = BASE_DIR / "models" / "gbr_pipeline.joblib"
if not model_path.exists():
    st.error(f"Cannot find model file at {model_path}")
    st.stop()
model = joblib.load(model_path)

st.title("---Morocco Used Car Price Predictor---")

# --- EDA Section ---
st.header("Dataset Overview")
col_img1, col_img2 = st.columns(2)
with col_img1:
    image_path = Path("figures/fig_02_price_distribution_up_to_95th_percentile.png")
    st.image(str(image_path), use_column_width=True)
with col_img2:
    image_path =Path("figures/fig_17_correlation_matrix.png")
    st.image(str(image_path), use_column_width=True)

# --- Prediction Section ---
st.header("Predict a Price")

col1, col2 = st.columns(2)

with col1:
    brands = sorted(df['Brand'].unique().tolist())
    brand = st.selectbox("Brand", brands)
    
    models_for_brand = sorted(df[df['Brand'] == brand]['Model'].unique().tolist())
    model_name = st.selectbox("Model", models_for_brand)
    
    year = st.slider("Year", 1981, 2024, 2018)
    mileage_mean = st.number_input("Mileage (km)", 0.0, 500000.0, 80000.0, step=1000.0)
    
    fiscal_power = st.number_input("Fiscal Power", 4, 41, 7)
    num_features = st.slider("Number of Features", 1, 15, 6)

with col2:
    gearbox = st.radio("Gearbox", ['Manual', 'Automatic'])
    fuel = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'Electrique', 'Hybrid', 'LPG'])
    origin = st.selectbox("Origin", ['Unknown', 'WW in Morocco', 'Customs-cleared car', 'Car not yet customs-cleared', 'Imported New'])
    fo = st.radio("First Owner (1=Yes, 0=No)", [1, 0])
    condition_numeric = st.slider("Condition (0-6)", 0, 6, 4)


if st.button("Estimate Price", type="primary"):
    mileage_log = np.log1p(mileage_mean)
    
    input_data = {
        'Brand': [brand],
        'Model': [model_name],
        'Year': [year],
        'Gearbox': [gearbox],
        'Fiscal Power': [fiscal_power],
        'Fuel': [fuel],
        'Origin': [origin],
        'FO': [fo],
        'num_features': [num_features],
        'condition_numeric': [condition_numeric],
        'Mileage_mean': [mileage_mean],
        'Mileage_log': [mileage_log]
    }
    
    X = pd.DataFrame(input_data)
    
    # Predict Log Price and convert back
    price_pred_log = model.predict(X)[0]
    price = np.expm1(price_pred_log)
    
    st.success(f"### Estimated Price: **{price:,.0f} MAD**")
    