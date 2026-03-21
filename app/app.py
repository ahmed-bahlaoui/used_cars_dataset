import streamlit as st
import joblib, pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor

# Compute the absolute root of the project to reliably find assets
BASE_DIR = Path(__file__).resolve().parent.parent

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data" / "processed" / "cleaned_data_final.csv")

# Load data to get real categories
df = load_data()

preprocessor_path = BASE_DIR / "models" / "preprocessor.joblib"
model_path = BASE_DIR / "models" / "car_price_model.json"
if not preprocessor_path.exists() or not model_path.exists():
    st.error(f"Cannot find model files. Please run train_xgboost.py first.")
    st.stop()
preprocessor = joblib.load(preprocessor_path)
model = XGBRegressor()
model.load_model(model_path)

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
    
    # Automatically determine specs based on brand, model and year
    matching_cars = df[(df['Brand'] == brand) & (df['Model'] == model_name) & (df['Year'] == year)]
    if matching_cars.empty:
        matching_cars = df[(df['Brand'] == brand) & (df['Model'] == model_name)]
        if matching_cars.empty:
            matching_cars = df[df['Brand'] == brand]

    if not matching_cars.empty:
        fiscal_power = int(matching_cars['Fiscal Power'].mode()[0])
        num_features = int(matching_cars['num_features'].mode()[0])
        gearbox = matching_cars['Gearbox'].mode()[0]
        fuel = matching_cars['Fuel'].mode()[0]
    else:
        fiscal_power, num_features, gearbox, fuel = 7, 6, "Manual", "Diesel"
            
    st.info(f"**Auto-Selected Specifications:**\n\n"
            f"- **Fiscal Power:** {fiscal_power} CV\n"
            f"- **Features:** {num_features}\n"
            f"- **Gearbox:** {gearbox}\n"
            f"- **Fuel Type:** {fuel}")

with col2:
    origin = st.selectbox("Origin", ['Unknown', 'WW in Morocco', 'Customs-cleared car', 'Car not yet customs-cleared', 'Imported New'])
    fo = st.radio("First Owner (1=Yes, 0=No)", [1, 0])
    
    conditions_list = ["For Parts", "Damaged", "Fair", "Good", "Very Good", "Excellent", "New"]
    condition_selected = st.selectbox("Condition", conditions_list, index=4) # Default to 'Very Good' map to 4
    condition_numeric = conditions_list.index(condition_selected)

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
    X_proc = preprocessor.transform(X)
    price_pred_log = model.predict(X_proc)[0]
    price = np.expm1(price_pred_log)
    
    st.success(f"### Estimated Price: **{price:,.0f} MAD**")
    