import pandas as pd
import numpy as np
from loguru import logger
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model(data_path, model_output_path):
    logger.info(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}.")
        return

    df = pd.read_csv(data_path)
    
    logger.info("Applying log transformations...")
    # Log-transform skewed columns
    df['Price_log'] = np.log1p(df['Price'])
    df['Mileage_log'] = np.log1p(df['Mileage_mean'])

    logger.info("Splitting data into features and target...")
    drop_cols = ['Price', 'Price_log', 'Condition', 'Equipment', 
                 'Mileage', 'Location', 'Sector', 'NoD']
    
    # Drop columns that aren't necessary or are already encoded
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['Price_log']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info("Building the preprocessing pipeline...")
    numeric_cols    = ['Year', 'Fiscal Power', 'Mileage_log', 'num_features', 'condition_numeric']
    ohe_cols        = ['Gearbox', 'Fuel', 'Origin']
    target_enc_cols = ['Brand', 'Model']

    # Note: FO (First Owner) is passed through since it's already binary encoded
    preprocessor = ColumnTransformer(transformers=[
        ('num',    StandardScaler(), numeric_cols),
        ('ohe',    OneHotEncoder(drop='first', handle_unknown='ignore'), ohe_cols),
        ('target', TargetEncoder(target_type='continuous'), target_enc_cols),
    ], remainder='passthrough')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model',        GradientBoostingRegressor(random_state=42))
    ])

    logger.info("Training GradientBoostingRegressor... this might take a minute.")
    pipeline.fit(X_train, y_train)
    logger.success("Model training complete!")

    logger.info("Evaluating the model on the test set...")
    y_pred_log = pipeline.predict(X_test)
    
    # Convert predictions back to original scale (MAD)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    logger.info(f"Test Set Metrics:")
    logger.info(f"  MAE:  {mae:,.0f} MAD")
    logger.info(f"  RMSE: {rmse:,.0f} MAD")
    logger.info(f"  R²:   {r2:.4f}")
    logger.info(f"  MAPE: {mape:.2f}%")

    logger.info(f"Saving the trained model to {model_output_path}...")
    out_dir = os.path.dirname(model_output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    joblib.dump(pipeline, model_output_path)
    logger.success(f"Pipeline successfully saved to {model_output_path}!")

if __name__ == '__main__':
    data_path = "../data/processed/cleaned_data_final.csv"
    model_output_path = "../models/gbr_pipeline.joblib"
    
    # Fallback to absolute/relative paths if called from different CWD
    if not os.path.exists(data_path):
        if os.path.exists("data/processed/cleaned_data_final.csv"):
            data_path = "data/processed/cleaned_data_final.csv"
            model_output_path = "models/gbr_pipeline.joblib"
            
    train_model(data_path, model_output_path)
