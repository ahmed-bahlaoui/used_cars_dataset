import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from loguru import logger
import joblib

def main():
    # Setup paths
    PROJECT_ROOT = Path(__file__).parent.parent
    filepath = PROJECT_ROOT / "data" / "processed" / "cleaned_data_final.csv"
    
    # 1. Load data
    logger.info(f"Importing data from {filepath}")
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info("Imported data successfully")

    # 2. Preprocess data
    if "Equipment" in df.columns:
        df.drop(columns=["Equipment"], inplace=True)
        
    # Log-transform skewed columns
    df['Price_log']   = np.log1p(df['Price'])
    df['Mileage_log'] = np.log1p(df['Mileage_mean'])

    # Car age is more intuitive than raw year
    df['Age'] = 2025 - df['Year']
    # Age × Mileage interaction — an old high-mileage car is much worse
    df['Age_x_Mileage'] = df['Age'] * df['Mileage_log']

    # Handling outliers
    # Train only on cars under 98th percentile price
    q98 = df['Price'].quantile(0.98)
    df_train_clean = df[df['Price'] <= q98]

    # 3. Define your feature matrix and target
    drop_cols = ['Price', 'Price_log', 'Condition',
                 'Mileage', 'Location', 'Sector', 'NoD']

    X = df_train_clean.drop(columns=drop_cols)
    y = df_train_clean['Price_log']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_cols    = ['Year', 'Fiscal Power', 'Mileage_log', 'num_features', 'condition_numeric']
    ohe_cols        = ['Gearbox', 'Fuel', 'Origin']
    target_enc_cols = ['Brand', 'Model']

    preprocessor = ColumnTransformer(transformers=[
        ('num',    StandardScaler(), numeric_cols),
        ('ohe',    OneHotEncoder(drop='first', handle_unknown='ignore'), ohe_cols),
        ('target', TargetEncoder(target_type='continuous'), target_enc_cols),
    ])

    # Fit preprocessor on train, transform both
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc  = preprocessor.transform(X_test)

    # 4. Train model
    optimized_params = {
        'subsample': 0.9,
        'reg_lambda': 5,
        'reg_alpha': 0,
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'colsample_bytree': 0.7,
    }
    optimized_params.update({
    'device': 'cuda',
    'random_state': 42,
    'early_stopping_rounds': 50,
})

    logger.info("Initializing XGBRegressor with optimized parameters")
    model = XGBRegressor(**optimized_params)

    logger.info("Training model...")
    model.fit(
        X_train_proc, y_train,
        eval_set=[(X_test_proc, y_test)],
        verbose=50
    )

    logger.success(f"Successfully trained model on optimized parameters")

    # 5. Evaluate model
    logger.info("Setting model to CPU")
    model.set_params(device='cpu')
    logger.info("Predicting on test set...")
    y_pred_log = model.predict(X_test_proc)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    # --- Metrics ---
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"MAE:  {mae:,.0f} MAD")
    print(f"RMSE: {rmse:,.0f} MAD")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # 6. Save model
    model_filename = PROJECT_ROOT / "models" / "car_price_model.json"
    model_filename.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_filename)
    logger.success(f"Successfully saved model to disk filename {model_filename}")

    preprocessor_filename = PROJECT_ROOT / "models" / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_filename)
    logger.success(f"Successfully saved preprocessor to disk filename {preprocessor_filename}")

if __name__ == "__main__":
    main()
