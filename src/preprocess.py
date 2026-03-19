import pandas as pd
import numpy as np
import re
import os
from  loguru import logger

def fill_doors(df):
    col = 'Number of Doors'
    def group_mode(group_cols):
        return (
            df.dropna(subset=[col])
              .groupby(group_cols)[col]
              .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        )
    mode_bmy = group_mode(['Brand', 'Model', 'Year'])
    mode_bm  = group_mode(['Brand', 'Model'])
    mode_b   = group_mode(['Brand'])
    overall_mode = df[col].mode().iloc[0]

    def impute(row):
        if pd.notna(row[col]):
            return row[col]
        for key, lookup in [
            ((row['Brand'], row['Model'], row['Year']), mode_bmy),
            ((row['Brand'], row['Model']), mode_bm),
            (row['Brand'], mode_b),
        ]:
            try:
                val = lookup.loc[key]
                if pd.notna(val): return val
            except KeyError:
                pass
        return overall_mode

    df[col] = df.apply(impute, axis=1)
    df[col] = df[col].astype('int64')
    return df

def fill_fiscal_power(df):
    col = 'Fiscal Power'
    def group_mode(group_cols):
        return (
            df.dropna(subset=[col])
              .groupby(group_cols)[col]
              .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        )
    mode_bmy = group_mode(['Brand', 'Model', 'Year'])   # most specific
    mode_bm  = group_mode(['Brand', 'Model'])           # fallback 1
    mode_b   = group_mode(['Brand'])                    # fallback 2
    overall_mode = df[col].mode().iloc[0]               # fallback 3

    def impute(row):
        if pd.notna(row[col]):
            return row[col]
        for key, lookup in [
            ((row['Brand'], row['Model'], row['Year']), mode_bmy),
            ((row['Brand'], row['Model']), mode_bm),
            (row['Brand'], mode_b),
        ]:
            try:
                val = lookup.loc[key]
                if pd.notna(val): return val
            except KeyError:
                pass
        return overall_mode

    df[col] = df.apply(impute, axis=1)
    return df

def mileage_to_mean(mileage_str):
    try:
        if pd.isna(mileage_str): return np.nan
        mileage_str = mileage_str.replace(" ", "").lower()
        if "plusde" in mileage_str:
            return int(re.findall(r"\d+", mileage_str)[0])
        elif "-" in mileage_str:
            low, high = mileage_str.split("-")
            return (int(low) + int(high)) / 2
        return int(mileage_str)
    except:
        return np.nan

def clean_data(df):
    print("Starting data cleaning...")
    
    # 1. Price
    print("Cleaning Price...")
    df = df.dropna(subset=['Price'])
    df["Price"] = df["Price"].replace(1, np.nan)
    lower = df["Price"].quantile(0.01)
    upper = df["Price"].quantile(0.99)
    df = df[(df["Price"] >= lower) & (df["Price"] <= upper)]
    # Removing luxury outliers
    upper = df["Price"].quantile(0.99)
    df = df[df["Price"] <= upper]
    
    # 2. Number of Doors
    print("Imputing Number of Doors...")
    df = fill_doors(df)
    
    # 3. Simple fillna & drops
    print("Handling missing values in other columns...")
    df = df.dropna(subset=['Condition'])
    df['Origin'] = df['Origin'].fillna('Unknown')
    df['First Owner'] = df['First Owner'].fillna('Unknown')
    df['Sector'] = df['Sector'].fillna('Unknown')
    df = df.drop(index=6714, errors='ignore')
    df.drop_duplicates(inplace=True)
    
    # 4. Year
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    
    # 5. Brand
    df = df[~df['Brand'].str.isnumeric()]
    
    # 6. Location
    print("Mapping City names...")
    city_mapping = {
        "الدار البيضاء": "Casablanca", "الرباط": "Rabat", "فاس": "Fès", "مراكش": "Marrakech",
        "طنجة": "Tanger", "أكادير": "Agadir", "وجدة": "Oujda", "القنيطرة": "Kénitra",
        "تطوان": "Tétouan", "آسفي": "Safi", "الجديدة": "El Jadida", "العرائش": "Larache",
        "خريبكة": "Khouribga", "بني ملال": "Beni Mellal", "الناظور": "Nador", "العيون": "Laâyoune",
        "الداخلة": "Dakhla", "المحمدية": "Mohammedia", "مكناس": "Meknès", "تمارة": "Temara"
    }
    df["Location"] = df["Location"].map(city_mapping).fillna(df["Location"])
    
    # 7. Renaming
    new_names_mapping = {'Number of Doors': 'NoD', 'First Owner': 'FO'}
    df.rename(columns=new_names_mapping, inplace=True)
    
    # 8. Fiscal Power
    print("Imputing Fiscal Power...")
    df = fill_fiscal_power(df)
    
    # Feature Engineering steps (Mileage, categorical maps, etc.)
    print("Engineering features (Mileage, categorical maps, numeric casting)...")
    
    # Equipment features count
    df['num_features'] = df['Equipment'].fillna('').apply(
        lambda s: len([x for x in s.split(',') if x.strip() != ''])
    )
    
    # Categorical Maps
    FO_mapping = {"Yes": 1, "No": 0, "Unknown": 0}
    df['FO'] = df['FO'].map(FO_mapping)
    
    condition_mapping = {
        'For Parts': 0, 'Damaged': 1, 'Fair': 2, 'Good': 3, 
        'Very Good': 4, 'Excellent': 5, 'New': 6
    }
    df['condition_numeric'] = df['Condition'].map(condition_mapping)
    
    # Mileage string to numeric mean
    df["Mileage_mean"] = df["Mileage"].apply(mileage_to_mean)
    df = df[df["Mileage_mean"] < 400000]
    
    # Fiscal power numeric extraction
    df["Fiscal Power"] = df["Fiscal Power"].astype(str).str.extract(r"(\d+)\s*CV").astype(float)
    
    print(f"Cleaning complete! Final dataset shape: {df.shape}")
    return df


if __name__ == '__main__':
    print("--- Used Cars Preprocessing Script ---")
    #csv_path = input("Enter the path to the raw dataset (CSV) [default: ../cars_dataframe.csv]: ").strip()
    csv_path = '../data/raw/cars_dataframe.csv'
    
    # Fallback to current directory or one level up based on execution directory
    if not csv_path:
        if os.path.exists("cars_dataframe.csv"):
            csv_path = "cars_dataframe.csv"
        elif os.path.exists("../cars_dataframe.csv"):
            csv_path = "../cars_dataframe.csv"
        else:
            csv_path = "cars_dataframe.csv"
            
    if not os.path.exists(csv_path):
        logger.error(f"Error: File '{csv_path}' not found. Please check your path.")
    else:
        logger.info(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        cleaned_df = clean_data(df)
        
        output_path = input("Enter the path to save the cleaned dataset [default: ../data/processed/cleaned_data.csv]: ").strip()
        if not output_path:
            output_path = "../data/processed/cleaned_data_final.csv"
            
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            
        cleaned_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.success(f"Cleaned dataset successfully saved to {output_path}!")