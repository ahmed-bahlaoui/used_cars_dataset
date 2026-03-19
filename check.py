import pandas as pd
import numpy as np
import re

df = pd.read_csv("cars_dataframe.csv")
df = df.dropna(subset=['Price'])

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

df = fill_doors(df)
df = df.dropna(subset=['Condition'])
df['Origin'] = df['Origin'].fillna('Unknown')
df['First Owner'] = df['First Owner'].fillna('Unknown')
df['Sector'] = df['Sector'].fillna('Unknown')
df = df.drop(index=6714, errors='ignore')
df.drop_duplicates(inplace=True)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df.dropna(subset=['Year'])
df['Year'] = df['Year'].astype(int)
df = df[~df['Brand'].str.isnumeric()]

city_mapping = {
    "الدار البيضاء": "Casablanca", "الرباط": "Rabat", "فاس": "Fès", "مراكش": "Marrakech",
    "طنجة": "Tanger", "أكادير": "Agadir", "وجدة": "Oujda", "القنيطرة": "Kénitra",
    "تطوان": "Tétouan", "آسفي": "Safi", "الجديدة": "El Jadida", "العرائش": "Larache",
    "خريبكة": "Khouribga", "بني ملال": "Beni Mellal", "الناظور": "Nador", "العيون": "Laâyoune",
    "الداخلة": "Dakhla", "المحمدية": "Mohammedia", "مكناس": "Meknès", "تمارة": "Temara"
}
df["Location"] = df["Location"].map(city_mapping).fillna(df["Location"])

new_names_mapping = {'Number of Doors': 'NoD', 'First Owner': 'FO'}
df.rename(columns=new_names_mapping, inplace=True)

def fill_fiscal_power(df):
    col = 'Fiscal Power'
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
    return df
df = fill_fiscal_power(df)

df["Price"] = df["Price"].replace(1, np.nan)
lower = df["Price"].quantile(0.01)
upper = df["Price"].quantile(0.99)
df = df[(df["Price"] >= lower) & (df["Price"] <= upper)]
upper = df["Price"].quantile(0.99)
df = df[df["Price"] <= upper]

df['num_features'] = df['Equipment'].fillna('').apply(
    lambda s: len([x for x in s.split(',') if x.strip() != ''])
)

FO_mapping = {"Yes": 1, "No": 0, "Unknown": 0}
df['FO'] = df['FO'].map(FO_mapping)

df['condition_numeric'] = df['Condition'].map({
    'Fair': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4
})

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

df["Mileage_mean"] = df["Mileage"].apply(mileage_to_mean)
df = df[df["Mileage_mean"] < 400000]

df["Fiscal Power"] = df["Fiscal Power"].astype(str).str.extract(r"(\d+)\s*CV").astype(float)

df['Price_log']   = np.log1p(df['Price'])
df['Mileage_log'] = np.log1p(df['Mileage_mean'])

drop_cols = ['Price', 'Price_log', 'Condition', 'Equipment', 
             'Mileage', 'Location', 'Sector', 'NoD']
X = df.drop(columns=drop_cols)
y = df['Price_log']

print("X NaN counts:")
print(X.isna().sum())
