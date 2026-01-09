import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Direct extraction based on your CSV structure
csv_path = '\Pluviométrie en millimètre par principale station (Année Agricole).csv'

# Read CSV ignoring first few rows - the data starts at row 4 (0-indexed)
df = pd.read_csv(csv_path, sep=';', skiprows=4, header=None, encoding='utf-8')

# Clean the data
df = df.dropna(how='all')
df.columns = ['City', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']

# Clean city names
df['City'] = df['City'].astype(str).str.strip()

# Convert rainfall to numeric, handling '--'
for year in ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']:
    df[year] = pd.to_numeric(df[year], errors='coerce')

print("All cities in data:")
print(df['City'].tolist())

# Target cities
target_cities = ['Gabes', 'Gafsa', 'Sidi-bouzid', 'Sfax', 'Medenine']

# Filter for target cities
df_target = df[df['City'].isin(target_cities)]

if df_target.empty:
    print("\nCities not found with exact names. Searching with partial match...")
    df_target = df[df['City'].str.contains('|'.join(target_cities), case=False, na=False)]

print(f"\nFound {len(df_target)} cities:")

# Prepare predictions
predictions = {}
years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
future_years = [2024, 2025, 2026, 2027]

for _, row in df_target.iterrows():
    city = row['City']
    rainfall = [row[str(year)] for year in years]
    
    # Filter out NaN values
    valid_data = [(year, value) for year, value in zip(years, rainfall) if not np.isnan(value)]
    
    if len(valid_data) >= 3:
        # Prepare data for regression
        X = np.array([year for year, _ in valid_data]).reshape(-1, 1)
        y = np.array([value for _, value in valid_data])
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict
        future_X = np.array(future_years).reshape(-1, 1)
        future_pred = model.predict(future_X)
        
        predictions[city] = dict(zip(future_years, future_pred))
        
        print(f"\n{city}:")
        print(f"  Data: {len(valid_data)} years available")
        print(f"  Historical avg: {np.mean(y):.1f} mm")
        print(f"  2026 prediction: {future_pred[2]:.1f} mm")
        print(f"  2027 prediction: {future_pred[3]:.1f} mm")
    else:
        print(f"\n{city}: Not enough data ({len(valid_data)} years) for prediction")

print("\n" + "="*60)
print("FINAL PREDICTIONS (mm)")
print("="*60)
print(f"{'City':<15} {'2026':<10} {'2027':<10} {'Notes':<20}")
print("-" * 60)

for city in target_cities:
    if city in predictions:
        pred_2026 = predictions[city][2026]
        pred_2027 = predictions[city][2027]
        
        # Get historical average
        city_row = df_target[df_target['City'] == city].iloc[0]
        hist_values = [city_row[str(year)] for year in years if not np.isnan(city_row[str(year)])]
        hist_avg = np.mean(hist_values) if hist_values else 0
        
        trend_2026 = "↑" if pred_2026 > hist_avg * 1.1 else "↓" if pred_2026 < hist_avg * 0.9 else "≈"
        trend_2027 = "↑" if pred_2027 > hist_avg * 1.1 else "↓" if pred_2027 < hist_avg * 0.9 else "≈"
        
        print(f"{city:<15} {pred_2026:<9.1f}{trend_2026} {pred_2027:<9.1f}{trend_2027} Based on {len(hist_values)} years")
    else:
        print(f"{city:<15} {'N/A':<10} {'N/A':<10} Insufficient data")