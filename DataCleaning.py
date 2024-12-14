# DataCleaning.py

import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Reading the Hyderabad Dataset
try:
    df = pd.read_csv('Hyderabad.csv')
except FileNotFoundError:
    print("Error: Hyderabad.csv file not found. Please ensure the file exists in the directory.")
    exit()

# Initialize geolocator with rate limiter to handle request rate
geolocator = Nominatim(user_agent='HousePricePrediction.py')
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# Ensure key columns exist: Price, Area, Latitude, Longitude, and Bedroom (BED)
required_columns = ['Price', 'Area', 'Location', 'BED', 'Latitude', 'Longitude']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"Error: The following required columns are missing: {missing_columns}")
    exit()

# Feature Engineering: Calculate price per square foot
df['Price_per_sqft'] = df['Price'] / df['Area']

# Handle missing data or drop rows with NA
df.dropna(subset=['Price', 'Area', 'Location', 'BED', 'Latitude', 'Longitude'], inplace=True)

# Save updated dataframe to new CSV file
df.to_csv('CleanedData.csv', index=False)

print('File saved as CleanedData.csv')
