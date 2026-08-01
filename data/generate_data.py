"""
House Price Dataset Generator
Generates a realistic synthetic housing dataset with 5,000 samples.
"""

import numpy as np
import pandas as pd
import os

def generate_housing_data(n_samples=5000, seed=42):
    """Generate realistic synthetic housing data."""
    np.random.seed(seed)

    # --- Feature generation ---
    sqft = np.random.randint(800, 5001, n_samples)
    bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.15, 0.30, 0.30, 0.15, 0.05])
    bathrooms = np.random.choice([1, 2, 3, 4], n_samples, p=[0.15, 0.40, 0.30, 0.15])
    age = np.random.randint(0, 101, n_samples)
    lot_size = np.round(np.random.uniform(0.1, 2.5, n_samples), 2)
    garage = np.random.choice([0, 1, 2, 3], n_samples, p=[0.10, 0.30, 0.40, 0.20])
    neighborhood = np.random.randint(1, 11, n_samples)
    has_pool = np.random.choice([0, 1], n_samples, p=[0.70, 0.30])
    distance_city = np.round(np.random.uniform(0.5, 40.0, n_samples), 1)
    condition = np.random.randint(1, 11, n_samples)

    # --- Price calculation with realistic coefficients ---
    price = (
        50_000                                  # base price
        + sqft * 150                            # $150 per sqft
        + bedrooms * 15_000                     # $15k per bedroom
        + bathrooms * 20_000                    # $20k per bathroom
        - age * 1_500                           # loses $1.5k per year of age
        + lot_size * 30_000                     # $30k per acre
        + garage * 25_000                       # $25k per garage spot
        + neighborhood * 20_000                 # $20k per neighborhood rating point
        + has_pool * 40_000                     # $40k for a pool
        - distance_city * 2_000                 # loses $2k per mile from city
        + condition * 10_000                    # $10k per condition point
    )

    # Add realistic noise (±15%)
    noise = np.random.normal(1.0, 0.15, n_samples)
    price = np.round(price * noise, -3)  # round to nearest $1,000
    price = np.clip(price, 50_000, None)  # minimum price $50k

    # --- Build DataFrame ---
    df = pd.DataFrame({
        'sqft': sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'lot_size': lot_size,
        'garage': garage,
        'neighborhood': neighborhood,
        'has_pool': has_pool,
        'distance_city': distance_city,
        'condition': condition,
        'price': price.astype(int)
    })

    return df


if __name__ == '__main__':
    print("🏠 Generating housing dataset...")
    df = generate_housing_data()

    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), 'housing_data.csv')
    df.to_csv(output_path, index=False)

    print(f"✅ Dataset saved to {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"\n📊 Dataset Statistics:")
    print(df.describe().round(2).to_string())
    print(f"\n💰 Price Range: ${df['price'].min():,.0f} — ${df['price'].max():,.0f}")
    print(f"   Mean Price:  ${df['price'].mean():,.0f}")
