"""
House Price Model Trainer
Trains and evaluates multiple ML models, saves the best one.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def load_data():
    """Load the housing dataset."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'housing_data.csv')
    if not os.path.exists(data_path):
        print("❌ Dataset not found! Run 'python data/generate_data.py' first.")
        sys.exit(1)
    df = pd.read_csv(data_path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_data(df):
    """Print exploratory data analysis."""
    print("\n" + "=" * 60)
    print("📊 EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\n🔹 Feature Correlations with Price:")
    correlations = df.corr()['price'].drop('price').sort_values(ascending=False)
    for feature, corr in correlations.items():
        bar = "█" * int(abs(corr) * 30)
        sign = "+" if corr > 0 else "-"
        print(f"   {feature:>18s}: {sign}{abs(corr):.3f}  {bar}")

    print(f"\n🔹 Missing Values: {df.isnull().sum().sum()}")


def train_and_evaluate(df):
    """Train multiple models and return the best one."""
    X = df.drop('price', axis=1)
    y = df['price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42, learning_rate=0.1),
    }

    print("\n" + "=" * 60)
    print("🤖 MODEL TRAINING & EVALUATION")
    print("=" * 60)

    results = {}
    best_model_name = None
    best_r2 = -np.inf

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
        print(f"   MAE:  ${mae:>12,.0f}")
        print(f"   RMSE: ${rmse:>12,.0f}")
        print(f"   R²:    {r2:>11.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    # --- Summary Table ---
    print("\n" + "=" * 60)
    print("📋 MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<22s} {'MAE':>12s} {'RMSE':>12s} {'R²':>8s}")
    print("-" * 56)
    for name, metrics in results.items():
        marker = " 🏆" if name == best_model_name else ""
        print(f"{name:<22s} ${metrics['MAE']:>10,.0f} ${metrics['RMSE']:>10,.0f} {metrics['R²']:>7.4f}{marker}")

    print(f"\n🏆 Best Model: {best_model_name} (R² = {best_r2:.4f})")

    # --- Feature Importance (for tree-based models) ---
    best_model = models[best_model_name]
    if hasattr(best_model, 'feature_importances_'):
        print("\n📊 Feature Importance (Best Model):")
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        for feat, imp in importances.items():
            bar = "█" * int(imp * 50)
            print(f"   {feat:>18s}: {imp:.4f}  {bar}")

    return models[best_model_name], scaler, X.columns.tolist()


def save_model(model, scaler, feature_names):
    """Save the trained model, scaler, and feature names."""
    model_dir = os.path.dirname(__file__)
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
    }
    model_path = os.path.join(model_dir, 'house_price_model.pkl')
    joblib.dump(model_data, model_path)
    print(f"\n💾 Model saved to {model_path}")


if __name__ == '__main__':
    df = load_data()
    explore_data(df)
    model, scaler, feature_names = train_and_evaluate(df)
    save_model(model, scaler, feature_names)
    print("\n✅ Training complete! You can now run the web app with: python app/server.py")
