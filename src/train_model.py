import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

if __name__ == "__main__":
    print("Starting Housing Price Prediction Pipeline...")

    # 1. Load Data
    df = pd.read_csv("../Projects/Housing_data.csv")
    df = df.drop("Id", axis=1)

    # 2. Feature Engineering
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["TotalBathrooms"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    # Target variable log transformation
    X = df.drop("SalePrice", axis=1)
    y = np.log1p(df["SalePrice"])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Building an Automated Preprocessing Pipeline
    
    # Separate numeric and categorical columns dynamically
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Pipeline for numbers: Fill missing with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categories: Fill missing with 'None', then One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine both into a single processor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. Defining the Final Pipeline & Tune
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', HistGradientBoostingRegressor(random_state=42, max_iter=100, learning_rate=0.1))
    ])

    pipeline.fit(X_train, y_train)

    # 5. Evaluate the Model
    preds = pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    
    print("\nFinal Model Evaluation")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # 6. Generate & Save Visualizations
    # A. Residual Plot
    residuals = y_val - preds
    plt.figure(figsize=(10, 6))
    plt.scatter(preds, residuals, alpha=0.5, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Sale Price (Log)")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot: Model Accuracy Check")
    plt.savefig("../Projects/residual_plot.png", bbox_inches='tight')
    plt.close()

    # B. SHAP Values
    X_val_processed = pipeline.named_steps['preprocessor'].transform(X_val)
    model_only = pipeline.named_steps['model']    
    
    cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_features = cat_encoder.get_feature_names_out(categorical_features)
    all_feature_names = list(numeric_features) + list(cat_features)

    # Calculate and save SHAP plot
    explainer = shap.TreeExplainer(model_only)
    shap_values = explainer.shap_values(X_val_processed)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_val_processed, feature_names=all_feature_names, show=False)
    plt.savefig("../Projects/shap_summary_plot.png", bbox_inches='tight')
    plt.close()

    # 7. Save the Model
    joblib.dump(pipeline, '../Projects/housing_pipeline.pkl')
    print("\nPipeline Complete!")