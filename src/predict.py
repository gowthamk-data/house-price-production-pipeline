import pandas as pd
import numpy as np
import joblib
import os
import argparse

def load_model(model_path="../Projects/housing_pipeline.pkl"):
    return joblib.load(model_path)

def generate_features(df):
    df_processed = df.copy()
    
    # Creating columns because the pipeline expects them
    df_processed["HouseAge"] = df_processed["YrSold"] - df_processed["YearBuilt"]
    df_processed["RemodAge"] = df_processed["YrSold"] - df_processed["YearRemodAdd"]
    df_processed["TotalBathrooms"] = df_processed["FullBath"] + 0.5 * df_processed["HalfBath"] + df_processed["BsmtFullBath"] + 0.5 * df_processed["BsmtHalfBath"]
    df_processed["TotalSF"] = df_processed["TotalBsmtSF"] + df_processed["1stFlrSF"] + df_processed["2ndFlrSF"]
    
    return df_processed

def predict(input_data_path, output_data_path, model_path="../Projects/housing_pipeline.pkl"):
    
    # 1. Load the new
    print(f"Loading new listings from {input_data_path}...")
    new_houses = pd.read_csv(input_data_path)
    
    # Keep the IDs for the final report, but drop them from the prediction data
    house_ids = new_houses["Id"]
    X_new = new_houses.drop("Id", axis=1)
    
    # 2. Applying our custom features
    X_new = generate_features(X_new)
    
    # 3. model loading
    pipeline = load_model(model_path)
    
    # 4. Make Predictions!
    print("Generating predictions...")
    log_predictions = pipeline.predict(X_new)
    
    # 5. Reverse the Log Transformation
    # Since we trained the model on np.log1p(), it predicts log values. 
    # We must use np.expm1() to convert them back to normal dollar amounts!
    actual_price_predictions = np.expm1(log_predictions)
    
    # 6. Save the results
    results_df = pd.DataFrame({
        "Id": house_ids,
        "Predicted_SalePrice": actual_price_predictions
    })
    
    results_df.to_csv(output_data_path, index=False)
    print(f"Success! Predictions saved to {output_data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict housing prices using the trained pipeline.")
    parser.add_argument("--input", type=str, default="../Projects/Housing_data.csv", help="Path to new listings CSV")
    parser.add_argument("--output", type=str, default="../Projects/predictions.csv", help="Path to save predictions")
    
    # Added args=[] here to stop Jupyter from crashing it!
    args = parser.parse_args(args=[]) 
    
    predict(input_data_path=args.input, output_data_path=args.output)