"""
Inference for Volleyball Reception Binary Classifier
------------------------------------------------------

This script loads a trained volleyball reception binary classifier (saved as a .h5 model) 
and a fitted StandardScaler (saved as a pickle file) to perform inference on new keypoint data.
The input CSV file can optionally contain labels in the last column; if present, they will be dropped.
Each row should otherwise contain flattened keypoint coordinates (e.g., kp1_x, kp1_y, kp2_x, kp2_y, ...).

The script:
  - Loads the CSV file using pandas.
  - Checks if the last column is non-numeric (assumed to be labels) and drops it.
  - Transforms the keypoints using the loaded StandardScaler.
  - Loads the trained model from a .h5 file.
  - Predicts the probability for the "upper" class (and outputs binary predictions).
  - Prints and optionally saves the predictions to an output CSV file.

Usage:
    python inference_receive_classifier.py --csv_path path/to/new_data.csv --model_path volleyball_receive_classifier.h5 --scaler_path scaler.pkl [--output_csv predictions.csv]
"""

import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os

def load_scaler(scaler_path):
    """
    Load a previously saved StandardScaler from a pickle file.
    
    Args:
        scaler_path (str): Path to the scaler pickle file.
        
    Returns:
        scaler: The loaded StandardScaler object.
    """
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def load_and_preprocess_inference_data(csv_path, scaler):
    """
    Load the new data CSV file and preprocess it using the provided scaler.
    
    Expected CSV format per row:
        kp1_x, kp1_y, kp2_x, kp2_y, ..., kpN_x, kpN_y[, label]
    If the last column is non-numeric, it is assumed to be a label and dropped.
    
    Args:
        csv_path (str): Path to the CSV file with new keypoints.
        scaler: A fitted StandardScaler used to transform the features.
    
    Returns:
        tuple: (X_scaled, df) where X_scaled is the scaled feature matrix,
               and df is the original DataFrame (with label column intact if present).
    """
    df = pd.read_csv(csv_path)
    # Check if the last column is non-numeric; if so, drop it for inference features.
    if df.iloc[:, -1].dtype == 'object':
        X = df.iloc[:, :-1].values.astype(np.float32)
    else:
        X = df.values.astype(np.float32)
    X_scaled = scaler.transform(X)
    return X_scaled, df

def main():
    parser = argparse.ArgumentParser(
        description="Run inference using a trained volleyball reception classifier on new keypoint data."
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file with keypoints (optionally with a label column).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.h5 file).")
    parser.add_argument("--scaler_path", type=str, required=True, help="Path to the saved StandardScaler (pickle file).")
    parser.add_argument("--output_csv", type=str, help="(Optional) Path to save predictions as a CSV file.")
    args = parser.parse_args()

    # Load the scaler and preprocess new data
    scaler = load_scaler(args.scaler_path)
    X_new, df_orig = load_and_preprocess_inference_data(args.csv_path, scaler)
    
    # Load the trained model
    model = load_model(args.model_path)
    
    # Run inference
    predictions = model.predict(X_new)
    # Convert probabilities to binary predictions: threshold of 0.5
    binary_preds = (predictions >= 0.5).astype(int).flatten()
    
    # Map binary predictions to class labels: 1 -> "upper", 0 -> "lower"
    pred_labels = np.where(binary_preds == 1, "upper", "lower")
    
    # Print predictions
    for idx, label in enumerate(pred_labels):
        print(f"Sample {idx}: {label} (probability: {predictions[idx][0]:.4f})")
    
    # Optionally, save the predictions to a CSV
    if args.output_csv:
        df_out = df_orig.copy()
        # If a label column exists, we remove it from the output to avoid confusion.
        if df_out.iloc[:, -1].dtype == 'object':
            df_out = df_out.iloc[:, :-1]
        df_out['predicted_label'] = pred_labels
        df_out.to_csv(args.output_csv, index=False)
        print(f"Predictions saved to {args.output_csv}")

if __name__ == "__main__":
    main()
