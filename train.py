
 
import os

import json

import joblib

import pickle

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder
 
# Import model architectures

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier
 
def generate_synthetic_data(num_samples=2000):

    """Generates an agricultural chemical and environmental dataset if agro_data.csv is absent."""

    print("Generating synthetic 'agro_data.csv' dataset...")

    np.random.seed(42)

    df = pd.DataFrame({

        'N': np.random.randint(10, 140, size=num_samples),

        'P': np.random.randint(10, 130, size=num_samples),

        'K': np.random.randint(10, 220, size=num_samples),

        'temperature': np.random.uniform(12.0, 38.0, size=num_samples),

        'humidity': np.random.uniform(35.0, 95.0, size=num_samples),

        'ph': np.random.uniform(4.5, 8.5, size=num_samples),

        'rainfall': np.random.uniform(40.0, 280.0, size=num_samples)

    })

    # Real-world inspired crop and soil amendment mapping logic

    targets = [

        "Rice || Urea", "Maize || NPK 15-15-15", "Wheat || DAP", 

        "Cotton || Ammonium Sulfate", "Sugarcane || Muriate of Potash"

    ]

    target_labels = []

    for _, row in df.iterrows():

        if row['N'] > 80 and row['rainfall'] > 180:

            target_labels.append(targets[0])

        elif row['P'] > 60 and row['K'] < 100:

            target_labels.append(targets[1])

        elif row['temperature'] < 22 and row['ph'] > 6.5:

            target_labels.append(targets[2])

        elif row['K'] > 120 and row['humidity'] > 60:

            target_labels.append(targets[4])

        else:

            target_labels.append(targets[3])

    df['crop_fertilizer'] = target_labels

    df.to_csv('agro_data.csv', index=False)

    print("Dataset generated and saved to 'agro_data.csv'.")

    return df
 
def main():

    # Acquire Data Source

    if os.path.exists('agro_data.csv'):

        print("Found existing 'agro_data.csv'. Loading...")

        df = pd.read_csv('agro_data.csv')

    else:

        df = generate_synthetic_data()
 
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

    y = df['crop_fertilizer']
 
    # Preprocessing Pipelines

    print("Fitting preprocessing encoders and scalers...")

    scaler = StandardScaler()

    # Scale the features and convert the NumPy array back into a DataFrame with column names

    X_scaled_array = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)
 
    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)
 
 
    # Save preprocessing binaries

    joblib.dump(scaler, 'scaler.pkl')

    joblib.dump(encoder, 'label_encoder.pkl')

    print("Saved 'scaler.pkl' and 'label_encoder.pkl'.")
 
    #  Stratified Train-Test Partition Split

    X_train, X_test, y_train, y_test = train_test_split(

        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded

    )
 
    #  Model Definition

    models = {

        'Logistic Regression (Aditi - M1)': LogisticRegression(max_iter=1000, random_state=42),

        'K-Nearest Neighbors (Kaustubh - M2)': KNeighborsClassifier(n_neighbors=5),

        'Random Forest (Abhiram - M3)': RandomForestClassifier(n_estimators=100, random_state=42),

        'LightGBM (Vedant - M4)': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

    }
 
    # Model filenames mapping

    filenames = {

        'Logistic Regression (Aditi - M1)': 'logistic_regression_model.pkl',

        'K-Nearest Neighbors (Kaustubh - M2)': 'knn_model.pkl',

        'Random Forest (Abhiram - M3)': 'random_forest_model.pkl',

        'LightGBM (Vedant - M4)': 'lightgbm_model.pkl'

    }
 
    # Training, Evaluation, and Serialization loop

    accuracy_metrics = {}
 
    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        # Calculate dynamic real-world accuracy on test set

        score = float(model.score(X_test, y_test))

        accuracy_metrics[name] = score

        print(f"{name} Evaluation Test Accuracy: {score * 100:.2f}%")

        # Export model binary artifact

        filepath = filenames[name]

        joblib.dump(model, filepath)

        print(f"Exported system weight artifact to '{filepath}'.")
 
    # Save Metric Configurations for Dashboard Consumption

    with open('accuracy_metrics.json', 'w') as f:

        json.dump(accuracy_metrics, f, indent=4)

    print("\nDynamic 'accuracy_metrics.json' performance registry written successfully.")

    print("System artifact training pipeline completes successfully! 🎉")
 
if __name__ == '__main__':

    main()