
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def initialize_shared_pipeline(dataset_path='crop_fertilizer_recommendation_dataset 1.csv'):
    print("[SYSTEM] Executing Central Masters-Level Data Preparation Pipeline...")

    df = pd.read_csv(dataset_path)

    df['joint_target'] = df['label'].astype(str) + " || " + df['fertilizer'].astype(str)

    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['joint_target']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoder, 'label_encoder.pkl')

    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('X_train_scaled.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('X_test_scaled.csv', index=False)
    pd.Series(y_train).to_csv('y_train.csv', index=False)
    pd.Series(y_test).to_csv('y_test.csv', index=False)
    
    print(f"[SUCCESS] Shared artifacts and data splits generated successfully!")
    print(f"Total Unique Co-joint Target Classes: {len(encoder.classes_)}")

if __name__ == "__main__":
    initialize_shared_pipeline()