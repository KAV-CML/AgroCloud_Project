"""
Cloud Machine Learning - Individual Assignment Pipeline
Algorithm: K-Nearest Neighbors Classifier (Distance-Based Instance Node)
Author: Group Member 2
"""
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve
 
def run_member2_pipeline():
    print("="*70)
    print(" [MEMBER 2] RUNNING K-NEAREST NEIGHBORS (KNN) PIPELINE")
    print("="*70)
 
    # 1. Import Shared Data Node Splits
    try:
        X_train = pd.read_csv('X_train_scaled.csv')
        X_test = pd.read_csv('X_test_scaled.csv')
        y_train = pd.read_csv('y_train.csv').values.ravel()
        y_test = pd.read_csv('y_test.csv').values.ravel()
        encoder = joblib.load('label_encoder.pkl')
    except FileNotFoundError:
        print("[ERROR] Split matrices missing. Run 'prepare_data.py' first.")
        return
 
    # 2. Model Initialization
    # Distance weights ensure closer geographic environmental records contribute 
    # more heavily to multi-class target resolution.
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='minkowski',
        p=2,
        n_jobs=-1
    )
 
    # 3. Training Microsecond Profiling Telemetry
    # KNN is a lazy learner; training should be nearly instant as it just maps data references.
    print("\n[TELEMETRY] Initiating Model Training Execution Loop...")
    t_start_train = time.perf_counter()
    model.fit(X_train, y_train)
    t_end_train = time.perf_counter()
    training_latency = t_end_train - t_start_train
 
    # 4. High-Throughput Inference Profiling Telemetry
    # Inference is heavier as it calculates distance metrics against all points.
    t_start_inf = time.perf_counter()
    y_pred = model.predict(X_test)
    t_end_inf = time.perf_counter()
    inference_latency = t_end_inf - t_start_inf
 
    # 5. Advanced Cloud Metrics Extraction
    throughput = len(X_test) / inference_latency if inference_latency > 0 else 0
    accuracy = accuracy_score(y_test, y_pred)
 
    print("\n" + "#"*40)
    print("      MEMBER 2 EXPERIMENTAL LOGS")
    print("#"*40)
    print(f"• Model Index/Fit Latency   : {training_latency:.6f} Seconds")
    print(f"• Batch Inference Test Latency: {inference_latency:.6f} Seconds")
    print(f"• Computed Node Throughput   : {throughput:.2f} Transactions/Sec")
    print(f"• Empirical Validation Accuracy: {accuracy * 100:.2f}%")
    print("#"*40)
 
    # 6. Detailed Analytics for Technical Error Reports
    print("\n[METRIC REPORT] Compiling Class F1-Score Metrics Profiles...")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_[:len(np.unique(y_test))]))
 
    # 7. Serialize Weights for Central Dashboard Mounting
    joblib.dump(model, 'knn_model.pkl')
    print("[SAVED] Exported 'knn_model.pkl' successfully.")
 
    # 8. Automated Academic Learning Curve Generation (Required by Project Brief)
    print("\n[GRAPHICS] Constructing Validation Learning Curve Array...")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=3, scoring='accuracy', 
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42, n_jobs=-1
    )
 
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='purple', label='Training Profile Metric')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='teal', label='Stratified CV Metric')
    plt.title('Member 2: KNN Empirical Learning Curves')
    plt.xlabel('Volumetric Sample Training Dimensions')
    plt.ylabel('Mathematical Accuracy Score Metric')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('member2_learning_curve.png', dpi=300)
    print("[SAVED] Learning curve plot exported as 'member2_learning_curve.png'.")
    print("="*70)
 
if __name__ == "__main__":
    run_member2_pipeline()