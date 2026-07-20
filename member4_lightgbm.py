
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve

def run_member4_pipeline():
    print("="*70)
    print(" [MEMBER 4] RUNNING LIGHTGBM GRADIENT BOOSTING PIPELINE")
    print("="*70)

    # Import shared data 
    try:
        X_train = pd.read_csv('X_train_scaled.csv')
        X_test = pd.read_csv('X_test_scaled.csv')
        y_train = pd.read_csv('y_train.csv').values.ravel()
        y_test = pd.read_csv('y_test.csv').values.ravel()
        encoder = joblib.load('label_encoder.pkl')
    except FileNotFoundError:
        print("[ERROR] Split matrices missing. Run 'prepare_data.py' first.")
        return

    #  Model initialization

    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(np.unique(y_train)),
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )


    print("\n[TELEMETRY] Initiating Model Training Execution Loop...")
    t_start_train = time.perf_counter()
    model.fit(X_train, y_train)
    t_end_train = time.perf_counter()
    training_latency = t_end_train - t_start_train
    t_start_inf = time.perf_counter()
    y_pred = model.predict(X_test)
    t_end_inf = time.perf_counter()
    inference_latency = t_end_inf - t_start_inf

    throughput = len(X_test) / inference_latency if inference_latency > 0 else 0
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "#"*40)
    print("      MEMBER 4 EXPERIMENTAL LOGS")
    print("#"*40)
    print(f"• Model Training Latency     : {training_latency:.6f} Seconds")
    print(f"• Batch Inference Test Latency: {inference_latency:.6f} Seconds")
    print(f"• Computed Node Throughput   : {throughput:.2f} Transactions/Sec")
    print(f"• Empirical Validation Accuracy: {accuracy * 100:.2f}%")
    print("#"*40)

    print("\n[METRIC REPORT] Compiling Class F1-Score Metrics Profiles...")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_[:len(np.unique(y_test))]))

    joblib.dump(model, 'lightgbm_model.pkl')
    print("[SAVED] Exported 'lightgbm_model.pkl' successfully.")

    #  Academic learning curve generation 
    print("\n[GRAPHICS] Constructing Validation Learning Curve Array...")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=3, scoring='accuracy', 
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42, n_jobs=-1
    )

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='crimson', label='Training Profile Metric')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='navy', label='Stratified CV Metric')
    plt.title('Member 4: LightGBM Empirical Learning Curves')
    plt.xlabel('Volumetric Sample Training Dimensions')
    plt.ylabel('Mathematical Accuracy Score Metric')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('member4_learning_curve.png', dpi=300)
    print("[SAVED] Learning curve plot exported as 'member4_learning_curve.png'.")
    print("="*70)

if __name__ == "__main__":
    run_member4_pipeline()