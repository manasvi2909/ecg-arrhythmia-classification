import os
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.label_utils import convert_to_binary


def main():
    #Paths
    PROCESSED_DIR = os.path.join("data", "processed")
    FIG_DIR = os.path.join("results", "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Load raw data 
    print("Loading data...")
    X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
    y_raw = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

    # 2. Binary label conversion (centralized) 
    y = convert_to_binary(y_raw)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\nClass distribution (0=Normal, 1=Abnormal):")
    for label, count in Counter(y).items():
        print(f"  {label}: {count}")

    # 3. Engineered Features 
    def extract_features(beats):
        mean = beats.mean(axis=1)
        std = beats.std(axis=1)
        max_val = beats.max(axis=1)
        min_val = beats.min(axis=1)
        energy = (beats ** 2).sum(axis=1)
        return np.vstack([mean, std, max_val, min_val, energy]).T

    print("\nExtracting features...")
    X_feat = extract_features(X)
    print(f"Feature matrix shape: {X_feat.shape}")

    # 4. Train / Validation / Test split (70/15/15) 
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_feat, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Split sizes → Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # 5. Feature scaling 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Helper: evaluate, interpret, and visualize 
    def evaluate_model(name, y_true, y_pred, y_prob, fig_prefix):
        target_names = ["Normal", "Abnormal"]

        print("\n" + "=" * 55)
        print(f"  {name} - FOCUS ON RECALL / F1")
        print("=" * 55)
        print(classification_report(y_true, y_pred, target_names=target_names))

        # ROC-AUC
        roc_auc = roc_auc_score(y_true, y_prob)
        print(f"  ROC-AUC Score: {roc_auc:.4f}")

        # Confusion matrix interpretation
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print("\nConfusion Matrix:")
        print(cm)
        print(f"\n  True Negatives  (Normal → Normal):    {tn}")
        print(f"  False Positives (Normal → Abnormal):   {fp}")
        print(f"  False Negatives (Abnormal → Normal):   {fn}")
        print(f"  True Positives  (Abnormal → Abnormal): {tp}")
        print("=" * 55)

        # Heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_path = os.path.join(FIG_DIR, f"{fig_prefix}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"  Heatmap saved → {cm_path}")

    # 6. Logistic Regression (with class_weight) 
    print("\nTraining Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train_scaled, y_train)

    # Validate on val set first
    y_val_pred_lr = log_reg.predict(X_val_scaled)
    print(f"\n[Val] Logistic Regression:")
    print(classification_report(y_val, y_val_pred_lr, target_names=["Normal", "Abnormal"]))

    # Final evaluation on test set
    y_pred_lr = log_reg.predict(X_test_scaled)
    y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
    evaluate_model("LOGISTIC REGRESSION (TEST)", y_test, y_pred_lr, y_prob_lr, "logreg")

    # 7. Random Forest (with class_weight) 
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced_subsample"
    )
    rf.fit(X_train_scaled, y_train)

    # Validate on val set first
    y_val_pred_rf = rf.predict(X_val_scaled)
    print(f"\n[Val] Random Forest:")
    print(classification_report(y_val, y_val_pred_rf, target_names=["Normal", "Abnormal"]))

    # Final evaluation on test set
    y_pred_rf = rf.predict(X_test_scaled)
    y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
    evaluate_model("RANDOM FOREST (TEST)", y_test, y_pred_rf, y_prob_rf, "rf")


if __name__ == "__main__":
    main()
