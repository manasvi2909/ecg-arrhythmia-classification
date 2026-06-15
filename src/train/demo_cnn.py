"""
demo_cnn.py - Complete demo script for ECG arrhythmia detection.

  1. ECG beat visualization
  2. Pre-trained CNN inference
  3. Classification report + ROC-AUC
  4. Confusion matrix heatmap
  5. Loss curve display (from training)

"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import random
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.cnn_model import ECGCNN
from src.data.label_utils import convert_to_binary


def demo():
    # Paths 
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "models", "best_cnn.pth")
    FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Verify model exists 
    if not os.path.exists(MODEL_PATH):
        print("=" * 55)
        print("    No pre-trained model found!")
        print(f"  Expected: {MODEL_PATH}")
        print()
        print("  Run training first:")
        print("    python3 -m src.train.train_cnn")
        print("=" * 55)
        return

    # 2. Load data 
    print("=" * 55)
    print("   ECG Arrhythmia Detection - CNN Demo")
    print("=" * 55)

    print("\n Loading ECG data...")
    X_raw = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
    y_raw = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

    y = convert_to_binary(y_raw)
    print(f"   Dataset: {X_raw.shape[0]} beats × {X_raw.shape[1]} samples")
    print(f"   Normal:   {(y == 0).sum()}")
    print(f"   Abnormal: {(y == 1).sum()}")

    # ── 3. ECG beat visualization (Step 2 of demo) ──
    print("\n Generating ECG beat visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Normal beat (raw)
    normal_idx = np.where(y == 0)[0]
    axes[0].plot(X_raw[normal_idx[0]], color="#2196F3", linewidth=1.5)
    axes[0].set_title("Normal Beat (Class 0)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].grid(True, alpha=0.3)

    # Abnormal beat
    abnormal_idx = np.where(y == 1)[0]
    axes[1].plot(X_raw[abnormal_idx[0]], color="#F44336", linewidth=1.5)
    axes[1].set_title("Abnormal Beat (Class 1)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Segmented ECG Beats - Centered on R-Peak (360 samples)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    ecg_path = os.path.join(FIG_DIR, "demo_ecg_samples.png")
    plt.savefig(ecg_path, dpi=150)
    plt.close()
    print(f"   Saved → {ecg_path}")

    # Single beat plot (for Step 2 talking point)
    plt.figure(figsize=(10, 5))
    plt.plot(X_raw[0], color="#1976D2", linewidth=1.8)
    plt.title("Segmented ECG Beat (360 samples)", fontsize=14, fontweight="bold")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    single_path = os.path.join(FIG_DIR, "demo_single_beat.png")
    plt.savefig(single_path, dpi=150)
    plt.close()
    print(f"   Saved → {single_path}")

    # 3.1 Data Analysis: Class Distribution 
    print("\n Analyzing class distribution...")
    plt.figure(figsize=(8, 5))
    counts = [(y == 0).sum(), (y == 1).sum()]
    labels = ["Normal (0)", "Abnormal (1)"]
    sns.barplot(x=labels, y=counts, palette=["#2196F3", "#F44336"])
    plt.title("Class Distribution (Full Dataset)", fontsize=14, fontweight="bold")
    plt.ylabel("Count")
    for i, count in enumerate(counts):
        plt.text(i, count + 50, f"{count}", ha='center', fontweight='bold')
    plt.tight_layout()
    dist_path = os.path.join(FIG_DIR, "cnn_class_distribution.png")
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"   Saved → {dist_path}")

    # 3.2 Signal Statistics (Data Analysis) 
    print(" Calculating signal statistics...")
    means = X_raw.mean(axis=1)
    stds = X_raw.std(axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(means, bins=50, ax=axes[0], color="#4CAF50", kde=True)
    axes[0].set_title("Distribution of Sample Means", fontweight="bold")
    axes[0].set_xlabel("Mean Amplitude")
    
    sns.histplot(stds, bins=50, ax=axes[1], color="#FF9800", kde=True)
    axes[1].set_title("Distribution of Sample Stds", fontweight="bold")
    axes[1].set_xlabel("Std Deviation")
    
    plt.tight_layout()
    stats_path = os.path.join(FIG_DIR, "cnn_signal_statistics.png")
    plt.savefig(stats_path, dpi=150)
    plt.close()
    print(f"   Saved → {stats_path}")

    # 4. Preprocessing: Normalization Comparison 
    print("\n Preprocessing (Normalization)...")
    X = (X_raw - X_raw.mean()) / X_raw.std()
    
    # Visualization of preprocessing
    sample_idx = 0
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(X_raw[sample_idx], color="#607D8B")
    plt.title("Raw Signal (Original)", fontweight="bold")
    plt.ylabel("mV")
    
    plt.subplot(1, 2, 2)
    plt.plot(X[sample_idx], color="#3F51B5")
    plt.title("Normalized Signal (Z-score)", fontweight="bold")
    plt.ylabel("Standardized Units")
    
    plt.tight_layout()
    norm_path = os.path.join(FIG_DIR, "demo_preprocessing_comparison.png")
    plt.savefig(norm_path, dpi=150)
    plt.close()
    print(f"   Saved → {norm_path}")

    # 5. Load pre-trained model 
    print(f"\n Loading pre-trained model...")
    device = torch.device("cpu")  
    model = ECGCNN(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("   Model loaded successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Run inference 
    DEMO_SIZE = min(2000, len(X))
    X_sample = X[:DEMO_SIZE]
    y_sample = y[:DEMO_SIZE]

    X_tensor = torch.tensor(X_sample, dtype=torch.float32)

    print(f"\n Running inference on {DEMO_SIZE} beats...")
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

    preds_np = preds.numpy()
    probs_np = probs[:, 1].numpy()
    target_names = ["Normal (N)", "Abnormal"]

    # 7. Classification report 
    print("\n" + "=" * 55)
    print("   CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(y_sample, preds_np, target_names=target_names))

    # ROC-AUC
    roc_auc = roc_auc_score(y_sample, probs_np)
    print(f"   ROC-AUC Score: {roc_auc:.4f}")

    # 8. ROC and PR Curves 
    print("\n Generating ROC and Precision-Recall curves...")
    fpr, tpr, _ = roc_curve(y_sample, probs_np)
    precision, recall, _ = precision_recall_curve(y_sample, probs_np)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    axes[0].plot(fpr, tpr, color='#E91E63', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_title("Receiver Operating Characteristic", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)
    
    # PR Curve
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, color='#9C27B0', lw=2, label=f'PR (AUC = {pr_auc:.3f})')
    axes[1].set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    curve_path = os.path.join(FIG_DIR, "cnn_performance_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"   Saved → {curve_path}")

    # 8.1 Probability Distribution 
    plt.figure(figsize=(10, 5))
    sns.histplot(probs_np[y_sample == 0], bins=30, color="#2196F3", label="Actual Normal", kde=True, alpha=0.5)
    sns.histplot(probs_np[y_sample == 1], bins=30, color="#F44336", label="Actual Abnormal", kde=True, alpha=0.5)
    plt.title("Distribution of Predicted Probabilities (Confidence)", fontsize=14, fontweight="bold")
    plt.xlabel("Probability of being Abnormal")
    plt.ylabel("Frequency")
    plt.legend()
    prob_path = os.path.join(FIG_DIR, "cnn_prob_distribution.png")
    plt.savefig(prob_path, dpi=150)
    plt.close()
    print(f"   Saved → {prob_path}")

    # 8. Confusion matrix with interpretation 
    cm = confusion_matrix(y_sample, preds_np)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "-" * 55)
    print("  Confusion Matrix Breakdown:")
    print("-" * 55)
    print(f"  True Negatives  (Normal -> Normal):    {tn}")
    print(f"  False Positives (Normal -> Abnormal):   {fp}")
    print(f"  False Negatives (Abnormal -> Normal):   {fn}")
    print(f"  True Positives  (Abnormal -> Abnormal): {tp}")
    print("=" * 55)

    # 9. Confusion matrix heatmap 
    print("\n Generating confusion matrix heatmap...")
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 18})
    plt.title("CNN - Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, "cnn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"   Saved → {cm_path}")

    # 9.1 Prediction Grid 
    print("\n Generating prediction sample grid...")
    plt.figure(figsize=(12, 12))
    indices = random.sample(range(len(y_sample)), 9)
    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i+1)
        plt.plot(X_sample[idx], color="#555555", linewidth=1)
        gt = target_names[int(y_sample[idx])]
        pred = target_names[int(preds_np[idx])]
        color = "green" if y_sample[idx] == preds_np[idx] else "red"
        
        plt.title(f"GT: {gt}\nPred: {pred}", color=color, fontweight="bold", fontsize=10)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

    plt.suptitle("Sample Predictions (Random Selection)", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    grid_path = os.path.join(FIG_DIR, "demo_prediction_grid.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f"   Saved → {grid_path}")

    # 10. Check if loss curve exists (from training)
    loss_curve_path = os.path.join(FIG_DIR, "cnn_loss_curve.png")
    if os.path.exists(loss_curve_path):
        print(f"   Loss curve already exists → {loss_curve_path}")
    else:
        print(f"     Loss curve not found (run full training to generate)")

    # Summary
    print("\n" + "=" * 55)
    print("   DEMO COMPLETE")
    print("=" * 55)
    print(f"\n   Visual outputs in: {FIG_DIR}/")
    print(f"     • demo_single_beat.png      - Single ECG beat")
    print(f"     • demo_ecg_samples.png      - Normal vs Abnormal comparison")
    print(f"     • cnn_class_distribution.png - Data breakdown")
    print(f"     • cnn_signal_statistics.png  - Mean/Std distributions")
    print(f"     • demo_preprocessing_comparison.png - Normalization effect")
    print(f"     • cnn_performance_curves.png  - ROC & PR curves")
    print(f"     • cnn_prob_distribution.png   - Prediction confidence")
    print(f"     • cnn_confusion_matrix.png   - Heatmap")
    print(f"     • demo_prediction_grid.png   - 9 random samples with labels")
    if os.path.exists(loss_curve_path):
        print(f"     • cnn_loss_curve.png        - Train vs Val loss")
    print()


if __name__ == "__main__":
    demo()
