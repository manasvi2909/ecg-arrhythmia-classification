import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.cnn_model import ECGCNN
from src.data.label_utils import convert_to_binary


def train():
    # Paths
    PROCESSED_DIR = os.path.join("data", "processed")
    MODEL_DIR = os.path.join("results", "models")
    FIG_DIR = os.path.join("results", "figures")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Load raw data 
    print("Loading data...")
    X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
    y_raw = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

    print(f"Raw X shape: {X.shape}")
    print(f"Raw y shape: {y_raw.shape}")

    # 2. Binary label conversion (centralized) 
    y = convert_to_binary(y_raw)
    num_classes = 2
    class_map = {0: "Normal (N)", 1: "Abnormal"}
    print(f"Classes: {class_map}")
    print(f"Distribution → Normal: {(y == 0).sum()}, Abnormal: {(y == 1).sum()}")

    # 3. Normalize ECG signals (z-score) 
    mean = X.mean()
    std = X.std()
    X = (X - mean) / std
    print(f"Normalized X - mean: {X.mean():.4f}, std: {X.std():.4f}")

    # 4. Train / Validation / Test split (70/15/15) 
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nSplit sizes → Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # 5. PyTorch tensors & loaders 
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    BATCH_SIZE = 32

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t),
                             batch_size=BATCH_SIZE, shuffle=False)

    # 6. Device 
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # 7. Class weights 
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {dict(zip(class_map.values(), class_weights))}")

    # 8. Model, loss, optimizer
    model = ECGCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 9. Training loop 
    NUM_EPOCHS = 20
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = os.path.join(MODEL_DIR, "best_cnn.pth")

    print("\nStarting training...")
    print("-" * 55)

    for epoch in range(NUM_EPOCHS):
        # Train phase 
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Save best model checkpoint 
        saved_tag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            saved_tag = "  ✓ saved"

        print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}]  "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}{saved_tag}")

    print("-" * 55)

    # 10. Load best model for evaluation 
    print(f"\nLoading best model (val_loss={best_val_loss:.4f})...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    # 11. Evaluation on TEST set 
    print("Evaluating on held-out TEST set...")
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # P(Abnormal)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    target_names = ["Normal (N)", "Abnormal"]

    print("\n" + "=" * 55)
    print("  CNN EVALUATION - FOCUS ON RECALL / F1")
    print("=" * 55)
    print(classification_report(all_targets, all_preds, target_names=target_names))

    # 12. ROC-AUC (threshold-independent) 
    roc_auc = roc_auc_score(all_targets, all_probs)
    print(f"  ROC-AUC Score: {roc_auc:.4f}")

    # 13. Confusion matrix with interpretation 
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(cm)
    print(f"\n  True Negatives  (Normal → Normal):    {tn}")
    print(f"  False Positives (Normal → Abnormal):   {fp}")
    print(f"  False Negatives (Abnormal → Normal):   {fn}")
    print(f"  True Positives  (Abnormal → Abnormal): {tp}")
    print("=" * 55)

    # 14. Confusion matrix heatmap 
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("CNN - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, "cnn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix heatmap saved → {cm_path}")

    # 15. Loss curve plot 
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, NUM_EPOCHS + 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs_range, val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN - Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(FIG_DIR, "cnn_loss_curve.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"Loss curve saved → {loss_path}")

    # 16. Loss trajectory summary 
    print("\nLoss Trajectory:")
    print(f"  Train - start: {train_losses[0]:.4f}, end: {train_losses[-1]:.4f}")
    print(f"  Val   - start: {val_losses[0]:.4f}, end: {val_losses[-1]:.4f}")

    if val_losses[-1] > val_losses[-3]:
        print(" Validation loss increased in last epochs - potential overfitting")
    else:
        print(" Validation loss is stable or decreasing")

    print(f"\nBest model saved → {best_model_path}")


if __name__ == "__main__":
    train()
