import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

PROCESSED_DIR = os.path.join("data", "processed")
FIG_DIR = os.path.join("results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

print("X shape:", X.shape)
print("y shape:", y.shape)

class_counts = Counter(y)
print("\nClass distribution:")
for label, count in class_counts.items():
    print(f"{label}: {count}")

# 1. Class Distribution Plot
plt.figure(figsize=(8, 5))
plt.bar(class_counts.keys(), class_counts.values(), color="skyblue")
plt.title("Class Distribution (MIT-BIH Dataset)")
plt.xlabel("Beat Type (AAMI Class)")
plt.ylabel("Count")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
dist_path = os.path.join(FIG_DIR, "class_distribution.png")
plt.savefig(dist_path, dpi=150)
plt.close()
print(f"Saved distribution plot → {dist_path}")

# 2. Sample Beats Plot
classes_to_plot = ["N", "S", "V", "F", "Q"]
plt.figure(figsize=(10, 8))
for i, cls in enumerate(classes_to_plot):
    idx = np.where(y == cls)[0]
    if len(idx) == 0:
        continue
    
    plt.subplot(len(classes_to_plot), 1, i+1)
    plt.plot(X[idx[0]], label=f"Class {cls}", color="C"+str(i))
    plt.title(f"Example Beat: Class {cls}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

plt.tight_layout()
samples_path = os.path.join(FIG_DIR, "dataset_samples.png")
plt.savefig(samples_path, dpi=150)
plt.close()
print(f"Saved dataset samples plot → {samples_path}")
    