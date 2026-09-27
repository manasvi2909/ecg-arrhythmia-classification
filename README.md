
# ECG Arrhythmia Detection

An end-to-end ECG heartbeat classification project using the **MIT-BIH Arrhythmia Database**, combining **classical machine learning** and **deep learning** to classify individual ECG beats as **Normal** or **Abnormal**.

The project covers the complete machine-learning workflow: ECG preprocessing, annotation-based labeling, feature engineering, classical ML modeling, 1D convolutional neural networks, class-imbalance handling, validation, test-set evaluation, and visual analysis.

## Overview

This project explores two different approaches to ECG heartbeat classification.

### Classical Machine Learning

The classical ML pipeline uses handcrafted statistical features extracted from each ECG heartbeat and evaluates two classifiers:

- Logistic Regression
- Random Forest

The extracted features are:

- Mean
- Standard deviation
- Maximum value
- Minimum value
- Signal energy

### Deep Learning

The deep-learning pipeline uses a custom **1D Convolutional Neural Network (CNN)** implemented in PyTorch.

Unlike the classical models, the CNN receives the complete 360-sample ECG waveform and learns feature representations directly from the signal.

This allows the project to compare:

**Handcrafted statistical representations vs learned waveform representations.**

---

## Dataset

The project uses the **MIT-BIH Arrhythmia Database** from PhysioNet.

The preprocessing pipeline reads ECG recordings and their corresponding beat annotations and extracts individual heartbeats from the first ECG channel.

### Beat Extraction

Each heartbeat is centered around its annotated R-peak:

- 180 samples before the R-peak
- 180 samples after the R-peak
- **360 samples per heartbeat**

Segments that fall outside the recording boundaries are discarded.

---

## Label Processing

The original MIT-BIH beat annotations are mapped into five AAMI-style classes.

| AAMI Class | Beat Symbols |
|---|---|
| N | N, L, R, e, j |
| S | A, a, J, S |
| V | V, E |
| F | F |
| Q | /, f, Q, P |

For model training, these five classes are converted into a binary classification problem:

```text
N → 0 : Normal

S, V, F, Q → 1 : Abnormal
````

The binary conversion is centralized in:

```text
src/data/label_utils.py
```

This ensures that both the classical ML and CNN pipelines use the same target definition.

---

## End-to-End Pipeline

```text
MIT-BIH Arrhythmia Database
            │
            ▼
     ECG Recordings
            │
            ▼
      Beat Annotations
            │
            ▼
      AAMI Label Mapping
       (N/S/V/F/Q)
            │
            ▼
   R-Peak-Centered Segments
         360 samples
            │
            ▼
     Binary Conversion
   Normal vs Abnormal
            │
      ┌─────┴─────┐
      │           │
      ▼           ▼
 Classical ML   Deep Learning
      │           │
      ▼           ▼
 Statistical    Raw ECG
  Features      Waveform
      │           │
 ┌────┴────┐      ▼
 │         │    1D CNN
 ▼         ▼      │
Logistic  Random  │
Regression Forest │
 │         │      │
 └────┬────┘      │
      │           │
      └─────┬─────┘
            ▼
     Held-Out Test Set
            │
            ▼
 Accuracy / Precision
 Recall / F1 / ROC-AUC
 Confusion Matrix
```

---

# Data Preprocessing

## 1. ECG Segmentation

`src/data/preprocess.py` loads the MIT-BIH recordings and annotations using `wfdb`.

For each annotated heartbeat:

1. The annotation's sample index is treated as the R-peak.
2. 180 samples are taken before the peak.
3. 180 samples are taken after the peak.
4. The resulting 360-sample segment is retained if it lies within the recording.
5. The corresponding annotation is mapped to an AAMI class.

The processed data is saved as:

```text
data/processed/X.npy
data/processed/y.npy
```

---

## 2. AAMI Mapping

The preprocessing stage groups individual MIT-BIH annotation symbols into:

```text
N
S
V
F
Q
```

This preserves the AAMI grouping before the final binary classification step.

---

## 3. Binary Classification

`src/data/label_utils.py` converts the five-class representation into:

```text
0 → Normal
1 → Abnormal
```

This function is shared by both the classical ML and CNN pipelines.

---

# Classical Machine Learning

The classical ML pipeline is implemented in:

```text
src/models/classical_ml.py
```

Instead of passing all 360 waveform samples directly to the classifiers, the project extracts five statistical descriptors from every heartbeat.

### Feature Engineering

For each heartbeat:

* Mean
* Standard deviation
* Maximum
* Minimum
* Signal energy

The resulting feature vector is:

```text
[mean, std, max, min, energy]
```

The features are standardized using `StandardScaler`.

---

## Logistic Regression

The first classical classifier is Logistic Regression.

Configuration:

```text
max_iter = 1000
class_weight = "balanced"
```

Class weighting is used to account for the imbalance between Normal and Abnormal beats.

---

## Random Forest

The second classical classifier is Random Forest.

Configuration:

```text
n_estimators = 100
random_state = 42
class_weight = "balanced_subsample"
```

The Random Forest operates on the same five engineered features used for Logistic Regression.

---

# Deep Learning: 1D CNN

The deep-learning model is implemented in:

```text
src/models/cnn_model.py
```

The CNN works directly on the 360-sample ECG waveform rather than the handcrafted statistical feature vector.

## Input

Each ECG beat is represented as:

```text
(Batch, 1, 360)
```

---

## Architecture

The network consists of three convolutional blocks followed by fully connected layers.

```text
Input ECG
360 samples
     │
     ▼
Conv1D
1 → 32 channels
kernel = 5
     │
BatchNorm
     │
ReLU
     │
MaxPool
     │
     ▼
Conv1D
32 → 64 channels
kernel = 5
     │
BatchNorm
     │
ReLU
     │
MaxPool
     │
     ▼
Conv1D
64 → 128 channels
kernel = 5
     │
BatchNorm
     │
ReLU
     │
MaxPool
     │
     ▼
Flatten
128 × 45 = 5760
     │
     ▼
Fully Connected
5760 → 256
     │
ReLU
     │
Dropout = 0.5
     │
     ▼
Fully Connected
256 → 2
     │
     ▼
Normal / Abnormal
```

---

# CNN Training

The training pipeline is implemented in:

```text
src/train/train_cnn.py
```

### Training Configuration

| Parameter     | Value                     |
| ------------- | ------------------------- |
| Epochs        | 20                        |
| Batch size    | 32                        |
| Optimizer     | Adam                      |
| Learning rate | 0.001                     |
| Loss          | Weighted CrossEntropyLoss |
| Data split    | 70% / 15% / 15%           |
| Random state  | 42                        |

The train/validation/test split is stratified to preserve the Normal/Abnormal class proportions.

---

## Signal Normalization

Before CNN training, the ECG signals are standardized using global Z-score normalization:

```text
X_normalized = (X - mean(X)) / std(X)
```

This produces approximately:

```text
Mean ≈ 0
Standard Deviation ≈ 1
```

---

## Class Imbalance Handling

The CNN computes class weights from the training data using:

```text
compute_class_weight(class_weight="balanced")
```

These weights are incorporated into:

```text
CrossEntropyLoss
```

The classical models similarly use class weighting.

This makes class imbalance an explicit part of both modeling pipelines.

---

# Model Selection

During CNN training, both training and validation loss are tracked across epochs.

Whenever validation loss improves, the model weights are saved.

The best checkpoint is stored as:

```text
results/models/best_cnn.pth
```

The saved best model is then loaded before evaluation on the held-out test set.

---

# Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

The evaluation code also reports:

```text
True Negatives
False Positives
False Negatives
True Positives
```

ROC-AUC is calculated using the predicted abnormal-class probabilities.

The CNN demo additionally generates:

* ROC curve
* Precision-Recall curve
* Predicted-probability distribution
* Sample prediction visualizations

---

# Results

Using the recorded project evaluation results:

| Model               |  Accuracy | Precision |   Recall | F1-score |  ROC-AUC |
| ------------------- | --------: | --------: | -------: | -------: | -------: |
| Logistic Regression |       74% |      0.73 |     0.72 |     0.73 |     0.79 |
| Random Forest       |       88% |      0.88 |     0.87 |     0.87 |     0.92 |
| 1D CNN              | **95.3%** |  **0.95** | **0.95** | **0.95** | **0.98** |

The experiment provides a direct comparison between two modeling strategies:

* Classical ML learns from a small set of handcrafted statistical descriptors.
* The CNN learns representations directly from the ECG waveform.

In the recorded experiment, the CNN achieved the highest reported performance across the listed metrics.

---

# Visualization and Analysis

The repository contains a dedicated set of analysis and visualization outputs.

## Dataset Analysis

The project generates:

* Class distribution plots
* Example ECG beats
* Signal mean distributions
* Signal standard-deviation distributions

These are generated through:

```text
src/data/analyze.py
```

---

## CNN Analysis

The CNN demo generates:

* Normal vs Abnormal ECG examples
* Single-beat waveform visualization
* Raw vs normalized signal comparison
* CNN class distribution
* Signal statistics
* Confusion matrix
* Training/validation loss curve
* ROC curve
* Precision-Recall curve
* Prediction probability distribution
* Random prediction grid

The generated figures are stored in:

```text
results/figures/
```

---

# Project Structure

```text
ecg-arrhythmia-classification/
│
├── src/
│   ├── data/
│   │   ├── analyze.py
│   │   ├── label_utils.py
│   │   └── preprocess.py
│   │
│   ├── models/
│   │   ├── classical_ml.py
│   │   └── cnn_model.py
│   │
│   └── train/
│       ├── demo_cnn.ipynb
│       ├── demo_cnn.py
│       └── train_cnn.py
│
├── scripts/
│   └── generate_visuals.py
│
├── results/
│   └── figures/
│
├── data/
│   ├── raw/
│   │   └── mit-bih/
│   │
│   └── processed/
│       ├── X.npy
│       └── y.npy
│
└── README.md
```

---

# File Descriptions

### `src/data/preprocess.py`

Loads MIT-BIH ECG recordings and annotations, maps beat symbols to AAMI classes, extracts R-peak-centered 360-sample heartbeat segments, and saves the processed arrays.

### `src/data/label_utils.py`

Provides the shared conversion from the five AAMI-style classes to the binary Normal/Abnormal target.

### `src/data/analyze.py`

Performs dataset-level analysis and generates class-distribution and sample-beat visualizations.

### `src/models/classical_ml.py`

Implements the classical machine-learning pipeline:

* Feature extraction
* Train/validation/test split
* Feature standardization
* Logistic Regression
* Random Forest
* Classification reports
* ROC-AUC
* Confusion matrices

### `src/models/cnn_model.py`

Defines the custom 1D CNN architecture used for raw ECG waveform classification.

### `src/train/train_cnn.py`

Implements the CNN training workflow:

* Data loading
* Binary label conversion
* Z-score normalization
* Stratified splitting
* PyTorch data loaders
* Class-weight calculation
* Weighted loss
* Adam optimization
* Validation
* Best-model checkpointing
* Test evaluation
* Loss visualization

### `src/train/demo_cnn.py`

Loads the trained CNN and provides an end-to-end inference and visualization demo, including classification metrics, ROC/PR curves, probability analysis, confusion matrix visualization, and sample predictions.

### `src/train/demo_cnn.ipynb`

Notebook version of the CNN demonstration and visualization workflow.

### `scripts/generate_visuals.py`

Coordinates dataset analysis, CNN training/demo execution, and generation of the visualization outputs.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/manasvi2909/ecg-arrhythmia-classification.git
cd ecg-arrhythmia-classification
```

Install the required dependencies:

```bash
pip install torch numpy scikit-learn wfdb matplotlib seaborn
```

---

# Dataset Setup

Download the required MIT-BIH Arrhythmia Database records from PhysioNet.

Place the corresponding:

```text
.dat
.hea
.atr
```

files inside:

```text
data/raw/mit-bih/
```

The records used by the project are defined in:

```text
src/data/preprocess.py
```

---

# Running the Project

## 1. Preprocess the ECG data

```bash
python -m src.data.preprocess
```

This generates:

```text
data/processed/X.npy
data/processed/y.npy
```

---

## 2. Analyze the dataset

```bash
python -m src.data.analyze
```

---

## 3. Train the classical ML models

```bash
python -m src.models.classical_ml
```

This trains:

```text
Logistic Regression
Random Forest
```

and generates their evaluation outputs and confusion matrices.

---

## 4. Train the CNN

```bash
python -m src.train.train_cnn
```

The best validation-loss checkpoint will be saved to:

```text
results/models/best_cnn.pth
```

---

## 5. Run the CNN demo

```bash
python src/train/demo_cnn.py
```

This loads the saved model and generates the available evaluation and visualization outputs.

---

# Reproducibility

The project uses:

```text
random_state = 42
```

for the dataset splitting and Random Forest configuration.

The CNN training script automatically selects:

```text
CUDA
↓
Apple MPS
↓
CPU
```

depending on hardware availability.

---

# Scope

This project is an educational/research implementation for experimenting with:

* Biomedical time-series data
* ECG preprocessing
* Signal segmentation
* AAMI-based label mapping
* Feature engineering
* Classical machine learning
* Deep learning
* Class-imbalance handling
* Model comparison
* Model evaluation and visualization

It is **not a clinical diagnostic system** and should not be used for medical decision-making.

