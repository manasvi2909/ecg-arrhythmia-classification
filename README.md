# ECG Arrhythmia Detection

This was a two-person group project aimed at classifying ECG heartbeats as normal or abnormal using the MIT-BIH Arrhythmia Database from PhysioNet.

My contributions to the project included:
- Building the 1D CNN model in PyTorch for heartbeat classification.
- Implementing the training workflow, including data normalization, class-weighted loss, train-validation-test splitting, and model checkpointing.
- Preparing ECG heartbeat segments for deep learning training and inference.
- Evaluating model performance using ROC-AUC, confusion matrices, precision, recall, and F1-score metrics.
- Monitoring training and validation loss trends to analyze model generalization and overfitting.
- Creating scripts and visualizations for prediction analysis and model evaluation.

The dataset used is the MIT-BIH Arrhythmia Database from PhysioNet.

## Project Files

- `src/data/preprocess.py`: Extracts individual heartbeats centered around the R-peak (360 samples per beat) and labels them.
- `src/models/cnn_model.py`: The 1D CNN architecture in PyTorch.
- `src/models/classical_ml.py`: Trains and evaluates the baseline Logistic Regression and Random Forest models on handcrafted statistical features (mean, std, max, min, energy).
- `src/train/train_cnn.py`: Training script for the PyTorch CNN (handles data splits, class weights, and checkpointing).
- `src/train/demo_cnn.py`: A quick demo script to visualize sample beats and run inference using the trained CNN model.

## Setup and Running

1. **Install dependencies**:
   ```bash
   pip install torch numpy scikit-learn wfdb matplotlib seaborn
   ```

2. **Preprocess the data**:
   Download the MIT-BIH database files (.dat, .hea, .atr) and place them in `data/raw/mit-bih/`. Then run:
   ```bash
   python -m src.data.preprocess
   ```

3. **Train the CNN**:
   ```bash
   python -m src.train.train_cnn
   ```

4. **Run baseline models**:
   ```bash
   python -m src.models.classical_ml
   ```

5. **Run the demo**:
   ```bash
   python src/train/demo_cnn.py
   ```

## Results

I split the data into 70% training, 15% validation, and 15% testing sets. Here is how the models compared on the test set:

- **Logistic Regression**: 74% accuracy.
- **Random Forest**: 88% accuracy.
- **1D CNN**: 99.6% accuracy (F1-score of 0.99 for abnormal beats).

The CNN performs significantly better because it learns features directly from the raw heartbeat shape rather than relying on simple statistical aggregates.


