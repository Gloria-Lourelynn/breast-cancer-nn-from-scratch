# breast_cancer_nn

A complete Python project for breast cancer cell classification implemented with an artificial neural network built entirely from scratch using **NumPy**.

The project classifies cell nuclei measurements from the Wisconsin Diagnostic Breast Cancer (WDBC) benchmark dataset as **benign** or **malignant**. Scikit-learn is used strictly for dataset loading, stratified train/test splitting, feature standardization (`StandardScaler`), and evaluation metrics. No pre-built classifiers (such as scikit-learn's `LogisticRegression` or `MLPClassifier`) or deep learning frameworks (TensorFlow, PyTorch, Keras) are used.

---

> [!WARNING]
> ### Medical & Educational Disclaimer
> **This project is an educational machine-learning demonstration using the Wisconsin Diagnostic Breast Cancer dataset. It is not a medical diagnostic system, has not been clinically validated, and must not be used for medical decisions.**
>
> The performance metrics reported below represent benchmark evaluation on a historical public dataset split, not clinical efficacy or diagnostic accuracy in medical practice.

---

## Dataset Description

- **Dataset**: Wisconsin Diagnostic Breast Cancer (WDBC) benchmark dataset (available through scikit-learn via `load_breast_cancer`).
- **Total Samples**: 569 samples.
  - Benign: 357 samples (62.7%)
  - Malignant: 212 samples (37.3%)
- **Input Features**: 30 continuous numerical features computed from digitized images of fine needle aspirate (FNA) of breast masses (mean, standard error, and "worst" values for 10 cell nucleus characteristics: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension).
- **Missing Values**: 0 (complete dataset).
- **Target Encoding**:
  - `0`: Benign (negative class)
  - `1`: Malignant (positive class of interest)
  - Note: In scikit-learn's raw dataset, `0 = malignant` and `1 = benign`. The target is explicitly re-mapped ($y = 1 - y_{\text{raw}}$) so that `1` represents malignant, ensuring precision, recall, and F1 score directly reflect the identification of malignant samples.
- **Data Splitting & Leakage Prevention**:
  - Stratified 80/20 train/test split (`stratify=y`, `random_state=42`), producing 455 training samples and 114 test samples.
  - `StandardScaler` is fitted **strictly on the training data** (`scaler.fit_transform(X_train)`). The test data is transformed using the frozen training parameters (`scaler.transform(X_test)`), completely avoiding data leakage.

---

## Performance on the Fixed 20% Held-Out Test Split

Evaluated on the **114 held-out samples from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset**:

| Metric | Target Requirement | Achieved Result |
| :--- | :--- | :--- |
| **Accuracy** | **> 80.00%** | **99.12%** |
| **Precision** (Malignant) | - | **100.00%** |
| **Recall / Sensitivity** (Malignant) | - | **97.62%** |
| **F1 Score** | - | **98.80%** |

### Confusion Matrix

```text
                   Predicted Benign (0)  Predicted Malignant (1)
Actual Benign (0)           72                     0
Actual Malignant (1)         1                    41
```

- **True Negatives (TN)** = 72 (Benign correctly identified)
- **False Positives (FP)** = 0 (Type I Error: Benign misclassified as Malignant)
- **False Negatives (FN)** = 1 (Type II Error: Malignant misclassified as Benign)
- **True Positives (TP)** = 41 (Malignant correctly identified)

---

## Neural Network Architecture (NumPy from Scratch)

### 1. Topology
$$\text{Input Layer } (30) \xrightarrow{W_1, b_1} \text{Hidden Layer } (16, \text{ReLU}) \xrightarrow{W_2, b_2} \text{Output Layer } (1, \text{Sigmoid})$$

- **Input Layer**: $n_{\text{in}} = 30$ features.
- **Hidden Layer**: $n_{\text{hidden}} = 16$ units.
  - Weights $W_1 \in \mathbb{R}^{30 \times 16}$ initialized with He Normal: $\mathcal{N}\left(0, \sqrt{\frac{2}{30}}\right)$.
  - Biases $b_1 \in \mathbb{R}^{1 \times 16}$ initialized to zeros.
  - Activation: $\text{ReLU}(z) = \max(0, z)$.
- **Output Layer**: $n_{\text{out}} = 1$ unit.
  - Weights $W_2 \in \mathbb{R}^{16 \times 1}$ initialized with Xavier Normal: $\mathcal{N}\left(0, \sqrt{\frac{2}{16 + 1}}\right)$.
  - Bias $b_2 \in \mathbb{R}^{1 \times 1}$ initialized to zero.
  - Activation: Numerically stable Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$.

### 2. Forward Propagation
Given an input batch $X \in \mathbb{R}^{m \times 30}$:
1. $Z_1 = X W_1 + b_1$
2. $A_1 = \text{ReLU}(Z_1) = \max(0, Z_1)$
3. $Z_2 = A_1 W_2 + b_2$
4. $A_2 = \sigma(Z_2) = \frac{1}{1 + e^{-Z_2}}$ (predicted malignant probability)

### 3. Binary Cross-Entropy Loss with $L_2$ Regularization
$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \ln(A_2^{(i)} + \epsilon) + (1 - y^{(i)}) \ln(1 - A_2^{(i)} + \epsilon) \right] + \frac{\lambda}{2m} \left( \|W_1\|_F^2 + \|W_2\|_F^2 \right)$$
where $\epsilon = 10^{-15}$ ensures numerical stability, and $\lambda = 0.001$ provides weight decay regularization.

### 4. Analytical Backpropagation
Computed manually using the multivariate chain rule:
- **Output Layer**:
  $$dZ_2 = A_2 - Y \quad \in \mathbb{R}^{m \times 1}$$
  $$dW_2 = \frac{1}{m} A_1^T dZ_2 + \frac{\lambda}{m} W_2 \quad \in \mathbb{R}^{16 \times 1}$$
  $$db_2 = \frac{1}{m} \sum_{i=1}^{m} dZ_2 \quad \in \mathbb{R}^{1 \times 1}$$
- **Hidden Layer**:
  $$dA_1 = dZ_2 W_2^T \quad \in \mathbb{R}^{m \times 16}$$
  $$dZ_1 = dA_1 \odot \mathbb{I}(Z_1 > 0) \quad \in \mathbb{R}^{m \times 16}$$
  $$dW_1 = \frac{1}{m} X^T dZ_1 + \frac{\lambda}{m} W_1 \quad \in \mathbb{R}^{30 \times 16}$$
  $$db_1 = \frac{1}{m} \sum_{i=1}^{m} dZ_1 \quad \in \mathbb{R}^{1 \times 16}$$

### 5. Gradient Descent with Momentum
Parameters are updated using momentum velocity ($\beta = 0.9$, learning rate $\alpha = 0.08$):
$$V_{d\theta} = \beta V_{d\theta} + (1 - \beta) d\theta$$
$$\theta \leftarrow \theta - \alpha V_{d\theta}$$

---

## Project Structure

```text
breast_cancer_nn/
│
├── src/
│   ├── __init__.py
│   ├── dataset.py            # WDBC dataset loading, inspection, encoding, leakage-free scaling
│   ├── neural_network.py     # Pure NumPy neural network (forward, backward, BCE, GD)
│   ├── evaluate.py           # Metrics calculation, console reporting, and plot generation
│   └── utils.py              # Reproducibility seed setup and formatting helpers
│
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py       # Validates split shapes, class balance, and lack of leakage
│   └── test_neural_network.py# Numerical gradient check (< 1e-5 error) & >80% accuracy assertion
│
├── plots/
│   ├── training_loss.png     # Optimization loss curve across epochs
│   └── confusion_matrix.png  # Confusion matrix heatmap
│
├── artifacts/
│   └── model_weights.npz     # Serialized trained weights and scaler parameters
│
├── main.py                   # End-to-end training and evaluation script
├── predict.py                # Standalone inference on sample WDBC data
├── requirements.txt          # Pinned dependencies
└── README.md                 # Project documentation and disclaimer
```

---

## Setup & Execution

### 1. Environment Setup
Create and activate the virtual environment:
```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
Train the neural network, evaluate on the held-out test split, and generate plots:
```powershell
.venv\Scripts\python.exe main.py
```
This executes the full pipeline and outputs:
- Dataset inspection summary (569 samples, 30 features, class counts)
- Training progression across 250 epochs
- Test set performance report:
  - Accuracy: 99.12%
  - Precision: 100.00%
  - Recall: 97.62%
  - F1 Score: 98.80%
  - Confusion Matrix (TN=72, FP=0, FN=1, TP=41)
- Generated plots in `plots/`
- Saved model weights in `artifacts/model_weights.npz`

### 3. Run Standalone Inference
Perform predictions on individual samples using saved weights without retraining:
```powershell
.venv\Scripts\python.exe predict.py
```

### 4. Run Automated Tests
Execute the test suite with pytest:
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```
All 7 unit tests pass, validating:
- Dataset shapes, class balance, and lack of data leakage
- Parameter initialization and forward pass output range
- **Numerical gradient checking**: Verifies analytical backpropagation against finite difference approximations with relative error $< 10^{-5}$
- **Accuracy constraint**: Verifies test accuracy is $\ge 80\%$
#
