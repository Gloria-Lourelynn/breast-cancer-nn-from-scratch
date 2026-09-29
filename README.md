# WDBC MLP from Scratch

A multilayer perceptron (MLP) implemented **from scratch using NumPy** for binary classification on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

The project demonstrates the fundamentals of neural-network implementation without relying on pre-built classifiers or deep-learning frameworks. Forward propagation, backpropagation, loss computation, gradient descent with momentum, regularization, and numerical gradient checking are implemented manually.

> **Educational project:** This is a machine-learning demonstration and is **not a medical diagnostic system**. It has not been clinically validated and must not be used for medical decisions.

---

## Overview

This project classifies breast cell nuclei measurements as:

* **Benign**
* **Malignant**

The neural network takes 30 numerical features from the WDBC dataset and predicts the probability that a sample belongs to the malignant class.

### Pipeline

```text
WDBC Dataset
     │
     ▼
Train/Test Split
     │
     ▼
Standardization
     │
     ▼
30 Input Features
     │
     ▼
16-Unit Hidden Layer
     │
   ReLU
     │
     ▼
1 Output Neuron
     │
  Sigmoid
     │
     ▼
Malignant Probability
     │
     ▼
Binary Classification
```

---

## Key Features

* Neural network implemented entirely with **NumPy**
* Manual forward propagation
* Manual analytical backpropagation
* Binary cross-entropy loss
* L2 weight regularization
* He and Xavier weight initialization
* ReLU hidden-layer activation
* Numerically stable sigmoid activation
* Gradient descent with momentum
* Leakage-free feature standardization
* Stratified train/test split
* Numerical gradient checking
* Automated unit tests with `pytest`
* Confusion matrix and training-loss visualization
* Standalone inference using saved model weights

---

## Dataset

The project uses the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset available through scikit-learn's `load_breast_cancer` dataset loader.

| Property         | Value |
| ---------------- | ----: |
| Total samples    |   569 |
| Features         |    30 |
| Benign           |   357 |
| Malignant        |   212 |
| Missing values   |     0 |
| Training samples |   455 |
| Test samples     |   114 |

The 30 features represent measurements of cell nuclei obtained from digitized images of fine needle aspirate (FNA) samples.

They include measurements such as:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave points
* Symmetry
* Fractal dimension

For each characteristic, the dataset provides mean, standard error, and worst-case measurements.

### Target Encoding

The target is explicitly encoded as:

```text
0 → Benign
1 → Malignant
```

This makes the malignant class the positive class for precision, recall, and F1-score calculations.

---

## Data Preprocessing

The dataset is divided using a **stratified 80/20 train/test split**:

```text
Training: 455 samples
Testing:  114 samples
```

A `StandardScaler` is fitted **only on the training data**:

```python
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

This prevents information from the test set from influencing the preprocessing stage.

### Why this matters

Fitting the scaler on the entire dataset before splitting would allow information from the test set to leak into training.

The project deliberately avoids this form of data leakage.

---

# Neural Network

The model is a small **Multi-Layer Perceptron (MLP)**:

```text
30 Input Features
       │
       ▼
16 Hidden Neurons
       │
      ReLU
       │
       ▼
1 Output Neuron
       │
     Sigmoid
       │
       ▼
Malignant Probability
```

### Architecture

| Layer  | Units | Activation |
| ------ | ----: | ---------- |
| Input  |    30 | —          |
| Hidden |    16 | ReLU       |
| Output |     1 | Sigmoid    |

---

## Forward Propagation

For an input matrix:

```text
X ∈ R^(m × 30)
```

the network computes:

### Hidden layer

```text
Z₁ = XW₁ + b₁
```

```text
A₁ = ReLU(Z₁)
```

where:

```text
ReLU(z) = max(0, z)
```

### Output layer

```text
Z₂ = A₁W₂ + b₂
```

```text
A₂ = sigmoid(Z₂)
```

The output represents the predicted probability of the malignant class.

---

# Loss Function

The network uses **Binary Cross-Entropy (BCE)** with L2 regularization.

```text
L = BCE + L2 penalty
```

The BCE component is:

```text
       1
L = - ─── Σ [y log(p) + (1-y) log(1-p)]
       m
```

L2 regularization is applied to the weights:

```text
λ
── (||W₁||² + ||W₂||²)
2m
```

The implementation uses:

```text
λ = 0.001
```

A small epsilon is also used when computing logarithms to improve numerical stability.

---

# Backpropagation

The gradients are derived and implemented manually using the chain rule.

For the output layer:

```text
dZ₂ = A₂ - Y
```

```text
dW₂ = (1/m) A₁ᵀdZ₂ + (λ/m)W₂
```

```text
db₂ = (1/m) ΣdZ₂
```

For the hidden layer:

```text
dA₁ = dZ₂W₂ᵀ
```

```text
dZ₁ = dA₁ ⊙ ReLU'(Z₁)
```

```text
dW₁ = (1/m) XᵀdZ₁ + (λ/m)W₁
```

```text
db₁ = (1/m) ΣdZ₁
```

This means the project does not rely on automatic differentiation from frameworks such as PyTorch or TensorFlow.

---

# Optimization

Parameters are updated using **gradient descent with momentum**.

The momentum coefficient is:

```text
β = 0.9
```

and the learning rate is:

```text
α = 0.08
```

The velocity update is:

```text
V = βV + (1 - β)dθ
```

followed by:

```text
θ = θ - αV
```

The network is trained for:

```text
250 epochs
```

---

# Weight Initialization

Different initialization strategies are used for the network layers.

### Hidden layer

The weights use **He Normal initialization**, which is suitable for ReLU activations.

```text
W₁ ~ N(0, √(2/30))
```

### Output layer

The weights use **Xavier Normal initialization**:

```text
W₂ ~ N(0, √(2/(16+1)))
```

Biases are initialized to zero.

---

# Results

The model was evaluated on the fixed **114-sample held-out test set**.

| Metric               |      Result |
| -------------------- | ----------: |
| Accuracy             |  **99.12%** |
| Precision            | **100.00%** |
| Recall / Sensitivity |  **97.62%** |
| F1 Score             |  **98.80%** |

### Confusion Matrix

```text
                    Predicted
                 Benign  Malignant
Actual Benign       72       0
Actual Malignant     1      41
```

This corresponds to:

```text
TN = 72
FP = 0
FN = 1
TP = 41
```

The model therefore correctly classified 113 of the 114 held-out samples.

> These results describe performance on this particular train/test split of a historical benchmark dataset. They should not be interpreted as clinical diagnostic performance.

---

# Numerical Gradient Checking

One of the main goals of the project is to verify that the manually implemented backpropagation is mathematically correct.

The analytical gradients are compared against numerical gradients calculated using finite differences.

Conceptually:

```text
Numerical Gradient ≈
    [L(θ + ε) - L(θ - ε)]
    ─────────────────────────
             2ε
```

The test suite checks that the relative error between the analytical and numerical gradients remains below:

```text
1 × 10⁻⁵
```

This provides an independent check of the manually derived backpropagation implementation.

---

# Project Structure

```text
wdbc-mlp-from-scratch/
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   │   └── Dataset loading, encoding, splitting and scaling
│   │
│   ├── neural_network.py
│   │   └── NumPy MLP, forward pass, loss and backpropagation
│   │
│   ├── evaluate.py
│   │   └── Metrics, reporting and visualization
│   │
│   └── utils.py
│       └── Reproducibility and helper functions
│
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   │   └── Dataset and preprocessing tests
│   │
│   └── test_neural_network.py
│       └── Network, gradient and accuracy tests
│
├── plots/
│   ├── training_loss.png
│   └── confusion_matrix.png
│
├── artifacts/
│   └── model_weights.npz
│
├── main.py
│   └── End-to-end training and evaluation
│
├── predict.py
│   └── Standalone inference
│
├── requirements.txt
└── README.md
```

---

# Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/wdbc-mlp-from-scratch.git
cd wdbc-mlp-from-scratch
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

# Usage

## Train and evaluate

Run:

```powershell
python main.py
```

This will:

1. Load the WDBC dataset
2. Inspect the dataset
3. Create the stratified train/test split
4. Standardize the features
5. Initialize the MLP
6. Train for 250 epochs
7. Evaluate the test set
8. Display classification metrics
9. Generate the training-loss plot
10. Generate the confusion matrix
11. Save model parameters

---

## Run inference

After training:

```powershell
python predict.py
```

The inference script loads the saved model parameters and performs prediction without retraining the network.

---

## Run tests

Run the complete test suite:

```powershell
python -m pytest tests/ -v
```

The tests cover:

* Dataset dimensions
* Class distribution
* Train/test split
* Preprocessing
* Parameter initialization
* Forward-pass output
* Numerical gradient checking
* Model accuracy

---

# Technologies

* **Python**
* **NumPy**
* **scikit-learn**
* **Matplotlib**
* **pytest**

### Frameworks deliberately not used

The neural network itself does **not** use:

* TensorFlow
* PyTorch
* Keras
* `sklearn.neural_network.MLPClassifier`
* `sklearn.linear_model.LogisticRegression`

Scikit-learn is used only for dataset access, preprocessing, splitting, and evaluation utilities.

---

# What This Project Demonstrates

This project focuses on understanding what happens **inside a neural network**, rather than simply calling a pre-built classifier.

It demonstrates practical understanding of:

* Matrix multiplication
* Neural-network architecture
* Activation functions
* Forward propagation
* Loss functions
* Partial derivatives
* Chain rule
* Backpropagation
* Gradient descent
* Momentum optimization
* Weight initialization
* L2 regularization
* Numerical gradient checking
* Data leakage prevention
* Classification metrics
* Model serialization
* Automated testing

---

# Limitations

This project has several important limitations.

### Dataset size

The WDBC dataset contains only 569 samples, so the evaluation should not be treated as evidence of real-world clinical performance.

### Fixed test split

The reported metrics come from one fixed 80/20 train/test split. Performance may vary with different splits or cross-validation procedures.

### Simple architecture

The model contains only one hidden layer with 16 neurons. It is intentionally small because the goal is to understand the underlying mechanics rather than maximize model complexity.

### No clinical validation

The model has not undergone clinical validation, prospective testing, external validation, or regulatory evaluation.

### Benchmark ≠ diagnosis

A high score on a public benchmark does not imply that the model can safely diagnose patients.

---

# Future Improvements

Possible extensions include:

* K-fold cross-validation
* Hyperparameter experiments
* Learning-rate scheduling
* Early stopping
* Additional hidden layers
* Dropout implementation from scratch
* Class-weighted loss
* ROC-AUC and PR-AUC evaluation
* Experiment tracking
* Model calibration
* Comparison against simpler baselines
* External dataset evaluation
* More extensive ablation studies

---

# Disclaimer

**This project is for educational and research purposes only.**

It uses the Wisconsin Diagnostic Breast Cancer benchmark dataset and is **not a medical diagnostic system**.

The reported metrics represent performance on a historical public dataset split and should not be interpreted as clinical efficacy, medical advice, or diagnostic accuracy.

**Do not use this software to make medical decisions.**

---

## License

Add your preferred license here, for example:

```text
MIT License
```

if you choose to release the project under the MIT License.

- **Numerical gradient checking**: Verifies analytical backpropagation against finite difference approximations with relative error $< 10^{-5}$
- **Accuracy constraint**: Verifies test accuracy is $\ge 80\%$
#   b r e a s t - c a n c e r - n n - f r o m - s c r a t c h 
 
 
