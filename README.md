# WDBC Breast Cancer Classification — Neural Network from Scratch

A multilayer perceptron (MLP) implemented from scratch using **NumPy** for binary classification on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

The project demonstrates the fundamentals of neural-network implementation without relying on pre-built classifiers or deep-learning frameworks. Forward propagation, backpropagation, loss computation, gradient descent with momentum, L2 regularization, and numerical gradient checking are implemented manually.

> [!WARNING]
> **Educational project:** This is a machine-learning demonstration and is **not a medical diagnostic system**. It has not been clinically validated and must not be used for medical decisions.

---

## Overview

The model classifies cell nuclei measurements from the WDBC dataset into:

- **Benign**
- **Malignant**

The network takes **30 numerical features** as input and produces a probability representing the likelihood that a sample belongs to the malignant class.

### Model Pipeline

```text
WDBC Dataset
     |
     v
Train/Test Split
     |
     v
Feature Standardization
     |
     v
30 Input Features
     |
     v
16-Unit Hidden Layer
     |
    ReLU
     |
     v
1 Output Neuron
     |
   Sigmoid
     |
     v
Malignant Probability
     |
     v
Binary Classification
