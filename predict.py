"""
predict.py - Standalone inference script for classifying new breast cancer cell samples.

Usage:
    python predict.py
"""

import os
import sys
import numpy as np
from src.neural_network import Activation


def load_model(weights_path: str = "artifacts/model_weights.npz"):
    """Loads saved neural network weights and scaler parameters."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file not found at '{weights_path}'. Please execute 'python main.py' first."
        )
    data = np.load(weights_path)
    model_params = {
        "W1": data["W1"],
        "b1": data["b1"],
        "W2": data["W2"],
        "b2": data["b2"],
        "scaler_mean": data["scaler_mean"],
        "scaler_scale": data["scaler_scale"],
    }
    return model_params


def predict_sample(
    sample_features: np.ndarray,
    model_params: dict,
    threshold: float = 0.5
) -> dict:
    """
    Classifies a raw (unscaled) cell sample.

    Args:
        sample_features: 1D or 2D array of shape (n_features,) or (m, n_features).
        model_params: Dictionary containing W1, b1, W2, b2, and scaler statistics.
        threshold: Probability cutoff for malignant diagnosis.

    Returns:
        result: Dictionary with predicted class, label, and malignant probability.
    """
    if sample_features.ndim == 1:
        sample_features = sample_features.reshape(1, -1)

    # 1. Standardize using training distribution statistics
    mean = model_params["scaler_mean"]
    scale = model_params["scaler_scale"]
    X_scaled = (sample_features - mean) / scale

    # 2. Forward Propagation in pure NumPy
    W1 = model_params["W1"]
    b1 = model_params["b1"]
    W2 = model_params["W2"]
    b2 = model_params["b2"]

    # Hidden Layer (ReLU)
    Z1 = np.dot(X_scaled, W1) + b1
    A1 = Activation.relu(Z1)

    # Output Layer (Sigmoid)
    Z2 = np.dot(A1, W2) + b2
    prob_malignant = Activation.sigmoid(Z2)

    predictions = []
    for prob in prob_malignant.flatten():
        is_malignant = bool(prob >= threshold)
        predictions.append({
            "probability_malignant": float(prob),
            "probability_benign": float(1.0 - prob),
            "class": 1 if is_malignant else 0,
            "diagnosis": "Malignant" if is_malignant else "Benign",
            "confidence": float(prob if is_malignant else (1.0 - prob)) * 100
        })

    return predictions[0] if len(predictions) == 1 else predictions


def demo_inference() -> None:
    """Demonstrates inference on sample cases from the dataset."""
    from sklearn.datasets import load_breast_cancer

    print("=" * 65)
    print("      BREAST CANCER CELL INFERENCE ENGINE (SCRATCH NUMPY)")
    print("=" * 65)

    params = load_model("artifacts/model_weights.npz")
    dataset = load_breast_cancer()

    # Find one known benign and one known malignant sample
    # Note: in raw sklearn, 0 = malignant, 1 = benign
    idx_malignant = np.where(dataset.target == 0)[0][0]
    idx_benign = np.where(dataset.target == 1)[0][0]

    test_samples = [
        ("WDBC Benchmark Malignant Sample", dataset.data[idx_malignant], "Malignant"),
        ("WDBC Benchmark Benign Sample", dataset.data[idx_benign], "Benign"),
    ]

    for title, sample, true_label in test_samples:
        result = predict_sample(sample, params)
        print(f"\nEvaluating: {title}")
        print(f"  Expected True Class   : {true_label}")
        print(f"  Model Prediction      : {result['diagnosis']}")
        print(f"  Malignant Probability : {result['probability_malignant']:.4f}")
        print(f"  Benign Probability    : {result['probability_benign']:.4f}")
        print(f"  Diagnosis Confidence  : {result['confidence']:.2f}%")
        match = "MATCH" if result["diagnosis"] == true_label else "MISMATCH"
        print(f"  Result Validation     : [{match}]")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    demo_inference()
