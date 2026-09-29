"""
main.py - End-to-end execution of Breast Cancer Cell Classification using a
Neural Network implemented entirely from scratch in NumPy.

Workflow:
1. Load & inspect the Wisconsin Breast Cancer dataset.
2. Encode target (0 = Benign, 1 = Malignant).
3. Stratified split into Train (80%) and Test (20%).
4. Standardize features avoiding data leakage.
5. Instantiate & train scratch NumPy Neural Network.
6. Evaluate performance on Test Set (Accuracy, Precision, Recall, F1, Confusion Matrix).
7. Assert accuracy requirement (> 80%).
8. Generate and save diagnostic visualizations (Loss curve & Confusion Matrix).
"""

import os
import sys
import numpy as np

from src.dataset import prepare_breast_cancer_data
from src.neural_network import NeuralNetwork
from src.evaluate import (
    compute_metrics,
    print_evaluation_report,
    plot_loss_curve,
    plot_confusion_matrix,
)
from src.utils import set_seed, format_header


def run_pipeline() -> None:
    # 0. Set seed for complete reproducibility
    set_seed(42)
    print(format_header("BREAST CANCER CELL CLASSIFICATION - NUMPY FROM SCRATCH"))

    # 1. Load, inspect, encode, split, and scale data
    X_train, X_test, y_train, y_test, scaler, feature_names, target_names = prepare_breast_cancer_data(
        test_size=0.20,
        random_state=42,
        verbose=True
    )

    # 2. Build scratch Neural Network model
    # Architecture: Input (30) -> Hidden (16, ReLU) -> Output (1, Sigmoid)
    n_features = X_train.shape[1]
    nn = NeuralNetwork(
        n_input=n_features,
        n_hidden=16,
        n_output=1,
        learning_rate=0.08,
        l2_lambda=0.001,
        activation_type="relu",
        random_state=42
    )

    # 3. Train model with mini-batch gradient descent and momentum
    epochs = 250
    batch_size = 32
    history = nn.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=epochs,
        batch_size=batch_size,
        momentum=0.9,
        verbose=True,
        print_every=25
    )

    # 4. Predict on Test Set
    test_probs = nn.predict_proba(X_test)
    test_preds = nn.predict(X_test, threshold=0.5)

    # 5. Evaluate Metrics
    metrics = compute_metrics(y_test, test_preds, target_names=target_names)
    print_evaluation_report(metrics, split_name="Fixed 20% Held-Out Test Split")

    # 6. Verify accuracy constraint (> 80%)
    accuracy = metrics["accuracy"]
    print("\n" + "-" * 70)
    print(f"CRITICAL REQUIREMENT CHECK: Test Accuracy > 80.00%")
    print(f"Actual Achieved Test Accuracy: {accuracy * 100:.2f}%")
    if accuracy >= 0.80:
        print(">>> STATUS: PASSED (Accuracy exceeds minimum threshold!) <<<")
    else:
        print(">>> STATUS: FAILED (Accuracy below 80%) <<<")
        sys.exit(1)
    print("-" * 70 + "\n")

    # 7. Generate and save plots
    os.makedirs("plots", exist_ok=True)
    plot_loss_curve(history, save_path="plots/training_loss.png")
    plot_confusion_matrix(metrics["confusion_matrix"], target_names=["Benign", "Malignant"], save_path="plots/confusion_matrix.png")

    # 8. Save trained weights and scaler parameters for standalone inference
    os.makedirs("artifacts", exist_ok=True)
    np.savez(
        "artifacts/model_weights.npz",
        W1=nn.W1,
        b1=nn.b1,
        W2=nn.W2,
        b2=nn.b2,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_
    )
    print("[Model Saved] Model weights and scaler saved to artifacts/model_weights.npz")
    print(format_header("EXECUTION PIPELINE SUCCESSFULLY COMPLETED"))


if __name__ == "__main__":
    run_pipeline()
