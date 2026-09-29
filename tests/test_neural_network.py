"""
test_neural_network.py - Unit tests and numerical gradient checks for scratch NeuralNetwork.
"""

import numpy as np
import pytest
from src.neural_network import NeuralNetwork, Activation
from src.dataset import prepare_breast_cancer_data


def test_parameter_initialization():
    nn = NeuralNetwork(n_input=30, n_hidden=16, n_output=1, random_state=42)
    assert nn.W1.shape == (30, 16)
    assert nn.b1.shape == (1, 16)
    assert nn.W2.shape == (16, 1)
    assert nn.b2.shape == (1, 1)
    assert np.all(nn.b1 == 0.0)
    assert np.all(nn.b2 == 0.0)


def test_forward_propagation_range_and_shapes():
    nn = NeuralNetwork(n_input=30, n_hidden=16, n_output=1, random_state=42)
    X = np.random.randn(10, 30)
    A2, cache = nn.forward(X)

    assert A2.shape == (10, 1)
    # Sigmoid outputs must be bounded in (0, 1)
    assert np.all(A2 > 0.0)
    assert np.all(A2 < 1.0)
    assert cache["Z1"].shape == (10, 16)
    assert cache["A1"].shape == (10, 16)
    assert cache["Z2"].shape == (10, 1)


def test_numerical_gradient_checking():
    """
    Validates analytical backpropagation gradients against two-sided finite difference approximations:
    grad_approx = (J(theta + eps) - J(theta - eps)) / (2 * eps)

    Relative error should be < 1e-6.
    """
    np.random.seed(123)
    n_in, n_h, n_out = 5, 4, 1
    m = 8

    X = np.random.randn(m, n_in)
    y = np.random.randint(0, 2, size=(m, 1)).astype(np.float64)

    # Disable regularization for pure gradient check
    nn = NeuralNetwork(n_input=n_in, n_hidden=n_h, n_output=n_out, l2_lambda=0.0, random_state=123)

    # 1. Compute analytical gradients via backpropagation
    A2, cache = nn.forward(X)
    grads = nn.backward(y, cache)

    # 2. Check each parameter tensor using finite differences
    eps = 1e-6
    for param_name, grad_name in [("W1", "dW1"), ("b1", "db1"), ("W2", "dW2"), ("b2", "db2")]:
        param = getattr(nn, param_name)
        analytical_grad = grads[grad_name]
        numerical_grad = np.zeros_like(param)

        # Iterate over all scalar elements of the parameter tensor
        it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig_val = param[idx]

            # J(theta + eps)
            param[idx] = orig_val + eps
            a_plus, _ = nn.forward(X)
            loss_plus = nn.compute_loss(y, a_plus)

            # J(theta - eps)
            param[idx] = orig_val - eps
            a_minus, _ = nn.forward(X)
            loss_minus = nn.compute_loss(y, a_minus)

            # Two-sided numerical gradient
            numerical_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)

            # Reset parameter
            param[idx] = orig_val
            it.iternext()

        # Compute relative Euclidean error
        numerator = np.linalg.norm(analytical_grad - numerical_grad)
        denominator = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad) + 1e-12
        relative_error = numerator / denominator

        assert relative_error < 1e-5, (
            f"Gradient check failed for {param_name}! Relative error: {relative_error:.2e}"
        )


def test_model_training_and_accuracy_above_80_percent():
    """
    Verifies that the neural network converges and achieves > 80% test accuracy
    on the Wisconsin Breast Cancer dataset as explicitly requested.
    """
    X_train, X_test, y_train, y_test, _, _, _ = prepare_breast_cancer_data(
        test_size=0.2, random_state=42, verbose=False
    )

    nn = NeuralNetwork(
        n_input=30,
        n_hidden=16,
        n_output=1,
        learning_rate=0.08,
        l2_lambda=0.001,
        random_state=42
    )

    nn.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=200,
        batch_size=32,
        momentum=0.9,
        verbose=False
    )

    preds = nn.predict(X_test)
    test_acc = float(np.mean(preds == y_test))

    print(f"\n[Test Result] Achieved Test Accuracy: {test_acc * 100:.2f}%")
    assert test_acc >= 0.80, f"Expected test accuracy >= 80%, got {test_acc * 100:.2f}%"
