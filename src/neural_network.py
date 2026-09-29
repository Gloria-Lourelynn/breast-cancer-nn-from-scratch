"""
neural_network.py - Artificial Neural Network implemented entirely from scratch in NumPy.

Architecture:
    Input Layer (n_input) -> Hidden Layer (n_hidden) -> Output Layer (1)

All components implemented manually:
- Weights and Biases (He & Xavier initialization)
- Forward Propagation (Linear -> Activation)
- Activation Functions (ReLU, Sigmoid, and their analytical derivatives)
- Binary Cross-Entropy Loss with numerical stability
- Analytical Backpropagation (Gradients computation)
- Gradient Descent Optimizer (Standard and Momentum-accelerated)
- Training Loop with Mini-batching and Convergence Tracking
- Binary Prediction and Probability Estimation
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class Activation:
    """Activation functions and their analytical derivatives implemented in NumPy."""

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit: max(0, z)"""
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU: 1 if z > 0 else 0"""
        return (z > 0.0).astype(np.float64)

    @staticmethod
    def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU to prevent dying neurons: z if z > 0 else alpha * z"""
        return np.where(z > 0.0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU: 1 if z > 0 else alpha"""
        return np.where(z > 0.0, 1.0, alpha)

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable Sigmoid function: 1 / (1 + exp(-z)).
        Clips input to [-500, 500] to prevent arithmetic overflow in exp.
        """
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        """Derivative of Sigmoid given activation output a = sigmoid(z): a * (1 - a)"""
        return a * (1.0 - a)


class NeuralNetwork:
    """
    Two-Layer Feedforward Neural Network (Single Hidden Layer MLP) from Scratch.

    Architecture:
        Input Layer (n_x) -> Hidden Layer (n_h, ReLU) -> Output Layer (1, Sigmoid)

    Attributes:
        n_input (int): Number of input features.
        n_hidden (int): Number of neurons in the hidden layer.
        n_output (int): Number of output neurons (1 for binary classification).
        learning_rate (float): Learning rate for gradient descent.
        l2_lambda (float): L2 regularization weight decay coefficient.
        activation_type (str): Hidden activation function ('relu' or 'leaky_relu').
        W1, b1 (np.ndarray): Weights and biases for Hidden Layer.
        W2, b2 (np.ndarray): Weights and biases for Output Layer.
        history (dict): Training history metrics across epochs.
    """

    def __init__(
        self,
        n_input: int = 30,
        n_hidden: int = 16,
        n_output: int = 1,
        learning_rate: float = 0.05,
        l2_lambda: float = 0.001,
        activation_type: str = "relu",
        random_state: Optional[int] = 42
    ) -> None:
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.activation_type = activation_type.lower()
        self.random_state = random_state

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize network parameters
        self._initialize_parameters()

        # Optimizer state (momentum velocities)
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

        # History tracker
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def _initialize_parameters(self) -> None:
        """
        Initializes weights and biases:
        - Hidden Layer (W1): He (Kaiming) Normal initialization: N(0, sqrt(2 / n_in))
          Optimal for ReLU non-linearities.
        - Output Layer (W2): Xavier (Glorot) Normal initialization: N(0, sqrt(2 / (n_in + n_out)))
          Optimal for Sigmoid non-linearities.
        - Biases (b1, b2): Initialized to zeros.
        """
        # Layer 1: Input -> Hidden
        std_W1 = np.sqrt(2.0 / self.n_input)
        self.W1 = np.random.randn(self.n_input, self.n_hidden) * std_W1
        self.b1 = np.zeros((1, self.n_hidden), dtype=np.float64)

        # Layer 2: Hidden -> Output
        std_W2 = np.sqrt(2.0 / (self.n_hidden + self.n_output))
        self.W2 = np.random.randn(self.n_hidden, self.n_output) * std_W2
        self.b2 = np.zeros((1, self.n_output), dtype=np.float64)

    def _hidden_activation(self, z: np.ndarray) -> np.ndarray:
        """Computes hidden layer activation."""
        if self.activation_type == "leaky_relu":
            return Activation.leaky_relu(z)
        return Activation.relu(z)

    def _hidden_activation_derivative(self, z: np.ndarray) -> np.ndarray:
        """Computes derivative of hidden layer activation."""
        if self.activation_type == "leaky_relu":
            return Activation.leaky_relu_derivative(z)
        return Activation.relu_derivative(z)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Performs forward propagation through the network.

        Mathematical steps:
            Z1 = X · W1 + b1
            A1 = ReLU(Z1)
            Z2 = A1 · W2 + b2
            A2 = Sigmoid(Z2)

        Args:
            X: Input feature matrix of shape (m, n_input).

        Returns:
            A2: Output predictions (probabilities) of shape (m, 1).
            cache: Dictionary storing intermediate activations and pre-activations
                   for analytical backpropagation.
        """
        # Hidden Layer
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._hidden_activation(Z1)

        # Output Layer
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = Activation.sigmoid(Z2)

        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2,
        }
        return A2, cache

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes Binary Cross-Entropy (BCE) Loss with optional L2 Regularization.

        Formula:
            BCE = -1/m * Σ [ y * log(y_pred + ε) + (1 - y) * log(1 - y_pred + ε) ]
            L2  = (λ / (2 * m)) * ( ||W1||_F^2 + ||W2||_F^2 )
            Total Loss = BCE + L2

        Args:
            y_true: Ground truth binary labels of shape (m, 1).
            y_pred: Predicted probabilities from Sigmoid of shape (m, 1).

        Returns:
            loss: Scalar float representing total regularized BCE loss.
        """
        m = y_true.shape[0]
        eps = 1e-15  # Avoid log(0) numerical instability
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

        bce_cost = - (1.0 / m) * np.sum(
            y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped)
        )

        # L2 Regularization penalty
        l2_cost = (self.l2_lambda / (2.0 * m)) * (
            np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))
        )

        total_loss = float(bce_cost + l2_cost)
        return total_loss

    def backward(
        self,
        y_true: np.ndarray,
        cache: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Executes analytical backpropagation using the chain rule.

        Derivation:
            1. Output Layer:
               For BCE loss combined with Sigmoid activation:
               ∂L/∂Z2 = dZ2 = A2 - Y                     shape (m, 1)
               ∂L/∂W2 = dW2 = (1/m) * A1^T · dZ2 + (λ/m) * W2  shape (n_h, 1)
               ∂L/∂b2 = db2 = (1/m) * Σ dZ2              shape (1, 1)

            2. Hidden Layer:
               dA1 = dZ2 · W2^T                          shape (m, n_h)
               dZ1 = dA1 ⊙ ReLU'(Z1)                     shape (m, n_h)
               ∂L/∂W1 = dW1 = (1/m) * X^T · dZ1 + (λ/m) * W1   shape (n_x, n_h)
               ∂L/∂b1 = db1 = (1/m) * Σ dZ1              shape (1, n_h)

        Args:
            y_true: True labels of shape (m, 1).
            cache: Stored forward pass tensors ('X', 'Z1', 'A1', 'Z2', 'A2').

        Returns:
            grads: Dictionary of analytical gradients ('dW1', 'db1', 'dW2', 'db2').
        """
        m = y_true.shape[0]
        X = cache["X"]
        Z1 = cache["Z1"]
        A1 = cache["A1"]
        A2 = cache["A2"]

        # Output Layer Gradients
        dZ2 = A2 - y_true
        dW2 = (1.0 / m) * np.dot(A1.T, dZ2) + (self.l2_lambda / m) * self.W2
        db2 = (1.0 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # Hidden Layer Gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._hidden_activation_derivative(Z1)
        dW1 = (1.0 / m) * np.dot(X.T, dZ1) + (self.l2_lambda / m) * self.W1
        db1 = (1.0 / m) * np.sum(dZ1, axis=0, keepdims=True)

        grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
        }
        return grads

    def update_parameters(
        self,
        grads: Dict[str, np.ndarray],
        momentum: float = 0.9
    ) -> None:
        """
        Updates weights and biases using Gradient Descent with Momentum.

        Update equations:
            v_θ = β * v_θ + (1 - β) * dθ
            θ   = θ - α * v_θ

        If momentum == 0, reduces to standard vanilla gradient descent:
            θ   = θ - α * dθ
        """
        if momentum > 0.0:
            # Momentum update
            self.v_W1 = momentum * self.v_W1 + (1.0 - momentum) * grads["dW1"]
            self.v_b1 = momentum * self.v_b1 + (1.0 - momentum) * grads["db1"]
            self.v_W2 = momentum * self.v_W2 + (1.0 - momentum) * grads["dW2"]
            self.v_b2 = momentum * self.v_b2 + (1.0 - momentum) * grads["db2"]

            self.W1 -= self.learning_rate * self.v_W1
            self.b1 -= self.learning_rate * self.v_b1
            self.W2 -= self.learning_rate * self.v_W2
            self.b2 -= self.learning_rate * self.v_b2
        else:
            # Standard Gradient Descent
            self.W1 -= self.learning_rate * grads["dW1"]
            self.b1 -= self.learning_rate * grads["db1"]
            self.W2 -= self.learning_rate * grads["dW2"]
            self.b2 -= self.learning_rate * grads["db2"]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 300,
        batch_size: Optional[int] = 32,
        momentum: float = 0.9,
        verbose: bool = True,
        print_every: int = 50
    ) -> Dict[str, List[float]]:
        """
        Trains the neural network using mini-batch gradient descent.

        Args:
            X_train: Training features of shape (m, n_input).
            y_train: Training labels of shape (m, 1).
            X_val: Optional validation features of shape (m_val, n_input).
            y_val: Optional validation labels of shape (m_val, 1).
            epochs: Number of training epochs (default 300).
            batch_size: Mini-batch size; if None, performs full batch gradient descent.
            momentum: Momentum factor for velocity accumulation (default 0.9).
            verbose: If True, prints training progress.
            print_every: Frequency of epoch progress prints.

        Returns:
            history: Dictionary with recorded losses and accuracies across epochs.
        """
        m = X_train.shape[0]
        best_val_loss = float("inf")
        best_params: Optional[Dict[str, np.ndarray]] = None

        if verbose:
            print("=" * 70)
            print("                STARTING NEURAL NETWORK TRAINING")
            print("=" * 70)
            print(f"Architecture : Input({self.n_input}) -> Hidden({self.n_hidden}, {self.activation_type.upper()}) -> Output({self.n_output}, SIGMOID)")
            print(f"Hyperparams  : Learning Rate={self.learning_rate}, L2 Reg={self.l2_lambda}, Momentum={momentum}")
            print(f"Dataset      : Train samples={m}, Epochs={epochs}, Batch size={batch_size if batch_size else 'Full Batch'}")
            print("-" * 70)

        for epoch in range(1, epochs + 1):
            # Shuffle indices for mini-batching
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            effective_batch_size = batch_size if (batch_size and batch_size < m) else m

            # Mini-batch gradient descent loop
            for start_idx in range(0, m, effective_batch_size):
                end_idx = min(start_idx + effective_batch_size, m)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward Propagation
                A2_batch, cache = self.forward(X_batch)

                # Analytical Backpropagation
                grads = self.backward(y_batch, cache)

                # Parameter Optimization
                self.update_parameters(grads, momentum=momentum)

            # Full train set evaluation for this epoch
            train_preds_proba, _ = self.forward(X_train)
            train_loss = self.compute_loss(y_train, train_preds_proba)
            train_acc = float(np.mean((train_preds_proba >= 0.5) == y_train))

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validation evaluation if validation data provided
            if X_val is not None and y_val is not None:
                val_preds_proba, _ = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_preds_proba)
                val_acc = float(np.mean((val_preds_proba >= 0.5) == y_val))
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                # Save best parameters
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = {
                        "W1": self.W1.copy(),
                        "b1": self.b1.copy(),
                        "W2": self.W2.copy(),
                        "b2": self.b2.copy(),
                    }

            if verbose and (epoch == 1 or epoch % print_every == 0 or epoch == epochs):
                val_str = ""
                if X_val is not None and y_val is not None:
                    val_str = f" | Val Loss: {val_loss:.4f} - Val Acc: {val_acc * 100:.2f}%"
                print(f"Epoch {epoch:4d}/{epochs:4d} | Train Loss: {train_loss:.4f} - Train Acc: {train_acc * 100:.2f}%{val_str}")

        # Restore best weights if validation was used
        if best_params is not None:
            self.W1 = best_params["W1"]
            self.b1 = best_params["b1"]
            self.W2 = best_params["W2"]
            self.b2 = best_params["b2"]
            if verbose:
                print(f"\nTraining Complete. Restored best model checkpoint (Val Loss: {best_val_loss:.4f}).")
        elif verbose:
            print("\nTraining Complete.")
        print("=" * 70)

        return self.history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes predicted probabilities P(Y = 1 | X) for benign/malignant classification.

        Args:
            X: Input feature matrix of shape (m, n_input).

        Returns:
            probabilities: Array of shape (m, 1) with values in range [0, 1].
        """
        A2, _ = self.forward(X)
        return A2

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predicts discrete binary classes (0 = Benign, 1 = Malignant).

        Args:
            X: Input feature matrix of shape (m, n_input).
            threshold: Decision probability boundary (default 0.5).

        Returns:
            binary_predictions: Array of shape (m, 1) with values {0, 1}.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(np.int32)
