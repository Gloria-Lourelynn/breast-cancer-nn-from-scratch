"""
dataset.py - Dataset Loading, Inspection, and Preprocessing for Breast Cancer Data.
"""

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_raw_data() -> Tuple[pd.DataFrame, pd.Series, Any]:
    """
    Loads the Wisconsin Diagnostic Breast Cancer (WDBC) benchmark dataset using scikit-learn.
    Returns:
        df: Pandas DataFrame containing 30 morphological features extracted from cell nuclei.
        target_series: Raw target Series.
        bunch: Raw scikit-learn Bunch object with metadata.
    """
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.data
    target_series = bunch.target
    return df, target_series, bunch


def inspect_dataset(df: pd.DataFrame, target: pd.Series, bunch: Any) -> Dict[str, Any]:
    """
    Performs thorough inspection of the raw dataset:
    - Shape and feature count
    - Missing values check
    - Class counts and proportions
    - Summary statistics
    """
    n_samples, n_features = df.shape
    missing_values = int(df.isnull().sum().sum())
    
    # Original sklearn encoding: 0 = malignant, 1 = benign
    raw_class_counts = {
        bunch.target_names[i]: int((target == i).sum())
        for i in range(len(bunch.target_names))
    }
    
    inspection_report = {
        "num_samples": n_samples,
        "num_features": n_features,
        "missing_values": missing_values,
        "raw_class_distribution": raw_class_counts,
        "feature_names": list(df.columns),
        "summary_stats": df.describe().round(4).to_dict(),
    }
    return inspection_report


def print_inspection_summary(report: Dict[str, Any]) -> None:
    """Prints a clean human-readable summary of the dataset inspection."""
    print("=" * 70)
    print("           WISCONSIN BREAST CANCER DATASET INSPECTION")
    print("=" * 70)
    print(f"Total Samples           : {report['num_samples']}")
    print(f"Total Features          : {report['num_features']}")
    print(f"Missing Values          : {report['num_values'] if 'num_values' in report else report['missing_values']}")
    print("-" * 70)
    print("Raw Class Distribution (scikit-learn defaults):")
    for cls_name, count in report["raw_class_distribution"].items():
        pct = (count / report["num_samples"]) * 100
        print(f"  - {cls_name:<12}: {count:>4} samples ({pct:>5.1f}%)")
    print("-" * 70)
    print(f"Feature Names (first 5 of {report['num_features']}):")
    for f in report["feature_names"][:5]:
        print(f"  * {f}")
    print("  ... [and 25 additional cellular nucleus morphological measurements]")
    print("=" * 70)


def prepare_breast_cancer_data(
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list, list]:
    """
    Loads, inspects, encodes, splits, and scales the breast cancer dataset.

    Data Leakage Prevention:
        - Train/test split is performed BEFORE any scaling.
        - StandardScaler is fitted strictly on X_train.
        - The fitted scaler is then applied to transform X_test without refitting.

    Target Encoding:
        - In scikit-learn: 0 = 'malignant', 1 = 'benign'.
        - Standard machine-learning binary classification convention:
            0 = 'benign'    (negative class)
            1 = 'malignant' (positive class of interest)
        - We explicitly map: y = 1 - raw_target so that 1 represents 'malignant'.
          This makes Recall, Precision, and F1 intuitive for detecting malignant samples.

    Args:
        test_size: Fraction of samples reserved for test evaluation (default 0.2).
        random_state: Random seed for reproducible splitting (default 42).
        verbose: If True, prints inspection summary and split shapes.

    Returns:
        X_train: Standardized training feature matrix of shape (m_train, 30).
        X_test: Standardized testing feature matrix of shape (m_test, 30).
        y_train: Encoded training labels of shape (m_train, 1).
        y_test: Encoded testing labels of shape (m_test, 1).
        scaler: The fitted StandardScaler instance (for subsequent inference).
        feature_names: List of all 30 feature names.
        target_names: Target label names ['benign', 'malignant'].
    """
    df, raw_target, bunch = load_raw_data()
    
    if verbose:
        inspection = inspect_dataset(df, raw_target, bunch)
        print_inspection_summary(inspection)

    # Standard target re-mapping: 0 -> Benign, 1 -> Malignant
    # In raw sklearn: 0 = malignant, 1 = benign. So 1 - raw yields 1 for malignant, 0 for benign.
    y_encoded = (1 - raw_target.to_numpy()).astype(np.float64).reshape(-1, 1)
    X_raw = df.to_numpy(dtype=np.float64)
    
    target_names = ["benign", "malignant"]
    feature_names = list(df.columns)

    # Stratified Train/Test split to maintain identical class ratios in both splits
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    # Feature Scaling: Fit strictly on training data to avoid data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    if verbose:
        print("\nDataset Splitting & Leakage-Free Standardization:")
        print(f"  Training Set : X_train = {X_train.shape}, y_train = {y_train.shape}")
        print(f"                 - Benign (0)   : {int((y_train == 0).sum())}")
        print(f"                 - Malignant (1): {int((y_train == 1).sum())}")
        print(f"  Testing Set  : X_test = {X_test.shape}, y_test = {y_test.shape}")
        print(f"                 - Benign (0)   : {int((y_test == 0).sum())}")
        print(f"                 - Malignant (1): {int((y_test == 1).sum())}")
        print("-" * 70)

    return X_train, X_test, y_train, y_test, scaler, feature_names, target_names
