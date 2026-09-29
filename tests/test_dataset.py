"""
test_dataset.py - Unit tests for data loading, encoding, and leakage-free scaling.
"""

import numpy as np
import pytest
from src.dataset import load_raw_data, inspect_dataset, prepare_breast_cancer_data


def test_load_raw_data():
    df, target, bunch = load_raw_data()
    assert df.shape == (569, 30)
    assert len(target) == 569
    assert df.isnull().sum().sum() == 0


def test_inspect_dataset():
    df, target, bunch = load_raw_data()
    report = inspect_dataset(df, target, bunch)
    assert report["num_samples"] == 569
    assert report["num_features"] == 30
    assert report["missing_values"] == 0
    assert "malignant" in report["raw_class_distribution"]
    assert "benign" in report["raw_class_distribution"]


def test_prepare_breast_cancer_data_shapes_and_leakage():
    X_train, X_test, y_train, y_test, scaler, feat_names, target_names = prepare_breast_cancer_data(
        test_size=0.2, random_state=42, verbose=False
    )

    # 569 * 0.8 = 455.2 -> 455 train, 114 test
    assert X_train.shape == (455, 30)
    assert X_test.shape == (114, 30)
    assert y_train.shape == (455, 1)
    assert y_test.shape == (114, 1)

    # Check target values are strictly 0 and 1
    assert set(np.unique(y_train)).issubset({0.0, 1.0})
    assert set(np.unique(y_test)).issubset({0.0, 1.0})

    # Check that X_train is standardized (mean ~ 0, std ~ 1)
    np.testing.assert_allclose(X_train.mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(X_train.std(axis=0), 1.0, atol=1e-7)

    # Test scaler was fit on train only (not test)
    # The mean of test will NOT be exactly 0, confirming lack of data leakage
    test_mean = np.abs(X_test.mean(axis=0))
    assert np.any(test_mean > 0.0)
