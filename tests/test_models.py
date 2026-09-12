"""Tests for baseline models."""
import pytest
import numpy as np
import pandas as pd

from src.models.baselines import (
    RidgeRegressor,
    RandomForestModel,
    XGBoostModel,
    MLPModel,
    SVMModel,
    GaussianProcessModel,
    BaselineEnsemble,
)


@pytest.fixture
def sample_data():
    """Create sample training data."""
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    return X, y


@pytest.fixture
def split_data():
    """Create sample train/val/test splits."""
    X = np.random.randn(200, 10)
    y = np.random.randn(200)
    
    X_train, y_train = X[:120], y[:120]
    X_val, y_val = X[120:160], y[120:160]
    X_test, y_test = X[160:], y[160:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


class TestRidgeRegressor:
    def test_fit_predict(self, sample_data):
        X, y = sample_data
        model = RidgeRegressor()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)
    
    def test_val_metrics(self, split_data):
        X_train, X_val, X_test, y_train, y_val, y_test = split_data
        model = RidgeRegressor()
        results = model.fit(X_train, y_train, X_val, y_val)
        assert 'train_mae' in results
        assert 'val_mae' in results


class TestRandomForestModel:
    def test_fit_predict(self, sample_data):
        X, y = sample_data
        model = RandomForestModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)


class TestXGBoostModel:
    def test_fit_predict(self, sample_data):
        X, y = sample_data
        model = XGBoostModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)


class TestMLPModel:
    def test_fit_predict(self, sample_data):
        X, y = sample_data
        model = MLPModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)


class TestSVMModel:
    def test_fit_predict(self, sample_data):
        X, y = sample_data
        model = SVMModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)


class TestGaussianProcessModel:
    def test_fit_predict(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        model = GaussianProcessModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)


class TestEnsembleModel:
    def test_init_and_predict(self, sample_data):
        X, y = sample_data
        model = BaselineEnsemble()
        assert model.models == {}
    
    def test_predict_not_fitted(self, sample_data):
        X, y = sample_data
        model = BaselineEnsemble()
        with pytest.raises(ValueError):
            model.predict(X)
