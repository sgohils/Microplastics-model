"""Tests for feature engineering."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.feature_engineering import (
    create_lag_features,
    add_temporal_features,
    normalize_features,
    engineer_features
)


@pytest.fixture
def sample_df():
    """Create sample dataframe for feature engineering tests."""
    dates = pd.date_range('2018-07-01', periods=30, freq='D')
    return pd.DataFrame({
        'date': dates,
        'target': np.random.lognormal(0, 1, 30),
        'discharge_m3s': np.random.uniform(10, 1000, 30),
        'temperature': np.random.uniform(10, 30, 30),
        'precipitation_mm': np.random.uniform(0, 50, 30),
        'location_name': ['loc_1'] * 30,
        'watershed': ['Delaware'] * 30,
        'latitude': [40.0] * 30,
        'longitude': [-75.0] * 30,
        'obs_index': range(30),
    })


class TestLagFeatures:
    def test_create_lag_features(self, sample_df):
        """Test lag feature creation."""
        df = sample_df.copy()
        result = create_lag_features(df, column='target', lags=[1, 3, 7])
        assert 'target_lag_1' in result.columns
        assert 'target_lag_3' in result.columns
        assert 'target_lag_7' in result.columns


class TestTemporalFeatures:
    def test_add_temporal_features(self, sample_df):
        """Test temporal feature addition."""
        df = sample_df.copy()
        result = add_temporal_features(df, date_col='date')
        
        assert 'day_of_week' in result.columns
        assert 'month' in result.columns
        assert 'day_of_year' in result.columns
        assert 'season' in result.columns


class TestNormalizeFeatures:
    def test_normalize_features(self, sample_df):
        """Test feature normalization."""
        df = sample_df.copy()
        feature_cols = ['discharge_m3s', 'temperature', 'precipitation_mm']
        result = normalize_features(df, feature_cols)
        
        for col in feature_cols:
            mean = result[col].mean()
            std = result[col].std()
            assert abs(mean) < 0.1
            assert abs(std - 1.0) < 0.1


class TestEngineerFeatures:
    def test_engineer_features_pipeline(self, sample_df):
        """Test full feature engineering pipeline."""
        result = engineer_features(sample_df)
        
        # Check that target is preserved
        assert 'target' in result.columns
        # Check that some features were created
        assert len(result.columns) > len(sample_df.columns)
