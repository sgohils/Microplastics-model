"""Tests for data validation."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.validate import (
    validate_microplastic_data,
    check_minimum_observations,
    identify_outliers
)
from src.data.clean import clean_microplastic_data


@pytest.fixture
def sample_data():
    """Create sample microplastic data for testing."""
    return pd.DataFrame({
        'target': np.random.lognormal(0, 1, 100),
        'location_name': [f'loc_{i % 10}' for i in range(100)],
        'latitude': np.random.uniform(39.5, 42.5, 100),
        'longitude': np.random.uniform(-75.5, -74.0, 100),
        'date': pd.date_range('2018-07-01', periods=100, freq='D').astype(str),
        'discharge_m3s': np.random.uniform(10, 1000, 100),
    })


class TestDataValidation:
    """Tests for data validation functions."""
    
    def test_validate_microplastic_data_valid(self, sample_data):
        """Test validation with valid data."""
        issues = validate_microplastic_data(sample_data)
        assert isinstance(issues, dict)
        assert 'errors' in issues
        assert 'warnings' in issues
    
    def test_validate_microplastic_data_missing_columns(self, sample_data):
        """Test validation detects missing columns."""
        bad_data = sample_data.drop('latitude', axis=1)
        issues = validate_microplastic_data(bad_data)
        assert 'latitude' in str(issues['errors'])
    
    def test_check_minimum_observations(self, sample_data):
        """Test minimum observations check."""
        result = check_minimum_observations(
            sample_data, min_obs=3, location_col='location_name'
        )
        assert isinstance(result, dict)
        assert 'valid' in result
    
    def test_identify_outliers(self, sample_data):
        """Test outlier detection."""
        outliers = identify_outliers(sample_data, 'target', threshold=3.0)
        assert isinstance(outliers, pd.Series)
        assert outliers.dtype == bool
