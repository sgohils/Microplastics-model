"""Data validation utilities."""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and integrity."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_microplastic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate microplastic observation data.
        
        Args:
            df: Raw microplastic dataframe
        
        Returns:
            Validated dataframe
        """
        required_columns = ['latitude', 'longitude', 'date', 'particles_per_m3']
        
        # Check required columns
        for col in required_columns:
            if col not in df.columns:
                self.errors.append(f"Missing required column: {col}")
        
        if self.errors:
            raise ValueError(f"Validation failed: {'; '.join(self.errors)}")
        
        # Check coordinate bounds
        lat_bounds = (-90, 90)
        lon_bounds = (-180, 180)
        
        invalid_lat = df[
            (df['latitude'] < lat_bounds[0]) | 
            (df['latitude'] > lat_bounds[1])
        ]
        if len(invalid_lat) > 0:
            self.errors.append(
                f"Invalid latitude values: {len(invalid_lat)} rows"
            )
        
        invalid_lon = df[
            (df['longitude'] < lon_bounds[0]) | 
            (df['longitude'] > lon_bounds[1])
        ]
        if len(invalid_lon) > 0:
            self.errors.append(
                f"Invalid longitude values: {len(invalid_lon)} rows"
            )
        
        # Check concentration values
        invalid_conc = df[df['particles_per_m3'] < 0]
        if len(invalid_conc) > 0:
            self.errors.append(
                f"Negative concentration values: {len(invalid_conc)} rows"
            )
        
        # Check timestamps
        try:
            df['date_parsed'] = pd.to_datetime(df['date'])
        except Exception as e:
            self.errors.append(f"Date parsing error: {e}")
        
        # Check for duplicate observations
        duplicates = df.duplicated(subset=['latitude', 'longitude', 'date'])
        if duplicates.sum() > 0:
            self.warnings.append(
                f"Duplicate observations (same location/date): {duplicates.sum()}"
            )
        
        # Check for missing values
        for col in required_columns:
            missing = df[col].isna().sum()
            if missing > 0:
                self.warnings.append(
                    f"Missing values in {col}: {missing}"
                )
        
        if self.errors:
            raise ValueError(f"Validation failed: {'; '.join(self.errors)}")
        
        logger.info(
            f"Microplastic data validation passed: {len(df)} valid observations"
        )
        if self.warnings:
            for w in self.warnings:
                logger.warning(w)
        
        return df
    
    def validate_streamflow_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate USGS streamflow data.
        
        Args:
            df: Streamflow dataframe
        
        Returns:
            Validated dataframe
        """
        required_columns = ['site_id', 'datetime', 'flow_cfs']
        
        for col in required_columns:
            if col not in df.columns:
                self.errors.append(f"Missing required column: {col}")
        
        if self.errors:
            raise ValueError(f"Streamflow validation failed: {'; '.join(self.errors)}")
        
        # Remove missing values (USGS uses -999999 for missing)
        df = df[df['flow_cfs'] != -999999]
        df = df.dropna(subset=['flow_cfs'])
        
        # Convert to datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Check for negative flow
        negative_flow = df[df['flow_cfs'] < 0]
        if len(negative_flow) > 0:
            self.warnings.append(
                f"Negative flow values found: {len(negative_flow)}"
            )
        
        logger.info(
            f"Streamflow data validation passed: {len(df)} valid records"
        )
        return df
    
    def validate_environmental_data(self, df: pd.DataFrame, 
                                     data_type: str) -> pd.DataFrame:
        """Validate environmental covariate data.
        
        Args:
            df: Environmental data dataframe
            data_type: Type of data ('daymet', 'nlcd', 'ned')
        
        Returns:
            Validated dataframe
        """
        if data_type == 'daymet':
            required = ['date', 'precipitation', 'temperature']
            for col in required:
                if col not in df.columns:
                    self.errors.append(f"Daymet missing column: {col}")
            
            df['date'] = pd.to_datetime(df['date'])
            
            # Check temperature range
            if 'temperature' in df.columns:
                extreme_temp = df[
                    (df['temperature'] > 50) | (df['temperature'] < -60)
                ]
                if len(extreme_temp) > 0:
                    self.warnings.append(
                        f"Extreme temperature values: {len(extreme_temp)}"
                    )
            
            # Check precipitation range
            if 'precipitation' in df.columns:
                negative_precip = df[df['precipitation'] < 0]
                if len(negative_precip) > 0:
                    self.warnings.append(
                        f"Negative precipitation: {len(negative_precip)}"
                    )
        
        logger.info(f"Environmental data validation passed for {data_type}")
        return df
    
    def get_validation_report(self) -> Dict[str, List[str]]:
        """Get validation report.
        
        Returns:
            Dictionary with errors and warnings
        """
        return {
            'errors': self.errors,
            'warnings': self.warnings
        }


def run_validation_pipeline(raw_dir: Path, 
                            output_dir: Path) -> Dict[str, bool]:
    """Run complete validation pipeline on all raw datasets.
    
    Args:
        raw_dir: Directory containing raw data files
        output_dir: Directory for cleaned/validated data
    
    Returns:
        Dictionary mapping dataset names to validation status
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    validator = DataValidator({})
    
    # Validate microplastic data
    mp_files = list(raw_dir.glob("microplastic_*.csv"))
    for f in mp_files:
        try:
            df = pd.read_csv(f)
            df_validated = validator.validate_microplastic_data(df)
            df_validated.to_csv(
                output_dir / f.name.replace('.csv', '_validated.csv"), 
                index=False
            )
            results[f.name] = True
        except Exception as e:
            logger.error(f"Validation failed for {f.name}: {e}")
            results[f.name] = False
    
    # Validate streamgage data
    sg_file = raw_dir / "streamgage_delaware_basin.csv"
    if sg_file.exists():
        try:
            df = pd.read_csv(sg_file)
            df_validated = validator.validate_streamflow_data(df)
            df_validated.to_csv(
                output_dir / "streamgage_validated.csv", 
                index=False
            )
            results['streamgage_delaware_basin.csv'] = True
        except Exception as e:
            logger.error(f"Streamgage validation failed: {e}")
            results['streamgage_delaware_basin.csv'] = False
    
    return results
