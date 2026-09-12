"""Main data processing pipeline for microplastic observations."""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import yaml

from .download import download_all_data
from .validate import DataValidator, run_validation_pipeline
from ..utils.config import load_config

logger = logging.getLogger(__name__)


class MicroplasticDataProcessor:
    """Processes raw microplastic data into standardized format."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.validator = DataValidator(config)
        self.processed_data: Optional[pd.DataFrame] = None
    
    def process(self, raw_dir: Path, output_path: Path) -> pd.DataFrame:
        """Process all microplastic datasets into a unified format.
        
        Args:
            raw_dir: Directory with raw data files
            output_path: Output file path
        
        Returns:
            Processed dataframe with standardized columns
        """
        all_data = []
        
        # Process Delaware River data
        delaware_file = raw_dir / "microplastic_delaware_river_2018.csv"
        if delaware_file.exists():
            df = self._process_delaware_river(delaware_file)
            all_data.append(df)
            logger.info(f"Processed Delaware River data: {len(df)} observations")
        
        # Process Great Lakes data
        gl_file = raw_dir / "microplastic_great_lakes_2014_2015.csv"
        if gl_file.exists():
            df = self._process_great_lakes(gl_file)
            all_data.append(df)
            logger.info(f"Processed Great Lakes data: {len(df)} observations")
        
        if not all_data:
            raise ValueError("No microplastic data files found")
        
        # Combine datasets
        combined = pd.concat(all_data, ignore_index=True)
        
        # Standardize columns
        combined = self._standardize_columns(combined)
        
        # Apply data quality filters
        combined = self._apply_quality_filters(combined)
        
        # Validate
        combined = self.validator.validate_microplastic_data(combined)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        
        logger.info(f"Processed microplastic data saved to {output_path}")
        logger.info(f"Total observations: {len(combined)}")
        
        self.processed_data = combined
        return combined
    
    def _process_delaware_river(self, filepath: Path) -> pd.DataFrame:
        """Process Delaware River microplastic data.
        
        Args:
            filepath: Path to Delaware River CSV
        
        Returns:
            Standardized dataframe
        """
        df = pd.read_csv(filepath)
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        
        # Standardize concentration to particles/m³
        # Already in particles/m³
        
        # Add watershed identifier
        df['watershed'] = 'Delaware River'
        
        # Extract particle count if available
        if 'max_particles_per_m3' in df.columns:
            df['has_max'] = df['max_particles_per_m3'].notna()
        
        return df
    
    def _process_great_lakes(self, filepath: Path) -> pd.DataFrame:
        """Process Great Lakes tributary microplastic data.
        
        Args:
            filepath: Path to Great Lakes CSV
        
        Returns:
            Standardized dataframe
        """
        df = pd.read_csv(filepath)
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        
        # Already in particles/m³ (estimated from published data)
        
        # Add watershed identifier
        df['watershed'] = df['basin'].map({
            'Lake Superior': 'Lake Superior',
            'Lake Michigan': 'Lake Michigan',
            'Mississippi': 'Mississippi River',
            'Ohio': 'Ohio River',
        })
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across datasets.
        
        Args:
            df: Input dataframe
        
        Returns:
            Dataframe with standardized columns
        """
        column_mapping = {
            'particles_per_m3': 'concentration_particles_per_m3',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'date': 'date',
            'location': 'location_name',
            'watershed': 'watershed',
            'source_dataset': 'source_dataset',
            'sampling_method': 'sampling_method',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required = ['latitude', 'longitude', 'date', 'concentration_particles_per_m3']
        for col in required:
            if col not in df.columns:
                df[col] = np.nan
        
        return df
    
    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scientifically justified quality filters.
        
        Args:
            df: Input dataframe
        
        Returns:
            Filtered dataframe
        """
        original_count = len(df)
        
        # Remove negative concentrations
        df = df[df['concentration_particles_per_m3'] >= 0]
        
        # Remove zero concentrations (possibly below detection limit)
        # Keep them but flag them - this is a data quality issue
        df['below_detection_limit'] = (
            df['concentration_particles_per_m3'] == 0
        )
        
        # Remove invalid coordinates
        df = df[
            (df['latitude'] >= -90) & (df['latitude'] <= 90) &
            (df['longitude'] >= -180) & (df['longitude'] <= 180)
        ]
        
        filtered_count = len(df)
        if original_count != filtered_count:
            logger.info(
                f"Quality filters removed {original_count - filtered_count} observations"
            )
        
        return df


def process_all_data(config: Dict) -> Dict[str, pd.DataFrame]:
    """Run the complete data processing pipeline.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Dictionary of processed dataframes
    """
    raw_dir = Path(config['data']['raw_dir'])
    processed_dir = Path(config['data']['processed_dir'])
    interim_dir = Path(config['data']['interim_dir'])
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Step 1: Download data if not present
    if not (raw_dir / "streamgage_delaware_basin.csv").exists():
        logger.info("Downloading datasets...")
        download_all_data(config)
    
    # Step 2: Validate raw data
    logger.info("Validating raw data...")
    validation_results = run_validation_pipeline(raw_dir, interim_dir)
    logger.info(f"Validation results: {validation_results}")
    
    # Step 3: Process microplastic data
    logger.info("Processing microplastic data...")
    processor = MicroplasticDataProcessor(config)
    mp_data = processor.process(
        raw_dir, 
        processed_dir / "microplastic_processed.csv"
    )
    results['microplastic'] = mp_data
    
    # Step 4: Process streamgage metadata
    logger.info("Processing streamgage metadata...")
    sg_file = raw_dir / "streamgauge_delaware_basin.csv"
    if sg_file.exists():
        sg_data = pd.read_csv(sg_file)
        sg_data.to_csv(processed_dir / "streamgauge_processed.csv", index=False)
        results['streamgages'] = sg_data
    
    logger.info("Data processing complete")
    return results
