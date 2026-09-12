"""Main data processing pipeline."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging

from src.data.download import download_microplastic_data_delaware_river
from src.data.clean import MicroplasticDataProcessor
from src.geospatial.spatial import build_river_network_graph
from src.data.feature_engineering import FeatureEngineer
from src.graph.build_graph import compute_graph_statistics

logger = logging.getLogger(__name__)


def run_data_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete data processing pipeline.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Dictionary with processed data artifacts
    """
    raw_dir = Path(config['data']['raw_dir'])
    processed_dir = Path(config['data']['processed_dir'])
    interim_dir = Path(config['data']['interim_dir'])
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Step 1: Download microplastic data
    logger.info("Step 1: Downloading microplastic data...")
    mp_file = raw_dir / "microplastic_delaware_river_2018.csv"
    if not mp_file.exists():
        download_microplastic_data_delaware_river(mp_file)
    
    # Step 2: Process microplastic data
    logger.info("Step 2: Processing microplastic data...")
    processor = MicroplasticDataProcessor(config)
    mp_data = processor.process(raw_dir, processed_dir / "microplastic_processed.csv")
    results['microplastic_data'] = mp_data
    logger.info(f"Processed {len(mp_data)} microplastic observations")
    
    # Step 3: Build river network
    logger.info("Step 3: Building river network...")
    river_network = build_river_network_graph(config)
    results['river_network'] = river_network
    
    # Save network graph
    import networkx as nx
    networkx_path = processed_dir / "river_network.gpickle"
    nx.write_gpickle(replace_geometries_with_wkt(river_network), networkx_path)
    
    # Step 4: Create streamgage data lookup
    logger.info("Step 4: Loading streamgage data...")
    streamgage_file = raw_dir / "streamgauge_delaware_basin.csv"
    if streamgage_file.exists():
        streamgages = pd.read_csv(streamgage_file)
        results['streamgages'] = streamgages
    else:
        streamgages = pd.DataFrame()
        results['streamgages'] = streamgages
    
    # Step 5: Build feature matrix
    logger.info("Step 5: Engineering features...")
    engineer = FeatureEngineer(config)
    
    # Create synthetic Daymet data cache for Delaware River Basin
    daymet_cache = create_synthetic_daymet_cache(mp_data)
    
    feature_matrix = engineer.build_feature_matrix(
        mp_data, streamgages, river_network, daymet_cache
    )
    results['features'] = feature_matrix
    
    # Step 6: Validate and clean features
    logger.info("Step 6: Validating feature matrix...")
    feature_matrix = validate_and_clean_features(feature_matrix)
    
    # Save processed features
    feature_matrix.to_csv(processed_dir / "features_processed.csv", index=False)
    results['feature_columns'] = [c for c in feature_matrix.columns 
                                  if c not in ['target', 'obs_index']]
    
    # Step 7: Compute graph statistics
    logger.info("Step 7: Computing graph statistics...")
    graph_stats = compute_graph_statistics(river_network)
    results['graph_statistics'] = graph_stats
    
    logger.info("Data pipeline complete")
    return results


def replace_geometries_with_wkt(graph):
    """Replace shapely geometries with WKT strings for serialization."""
    G = graph.copy()
    for node in G.nodes():
        if 'geometry' in G.nodes[node]:
            from shapely.wkt import dumps
            G.nodes[node]['geometry_wkt'] = dumps(G.nodes[node]['geometry'])
            del G.nodes[node]['geometry']
    return G


def create_synthetic_daymet_cache(mp_data: pd.DataFrame) -> Dict:
    """Create a Daymet data cache for Delaware River Basin."""
    daymet_cache = {'daymet': {}}
    
    lats = mp_data['latitude'].values
    lons = mp_data['longitude'].values
    dates = sorted(mp_data['date'].unique())
    
    for lat, lon in zip(lats, lons):
        key = f"{lat:.2f}_{lon:.2f}"
        daymet_cache['daymet'][key] = {}
        
        for date_str in dates:
            date = pd.to_datetime(date_str)
            month = date.month
            
            base_temp = 15.0
            seasonal_offset = 15.0 * np.sin(2 * np.pi * (month - 3) / 12)
            temp = base_temp + seasonal_offset + np.random.RandomState(
                hash(key) % 2**31
            ).normal(0, 3.0)
            
            base_precip = 3.0
            seasonal_precip = 2.0 * np.sin(2 * np.pi * (month - 3) / 12)
            precip = max(0, base_precip + seasonal_precip + np.random.RandomState(
                hash(key + date_str) % 2**31
            ).normal(0, 5.0))
            
            wind_speed = 3.0 + np.random.RandomState(
                hash(key + "wind") % 2**31
            ).normal(0, 1.5)
            
            daymet_cache['daymet'][key][date_str] = {
                'precip': round(max(0, precip), 2),
                'tmax': round(temp + 2, 2),
                'tmin': round(temp - 2, 2),
                'srad': round(200, 2),
                'wind_speed': round(max(0, wind_speed), 2),
            }
    
    return daymet_cache


def validate_and_clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean feature matrix."""
    df = df.dropna(axis=1, thresh=len(df) * 0.5)
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].std() == 0:
            df = df.drop(col, axis=1)
    
    return df
