"""Main data processing pipeline."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging

from src.data.download import (
    download_microplastic_data_delaware_river,
    download_microplastic_data_great_lakes,
    download_usgs_streamgage_metadata,
    download_daymet_data,
    download_usgs_streamflow,
    download_nhdplus_data
)
from src.data.clean import MicroplasticDataProcessor
from src.geospatial.spatial import build_river_network_graph
from src.data.feature_engineering import FeatureEngineer
from src.graph.river_topology import compute_graph_statistics

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
    
    # Step 1: Download all data
    logger.info("Step 1: Downloading all data...")
    download_all_data(config, raw_dir)
    
    # Step 2: Process microplastic data (Delaware River)
    logger.info("Step 2: Processing Delaware River microplastic data...")
    mp_file = raw_dir / "microplastic_delaware_river_2018.csv"
    processor = MicroplasticDataProcessor(config)
    mp_data = processor.process(raw_dir, processed_dir / "microplastic_processed.csv")
    results['microplastic_data'] = mp_data
    logger.info(f"Processed {len(mp_data)} microplastic observations")
    
    # Step 3: Process Great Lakes data (for validation)
    logger.info("Step 3: Processing Great Lakes tributary data...")
    gl_file = raw_dir / "microplastic_great_lakes_2014_2015.csv"
    gl_data = processor.process(raw_dir, processed_dir / "microplastic_great_lakes_processed.csv")
    results['great_lakes_data'] = gl_data
    logger.info(f"Processed {len(gl_data)} Great Lakes observations")
    
    # Step 4: Build river network
    logger.info("Step 4: Building river network...")
    river_network = build_river_network_graph(config)
    results['river_network'] = river_network
    
    # Save network graph
    import pickle
    networkx_path = processed_dir / "river_network.gpickle"
    with open(networkx_path, 'wb') as f:
        pickle.dump(replace_geometries_with_wkt(river_network), f)
    logger.info(f"River network saved to {networkx_path}")
    
    # Step 5: Download/load streamgage data
    logger.info("Step 5: Loading streamgage data...")
    streamgage_file = raw_dir / "streamgauge_delaware_basin.csv"
    if streamgage_file.exists():
        streamgages = pd.read_csv(streamgage_file)
        results['streamgages'] = streamgages
    else:
        streamgages = pd.DataFrame()
        results['streamgages'] = streamgages
    
    # Step 6: Download Daymet data for all observation locations
    logger.info("Step 6: Downloading Daymet meteorological data...")
    daymet_cache = download_daymet_for_observations(mp_data, config, raw_dir)
    
    # Step 7: Download USGS streamflow data for all streamgages
    logger.info("Step 7: Downloading USGS streamflow data...")
    streamflow_cache = download_streamflow_for_streamgages(streamgages, mp_data, config, raw_dir)
    
    # Step 8: Build feature matrix
    logger.info("Step 8: Engineering features...")
    engineer = FeatureEngineer(config)
    
    feature_matrix = engineer.build_feature_matrix(
        mp_data, streamgages, river_network, daymet_cache, streamflow_cache
    )
    results['features'] = feature_matrix
    
    # Step 9: Validate and clean features
    logger.info("Step 9: Validating feature matrix...")
    feature_matrix = validate_and_clean_features(feature_matrix)
    
    # Save processed features
    feature_matrix.to_csv(processed_dir / "features_processed.csv", index=False)
    results['feature_columns'] = [c for c in feature_matrix.columns 
                                  if c not in ['target', 'obs_index']]
    
    # Step 10: Compute graph statistics
    logger.info("Step 10: Computing graph statistics...")
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


def download_all_data(config: Dict[str, Any], raw_dir: Path) -> bool:
    """Download all required datasets.
    
    Args:
        config: Configuration dictionary
        raw_dir: Raw data directory
    
    Returns:
        True if all critical downloads succeed
    """
    success = True
    
    # 1. Delaware River microplastic data
    success &= download_microplastic_data_delaware_river(
        raw_dir / "microplastic_delaware_river_2018.csv"
    )
    
    # 2. Great Lakes tributary data
    success &= download_microplastic_data_great_lakes(
        raw_dir / "microplastic_great_lakes_2014_2015.csv"
    )
    
    # 3. USGS streamgage metadata
    success &= download_usgs_streamgage_metadata(
        raw_dir / "streamgauge_delaware_basin.csv"
    )
    
    # 4. NHDPlus data (placeholder - in real implementation, download actual shapefile)
    success &= download_nhdplus_data(raw_dir / "nhdplus")
    
    logger.info(f"Data download complete. Success: {success}")
    return success


def download_daymet_for_observations(
    mp_data: pd.DataFrame, 
    config: Dict[str, Any], 
    raw_dir: Path
) -> Dict:
    """Download Daymet data for all observation locations and dates.
    
    Args:
        mp_data: Microplastic observations with lat/lon/date
        config: Configuration dictionary
        raw_dir: Raw data directory
    
    Returns:
        Daymet cache dictionary
    """
    daymet_cache = {'daymet': {}}
    
    # Get unique locations and date range
    locations = mp_data[['latitude', 'longitude']].drop_duplicates()
    dates = sorted(mp_data['date'].unique())
    
    # Add 7-day buffer before earliest date for lag features
    min_date = pd.to_datetime(dates[0])
    max_date = pd.to_datetime(dates[-1])
    start_date = (min_date - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = max_date.strftime('%Y-%m-%d')
    
    logger.info(f"Downloading Daymet data for {len(locations)} locations from {start_date} to {end_date}")
    
    for _, row in locations.iterrows():
        lat, lon = row['latitude'], row['longitude']
        key = f"{lat:.2f}_{lon:.2f}"
        output_path = raw_dir / f"daymet_{key}.csv"
        
        daymet_cache['daymet'][key] = {}
        
        if output_path.exists():
            logger.info(f"Daymet data for ({lat}, {lon}) already exists, loading...")
            df = pd.read_csv(output_path)
        else:
            success = download_daymet_data(lat, lon, start_date, end_date, output_path)
            if not success:
                logger.warning(f"Failed to download Daymet for ({lat}, {lon}), using synthetic fallback")
                df = create_synthetic_daymet_fallback(lat, lon, start_date, end_date)
        
        # Parse into cache format
        for _, drow in df.iterrows():
            date_str = str(drow.get('date', drow.get('year', '') + '-' + 
                                      str(drow.get('yday', '')).zfill(3)))
            # Convert yday to date if needed
            if 'yday' in drow:
                date = pd.to_datetime(str(int(drow['year'])), format='%Y') + pd.Timedelta(days=int(drow['yday'])-1)
                date_str = date.strftime('%Y-%m-%d')
            
            daymet_cache['daymet'][key][date_str] = {
                'precip': float(drow.get('prcp', 0)),
                'tmax': float(drow.get('tmax', 15)),
                'tmin': float(drow.get('tmin', 5)),
                'srad': float(drow.get('srad', 200)),
                'wind_speed': float(drow.get('vp', 3)),  # vapor pressure as proxy
            }
    
    return daymet_cache


def create_synthetic_daymet_fallback(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Create synthetic Daymet data as fallback when API unavailable.
    
    NOTE: This is clearly labeled as synthetic fallback data.
    It should NOT be used for primary analysis.
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with synthetic meteorological data
    """
    logger.warning("Using SYNTHETIC Daymet fallback data - not for primary analysis!")
    
    dates = pd.date_range(start_date, end_date, freq='D')
    key = f"{lat:.2f}_{lon:.2f}"
    
    data = []
    for date in dates:
        month = date.month
        
        base_temp = 15.0
        seasonal_offset = 15.0 * np.sin(2 * np.pi * (month - 3) / 12)
        temp = base_temp + seasonal_offset + np.random.RandomState(
            hash(key + str(date)) % 2**31
        ).normal(0, 3.0)
        
        base_precip = 3.0
        seasonal_precip = 2.0 * np.sin(2 * np.pi * (month - 3) / 12)
        precip = max(0, base_precip + seasonal_precip + np.random.RandomState(
            hash(key + str(date) + "precip") % 2**31
        ).normal(0, 5.0))
        
        wind_speed = 3.0 + np.random.RandomState(
            hash(key + str(date) + "wind") % 2**31
        ).normal(0, 1.5)
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'prcp': round(max(0, precip), 2),
            'tmax': round(temp + 2, 2),
            'tmin': round(temp - 2, 2),
            'srad': round(200, 2),
            'vp': round(max(0, wind_speed), 2),
        })
    
    return pd.DataFrame(data)


def download_streamflow_for_streamgages(
    streamgages: pd.DataFrame,
    mp_data: pd.DataFrame,
    config: Dict[str, Any],
    raw_dir: Path
) -> Dict:
    """Download USGS streamflow data for relevant streamgages.
    
    Args:
        streamgages: Streamgage metadata
        mp_data: Microplastic observations (for date range)
        config: Configuration dictionary
        raw_dir: Raw data directory
    
    Returns:
        Streamflow cache dictionary
    """
    streamflow_cache = {'streamflow': {}}
    
    if streamgages.empty:
        logger.warning("No streamgage metadata available")
        return streamflow_cache
    
    dates = sorted(mp_data['date'].unique())
    min_date = pd.to_datetime(dates[0])
    max_date = pd.to_datetime(dates[-1])
    start_date = (min_date - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = max_date.strftime('%Y-%m-%d')
    
    logger.info(f"Downloading streamflow for {len(streamgages)} streamgages from {start_date} to {end_date}")
    
    for _, row in streamgages.iterrows():
        site_id = row['site_no']
        output_path = raw_dir / f"streamflow_{site_id}.csv"
        
        if output_path.exists():
            logger.info(f"Streamflow for {site_id} already exists, loading...")
            df = pd.read_csv(output_path)
        else:
            success = download_usgs_streamflow(site_id, start_date, end_date, output_path)
            if not success:
                logger.warning(f"Failed to download streamflow for {site_id}")
                continue
            df = pd.read_csv(output_path)
        
        streamflow_cache['streamflow'][site_id] = df
    
    return streamflow_cache


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
