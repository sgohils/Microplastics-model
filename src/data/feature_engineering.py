"""Feature engineering for the river network prediction model."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from shapely.geometry import Point
import networkx as nx

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineers features for microplastic prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        
    def build_feature_matrix(
        self,
        observations: pd.DataFrame,
        streamgages: pd.DataFrame,
        graph: nx.DiGraph,
        daymet_cache: Dict,
        streamflow_cache: Dict = None
    ) -> pd.DataFrame:
        """Build complete feature matrix for all observations.
        
        Args:
            observations: Microplastic observation data
            streamgages: Streamgage metadata
            graph: River network graph
            daymet_cache: Cached Daymet data dictionary
        
        Returns:
            Feature matrix dataframe with target column
        """
        features = []
        
        for idx, row in observations.iterrows():
            feature_row = self._extract_features_for_observation(
                row, streamgages, graph, daymet_cache, streamflow_cache
            )
            feature_row['obs_index'] = idx
            features.append(feature_row)
        
        feature_df = pd.DataFrame(features)
        
        # Ensure target column
        feature_df['target'] = observations['concentration_particles_per_m3'].values
        
        self._categorize_features(feature_df)
        
        logger.info(f"Feature matrix built: {feature_df.shape[0]} samples, {feature_df.shape[1]} columns")
        return feature_df
    
    def _extract_features_for_observation(
        self,
        obs: pd.Series,
        streamgages: pd.DataFrame,
        graph: nx.DiGraph,
        daymet_cache: Dict,
        streamflow_cache: Dict = None
    ) -> Dict:
        """Extract all features for a single observation.
        
        Args:
            obs: Observation row
            streamgages: Streamgage metadata
            graph: River network graph
            daymet_cache: Cached Daymet data
            streamflow_cache: Cached USGS streamflow data
        
        Returns:
            Dictionary of features
        """
        features = {}
        obs_point = Point(obs['longitude'], obs['latitude'])
        sample_date = pd.to_datetime(obs['date'])
        
        # Preserve watershed info for spatial split experiments
        features['watershed'] = obs.get('watershed', 'unknown')
        
        # Preserve date for temporal split experiments
        features['date'] = str(obs.get('date', ''))
        
        # Geographic features
        features['latitude'] = obs['latitude']
        features['longitude'] = obs['longitude']
        
        # Find nearest streamgage for hydrological data
        nearest_gage = self._find_nearest_streamgage(obs_point, streamgages)
        
        # Hydrological features from nearest gage using real streamflow data
        if nearest_gage is not None:
            gage_id = str(nearest_gage['site_no'])
            # Use real streamflow data if available, otherwise fall back to daymet_cache
            streamflow_data = {}
            if streamflow_cache and 'streamflow' in streamflow_cache:
                streamflow_data = streamflow_cache['streamflow'].get(gage_id, {})
            else:
                streamflow_data = daymet_cache.get('streamflow', {}).get(gage_id, {})
            features.update(self._extract_hydrological_features(
                streamflow_data, sample_date, gage_id
            ))
        
        # Meteorological features from Daymet
        daymet_data = self._get_daymet_for_location(
            obs['latitude'], obs['longitude'], daymet_cache
        )
        features.update(self._extract_meteorological_features(
            daymet_data, sample_date
        ))
        
        # Graph topology features
        features.update(self._extract_graph_features(obs_point, graph))
        
        # Temporal features
        features.update(self._extract_temporal_features(sample_date))
        
        # Static geographic features (derived from coordinates + known basin characteristics)
        features.update(self._extract_static_geographic_features(obs))
        
        # Lagged features
        if nearest_gage is not None and 'gage_data' in locals():
            pass  # Handled in hydrological features
        
        return features
    
    def _find_nearest_streamgage(
        self, 
        point: Point, 
        streamgages: pd.DataFrame
    ) -> Optional[pd.Series]:
        """Find the nearest streamgage to a point.
        
        Args:
            point: Shapely Point
            streamgages: Streamgage dataframe
        
        Returns:
            Nearest streamgage row
        """
        if streamgages.empty:
            return None
        
        min_dist = float('inf')
        nearest = None
        
        for _, gage in streamgages.iterrows():
            gage_point = Point(gage['longitude'], gage['latitude'])
            dist = point.distance(gage_point)
            if dist < min_dist:
                min_dist = dist
                nearest = gage
        
        max_distance_deg = 1.0  # ~111 km
        if nearest is not None and min_dist > max_distance_deg:
            logger.warning(
                f"Nearest streamgage is {min_dist * 111:.1f} km away"
            )
        
        return nearest
    
    def _extract_hydrological_features(
        self,
        gage_data: Dict,
        sample_date: pd.Timestamp,
        gage_id: str
    ) -> Dict:
        """Extract hydrological features from streamgage data.
        
        Args:
            gage_data: Streamgage data dictionary
            sample_date: Sampling date
            gage_id: Streamgage identifier
        
        Returns:
            Dictionary of hydrological features
        """
        features = {}
        
        if not gage_data:
            # Use default values from NHDPlus if available
            features['discharge_cfs'] = np.nan
            features['discharge_m3s'] = np.nan
            features['gage_height_ft'] = np.nan
        else:
            # Get closest date (use data from previous day to prevent leakage)
            prev_date = sample_date - pd.Timedelta(days=1)
            
            # Discharge
            discharge = self._get_closest_value(
                gage_data.get('discharge', {}), prev_date
            )
            features['discharge_cfs'] = discharge
            if discharge is not None and not np.isnan(discharge):
                features['discharge_m3s'] = discharge * 0.0283168  # cfs to m3/s
            else:
                features['discharge_m3s'] = np.nan
            
            # Gage height
            gage_height = self._get_closest_value(
                gage_data.get('gage_height', {}), prev_date
            )
            features['gage_height_ft'] = gage_height
            
            # Lag features for discharge
            for lag in [1, 3, 7]:
                lag_date = sample_date - pd.Timedelta(days=lag)
                lag_value = self._get_closest_value(
                    gage_data.get('discharge', {}), lag_date
                )
                features[f'discharge_lag_{lag}d'] = lag_value
                if lag_value is not None:
                    features[f'discharge_lag_{lag}d_m3s'] = lag_value * 0.0283168
        
        return features
    
    def _get_closest_value(
        self,
        date_dict: Dict[str, float],
        target_date: pd.Timestamp,
        max_gap_days: int = 7
    ) -> Optional[float]:
        """Get the closest available value to a target date.
        
        Args:
            date_dict: Dictionary mapping dates to values
            target_date: Target date
            max_gap_days: Maximum acceptable gap in days
        
        Returns:
            Closest value or None
        """
        if not date_dict:
            return None
        
        target_str = target_date.strftime('%Y-%m-%d')
        
        if target_str in date_dict:
            return date_dict[target_str]
        
        # Find nearest date within max_gap_days
        closest_dist = float('inf')
        closest_val = None
        
        for date_str, val in date_dict.items():
            try:
                date = pd.to_datetime(date_str)
                gap = abs((date - target_date).days)
                if gap <= max_gap_days and gap < closest_dist:
                    closest_dist = gap
                    closest_val = val
            except Exception:
                continue
        
        return closest_val
    
    def _get_daymet_for_location(
        self,
        lat: float,
        lon: float,
        daymet_cache: Dict
    ) -> Dict:
        """Get Daymet data for a location from cache.
        
        Args:
            lat: Latitude
            lon: Longitude
            daymet_cache: Cached Daymet data
        
        Returns:
            Dictionary of Daymet data
        """
        # In real implementation, extract from Daymet grids
        # For now, use cached representative values
        return daymet_cache.get('daymet', {}).get(f"{lat:.2f}_{lon:.2f}", {})
    
    def _extract_meteorological_features(
        self,
        daymet_data: Dict,
        sample_date: pd.Timestamp
    ) -> Dict:
        """Extract meteorological features.
        
        Args:
            daymet_data: Daymet data dictionary
            sample_date: Sampling date
        
        Returns:
            Dictionary of meteorological features
        """
        features = {}
        
        if not daymet_data:
            # Default values
            for feat in ['precipitation_mm', 'max_temp_c', 'min_temp_c', 
                         'shortwave_radiation', 'wind_speed']:
                features[feat] = np.nan
            
            for lag in [1, 3, 7]:
                for feat in ['precipitation_mm', 'max_temp_c']:
                    features[f'{feat}_lag_{lag}d'] = np.nan
            
            features['precip_cumsum_7d'] = np.nan
        else:
            # Get previous day's data (to prevent leakage)
            prev_date = sample_date - pd.Timedelta(days=1)
            prev_date_str = prev_date.strftime('%Y-%m-%d')
            
            features['precipitation_mm'] = daymet_data.get(prev_date_str, {}).get('precip', np.nan)
            features['max_temp_c'] = daymet_data.get(prev_date_str, {}).get('tmax', np.nan)
            features['min_temp_c'] = daymet_data.get(prev_date_str, {}).get('tmin', np.nan)
            features['shortwave_radiation'] = daymet_data.get(prev_date_str, {}).get('srad', np.nan)
            features['wind_speed'] = daymet_data.get(prev_date_str, {}).get('wind_speed', np.nan)
            
            # Lag features
            for lag in [1, 3, 7]:
                lag_date = sample_date - pd.Timedelta(days=lag)
                lag_date_str = lag_date.strftime('%Y-%m-%d')
                daymet_day = daymet_data.get(lag_date_str, {})
                
                features[f'precipitation_mm_lag_{lag}d'] = daymet_day.get('precip', np.nan)
                features[f'max_temp_c_lag_{lag}d'] = daymet_day.get('tmax', np.nan)
            
            # 7-day cumulative precipitation
            cum_precip = 0
            count = 0
            for days_back in range(1, 8):
                lag_date = sample_date - pd.Timedelta(days=days_back)
                lag_date_str = lag_date.strftime('%Y-%m-%d')
                precip = daymet_data.get(lag_date_str, {}).get('precip', np.nan)
                if not np.isnan(precip):
                    cum_precip += precip
                    count += 1
            
            features['precip_cumsum_7d'] = cum_precip if count > 0 else np.nan
        
        return features
    
    def _extract_graph_features(
        self,
        point: Point,
        graph: nx.DiGraph
    ) -> Dict:
        """Extract features from river network graph.
        
        Args:
            point: Observation point
            graph: River network graph
        
        Returns:
            Dictionary of graph features
        """
        features = {}
        
        # Find nearest segment
        min_dist = float('inf')
        nearest_node = None
        
        for node_id, node_data in graph.nodes(data=True):
            if 'geometry' not in node_data:
                continue
            
            dist = point.distance(node_data['geometry'])
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_id
        
        if nearest_node is not None:
            node_data = graph.nodes[nearest_node]
            features['nearest_segment_id'] = nearest_node
            features['distance_to_segment_km'] = min_dist * 111.0
            features['drainage_area_km2'] = node_data.get('drainage_area', np.nan)
            features['stream_order'] = node_data.get('stream_order', np.nan)
            features['segment_slope'] = node_data.get('slope', np.nan)
            features['avg_discharge_cfs'] = node_data.get('avg_discharge_cfs', np.nan)
            features['num_upstream_nodes'] = node_data.get('num_upstream', np.nan)
            features['num_downstream_nodes'] = node_data.get('num_downstream', np.nan)
            features['is_tributary'] = int(node_data.get('is_tributary', False))
        else:
            features['nearest_segment_id'] = -1
            features['distance_to_segment_km'] = np.nan
            features['drainage_area_km2'] = np.nan
            features['stream_order'] = np.nan
            features['segment_slope'] = np.nan
            features['avg_discharge_cfs'] = np.nan
            features['num_upstream_nodes'] = np.nan
            features['num_downstream_nodes'] = np.nan
            features['is_tributary'] = 0
        
        return features
    
    def _extract_temporal_features(
        self,
        sample_date: pd.Timestamp
    ) -> Dict:
        """Extract temporal/calendar features.
        
        Args:
            sample_date: Sampling date
        
        Returns:
            Dictionary of temporal features
        """
        return {
            'month': sample_date.month,
            'day_of_year': sample_date.dayofyear,
            'season': self._get_season(sample_date.month),
            'year': sample_date.year,
            'is_spring': int(sample_date.month in [3, 4, 5]),
            'is_summer': int(sample_date.month in [6, 7, 8]),
            'is_fall': int(sample_date.month in [9, 10, 11]),
            'is_winter': int(sample_date.month in [12, 1, 2]),
        }
    
    def _get_season(self, month: int) -> int:
        """Convert month to season category.
        
        Args:
            month: Month number (1-12)
        
        Returns:
            Season index (0=winter, 1=spring, 2=summer, 3=fall)
        """
        if month in [12, 1, 2]:
            return 0
        elif month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        else:
            return 3
    
    def _extract_static_geographic_features(
        self,
        obs: pd.Series
    ) -> Dict:
        """Extract static geographic features from known basin characteristics.
        
        Args:
            obs: Observation row
        
        Returns:
            Dictionary of geographic features
        """
        features = {}
        
        # Approximate basin characteristics based on location
        lat = obs['latitude']
        lon = obs['longitude']
        
        # Delaware River basin characteristics
        # Upper basin (lat > 41)
        if lat > 41:
            features['basin_type'] = 0  # Upper
            features['urban_pct'] = 15
            features['forest_pct'] = 70
            features['agriculture_pct'] = 10
            features['population_density'] = 50
            features['elevation_m'] = 400
            features['slope_degrees'] = 1.5
        # Middle basin (40 < lat <= 41)
        elif lat > 40:
            features['basin_type'] = 1  # Middle
            features['urban_pct'] = 45
            features['forest_pct'] = 40
            features['agriculture_pct'] = 10
            features['population_density'] = 250
            features['elevation_m'] = 100
            features['slope_degrees'] = 0.8
        # Lower basin (lat <= 40)
        else:
            features['basin_type'] = 2  # Lower
            features['urban_pct'] = 60
            features['forest_pct'] = 25
            features['agriculture_pct'] = 10
            features['population_density'] = 500
            features['elevation_m'] = 10
            features['slope_degrees'] = 0.3
        
        features['impervious_surface_pct'] = features['urban_pct'] * 0.5
        
        # Great Lakes basin characteristics
        basin = obs.get('watershed', '')
        if 'Great Lakes' in basin or 'Lake' in basin:
            features['basin_type'] = 3  # Great Lakes
            features['urban_pct'] = 25
            features['forest_pct'] = 60
            features['agriculture_pct'] = 10
            features['population_density'] = 100
            features['elevation_m'] = 200
            features['slope_degrees'] = 1.2
        
        return features
    
    def _categorize_features(self, df: pd.DataFrame):
        """Identify numerical and categorical features.
        
        Args:
            df: Feature dataframe
        """
        # Exclude target and identifier columns
        exclude = ['target', 'obs_index', 'location_name', 'date', 
                   'date_parsed', 'source_dataset', 'original_units', 
                   'citation', 'notes', 'particle_types', 'watershed',
                   'nearest_segment_id', 'has_max', 'below_detection_limit',
                   'basin', 'source_url', 'sampling_method']
        
        for col in df.columns:
            if col in exclude:
                continue
            
            if df[col].dtype in ['object', 'category', 'bool']:
                self.categorical_columns.append(col)
            elif df[col].dtype == 'datetime64[ns]':
                continue  # Skip datetime
            else:
                self.numerical_columns.append(col)
        
        self.feature_columns = self.numerical_columns + self.categorical_columns
        
        logger.info(
            f"Feature categorization: {len(self.numerical_columns)} numerical, "
            f"{len(self.categorical_columns)} categorical"
        )


def create_temporal_sequences(
    features: pd.DataFrame,
    graph: nx.DiGraph,
    sequence_length: int = 7
) -> List[Dict]:
    """Create temporal sequences for the GNN.
    
    Args:
        features: Feature dataframe
        graph: River network graph
        sequence_length: Number of timesteps (days) in sequence
    
    Returns:
        List of temporal sequence samples
    """
    sequences = []
    
    # Group by observation
    for idx, obs in features.iterrows():
        sequence = {
            'features': {},
            'target': obs['target'],
            'date': obs.get('date'),
            'obs_index': obs['obs_index'],
        }
        
        # For each timestep in the sequence
        for t in range(sequence_length):
            timestep_features = {}
            
            # Get features for this timestep
            for col in features.columns:
                if col.startswith('lag_') and f'_t{t}' in col:
                    timestep_features[col] = obs[col]
                elif col in ['latitude', 'longitude', 'discharge_m3s',
                            'precipitation_mm', 'max_temp_c', 'min_temp_c']:
                    timestep_features[col] = obs[col]
            
            sequence['features'][t] = timestep_features
        
        sequences.append(sequence)
    
    return sequences


def normalize_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Normalize features using training data statistics.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        feature_columns: Columns to normalize
    
    Returns:
        Tuple of (normalized_train, normalized_val, normalized_test, scaler_params)
    """
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    # Fit on training data only
    train_features = train_df[feature_columns].copy()
    
    # Handle NaN values - fill with median from training set
    medians = train_features.median()
    train_features = train_features.fillna(medians)
    
    # Fit scaler
    scaler.fit(train_features)
    
    # Transform all datasets
    train_norm = train_df.copy()
    val_norm = val_df.copy()
    test_norm = test_df.copy()
    
    train_norm[feature_columns] = scaler.transform(
        train_features.fillna(medians)
    )
    
    val_features = val_df[feature_columns].fillna(medians)
    val_norm[feature_columns] = scaler.transform(val_features)
    
    test_features = test_df[feature_columns].fillna(medians)
    test_norm[feature_columns] = scaler.transform(test_features)
    
    scaler_params = {
        'means': dict(zip(feature_columns, scaler.mean_.tolist())),
        'stds': dict(zip(feature_columns, scaler.scale_.tolist())),
        'medians': dict(medians),
    }
    
    return train_norm, val_norm, test_norm, scaler_params


def add_graph_topology_features(
    features: pd.DataFrame,
    graph: nx.DiGraph
) -> pd.DataFrame:
    """Add graph-derived topology features to the feature matrix.
    
    Args:
        features: Feature dataframe
        graph: River network graph
    
    Returns:
        Dataframe with added topology features
    """
    # This would normally use network embeddings, but for small graphs
    # we use simple topological features
    features = features.copy()
    
    # Add PageRank centrality
    try:
        pagerank = nx.pagerank(graph)
        features['pagerank_centrality'] = features['nearest_segment_id'].map(
            lambda x: pagerank.get(x, 0) if x >= 0 else 0
        )
    except Exception:
        features['pagerank_centrality'] = 0.0
    
    # Add betweenness centrality (simplified for directed graphs)
    try:
        betweenness = nx.betweenness_centrality(graph)
        features['betweenness_centrality'] = features['nearest_segment_id'].map(
            lambda x: betweenness.get(x, 0) if x >= 0 else 0
        )
    except Exception:
        features['betweenness_centrality'] = 0.0
    
    return features
