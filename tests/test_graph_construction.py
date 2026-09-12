"""Tests for graph construction."""
import pytest
import numpy as np
import pandas as pd
import torch
import networkx as nx

from src.graph.build_graph import GraphBuilder, get_river_graph_statistics
from src.graph.river_topology import build_haversine_graph


@pytest.fixture
def sample_observations():
    """Create sample observation data."""
    return pd.DataFrame({
        'obs_index': range(10),
        'location_name': [f'loc_{i}' for i in range(10)],
        'latitude': np.linspace(39.5, 42.5, 10),
        'longitude': np.linspace(-75.5, -74.0, 10),
        'watershed': ['Delaware'] * 10,
        'target': np.random.lognormal(0, 1, 10),
    })


@pytest.fixture
def sample_river_network():
    """Create sample river network graph."""
    G = nx.DiGraph()
    G.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),
        (4, 9),
    ])
    return G


class TestGraphConstruction:
    """Tests for graph construction."""
    
    def test_build_haversine_graph(self, sample_observations):
        """Test haversine graph creation."""
        graph = build_haversine_graph(
            sample_observations,
            coord_cols=('latitude', 'longitude'),
            threshold_km=50
        )
        assert isinstance(graph, nx.Graph)
        assert len(graph.nodes()) <= 10
        assert len(graph.edges()) >= 0
    
    def test_graph_builder_init(self):
        """Test GraphBuilder initialization."""
        config = {'graph': {'max_nodes': 1000}}
        builder = GraphBuilder(config)
        assert builder.config == config
        assert builder.graphs == {}
    
    def test_river_graph_statistics(self, sample_river_network):
        """Test river graph statistics."""
        stats = get_river_graph_statistics(sample_river_network)
        assert isinstance(stats, dict)
        assert 'n_nodes' in stats
        assert 'n_edges' in stats
        assert stats['n_nodes'] == 10
        assert stats['n_edges'] == 9
