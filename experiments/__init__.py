"""Experiment module - re-exports all experiment functions."""

from .e01_baselines import run_baseline_experiments, run_experiment_e1
from .e02_e5_baselines import (
    run_experiment_e2_svm_rbf,
    run_experiment_e3_gaussian_process,
    run_experiment_e4_mlp,
    run_experiment_e5_ensemble,
    run_experiment_e7_feature_importance,
)
from .e06_gnn import run_gnn_experiment, run_experiments_e2_e6
from .e11_e19_generalization import (
    run_experiment_e11_spatial_holdout,
    run_experiment_e12_temporal_holdout,
    run_experiment_e13_spatiotemporal_holdout,
    run_experiment_e18_unseen_watershed,
    run_experiment_e19_temporal_extrapolation,
)
from .e14_e15_topology_control import (
    run_experiment_e14_edge_ablation,
    run_experiment_e15_node_ablation,
    run_experiment_e21_topology_comparison,
)
from .e17_e20_special_experiments import (
    run_experiment_e17_sparse_data,
    run_experiment_e20_extreme_events,
)

__all__ = [
    'run_baseline_experiments',
    'run_experiment_e1',
    'run_experiment_e2_svm_rbf',
    'run_experiment_e3_gaussian_process',
    'run_experiment_e4_mlp',
    'run_experiment_e5_ensemble',
    'run_experiment_e7_feature_importance',
    'run_gnn_experiment',
    'run_experiments_e2_e6',
    'run_experiment_e11_spatial_holdout',
    'run_experiment_e12_temporal_holdout',
    'run_experiment_e13_spatiotemporal_holdout',
    'run_experiment_e18_unseen_watershed',
    'run_experiment_e19_temporal_extrapolation',
    'run_experiment_e14_edge_ablation',
    'run_experiment_e15_node_ablation',
    'run_experiment_e21_topology_comparison',
    'run_experiment_e17_sparse_data',
    'run_experiment_e20_extreme_events',
]
