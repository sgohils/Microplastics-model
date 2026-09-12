# Microplastic Transport Prediction System
# Research specification: docs/microplastic_gnn_research_specification.md

## Overview

A spatiotemporal graph neural network system for predicting microplastic concentrations in river networks, with explicit comparison against conventional machine learning baselines and graph-topology controls.

## Key Components

- **Study Region:** Delaware River Basin (primary), with Great Lakes tributaries for validation
- **Target Variable:** Microplastic concentration (particles/m³ in water samples)
- **Graph Representation:** River segments as nodes, flow-direction connectivity as edges
- **Models:** XGBoost, Random Forest, MLP baselines + spatiotemporal GNN (GraphSAGE with GRU)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python train.py

# Run specific experiments
python experiments/e01_baselines.py
python experiments/e07_topology_control.py
```

## Structure

See `docs/IMPLEMENTATION_SPEC.md` for the complete implementation specification.

## Data Sources

All data sources are documented in `docs/DATA_SOURCES.md` and `data/data_manifest.csv`.

## Important Notes

- All microplastic observations are real measurements from USGS datasets
- No synthetic data is used for the primary experiment
- Train/validation/test splits are strictly enforced with no leakage
- The model is not trained on test data under any circumstances
