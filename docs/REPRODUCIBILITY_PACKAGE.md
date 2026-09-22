# Reproducibility Package

This document explains exactly how another researcher could reproduce the project.

## Software and Versions
The following software versions were used (as specified in `environment.yml` and `requirements.txt`):
- Python 3.11
- PyTorch >=2.1 (via pytorch channel)
- PyTorch Geometric >=0.6
- NumPy >=1.24
- Pandas >=2.0
- SciPy >=1.10
- Scikit-learn >=1.3
- GeoPandas >=0.13
- Shapely >=2.0
- PyProj >=3.5
- NetworkX >=3.1
- Rasterio >=1.3
- Xarray >=2023.6
- Matplotlib >=3.7
- Seaborn >=0.12
- Plotly >=5.15
- Joblib >=4.13
- PyYAML >=6.0
- Tqdm >=4.65
- XGBoost >=1.7
- LightGBM >=4.0
- Statsmodels >=0.14
- Requests >=2.31

## Dependencies
All dependencies are listed in `environment.yml` (for conda) and `requirements.txt` (for pip). To reproduce the environment, run:
```bash
conda env create -f environment.yml
# or
pip install -r requirements.txt
```

## Data Sources
Microplastic observations:
- USGS ScienceBase: Delaware River, 2018 (DOI:10.5066/P9QVIVX3)
- USGS ScienceBase: Great Lakes Tributaries, 2014-2015
- USGS ScienceBase: Northeastern U.S. Streams, 2017-2018
River network: NHDPlus v2 and NHDPlus HR (EPA/USGS)
Hydrology: USGS NWIS via API
Meteorology: Daymet v4 (Oak Ridge National Laboratory)
Geography: NLCD 2019 (USGS), NED 10m (USGS), gridded population density

All data sources are publicly available and documented in `docs/DATA_SOURCES.md` (if created) and the research specification.

## Data Preparation
1. Download microplastic observation CSV files from the provided USGS DOIs.
2. Harmonize units to particles per cubic meter (particles/m³): convert particles/L by multiplying by 1000.
3. Exclude observations lacking volume measurements (cannot convert to concentration).
4. Spatially join observations to NHDPlus flowline reaches to assign nodes.
5. Extract environmental covariates:
   - Hydrology: daily discharge (cfs → m³/s) and gage height from nearest USGS streamgage (inverse-distance weighting within 50 km).
   - Meteorology: Daymet gridded data (precipitation, temperature, wind speed, snow water equivalent) at 1 km resolution, extracted at node centroid.
   - Geography: NLCD land cover fractions, NED elevation, impervious surface, population density.
6. Compute temporal features: lagged precipitation and discharge (1-, 3-, 7-day) using only past data to prevent leakage; month, season.
7. Apply log1p transformation to target variable for training.
8. Normalize features using z-score (mean and standard deviation) computed exclusively on the training set.
9. Save processed data as feature matrices and graph structure.

## Configuration
The central configuration file is `config.yaml`. It specifies:
- Model type (`graphsage_gru`)
- Hyperparameters (hidden_dim, learning_rate, weight_decay, dropout, sequence_length)
- Data paths
- Training parameters (validation split, early stopping patience, gradient clipping, device)
Example `config.yaml` is provided in the repository.

## Random Seeds
For reproducibility, the following fixed seeds were used in experiments: {42, 123, 456, 789, 101}. These seeds are referenced in the statistical analysis and ensemble training.

## Training Command
To train the model from scratch, run:
```bash
python experiments/run_experiment.py --config config.yaml --train
```
This script handles data loading, preprocessing, graph construction, model training, validation, and saving checkpoints. See `experiments/run_experiment.py` for details.

## Evaluation Command
To evaluate a trained model on the test set, run:
```bash
python experiments/run_experiment.py --config config.yaml --evaluate --checkpoint checkpoints/best_model.pt
```
Alternatively, use the prediction script:
```bash
python predict.py --model_path checkpoints/best_model.pt --input_csv data/processed/test_features.csv --output_csv predictions/test_predictions.csv --config config.yaml
```

## Figure Generation Command
Figures can be generated using the plotting scripts in `src/visualization/` (if available) or by running the analysis notebooks. Example:
```bash
python src/visualization/plot_results.py --results_dir experiments/results/ --output_dir presentation/figures/
```
Refer to `src/visualization/` for specific figure generation scripts.

## Prediction Command
To generate predictions for new data (e.g., for deployment), run:
```bash
python predict.py --model_path checkpoints/best_model.pt --input_csv data/processed/new_features.csv --output_csv predictions/new_predictions.csv --config config.yaml
```
The input CSV must contain the same features as the training data (static and temporal, with lagged features computed from historical data).

## Notes
- All preprocessing steps (normalization, lag feature computation) are fitted only on the training set to prevent data leakage.
- Graph construction for holdout experiments uses only training-set nodes to prevent leakage.
- The ensemble for uncertainty quantification uses 5 members with different random seeds and train/validation splits.
- For exact command-line arguments, see the help of each script (e.g., `python experiments/run_experiment.py --help`).