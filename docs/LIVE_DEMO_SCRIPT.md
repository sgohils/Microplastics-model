# Live Demo Script

This script demonstrates how to use the trained model to generate predictions for a given location and time, including uncertainty estimates from the deep ensemble.

## Prerequisites
- Ensure the environment is set up as described in `docs/REPRODUCIBILITY_PACKAGE.md`.
- A trained model checkpoint must be available (e.g., `checkpoints/best_model.pt` from running the experiments).
- The processed feature data for the desired prediction time and location must be prepared (see `docs/REPRODUCIBILITY_PACKAGE.md` for data preparation steps).

## Demo Script (Python)
```python
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config import load_config
from src.utils.seed import get_device
from predict import load_trained_model, predict_from_csv

def main():
    # Load configuration
    config_path = Path("config.yaml")
    config = load_config(config_path)
    
    # Set device (CPU or GPU if available)
    device = get_device()
    print(f"Using device: {device}")
    
    # Path to trained model checkpoint (ensemble member 0 as example)
    model_path = Path("checkpoints/best_model.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train the model first.")
    
    # Load the model
    model, checkpoint = load_trained_model(model_path, config, device=device)
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Example: Predict on the test set (replace with your own input CSV)
    input_csv = Path("data/processed/test_features.csv")  # Ensure this file exists
    output_csv = Path("predictions/test_predictions.csv")
    
    if not input_csv.exists():
        print(f"Warning: Input CSV {input_csv} not found. Please provide processed features.")
        print("You can generate features using the data preparation pipeline.")
        return
    
    # Generate predictions (this function returns a DataFrame with predictions and optionally uncertainty)
    # Note: The predict_from_csv function in predict.py currently returns predictions only.
    # To get uncertainty, we need to use the ensemble prediction approach.
    # For simplicity, we demonstrate loading multiple ensemble members and computing mean and std.
    
    # Load ensemble members (assuming checkpoints are named best_model_0.pt, best_model_1.pt, etc.)
    ensemble_predictions = []
    for i in range(5):  # 5 ensemble members as per uncertainty method
        member_path = Path(f"checkpoints/best_model_{i}.pt")
        if member_path.exists():
            member_model, _ = load_trained_model(member_path, config, device=device)
            member_model.eval()
            # Use predict_from_csv to get predictions for this member
            preds = predict_from_csv(member_model_path=member_path, input_csv=input_csv, config_path=config_path)
            ensemble_predictions.append(preds['prediction'].values)  # Adjust based on actual return format
        else:
            print(f"Warning: Ensemble member {member_path} not found. Using available members.")
    
    if ensemble_predictions:
        ensemble_array = np.array(ensemble_predictions)  # shape (n_members, n_samples)
        mean_pred = np.mean(ensemble_array, axis=0)
        std_pred = np.std(ensemble_array, axis=0)
        
        # Save results
        results_df = pd.DataFrame({
            'prediction_mean': mean_pred,
            'prediction_std': std_pred,
            'prediction_lower_95': mean_pred - 1.96 * std_pred,
            'prediction_upper_95': mean_pred + 1.96 * std_pred
        })
        results_df.to_csv(output_csv, index=False)
        print(f"Predictions with uncertainty saved to {output_csv}")
        print(f"Mean prediction: {mean_pred[:5]}")  # Show first 5 values
        print(f"Uncertainty (std): {std_pred[:5]}")
    else:
        print("No ensemble members available for uncertainty estimation.")

if __name__ == "__main__":
    main()
```

## How to Use This Demo
1. Train the model and generate ensemble checkpoints (see `docs/REPRODUCIBILITY_PACKAGE.md` for training command).
2. Ensure you have processed feature data for the prediction set (e.g., test set features).
3. Run the script: `python demo_live_prediction.py`
4. View the output CSV containing predictions and uncertainty bounds.

## Notes
- The actual implementation may vary depending on the exact structure of the prediction script and checkpoint files.
- For a true live interactive demo, consider wrapping this functionality in a simple Streamlit or GUI application.
- Always verify that the input features match those used during training (same preprocessing, no leakage).

## Fallback
If checkpoints or processed data are not available, the demo cannot be run. In such case, refer to the static figures in `presentation/figures/` for illustrative examples.