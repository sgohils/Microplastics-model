"""Check E2-E5 results."""
import json, os

p = 'experiments/results'
for f in ['E2_svm_rbf.json', 'E3_gaussian_process.json', 'E4_mlp.json', 'E5_ensemble.json']:
    fp = os.path.join(p, f)
    if os.path.exists(fp):
        with open(fp) as fh:
            r = json.load(fh)
        if r.get('metrics'):
            m = r['metrics']
            print(f'{f}: MAE={m["mae"]:.4f}, RMSE={m["rmse"]:.4f}, R2={m["r2"]:.4f}')
    else:
        print(f'{f}: NOT FOUND')

# Check E11/E12/E13
for f in ['E11_spatial_holdout.json', 'E12_temporal_holdout.json', 'E13_spatiotemporal_holdout.json']:
    fp = os.path.join(p, f)
    if os.path.exists(fp):
        with open(fp) as fh:
            r = json.load(fh)
        if r.get('metrics'):
            m = r['metrics']
            print(f'{f}: MAE={m["mae"]:.4f}, RMSE={m["rmse"]:.4f}, R2={m["r2"]:.4f}')
    else:
        print(f'{f}: NOT FOUND')

# Check E17/E20
for f in ['e17_sparse_data.json', 'e20_extreme_events.json']:
    fp = os.path.join(p, f)
    if os.path.exists(fp):
        with open(fp) as fh:
            r = json.load(fh)
        print(f'{f}: keys={list(r.keys())[:5]}')
    else:
        print(f'{f}: NOT FOUND')
