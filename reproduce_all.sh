#!/usr/bin/env bash
set -e

echo "=== Microplastic Transport GNN Pipeline ==="
echo ""

# Run the full pipeline
python train.py \
    --config config.yaml \
    --experiment baselines \
    --output experiments/results/

python train.py \
    --config config.yaml \
    --experiment gnn \
    --output experiments/results/

echo ""
echo "=== Pipeline Complete ==="
echo "Results saved to experiments/results/"
echo "Figures saved to figures/"
