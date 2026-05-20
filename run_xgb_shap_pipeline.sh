#!/usr/bin/env bash
set -euo pipefail

EPV=10
FEATURES=25
MODEL=xgb

echo "======================================="
echo "Running XGB-SHAP inferential pipeline"
echo "======================================="

echo "[1/3] Running regression coefficients..."

python regression_coefs.py \
    --model ${MODEL} \
    --epv ${EPV} \
    --num-features ${FEATURES}

echo "[2/3] Running significant feature analysis..."

python logit_regression_significant_features.py \
    --model ${MODEL} \
    --epv ${EPV}

echo "[3/3] Running AUROC analysis..."

python logit_regression_aucroc.py \
    --model ${MODEL} \
    --epv ${EPV}

echo "======================================="
echo "Pipeline completed successfully"
echo "======================================="
