#!/bin/sh
# Regenerates the submission from the provided data bundle in data/ (see DATA.md).
#   python3 -m venv .venv && .venv/bin/pip install -r requirements-solution.txt
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
OUT=${1:-submission.csv}
solution/download_dfl.sh                       # other matches: DFL open Bundesliga data (training)
$PY solution/train_dfl.py final                # -> models/gbm_dfl.joblib
$PY solution/predict.py --model dfl --out "$OUT"
$PY evaluation/validate_submission.py "$OUT"
