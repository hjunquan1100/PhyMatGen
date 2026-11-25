#!/usr/bin/env bash
#SBATCH --job-name=phymatgen-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python}
: "${DIT_CKPT:?Set DIT_CKPT to the correction checkpoint}"

cd "$ROOT_DIR"
"$PYTHON" -m phymatgen_correction.src.code.eval_basic_cond_dit \
  --ckpt "$DIT_CKPT" \
  --data-root "${DATA_ROOT:-$ROOT_DIR/data/correction}" \
  --output-dir "${OUTPUT_DIR:-$ROOT_DIR/outputs/evaluation}" \
  "$@"
