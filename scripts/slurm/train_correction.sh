#!/usr/bin/env bash
#SBATCH --job-name=phymatgen-correction
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python}
RUN_NAME=${RUN_NAME:-basic_cond_dit_${SLURM_JOB_ID:-local}}

cd "$ROOT_DIR"
"$PYTHON" -m phymatgen_correction.src.code.train_basic_cond_dit \
  --data-root "${DATA_ROOT:-$ROOT_DIR/data/correction}" \
  --output-dir "${OUTPUT_DIR:-$ROOT_DIR/outputs/correction/$RUN_NAME}" \
  --device "${DEVICE:-cuda}" \
  "$@"
