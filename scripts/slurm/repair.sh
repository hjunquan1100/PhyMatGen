#!/usr/bin/env bash
#SBATCH --job-name=phymatgen-repair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python}
: "${DIT_CKPT:?Set DIT_CKPT to the correction checkpoint}"

vae_args=()
if [[ -n "${VAE_CKPT:-}" ]]; then
  vae_args=(--vae-ckpt "$VAE_CKPT")
fi

cd "$ROOT_DIR"
"$PYTHON" -m phymatgen_correction.src.code.repair_ntg_with_dit \
  --dit-ckpt "$DIT_CKPT" \
  "${vae_args[@]}" \
  --data-root "${DATA_ROOT:-$ROOT_DIR/data/correction}" \
  --output-dir "${OUTPUT_DIR:-$ROOT_DIR/outputs/repair/${SLURM_JOB_ID:-local}}" \
  --device "${DEVICE:-cuda}" \
  "$@"
