#!/usr/bin/env bash
#SBATCH --job-name=phymatgen-generation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python}
: "${VAE_CKPT:?Set VAE_CKPT to the trained autoencoder checkpoint}"
export PROJECT_ROOT=${PROJECT_ROOT:-$ROOT_DIR}

cd "$ROOT_DIR"
"$PYTHON" src/train_diffusion.py \
  diffusion_module.autoencoder_ckpt="$VAE_CKPT" \
  "$@"
