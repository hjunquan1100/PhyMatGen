#!/usr/bin/env bash
#SBATCH --job-name=phymatgen-latents
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python}
: "${VAE_CKPT:?Set VAE_CKPT to the trained autoencoder checkpoint}"

cd "$ROOT_DIR"
"$PYTHON" -m phymatgen_correction.src.code.build_latent_bank \
  --vae-ckpt "$VAE_CKPT" \
  --data-root "${DATA_ROOT:-$ROOT_DIR/data/correction}" \
  --datasets qm9 mp20 perov5 carbon24 \
  --splits tg ntg \
  "$@"
