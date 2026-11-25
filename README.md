PhyMatGen combines an all-atom latent generative model with a validation-guided
DiT correction pipeline for molecules and periodic materials. The repository
contains training, evaluation, validation, and repair code only. Datasets,
checkpoints, generated structures, and experiment logs are intentionally not
included.

The `phymatgen_correction` package contains the project-specific latent-bank,
conditional DiT, global repair, masked repair, and local refinement workflow.

## Repository layout

```text
PhyMatGen/
├── configs/                 # Hydra configs for VAE and latent generation
├── src/                     # Generation backbone and training entry points
├── phymatgen_correction/src/ # Validators and correction implementation
│   └── code/                # Latent bank, DiT training, evaluation, repair
├── scripts/slurm/           # Portable Slurm templates
├── data/                    # Local datasets, ignored by Git
├── outputs/                 # Checkpoints and generated results, ignored by Git
└── requirements.txt
```

## Installation

Python 3.10 and a CUDA-enabled PyTorch environment are recommended. Install
PyTorch and PyTorch Geometric for the CUDA version available on the machine,
then install the remaining dependencies:

```bash
pip install -r requirements.txt
export PROJECT_ROOT="$PWD"
```

Open Babel, `torch_scatter`, and `torch_sparse` may need platform-specific
Conda or PyG wheels. CHGNet is needed for the heavy crystal validation stage.

## Training workflow

### 1. Train the autoencoder

```bash
python src/train_autoencoder.py
```

Hydra overrides can select a dataset or trainer, for example:

```bash
python src/train_autoencoder.py data=mp20_only trainer=gpu
```

### 2. Train the latent generation model

The VAE checkpoint is supplied at runtime and is not stored in this repository.

```bash
python src/train_diffusion.py \
  diffusion_module.autoencoder_ckpt=/path/to/vae.ckpt
```

For joint training on QM9, MP20, Carbon24, and Perov5, use the four-dataset
data config and five condition embeddings (index 0 is reserved by the model):

```bash
python src/train_diffusion.py \
  data=jointtrain \
  diffusion_module.autoencoder_ckpt=/path/to/vae.ckpt \
  diffusion_module.denoiser.num_datasets=5
```

### 3. Validate generated structures

The crystal validators read paths from environment variables. The following
example validates MP20 structures:

```bash
export PHYMATGEN_CORRECTION_CIF_DIR=/path/to/generated/mp20
export PHYMATGEN_CORRECTION_TG_DIR="$PWD/data/correction/mp20/tg"
export PHYMATGEN_CORRECTION_NTG_DIR="$PWD/data/correction/mp20/ntg"
export PHYMATGEN_CORRECTION_CSV_OUT="$PWD/data/correction/mp20/validation_combined_results.csv"
python -m phymatgen_correction.src.validate_mp20
```

Use `validate_carbon24`, `validate_perov5`, or `validate_qm9_first3` for the
other datasets.

### 4. Build latent banks

```bash
python -m phymatgen_correction.src.code.build_latent_bank \
  --vae-ckpt /path/to/vae.ckpt \
  --data-root data/correction \
  --datasets qm9 mp20 perov5 carbon24 \
  --splits tg ntg
```

Expected correction data layout:

```text
data/correction/<dataset>/
├── tg/                              # structures passing validation
├── ntg/                             # structures failing validation
├── validation_combined_results.csv  # QM9 uses qm9_first3_results.csv
└── latent/
    ├── tg_latents.pt
    └── ntg_latents.pt
```

### 5. Train the correction DiT

```bash
python -m phymatgen_correction.src.code.train_basic_cond_dit \
  --data-root data/correction \
  --output-dir outputs/correction/run_001
```

### 6. Repair failed samples

```bash
python -m phymatgen_correction.src.code.repair_ntg_with_dit \
  --dit-ckpt /path/to/correction_dit.pt \
  --vae-ckpt /path/to/vae.ckpt \
  --data-root data/correction \
  --output-dir outputs/repair/run_001
```

Masked reconstruction is available through
`phymatgen_correction.src.code.repair_masked_structures_with_dit`. Dataset-specific local
refinement entry points are under `phymatgen_correction/src/code/refine_*`.

## Slurm

The original Slurm scripts were useful as records of the actual workflow, but
they contained machine-specific paths and environment names. Portable versions
are provided in `scripts/slurm`; select an environment by setting `PYTHON` and
provide checkpoint paths through `VAE_CKPT` or `DIT_CKPT`.

## Excluded legacy code

The former `CHECK` directory is not included. It was not imported by the
training or correction pipeline, and its relevant validation logic is already
implemented in `phymatgen_correction/src/validate_*.py`. Plotting code, debug runs, paper
figures, duplicate experimental scripts, logs, and model weights are also
excluded from this repository.

## License and attribution

See `LICENSE` for the upstream licensing terms. The generation backbone is
derived from the All-atom Diffusion Transformers project; third-party portions
under `src/tools/ase_notebook` and `src/models/components/equiformer_v2` retain
their own license files.
