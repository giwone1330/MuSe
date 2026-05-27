# MuSe — Packing-Aware Causal Sequence Modeling under Fully Homomorphic Encryption

This repository contains the research code for my Master's thesis:

- Project page: [Link](https://giwone1330.github.io/publications/muse-packing-aware-causal-sequence-modeling-under-fully-homomorphic-encryption/)
- Full Paper: [Link](https://drive.google.com/file/d/1eSPaASRvVPgcqKmsVjx5fd-ehwWu81t2/view?usp=drive_link)

The codebase is intentionally split into **two parts**:

1. **Plaintext training + evaluation** (MuSe vs Transformer/GPT-2-style baselines) — everything under `src/`, driven by Hydra configs in `configs/`.
2. **FHE inference** — contained in `FHE/` (a self-contained pipeline built around the `desilofhe` CKKS stack).

## What is MuSe (thesis summary)

MuSe is a decoder-style causal sequence model designed to align with the arithmetic constraints of pure Fully Homomorphic Encryption (FHE) inference.
In the thesis:

- Softmax attention is replaced by a **softmax-free causal mixer** (Polynomial Toeplitz Mixer).
- The standard MLP/FFN block is replaced by a **multilinear operator**.
- The architecture is studied under **packing-aware** encrypted execution, including *outer-based* and *inner-based* ciphertext layouts.

Key thesis outcomes (high level):

- End-to-end pure-FHE causal inference is demonstrated on the **3-digit addition** setting under multiple packing regimes.
- **Inner-based cached generation** substantially reduces per-token runtime while maintaining extremely small drift vs plaintext.
- In plaintext evaluation, matched MuSe variants use **fewer non-embedding parameters** than Transformer references while staying competitive on selected downstream tasks.

## Repository layout

- `src/` — plaintext training + evaluation code (Hydra entrypoints, datasets, models, trainers, evaluators)
- `configs/` — Hydra configuration tree
	- `configs/presets/` — “paper-style” experiment presets (dataset + trainer + tokenizer defaults)
	- `configs/model/` — MuSe + baseline model configs
	- `configs/dataset/` — dataset configs (synthetic arithmetic, NLP datasets, FineWeb, …)
- `runs/` — reproducibility + sweep specs (`run.yaml` files used to generate local scripts / HTCondor submit files)
- `scripts/` — helper scripts (notably `scripts/create_exp_script.py` which turns `runs/*/run.yaml` into runnable scripts)
- `chtc/` — HTCondor / Apptainer tooling for UW-Madison CHTC-style execution
- `FHE/` — FHE inference pipeline and a runnable notebook

## Installation

### Environment

The repo is Python 3.10-oriented (see `environment.yml`).

```bash
conda env create -f environment.yml
conda activate p10t28
```

If you prefer pip-only, install from `requirements.txt` in a clean Python 3.10 environment.

### Environment variables

Some features rely on environment variables (loaded via `python-dotenv`).
Create a `.env` file in the repository root (or export variables in your shell):

- `HF_TOKEN` — Hugging Face token used by training and the FHE notebook (required if you load private checkpoints)
- `WANDB_API_KEY` — optional, for Weights & Biases logging
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — optional, for run notifications

Cluster/CHTC helpers additionally assume:

- `USERNAME`, `PROJECT` — used to form `/staging/<USERNAME>/<PROJECT>/...` paths on CHTC-like systems

## Part 1 — Plaintext training & evaluation

### Entry point

Plaintext training is driven by Hydra and the main entrypoint in:

- `src/main_lm5.py`

Run it as a module from the repository root:

```bash
python -m src.main_lm5 --help
```

### Hydra configuration model

- `configs/config.yaml` is the base config.
- Use `--config-path` and `--config-name` to select a preset (recommended).
- Override any field from the CLI using standard Hydra override syntax.

#### Synthetic arithmetic (3-digit addition)

Train MuSe:

```bash
python -m src.main_lm5 \
	--config-path configs/presets --config-name 3a3 \
	model=gpt2_muse model/size@model.config=small
```

Train the baseline (GPT-2-style decoder):

```bash
python -m src.main_lm5 \
	--config-path configs/presets --config-name 3a3 \
	model=gpt2_vanilla model/size@model.config=small
```

### Outputs and logging

Hydra controls output directories. If you don’t override `hydra.run.dir`, Hydra will use its default `outputs/` directory.

W&B logging is enabled via `configs/tracker/` and `configs/env/default.yaml` if your environment is set up.

Important:

- `start.sh` assumes a CHTC-style Linux filesystem layout and HTCondor tooling.
- For macOS/local development, prefer direct Hydra CLI runs (examples above).

## Part 2 — FHE inference (`FHE/`)

The FHE implementation lives under `FHE/` and is designed as a self-contained pipeline:

- `FHE/src/inference.py` — end-to-end FHE inference function (`run_fhe_muse_inference`)
- `FHE/src/engine.py` — `desilofhe` engine creation (GPU mode) and key generation
- `FHE/src/weights.py` — weight extraction + blockwise precomputation + LayerNorm variance profiling
- `FHE/muse_experiment.ipynb` — a runnable experiment notebook used for thesis measurements

### Platform note

The FHE stack uses `desilofhe-cu124` (CUDA build) and the engine is created in `mode="gpu"`.
In practice this means FHE inference is intended to run on **Linux + NVIDIA GPU + CUDA 12.4** environments.

### Recommended: run the notebook

After saving the trained Muse model to HuggingFace, open and run:

- `FHE/muse_experiment.ipynb`

The notebook loads a MuSe checkpoint from Hugging Face with `trust_remote_code=True` and lets you choose among:

- PyTorch reference execution
- FHE execution variants (different packing/fusion modes)
- Cached generation (inner-based packing)


## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{shin2026muse,
	title  = {MuSe: Packing-Aware Causal Sequence Modeling under Fully Homomorphic Encryption},
	author = {Shin, Giwon},
	year   = {2026},
	school = {University of Wisconsin--Madison},
	type   = {Thesis}
}
```

## License

Apache License 2.0 — see `LICENSE`.
