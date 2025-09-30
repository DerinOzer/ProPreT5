# ProPreT5 — Training

This repository contains a training pipeline for a T5‑based **ProPreT5** for molecular product prediction with typed inputs (reactants, reactions, etc.). It includes:
- A robust tokenizer builder from a plain text vocabulary.
- Single‑GPU and SLURM multi‑GPU (DDP) training.
- Deterministic, offline‑friendly HuggingFace setup.
- Efficient dataset preprocessing and caching.


---

## Project Layout

```
project_root/
├── data/                # CSV datasets live here
│   ├── hartenfeller_test_ds_reannotated_for_propret5.csv
│   └── ...
├── tokens.txt           # vocabulary file (one token per line)
├── training/
│   ├── main.py          # entrypoint (no argparse; all config in code)
│   ├── model.py         # LongContextMolFormer / ProPreT5
│   ├── trainer.py       # Trainer (data prep, DDP, loops, eval)
│   ├── dataCollator.py  # custom collator
└── README.md
```
---

## Environment & Dependencies

- Python 3.9+ (recommended)
- PyTorch (CUDA build matching your drivers)
- HuggingFace `transformers` and `datasets`
- `sacrebleu`, `tqdm`, `tensorboard`

Install (example):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets sacrebleu tqdm tensorboard
# idr_torch is cluster-specific; install per your HPC docs.
```

### Offline‑friendly Hugging Face caches

`main.py` sets sensible defaults:
- `HF_DATASETS_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- Caches under `${WORK}/.cache/huggingface` (fallback to `~` if `WORK` is unset)
- `HF_DATASETS_CACHE` and `HF_HOME` are created if missing

You can override by exporting your own env vars before launch.

---

## Data Format

CSV files should contain columns for inputs/targets. The `Trainer` automatically discovers fields with names like:
- Inputs: keys listed in `Config.inputs` (e.g., `"reactant"`, `"reaction"`) and any numbered variants (`reactant_1`, `reactant_2`, …).
- Targets: keys listed in `Config.targets` (e.g., `"product"`).

If the dataset stores **integer reaction indices**, `Trainer` maps them using `dataset.get_rxn_list()`.

Tokenization truncates to `Config.max_len` and constructs **type IDs** aligned to tokens. Labels append `<sep>` and replace it with `<eos>` at the end.

Processed datasets are cached in `HF_DATASETS_CACHE` with a name that includes dataset filename(s) and `fraction=` suffixes.

---

## Running

### Single‑GPU (or CPU fallback)
Set `Config.multigpu = False` in `main.py`, then:
```bash
cd training
python main.py
```

### Multi‑GPU on SLURM (DDP with `idr_torch`)
- Ensure the cluster exports SLURM env vars (`MASTER_ADDR`, `MASTER_PORT`, etc.).
- Set `Config.multigpu = True`.

Example (adapt to your HPC launcher):
```bash
srun --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=4 \
     --nodes=1 --time=04:00:00 \
     bash -lc 'cd training && python main.py'
```

The script will set the local CUDA device with `idr_torch.local_rank`, and synchronize tokenizer building / dataset preprocessing across ranks via `dist.barrier()`.

---

## Outputs

- **Checkpoints** in `weights/`
  - `model_epoch_{E}.pth`
  - `optimizer_epoch_{E}.pth`
- **Logs** in `logs/`
  - TensorBoard scalars: loss, accuracy, BLEU, parameter norm
  - CSV generations per epoch: `logs/generations_epoch{E}.csv` with columns
    `step, rank, full_input, input, target, generation`

---

## Citation

If you use this code, please consider citing our work :

```
@article{ozer2025transformer,
  title={A transformer model for predicting chemical reaction products from generic templates},
  author={Ozer, Derin and Lamprier, Sylvain and Cauchy, Thomas and Gutowski, Nicolas and Da Mota, Benoit},
  journal={arXiv preprint arXiv:2503.05810},
  year={2025}
}
```

---

## License

MIT License
