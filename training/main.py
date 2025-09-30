#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train ProPreT5 (camera-ready, no argparse).
- All settings live in the Config dataclass
- Robust single-GPU / multi-GPU (SLURM) launch
- Deterministic seeding
- Rank-aware logging & TB writer
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

# Project imports (assumed available in PYTHONPATH)
from model import ProPreT5, buildTokenizer
from trainer import Trainer

# Optional idr_torch (SLURM helper)
try:
    import idr_torch  # provides .rank, .local_rank, .size, .hostnames
    HAS_IDR = True
except Exception:
    HAS_IDR = False

DEFAULT_CUDA_NUMBER = 0
DEFAULT_DEVICE = f"cuda:{DEFAULT_CUDA_NUMBER}"


def is_master(rank: int) -> bool:
    return rank in (-1, 0)


def setup_hf_env():
    """Configure HF to run offline with explicit caches."""
    work = os.getenv("WORK", "~")
    work = os.path.expanduser(work)

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(work) / ".cache/huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(work) / ".cache/huggingface/datasets_chimie"))
    os.environ.setdefault("HF_HOME", str(Path(work) / "hf"))

    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reserve_gpu_memory(cuda_number=0):

    mem_frees = os.popen('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader').read().split()

    mem_free = int(mem_frees[cuda_number])

    mem_to_reserve = int(0.8 * mem_free)

    mem_reserved = torch.rand((256,1024,mem_to_reserve)).to("cuda:{}".format(cuda_number))
    mem_reserved = torch.rand((2,2)).to("cuda:{}".format(cuda_number))


@dataclass
class Config:
    # Data / vocab
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_dir = os.path.join(project_root, "../data")
    vocab_path = os.path.join(project_root, "../tokens2.txt")
    train_name = ["original_train_for_propret5.csv", "augmented_train_for_propret5.csv"]
    test_name = "original_test_for_propret5.csv"

    # Model / training
    epochs: int = 100
    batch_size: int = 8
    max_len: int = 512
    lr: float = 1e-4
    seed: int = 2
    num_beams: int = 4
    fraction: float = 1.0

    timeout_hours: int = 24

    # Checkpointing / evaluation
    start_from: Optional[str] = None # Path to pretrained weights (set to evaluate or fine-tune an existing model, leave None to start fresh)
    start_epoch: int = 0 # The epoch number to load (or 0 to start from scratch)
    retest_end: int = -1  # set to -1 to train instead of testing

    # Masking
    types: dict = field(default_factory=lambda: {"reactant": 0, "reaction": 1, "product": 2, "reagent": 3})
    inputs: List[str] = field(default_factory=lambda: ["reactant", "reaction"])
    targets: List[str] = field(default_factory=lambda: ["product"])

    # The following two are dictionaries of type IDs to mask during training
    # Keys = type IDs to mask
    # Values = (psample, mask_rate):
    #    - psample: probability of applying masking to a given sample
    #    - mask_rate: fraction of tokens to mask within a selected sample
    training_masked_types: dict = field(default_factory=lambda: {1: (1, 1)})
    testing_masked_types: dict = field(default_factory=lambda: {1: (1, 1)})
    remap_types: dict = field(default_factory=dict) # Simpler than redefining the types, which would require reprocessing the dataset

    multigpu: bool = True         # set False to force single-GPU
    local_test: bool = False      # for local test without GPU

    # Cache control
    rebuildCacheTrain: bool = True
    rebuildCacheTest: bool = True

    # Logging / I/O
    save_dir: str = "weights"
    output_dir: str = "logs"

    # Generation frequency (if used in Trainer)
    freq_gen: int = 100


def main():
    setup_hf_env()
    cfg = Config()

    # Local quick test setup
    if cfg.local_test:
        cfg.train_name = ["train.csv", "train_synthesis.csv"]
        cfg.test_name = "test.csv"
        cfg.vocab_path = "../tokens2.txt"
        cfg.data_dir = str(Path.cwd() / "../data/data_samples")

    # I/O
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Distributed / device
    # -------------------------
    rank = -1
    local_rank = 0
    device = torch.device("cpu")

    multigpu_requested = cfg.multigpu and HAS_IDR
    if multigpu_requested:
        rank = idr_torch.rank
        local_rank = idr_torch.local_rank
        world_size = idr_torch.size

        if is_master(rank):
            master_addr = os.getenv("MASTER_ADDR", "unknown")
            print(f">>> Training on {len(idr_torch.hostnames)} nodes, {world_size} processes. Master: {master_addr}")

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(hours=cfg.timeout_hours),
        )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda")
        if is_master(rank):
            print(f"[rank {rank}] dist initialized, using cuda:{local_rank}")
    else:
        # Single GPU / CPU fallback
        rank = 0
        local_rank = 0
        if torch.cuda.is_available():
            try:
                reserve_gpu_memory(DEFAULT_CUDA_NUMBER)
            except Exception:
                pass
            device = torch.device(DEFAULT_DEVICE)
        else:
            device = torch.device("cpu")
        if not HAS_IDR and cfg.multigpu:
            print("Warning: multigpu=True but idr_torch not available. Falling back to single-GPU/CPU.")

    # -------------------------
    # Reproducibility
    # -------------------------
    set_all_seeds(cfg.seed)

    # -------------------------
    # Tokenizer & Model
    # -------------------------
    tokenizer = buildTokenizer(cfg)
    model = ProPreT5(cfg, tokenizer)
    model.to(device)

    # -------------------------
    # Trainer + Data
    # -------------------------
    writer: Optional[SummaryWriter] = None
    if is_master(rank):
        writer = SummaryWriter(cfg.output_dir)
        print(f"device = {device}")

    trainer = Trainer(cfg, device, model.tokenizer, writer)

    dataset_train = trainer.load_and_prepare_dataset(
        cfg.train_name, fraction=cfg.fraction, rebuildCache=cfg.rebuildCacheTrain
    )
    dataset_test = trainer.load_and_prepare_dataset(
        cfg.test_name, fraction=cfg.fraction, rebuildCache=cfg.rebuildCacheTest
    )

    # -------------------------
    # Train or Re-test
    # -------------------------
    if cfg.start_from is not None and (cfg.start_epoch <= cfg.retest_end) and (cfg.retest_end >= 0):
        if is_master(rank):
            print(f"Recovering test curves from '{cfg.start_from}' epochs {cfg.start_epoch}..{cfg.retest_end}")
        trainer.recover_test_curves(model, dataset_test, device, cfg.start_epoch, cfg.retest_end)
    else:
        trainer.run_training(model, dataset_train, device, dataset_test)

    # Cleanup
    if writer is not None:
        writer.flush()
        writer.close()

    if multigpu_requested and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    if HAS_IDR:
        node_id = os.getenv("SLURM_NODEID", "NA")
        print(f"- Process {idr_torch.rank} corresponds to GPU {idr_torch.local_rank} of node {node_id}")
    print("cwd =", Path.cwd())
    main()
