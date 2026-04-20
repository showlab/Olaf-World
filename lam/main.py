"""LAM training / test entry point (PyTorch Lightning CLI)."""
import argparse
import sys
import torch
from lightning.pytorch.cli import LightningCLI

from lam.model import LAM
from lam.vq_model import VQLAM
from lam.model_align import LAMAlign


MODEL_REGISTRY = {
    "lam": LAM,
    "vq_lam": VQLAM,
    "lam_align": LAMAlign,
}


def main():
    """Usage:
        python -m lam.main fit --model-type lam --config configs/lam/lam.yaml
        python -m lam.main test --model-type lam_align --config ...
    """
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model-type", "--model_type",
        choices=list(MODEL_REGISTRY.keys()),
        default="lam",
    )
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    model_class = MODEL_REGISTRY[known.model_type.lower()]
    print(f"[lam.main] model: {model_class.__name__}")
    LightningCLI(model_class, seed_everything_default=32)


if __name__ == "__main__":
    main()
