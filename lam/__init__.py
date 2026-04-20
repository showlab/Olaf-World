"""Olaf-World Latent Action Model.

The stable inference interface consumed by `world_model/` is in `lam.inference`.
Training entry points live in `lam.main` (use `python -m lam.main fit ...`).
"""
from lam.inference import FrozenLAMEncoder, load_lam_encoder

__all__ = ["FrozenLAMEncoder", "load_lam_encoder"]
