# Lazy imports — VJEPAAligner requires external vjepa2 repo on PYTHONPATH;
# VideoMAEv2Aligner requires transformers. Neither is needed for inference.


def __getattr__(name):
    if name == "VJEPAAligner":
        from .vjepa_aligner import VJEPAAligner
        return VJEPAAligner
    if name == "VideoMAEv2Aligner":
        from .videomae_aligner import VideoMAEv2Aligner
        return VideoMAEv2Aligner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["VJEPAAligner", "VideoMAEv2Aligner"]
