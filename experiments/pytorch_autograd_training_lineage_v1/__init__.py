"""Frozen PyTorch Autograd projection and training-lineage experiment."""

from .pipeline import TrainingRun, TrainingSpec, run_training_step

__all__ = ["TrainingRun", "TrainingSpec", "run_training_step"]
