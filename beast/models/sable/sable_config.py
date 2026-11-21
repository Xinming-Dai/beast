"""Pydantic config schemas for the Sable multi-view 3DGS model.

Sable subclasses ``BaseBeastConfig`` but replaces the ``optimizer`` and ``data`` sections and
does not use ``BaseTrainingConfig``: its yaml files (``configs/sable/*.yaml``) have no ``data``
section and use a step-based training loop (``max_fwdbwd_passes``, ``val_every``, ...) instead of
epochs. Only the fields needed to dispatch and train are typed; every other key passes through
untouched (``extra='allow'``), so ``model_dump()`` returns the yaml content plus defaults.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from beast.config import BaseBeastConfig, DataConfig


class SableModelConfig(BaseModel):
    """Model section for Sable; architecture keys pass through as extras."""

    model_config = ConfigDict(extra='allow')

    model_class: Literal['sable']
    seed: int = 0


class SableTrainingConfig(BaseModel):
    """Training section for Sable; loop and dataset keys pass through as extras."""

    model_config = ConfigDict(extra='allow')

    dataset_name: str
    dataset_path: str | None = None
    batch_size_per_gpu: int = 1
    num_workers: int = 8


class SableOptimizerConfig(BaseModel):
    """Optimizer section for Sable; schedule keys pass through as extras."""

    model_config = ConfigDict(extra='allow')

    lr: float
    type: str = 'AdamW'


class SableBeastConfig(BaseBeastConfig):
    """Complete top-level config for Sable training and inference runs."""

    model_config = ConfigDict(extra='allow')

    model: SableModelConfig
    training: SableTrainingConfig
    optimizer: SableOptimizerConfig  # type: ignore[assignment]
    # sable configs do not use the shared data section
    data: DataConfig | None = None  # type: ignore[assignment]
