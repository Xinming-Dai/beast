"""Pydantic config models for the SABLE IBL stereo extraction pipeline."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from beast.preprocess.config_3d import DownsampleConfig, FrameConfig, TrimConfig


class SABLEFrameConfig(FrameConfig):
    """Frame extraction settings for SABLE, extending FrameConfig with split settings."""

    model_config = ConfigDict(extra='forbid')

    split_names: list[str] = ['train', 'val', 'test']
    split_ratios: list[float] = [0.7, 0.1, 0.2]


class VDAConfig(BaseModel):
    """VDA depth precomputation settings.

    Checkpoint resolution order:
    1. If checkpoint_path is set and file exists, use it
    2. Otherwise, look in third_party/VDA/checkpoints/video_depth_anything_{encoder}.pth
    3. If neither exists and vda.enabled is true, an error will be raised during processing
    """

    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    encoder: str = 'vitb'
    checkpoint_path: str | None = None
    device: str = 'auto'
    input_size: int = 518
    fp32: bool = False


class VideoNamingConfig(BaseModel):
    """Video file naming conventions for IBL-2view layout."""

    model_config = ConfigDict(extra='forbid')

    camera_video_subdir: str = '{cam}Camera.video'
    video_filename: str = '_iblrig_{cam}Camera.downsampled.{session_id}.mp4'
    timestamp_filename: str = '_ibl_{cam}Camera.times.{session_id}.npy'
    timestamp_subdir: str = 'timestamps'


class LitPoseConfig(BaseModel):
    """LitPose CSV-to-npy conversion settings."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    video_preds_dir: str | None = None
    keypoints: list[str] = ['pawL', 'pawR', 'nose']
    min_likelihood: float = 0.0
    keypoint_shifts: dict[str, dict[str, list[float]]] = {}


class SABLEConfig(BaseModel):
    """Top-level config for the SABLE IBL stereo extraction pipeline."""

    model_config = ConfigDict(extra='forbid')

    name: str = ''
    input_dir: str = ''
    output_dir: str = ''
    anchor_view: str = 'left'
    cameras: list[str] = ['left', 'right']
    max_workers: int = 4
    author: str = 'anonymous'
    seed: int = 42
    sessionids: list[str] | None = None

    frame: SABLEFrameConfig = SABLEFrameConfig()
    trim: TrimConfig = TrimConfig()
    downsample: DownsampleConfig = DownsampleConfig()
    vda: VDAConfig = VDAConfig()
    video_naming: VideoNamingConfig = VideoNamingConfig()
    litpose: LitPoseConfig = LitPoseConfig()


def load_sable_config(path: str | Path) -> SABLEConfig:
    """Load a SABLEConfig from a YAML file.

    Args:
        path: path to the YAML config file.

    Returns:
        validated SABLEConfig object.

    Raises:
        FileNotFoundError: if the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'config file not found: {path}')
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return SABLEConfig.model_validate(raw or {})


def validate_sable_config(cfg: SABLEConfig) -> None:
    """Check filesystem preconditions for a SABLEConfig.

    Args:
        cfg: config to validate.

    Raises:
        ValueError: if required fields are empty or required paths do not exist.
    """
    for field in ('name', 'input_dir', 'output_dir'):
        if not getattr(cfg, field):
            raise ValueError(f'config must specify {field}')

    if cfg.anchor_view not in cfg.cameras:
        raise ValueError(
            f'anchor_view "{cfg.anchor_view}" must be one of cameras {cfg.cameras}'
        )

    input_path = Path(cfg.input_dir)
    if not input_path.is_dir():
        raise ValueError(f'input_dir not found: {input_path}')

    if cfg.litpose.enabled and not cfg.litpose.video_preds_dir:
        raise ValueError(
            'litpose.video_preds_dir must be set when litpose.enabled is true'
        )
