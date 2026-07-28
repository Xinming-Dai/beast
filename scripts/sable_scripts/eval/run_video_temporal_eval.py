"""Run SABLE video-driven inference for temporal consistency evaluation.

This script iterates through the validation split of the IBLTwoViewDataset,
runs SABLE's forward pass for each pair, and saves the rendered frames
along with their metadata (pair_idx, source_frame_index) to an NPZ file
for later temporal metric computation.

Usage:
    # Full run with GPU:
    python run_video_temporal_eval.py \
        --model-dir /path/to/outputs/loss_weighting/cell_default \
        --dataset-path /localssd/jinqihang/datasets/beast3d-data/sable_ibl_4b00 \
        --vda-cache-root /localssd/jinqihang/datasets/beast3d-data/sable_ibl_4b00/vda_cache \
        --correspondence-cache-root /localssd/jinqihang/datasets/beast3d-data/sable_ibl_4b00/litpose_correspondences/processed_correspondences \
        --output-dir /path/to/outputs/loss_weighting/cell_default/temporal_eval \
        --device cuda:0

    # Resume interrupted run:
    python run_video_temporal_eval.py ... --resume
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add beast to path for imports
BEAST_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BEAST_ROOT / 'beast'))

from beast.data.sable_dataset import (
    IBLTwoViewDataset,
    collate_with_correspondence_padding,
)

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Directory containing config.yaml and checkpoint (e.g. outputs/loss_weighting/cell_default)',
    )
    p.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to IBL dataset root (e.g. /localssd/jinqihang/datasets/beast3d-data/sable_ibl_4b00)',
    )
    p.add_argument(
        '--vda-cache-root',
        type=str,
        required=True,
        help='Path to VDA cache root (e.g. /localssd/jinqihang/datasets/beast3d-data/sable_ibl_4b00/vda_cache)',
    )
    p.add_argument(
        '--correspondence-cache-root',
        type=str,
        default=None,
        help='Path to correspondence cache root (optional)',
    )
    p.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save video_frames_raw.npz',
    )
    p.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Device to use for inference (default: cuda:0 if available, else cpu)',
    )
    p.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for inference (default: 1)',
    )
    p.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of dataloader workers (default: 4)',
    )
    p.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing output if present',
    )
    p.add_argument(
        '--val-split-ratio',
        type=float,
        default=0.1,
        help='Validation split ratio used during training (default: 0.1)',
    )
    return p.parse_args(argv)


def load_model_from_checkpoint(
    model_dir: str | Path,
    checkpoint_name: str | None = None,
) -> tuple:  # type hint intentionally generic to avoid import issues
    """Load SABLE model from a checkpoint directory.

    Args:
        model_dir: Directory containing config.yaml and checkpoint
        checkpoint_name: Specific checkpoint file name (e.g. 'step=10000.ckpt').
                       If None, looks for *best.ckpt, then *step*.ckpt

    Returns:
        Tuple of (Model wrapper, config dict)
    """
    import lightning.pytorch as pl
    import yaml
    from beast.models.sable import Sable

    model_dir = Path(model_dir)

    # Load config
    config_path = model_dir / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f'config.yaml not found in {model_dir}')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Find checkpoint
    if checkpoint_name:
        checkpoint_path = model_dir / checkpoint_name
        if not checkpoint_path.exists():
            # Try subdirectories
            checkpoint_path = model_dir / 'tb_logs' / 'version_0' / 'checkpoints' / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_name}')
    else:
        # Try best.ckpt first
        best_ckpts = list(model_dir.rglob('*best.ckpt'))
        if best_ckpts:
            checkpoint_path = best_ckpts[0]
        else:
            # Fall back to step*.ckpt
            step_ckpts = sorted(model_dir.rglob('*step*.ckpt'))
            if not step_ckpts:
                raise FileNotFoundError(
                    f'No checkpoint found in {model_dir} (looked for *best.ckpt and *step*.ckpt)'
                )
            checkpoint_path = step_ckpts[0]

    # Initialize model
    model = Sable(config)

    # Load checkpoint
    _logger.info(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)

    _logger.info(f'Loaded model weights from {checkpoint_path}')

    # Create a minimal wrapper class
    class LoadedModel:
        def __init__(self, model, config, model_dir):
            self.model = model
            self.config = config
            self.model_dir = Path(model_dir)

    return LoadedModel(model, config, model_dir), config


def build_config(args: argparse.Namespace) -> dict:
    """Build the config dict from the model-dir config.yaml with overrides.

    Args:
        args: Parsed CLI arguments

    Returns:
        Config dict for IBLTwoViewDataset and model
    """
    import yaml

    model_dir = Path(args.model_dir)
    config_path = model_dir / 'config.yaml'

    if not config_path.exists():
        raise FileNotFoundError(f'config.yaml not found in {model_dir}')

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override paths
    config['training']['dataset_path'] = args.dataset_path
    config['model']['vda']['cache_root'] = args.vda_cache_root
    config['training']['val_split_ratio'] = args.val_split_ratio

    if args.correspondence_cache_root:
        config['model']['merge_pcd']['correspondence_cache_root'] = args.correspondence_cache_root

    return config


def run_inference(
    model: torch.nn.Module,
    dataset: IBLTwoViewDataset,
    device: torch.device,
    batch_size: int = 1,
    resume: bool = False,
    output_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Run SABLE inference on the validation split and collect rendered frames.

    Args:
        model: SABLE model
        dataset: IBLTwoViewDataset with val split
        device: torch device
        batch_size: inference batch size
        resume: Whether to resume from existing output
        output_path: Path to save raw frame NPZ

    Returns:
        Dict with arrays: pair_idx, source_frame_index, renders, targets
    """
    model.eval()

    # Data structures to collect all frames
    all_pair_idx: list[int] = []
    all_source_frame_idx: list[int] = []
    all_renders: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    # Try to resume from existing output
    existing_data = None
    if resume and output_path and output_path.exists():
        try:
            existing_data = dict(np.load(output_path, allow_pickle=True))
            all_pair_idx = list(existing_data['pair_idx'])
            all_source_frame_idx = list(existing_data['source_frame_index'])
            all_renders = [existing_data['renders'][i] for i in range(len(existing_data['renders']))]
            all_targets = [existing_data['targets'][i] for i in range(len(existing_data['targets']))]
            print(f'Resuming from existing output: {len(all_pair_idx)} frames already collected')
        except Exception as e:
            print(f'Failed to load existing output: {e}, starting fresh')
            existing_data = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for safety with this inference mode
        collate_fn=collate_with_correspondence_padding,
        drop_last=False,
    )

    n_batches = len(dataloader)
    print(f'Running inference on {len(dataset)} val samples in {n_batches} batches')

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            # Get model outputs
            out = model.get_model_outputs(batch)

            # Get renders and targets
            renders = out['render']  # [B, V, 3, H, W]
            targets = batch['image']  # [B, V, 3, H, W]

            # Get metadata from records
            # Note: We use the batch indices to access records
            for i in range(renders.shape[0]):
                dataset_idx = batch_idx * batch_size + i
                if dataset_idx >= len(dataset):
                    break

                # Skip if already processed (for resume)
                if resume and existing_data is not None:
                    if dataset_idx < len(existing_data['pair_idx']):
                        continue

                rec = dataset._records[dataset_idx]

                # Clamp renders to [0, 1]
                render_np = renders[i].clamp(0.0, 1.0).cpu().numpy()
                target_np = targets[i].cpu().numpy()

                # Use left_source_frame_index for temporal ordering
                source_frame_idx = rec.left_source_frame_index

                all_pair_idx.append(rec.pair_idx)
                all_source_frame_idx.append(source_frame_idx)
                all_renders.append(render_np)
                all_targets.append(target_np)

            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                print(f'  Batch {batch_idx + 1}/{n_batches} ({len(all_pair_idx)} frames collected)')

    # Convert to numpy arrays
    result = {
        'pair_idx': np.array(all_pair_idx, dtype=np.int64),
        'source_frame_index': np.array(all_source_frame_idx, dtype=np.int64),
        'renders': np.stack(all_renders, axis=0).astype(np.float32),
        'targets': np.stack(all_targets, axis=0).astype(np.float32),
    }

    return result


def main(argv: list[str] | None = None) -> None:
    """Run the temporal evaluation inference."""
    args = parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'video_frames_raw.npz'

    # Check if already complete
    if args.resume and output_path.exists():
        existing = dict(np.load(output_path, allow_pickle=True))
        print(f'Found existing output with {len(existing["pair_idx"])} frames')
        print('If you want to re-run, delete the existing file or omit --resume')

    # Build config
    print('Building configuration...')
    config = build_config(args)

    # Load model
    print(f'Loading model from {args.model_dir}...')
    wrapped, config = load_model_from_checkpoint(args.model_dir)
    config['inference'] = True
    config['evaluation'] = False

    device = torch.device(args.device)
    model = wrapped.model.to(device)
    model.eval()
    print(f'Model loaded on {device}')

    # Create dataset
    print(f'Creating IBLTwoViewDataset (val split) from {args.dataset_path}...')
    dataset = IBLTwoViewDataset(config, include_splits=['val'])
    print(f'Dataset: {len(dataset)} validation samples')

    # Show sample metadata
    if len(dataset) > 0:
        rec = dataset._records[0]
        print(f'  First record: pair_idx={rec.pair_idx}, '
              f'left_source_frame_index={rec.left_source_frame_index}, '
              f'session_id={rec.session_id}')

    # Run inference
    print('Starting inference...')
    result = run_inference(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        resume=args.resume,
        output_path=output_path,
    )

    # Save results
    print(f'Saving {len(result["pair_idx"])} frames to {output_path}')
    np.savez_compressed(output_path, **result)

    print('\nDone! Run aggregate_video_temporal.py next to compute temporal metrics.')


if __name__ == '__main__':
    main()
