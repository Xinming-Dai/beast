"""Extract temporal metrics from existing rendered visuals and ground truth images.

This script uses the existing rendered images from the training visuals directory
along with ground truth images from the dataset to compute temporal consistency metrics.
This approach bypasses the need for gsplat rendering by using pre-rendered frames.

Usage:
    python extract_temporal_from_visuals.py \
        --visuals-dir /path/to/outputs/loss_weighting/cell_default/visuals \
        --dataset-path /localssd/jinqihang/datasets/beast3d-data/sable_ibl_4b00 \
        --vda-cache-root /localssd/jinqihang/datasets/beast3d-data/sable_ibl_4b00/vda_cache \
        --output-dir /path/to/outputs/loss_weighting/cell_default/temporal_eval
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add beast to path for imports
BEAST_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BEAST_ROOT / 'beast'))

from beast.data.sable_dataset import IBLTwoViewDataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument(
        '--visuals-dir',
        type=str,
        required=True,
        help='Directory containing rendered PNG visuals (e.g., step_*.png)',
    )
    p.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to IBL dataset root',
    )
    p.add_argument(
        '--vda-cache-root',
        type=str,
        required=True,
        help='Path to VDA cache root',
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
        '--val-split-ratio',
        type=float,
        default=0.1,
        help='Validation split ratio used during training (default: 0.1)',
    )
    p.add_argument(
        '--checkpoint-step',
        type=int,
        default=10000,
        help='Checkpoint step number to filter visuals (default: 10000)',
    )
    return p.parse_args(argv)


def parse_visual_filename(filename: str) -> dict | None:
    """Parse a visual filename like step_000000_session_pair_000001_sample00.png.

    Args:
        filename: PNG filename to parse

    Returns:
        Dict with step, session_id, pair_idx, sample_idx, or None if not parseable
    """
    pattern = r'step_(\d+)_([0-9a-f-]+)_pair_(\d+)_sample(\d+)\.png'
    match = re.match(pattern, filename)
    if match:
        return {
            'step': int(match.group(1)),
            'session_id': match.group(2),
            'pair_idx': int(match.group(3)),
            'sample_idx': int(match.group(4)),
        }
    return None


def load_image_as_tensor(path: Path, size: int = 320) -> np.ndarray:
    """Load an image and convert to float32 numpy array in [0, 1].

    Args:
        path: Path to image file
        size: Target size (assumes square images)

    Returns:
        numpy array of shape [3, size, size] in [0, 1]
    """
    with Image.open(path) as img:
        img = img.convert('RGB')
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))  # [H, W, C] -> [C, H, W]


def build_config(args: argparse.Namespace) -> dict:
    """Build the config dict for the dataset."""
    import yaml

    config = {
        'training': {
            'dataset_path': args.dataset_path,
            'val_split_ratio': args.val_split_ratio,
        },
        'model': {
            'seed': 0,
            'vda': {'cache_root': args.vda_cache_root},
            'image_tokenizer': {'image_size': 320},
            'merge_pcd': {},
        },
    }

    if args.correspondence_cache_root:
        config['model']['merge_pcd']['correspondence_cache_root'] = args.correspondence_cache_root

    return config


def extract_frames_from_visuals(
    visuals_dir: Path,
    dataset: IBLTwoViewDataset,
    checkpoint_step: int = 10000,
) -> dict[str, np.ndarray]:
    """Extract rendered frames and corresponding ground truth from visuals.

    Args:
        visuals_dir: Directory containing rendered PNGs
        dataset: IBLTwoViewDataset for loading ground truth
        checkpoint_step: Only use visuals from this training step

    Returns:
        Dict with pair_idx, source_frame_index, renders, targets
    """
    # Find all PNG files matching the step pattern
    png_files = sorted(visuals_dir.glob('step_*.png'))

    # Group by step, only keep the checkpoint step
    step_prefix = f'step_{checkpoint_step:06d}_'
    target_files = [f for f in png_files if step_prefix in f.name]

    print(f'Found {len(target_files)} visuals at step {checkpoint_step}')

    # Build lookup from pair_idx to dataset record
    pair_to_record = {rec.pair_idx: rec for rec in dataset._records}

    all_pair_idx = []
    all_source_frame_idx = []
    all_renders = []
    all_targets = []

    for visual_path in target_files:
        parsed = parse_visual_filename(visual_path.name)
        if parsed is None:
            print(f'Could not parse: {visual_path.name}')
            continue

        pair_idx = parsed['pair_idx']

        # Find corresponding dataset record
        if pair_idx not in pair_to_record:
            print(f'Pair {pair_idx} not found in dataset')
            continue

        rec = pair_to_record[pair_idx]

        # Load rendered image
        try:
            render = load_image_as_tensor(visual_path)
        except Exception as e:
            print(f'Failed to load {visual_path}: {e}')
            continue

        # Load ground truth (left camera)
        try:
            target = load_image_as_tensor(rec.left_path)
        except Exception as e:
            print(f'Failed to load GT {rec.left_path}: {e}')
            continue

        all_pair_idx.append(pair_idx)
        all_source_frame_idx.append(rec.left_source_frame_index)
        all_renders.append(render)
        all_targets.append(target)

    if not all_pair_idx:
        raise RuntimeError('No valid frame pairs found')

    result = {
        'pair_idx': np.array(all_pair_idx, dtype=np.int64),
        'source_frame_index': np.array(all_source_frame_idx, dtype=np.int64),
        'renders': np.stack(all_renders, axis=0).astype(np.float32),
        'targets': np.stack(all_targets, axis=0).astype(np.float32),
    }

    return result


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    visuals_dir = Path(args.visuals_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'video_frames_raw.npz'

    if output_path.exists():
        print(f'Output already exists: {output_path}')
        print('Delete it or use a different output directory to recompute.')
        return

    # Build config
    print('Building configuration...')
    config = build_config(args)

    # Create dataset (val split only)
    print(f'Creating IBLTwoViewDataset (val split) from {args.dataset_path}...')
    dataset = IBLTwoViewDataset(config, include_splits=['val'])
    print(f'Dataset: {len(dataset)} validation samples')

    # Extract frames
    print(f'Extracting frames from {visuals_dir}...')
    result = extract_frames_from_visuals(
        visuals_dir=visuals_dir,
        dataset=dataset,
        checkpoint_step=args.checkpoint_step,
    )

    # Save results
    print(f'Saving {len(result["pair_idx"])} frames to {output_path}')
    np.savez_compressed(output_path, **result)

    print('\nDone! Run aggregate_video_temporal.py next to compute temporal metrics.')


if __name__ == '__main__':
    main()
