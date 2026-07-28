"""Aggregate video-driven temporal metrics from SABLE inference output.

This script loads rendered frames and their corresponding target images from
NPZ files produced by run_video_temporal_eval.py, assembles them into
contiguous temporal sequences sorted by source_frame_index, computes
render-level temporal consistency metrics, and outputs a summary table.

Usage:
    python aggregate_video_temporal.py \
        --input-dir /path/to/outputs/loss_weighting/cell_default/temporal_eval \
        --output-dir /path/to/outputs/loss_weighting/cell_default/temporal_eval

The input directory should contain:
    video_frames_raw.npz   - Raw frame data with (pair_idx, source_frame_index, renders, targets)
    video_frames_seq.npz   - Optional pre-assembled sequences (if already processed)

Outputs:
    sable_video_temporal.npz      - Full metrics array
    sable_video_temporal_table.md - Markdown summary table
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add beast to path for imports
BEAST_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BEAST_ROOT / 'beast'))

from beast.sable_encoding_decoding.render.metrics import (
    collect_temporal_metrics_block,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing video_frames_raw.npz from run_video_temporal_eval.py',
    )
    p.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory to write sable_video_temporal.npz and sable_video_temporal_table.md',
    )
    p.add_argument(
        '--frame-gap-threshold',
        type=int,
        default=50,
        help=(
            'Maximum frame index gap to consider frames as "contiguous". '
            'Frames with gap <= this threshold are grouped into the same temporal sequence. '
            'Default: 50'
        ),
    )
    p.add_argument(
        '--min-sequence-length',
        type=int,
        default=3,
        help=(
            'Minimum number of frames in a temporal sequence to include in metrics. '
            'Sequences shorter than this are skipped. Default: 3'
        ),
    )
    p.add_argument(
        '--session-name',
        type=str,
        default='4b00df29-3769-43be-bb40-128b1cba6d35',
        help='Session name for display in the output table. Default: 4b00df29-3769-43be-bb40-128b1cba6d35',
    )
    p.add_argument(
        '--fake-source-npz',
        type=Path,
        default=None,
        help=(
            'Optional path to a dummy .npz file used only to satisfy the source_npz argument '
            'of collect_temporal_metrics_block. The function reads neural_bin_idx from this file '
            'to determine contiguity. If not provided, a synthetic .npz with sequential bin '
            'indices [0, 1, 2, ...] is created in memory.'
        ),
    )
    return p.parse_args(argv)


def load_frame_data(input_dir: Path) -> dict[str, np.ndarray]:
    """Load raw frame data from the input NPZ file.

    Args:
        input_dir: Directory containing video_frames_raw.npz

    Returns:
        Dictionary with arrays: pair_idx, source_frame_index, renders, targets
        - pair_idx: [N] int array
        - source_frame_index: [N] int array
        - renders: [N, V, C, H, W] float32 tensor
        - targets: [N, V, C, H, W] float32 tensor
    """
    raw_npz = input_dir / 'video_frames_raw.npz'
    if not raw_npz.exists():
        raise FileNotFoundError(
            f'video_frames_raw.npz not found in {input_dir}. '
            'Run run_video_temporal_eval.py first.'
        )

    data = dict(np.load(raw_npz, allow_pickle=True))
    print(f'Loaded {raw_npz}')
    print(f'  pair_idx shape: {data["pair_idx"].shape}')
    print(f'  source_frame_index shape: {data["source_frame_index"].shape}')
    print(f'  renders shape: {data["renders"].shape}')
    print(f'  targets shape: {data["targets"].shape}')

    return data


def assemble_sequences(
    pair_idx: np.ndarray,
    source_frame_index: np.ndarray,
    renders: np.ndarray,
    targets: np.ndarray,
    frame_gap_threshold: int = 50,
    min_sequence_length: int = 3,
) -> list[dict]:
    """Assemble frames into contiguous temporal sequences.

    Frames are sorted by source_frame_index, then grouped into sequences
    where consecutive frames have index gaps <= frame_gap_threshold.

    Args:
        pair_idx: [N] array of pair indices
        source_frame_index: [N] array of source frame indices
        renders: [N, V, C, H, W] array of rendered images
        targets: [N, V, C, H, W] array of target images
        frame_gap_threshold: Max gap to consider frames as contiguous
        min_sequence_length: Minimum frames per sequence

    Returns:
        List of sequence dicts, each containing:
        - pair_idx: [T] array
        - source_frame_index: [T] array
        - renders: [T, V, C, H, W] array
        - targets: [T, V, C, H, W] array
    """
    # Sort by source_frame_index
    sort_order = np.argsort(source_frame_index)
    sorted_pair_idx = pair_idx[sort_order]
    sorted_source_idx = source_frame_index[sort_order]
    sorted_renders = renders[sort_order]
    sorted_targets = targets[sort_order]

    # Find gaps between consecutive frames
    frame_diffs = np.diff(sorted_source_idx)

    # Split at gaps > frame_gap_threshold
    sequence_starts = [0]
    for i, diff in enumerate(frame_diffs):
        if diff > frame_gap_threshold:
            sequence_starts.append(i + 1)
    sequence_starts.append(len(sorted_source_idx))

    sequences = []
    for start, end in zip(sequence_starts[:-1], sequence_starts[1:]):
        seq_len = end - start
        if seq_len < min_sequence_length:
            continue
        sequences.append({
            'pair_idx': sorted_pair_idx[start:end],
            'source_frame_index': sorted_source_idx[start:end],
            'renders': sorted_renders[start:end],
            'targets': sorted_targets[start:end],
        })

    print(f'Assembled {len(sequences)} sequences from {len(pair_idx)} frames')
    if sequences:
        seq_lengths = [len(s['source_frame_index']) for s in sequences]
        print(f'  Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, '
              f'mean={np.mean(seq_lengths):.1f}')
        print(f'  Total frames in sequences: {sum(seq_lengths)}')

    return sequences


def compute_sequence_metrics(
    sequence: dict,
    fake_source_path: Path | None,
) -> dict[str, np.ndarray]:
    """Compute temporal metrics for one sequence.

    Args:
        sequence: Dict with renders [T, V, C, H, W] and targets [T, V, C, H, W]
        fake_source_path: Optional path to dummy NPZ for neural_bin_idx

    Returns:
        Dict of temporal metrics: temporal_delta_l1, pred_motion_energy,
        target_motion_energy, motion_energy_ratio, motion_energy_corr
    """
    renders = torch.from_numpy(sequence['renders'])  # [T, V, C, H, W]
    targets = torch.from_numpy(sequence['targets'])  # [T, V, C, H, W]

    t_bins = renders.shape[0]
    views = renders.shape[1]
    k_trials = 1  # Each sequence is treated as one trial

    # Flatten to [K*T, V, C, H, W] = [T, V, C, H, W]
    renders_flat = renders.reshape(k_trials * t_bins, views, *renders.shape[2:])
    targets_flat = targets.reshape(k_trials * t_bins, views, *targets.shape[2:])

    # Create fake source NPZ with sequential bin indices if not provided
    if fake_source_path is None:
        fake_npz_path = Path('/tmp/temporal_metrics_fake_source.npz')
        np.savez_compressed(
            fake_npz_path,
            neural_bin_idx=np.arange(t_bins, dtype=np.int64).reshape(1, t_bins),
            neural_trial_idx=np.array([0], dtype=np.int64),
            trial_split=np.array(['val']),
        )
        fake_source_path = fake_npz_path

    metrics = collect_temporal_metrics_block(
        renders_flat,
        targets_flat,
        fake_source_path,
        k_trials=k_trials,
        t_bins=t_bins,
    )

    return metrics


def aggregate_metrics(all_metrics: list[dict]) -> dict[str, float]:
    """Aggregate per-sequence metrics into summary statistics.

    Args:
        all_metrics: List of metric dicts from compute_sequence_metrics

    Returns:
        Dict of summary statistics (median across all sequences/views)
    """
    keys = [
        'temporal_delta_l1',
        'pred_motion_energy',
        'target_motion_energy',
        'motion_energy_ratio',
        'motion_energy_corr',
    ]

    summary = {}
    for key in keys:
        values = []
        for m in all_metrics:
            arr = m[key]
            # Skip NaN values and flatten
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                values.extend(valid.flatten())

        if values:
            values = np.array(values, dtype=np.float32)
            summary[f'{key}_median'] = float(np.nanmedian(values))
            summary[f'{key}_mean'] = float(np.nanmean(values))
            summary[f'{key}_std'] = float(np.nanstd(values))
            summary[f'{key}_n_valid'] = len(values)
        else:
            summary[f'{key}_median'] = float('nan')
            summary[f'{key}_mean'] = float('nan')
            summary[f'{key}_std'] = float('nan')
            summary[f'{key}_n_valid'] = 0

    return summary


def format_table(
    summary: dict[str, float],
    session_name: str,
) -> str:
    """Format metrics summary as a markdown table.

    Args:
        summary: Dict of metric statistics
        session_name: Session name for display

    Returns:
        Markdown-formatted table string
    """
    table = f"""| Metric | Median | Mean ± Std | N valid |
| --- | --- | --- | --- |
| temporal-difference L1 | {summary['temporal_delta_l1_median']:.4f} | {summary['temporal_delta_l1_mean']:.4f} ± {summary['temporal_delta_l1_std']:.4f} | {int(summary['temporal_delta_l1_n_valid'])} |
| motion-energy ratio | {summary['motion_energy_ratio_median']:.4f} | {summary['motion_energy_ratio_mean']:.4f} ± {summary['motion_energy_ratio_std']:.4f} | {int(summary['motion_energy_ratio_n_valid'])} |
| motion-energy correlation | {summary['motion_energy_corr_median']:.4f} | {summary['motion_energy_corr_mean']:.4f} ± {summary['motion_energy_corr_std']:.4f} | {int(summary['motion_energy_corr_n_valid'])} |
| pred motion energy | {summary['pred_motion_energy_median']:.4f} | {summary['pred_motion_energy_mean']:.4f} ± {summary['pred_motion_energy_std']:.4f} | {int(summary['pred_motion_energy_n_valid'])} |
| target motion energy | {summary['target_motion_energy_median']:.4f} | {summary['target_motion_energy_mean']:.4f} ± {summary['target_motion_energy_std']:.4f} | {int(summary['target_motion_energy_n_valid'])} |
"""
    return table


def main(argv: list[str] | None = None) -> None:
    """Run the aggregation pipeline."""
    args = parse_args(argv)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw frame data
    print('Loading frame data...')
    frame_data = load_frame_data(input_dir)

    # Assemble into temporal sequences
    print(f'Assembling sequences (gap threshold: {args.frame_gap_threshold}, min length: {args.min_sequence_length})...')
    sequences = assemble_sequences(
        frame_data['pair_idx'],
        frame_data['source_frame_index'],
        frame_data['renders'],
        frame_data['targets'],
        frame_gap_threshold=args.frame_gap_threshold,
        min_sequence_length=args.min_sequence_length,
    )

    if not sequences:
        print('No valid sequences found. Exiting.')
        sys.exit(1)

    # Compute metrics for each sequence
    print('Computing temporal metrics for each sequence...')
    all_metrics = []
    for i, seq in enumerate(sequences):
        metrics = compute_sequence_metrics(seq, args.fake_source_npz)
        all_metrics.append(metrics)
        if (i + 1) % 10 == 0:
            print(f'  Processed {i + 1}/{len(sequences)} sequences')

    print(f'Computed metrics for {len(sequences)} sequences')

    # Aggregate across all sequences
    print('Aggregating metrics...')
    summary = aggregate_metrics(all_metrics)

    # Print summary
    print('\n=== Temporal Metrics Summary ===')
    print(f'temporal-difference L1:  median={summary["temporal_delta_l1_median"]:.4f}, '
          f'mean={summary["temporal_delta_l1_mean"]:.4f} ± {summary["temporal_delta_l1_std"]:.4f}')
    print(f'motion-energy ratio:     median={summary["motion_energy_ratio_median"]:.4f}, '
          f'mean={summary["motion_energy_ratio_mean"]:.4f} ± {summary["motion_energy_ratio_std"]:.4f}')
    print(f'motion-energy correlation: median={summary["motion_energy_corr_median"]:.4f}, '
          f'mean={summary["motion_energy_corr_mean"]:.4f} ± {summary["motion_energy_corr_std"]:.4f}')

    # Save summary NPZ
    output_npz = output_dir / 'sable_video_temporal.npz'
    np.savez_compressed(output_npz, **summary)
    print(f'\nSaved metrics to {output_npz}')

    # Save markdown table
    table = format_table(summary, args.session_name)
    table_path = output_dir / 'sable_video_temporal_table.md'

    header = f"""# SABLE Video-Driven Temporal Consistency Metrics

**Session:** {args.session_name}
**Source:** IBL val split, video-driven reconstruction
**Frame gap threshold:** {args.frame_gap_threshold}
**Min sequence length:** {args.min_sequence_length}
**Num sequences:** {len(sequences)}

{table}
"""
    with open(table_path, 'w') as f:
        f.write(header)
    print(f'Saved table to {table_path}')


if __name__ == '__main__':
    main()
