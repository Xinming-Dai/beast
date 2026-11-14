"""Score every frame of a Cheese3D split on an anchored novel view.

``cheese3d_nvs`` evaluates one frame per session. This module runs the same anchored novel-view
synthesis (the model's predicted poses, with the anchor camera placed from the GT rig when the
model has no pose for it) on every frame of a split, skipping frames that are also in the
train split, and keeps the whole-image PSNR/SSIM of the anchor camera only. Results use the
``psnr_ssim_metrics.npz`` layout of the neural-decode evaluation
(``save_psnr_ssim_metrics_npz``) with ``K`` = frames, ``T`` = 1 and one view.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from beast.logging import log_step
from beast.sable_encoding_decoding.render.cheese3d_nvs import (
    _BG_COLORS,
    _prepare_nvs,
    _run_nvs_session,
    held_out_variants,
)
from beast.sable_encoding_decoding.render.metrics import save_psnr_ssim_metrics_npz

_EVAL_CAMERAS = ('TL', 'TR', 'TC')
_SUMMARY_NAME = 'nvs_eval_summary.json'
_LOG_EVERY = 50


def score_session_frames(
    model,
    config: dict,
    dataset,
    model_dir: Path,
    session: str,
    cameras: list[str],
    device: str,
    score_camera: str,
    align_cameras: list[str] | None = None,
    amp: str = 'none',
    context_batch: int = 1,
    intrinsics: str = 'gt',
    anchor_source: str = 'auto',
    mask_held_out: bool = False,
    variant: str | None = None,
    hide_held_out_from_encoders: bool = False,
    bg_color: str = 'black',
    exclude_frames: set[int] | frozenset[int] = frozenset(),
) -> dict[str, np.ndarray | list[str]]:
    """Run anchored NVS on every frame of ``session`` and collect ``score_camera``'s metrics.

    Each source frame is scored once, even if a repeated session in the training config put it
    in the dataset twice, and frames in ``exclude_frames`` are skipped. A frame whose NVS raises
    ``ValueError`` (e.g. too few cameras to align) is logged and scored NaN so the frame count
    stays complete.

    Args:
        model: trained Sable model, already on ``device`` in eval mode.
        config: inference config from ``_prepare_nvs``.
        dataset: ``Cheese3DDataset`` for the split being scored.
        model_dir: model directory, recorded in the per-frame metrics.
        session: session directory name.
        cameras: GT cameras to render, see ``_run_nvs_session``.
        device: torch device string.
        score_camera: camera whose anchored render is scored; also the anchor camera.
        align_cameras: cameras for the pred-to-GT similarity fit; ``None`` uses the model's.
        amp: ``'none'`` or ``'bf16'``.
        context_batch: records collated per forward pass.
        intrinsics: intrinsics of the global-fit renders (not scored here).
        anchor_source: ``'auto'``, ``'predicted'`` or ``'derived'``.
        mask_held_out: strict holdout of the held-out view, see ``_run_nvs_session``.
        variant: variant name passed through to ``_run_nvs_session``.
        hide_held_out_from_encoders: with ``mask_held_out``, also hide it from the encoders.
        bg_color: ``'white'`` or ``'black'`` render background.
        exclude_frames: source frame indices of ``session`` to skip, e.g. the frames the model
            trained on.

    Returns:
        dict with ``frame_idx`` (int64 ``[K]``), ``psnr`` and ``ssim`` (float32 ``[K]``) and
        ``source_files`` (``K`` GT image paths of ``score_camera``); empty arrays when the
        session has no frame left to score.
    """
    records, seen = [], set()
    n_session = n_excluded = 0
    for rec in dataset._records:
        if rec.session_id != session:
            continue
        n_session += 1
        frame = int(rec.left_source_frame_index)
        if frame in exclude_frames:
            n_excluded += 1
        elif frame not in seen:
            seen.add(frame)
            records.append(rec)
    log_step(
        f'cheese3d_nvs_eval: {session}: {n_session} dataset records, {n_excluded} dropped '
        f'(frame in excluded set), {n_session - n_excluded - len(records)} dropped (repeated '
        f'frame); scoring {len(records)}',
        level='info',
    )
    frame_idx = np.array([rec.left_source_frame_index for rec in records], dtype=np.int64)
    psnr = np.full(len(records), np.nan, dtype=np.float32)
    ssim = np.full(len(records), np.nan, dtype=np.float32)
    source_files = [
        str(rec.left_path.parent.parent / score_camera / rec.left_path.name) for rec in records
    ]
    for pos, frame in enumerate(frame_idx.tolist()):
        try:
            metrics = _run_nvs_session(
                model, config, dataset, model_dir, Path(), session, frame, cameras,
                align_cameras, device, amp=amp, context_batch=context_batch,
                intrinsics=intrinsics, save_glb=False, pose_mode='anchor',
                anchor_camera=score_camera, anchor_source=anchor_source,
                mask_held_out=mask_held_out, variant=variant,
                hide_held_out_from_encoders=hide_held_out_from_encoders,
                bg_color=bg_color, save_outputs=False,
            )
        except ValueError as err:
            log_step(
                f'cheese3d_nvs_eval: {session} frame {frame} scored NaN: {err}',
                level='warning',
            )
            continue
        scored = metrics['projected_by_pose_mode']['anchor'][score_camera]
        psnr[pos], ssim[pos] = scored['psnr'], scored['ssim']
        if (pos + 1) % _LOG_EVERY == 0 or pos + 1 == len(records):
            log_step(
                f'cheese3d_nvs_eval: {session} {pos + 1}/{len(records)} frames; running mean '
                f'PSNR {np.nanmean(psnr[:pos + 1]):.3f} SSIM {np.nanmean(ssim[:pos + 1]):.4f}',
                level='info',
            )
    return {'frame_idx': frame_idx, 'psnr': psnr, 'ssim': ssim, 'source_files': source_files}


def train_frames_by_session(config: dict) -> dict[str, set[int]]:
    """Return the source frames of every session that the model's train split contains.

    Args:
        config: inference config from ``_prepare_nvs``.

    Returns:
        ``{session: {source frame index}}`` over the train split.
    """
    from beast.inference import _build_sable_dataset

    train = _build_sable_dataset(config, include_splits=['train'])
    frames: dict[str, set[int]] = {}
    for rec in train._records:
        frames.setdefault(rec.session_id, set()).add(int(rec.left_source_frame_index))
    return frames


def save_frame_metrics(
    metrics_npz: Path,
    blocks: list[dict[str, np.ndarray | list[str]]],
    split: str,
    score_camera: str,
) -> dict[str, np.ndarray]:
    """Save per-frame metrics as the shared ``psnr_ssim_metrics.npz`` structure.

    Frames are rows of ``K`` with ``T`` = 1 and one view. ``neural_trial_idx`` carries the
    source frame index (these frames have no neural trial) and ``neural_bin_idx`` is 0;
    ``source_files`` names each frame's GT image, which identifies session and frame.

    Args:
        metrics_npz: output path.
        blocks: one ``score_session_frames`` result per session, concatenated along ``K``.
        split: split label stored per row.
        score_camera: the single view's name.

    Returns:
        the arrays written, see ``save_psnr_ssim_metrics_npz``.
    """
    return save_psnr_ssim_metrics_npz(
        metrics_npz,
        psnr_blocks=[block['psnr'][:, None, None] for block in blocks],
        ssim_blocks=[block['ssim'][:, None, None] for block in blocks],
        neural_trial_blocks=[block['frame_idx'] for block in blocks],
        neural_bin_blocks=[
            np.zeros((len(block['frame_idx']), 1), dtype=np.int64) for block in blocks
        ],
        trial_split_blocks=[np.full(len(block['frame_idx']), split) for block in blocks],
        source_file_rows=[path for block in blocks for path in block['source_files']],
        view_names=(score_camera,),
    )


def _summary_row(arrays: dict[str, np.ndarray]) -> dict[str, float | int]:
    """Frame counts and mean / SE of PSNR and SSIM from the arrays of a saved metrics npz."""
    return {
        'n_frames': int(arrays['psnr'].shape[0]),
        'n_scored': int(np.sum(~np.isnan(arrays['psnr']))),
        'psnr_mean': float(arrays['average_psnr']),
        'psnr_se': float(arrays['se_psnr']),
        'ssim_mean': float(arrays['average_ssim']),
        'ssim_se': float(arrays['se_ssim']),
    }


def run_nvs_eval(
    model_dir: Path,
    dataset_path: Path,
    output_dir: Path,
    sessions: list[str] | None = None,
    split: str = 'val',
    cameras: tuple[str, ...] = _EVAL_CAMERAS,
    align_cameras: list[str] | None = None,
    amp: str = 'none',
    context_batch: int = 1,
    intrinsics: str = 'gt',
    anchor_camera: str = 'TC',
    anchor_source: str = 'auto',
    held_out_tc: str = 'masked',
    hide_held_out_from_encoders: bool = False,
    bg_color: str = 'black',
) -> dict[str, dict[str, float | int]]:
    """Score ``anchor_camera`` on every held-out ``split`` frame of each session.

    The split is the one the model trained with (see ``_prepare_nvs``). Frames that also occur
    in the train split (a repeated session in the training config duplicates its records) are
    skipped and every frame is scored once, so only frames the model never trained on count.

    For each held-out variant (one for models without a held-out view; see
    ``held_out_variants``) this writes ``<output_dir>[/<variant>]/<session>/psnr_ssim_metrics.npz``
    per session, ``<output_dir>[/<variant>]/psnr_ssim_metrics.npz`` over all sessions, and
    ``<output_dir>/nvs_eval_summary.json`` with the mean and SE of both metrics per variant.

    Args:
        model_dir: directory with ``config.yaml`` and the best checkpoint.
        dataset_path: root Cheese3D directory.
        output_dir: model-level output directory.
        sessions: session names, or ``None`` for every session in the saved training config.
        split: dataset split to score.
        cameras: GT cameras to render; must include the model's cameras and ``anchor_camera``.
        align_cameras: cameras for the pred-to-GT similarity fit; ``None`` uses the model's.
        amp: ``'none'`` or ``'bf16'``.
        context_batch: records collated per forward pass.
        intrinsics: intrinsics of the global-fit renders (not scored here).
        anchor_camera: camera that is placed from the anchor pose and scored.
        anchor_source: ``'auto'``, ``'predicted'`` or ``'derived'``.
        held_out_tc: ``'both'``, ``'visible'`` or ``'masked'``, see ``held_out_variants``.
        hide_held_out_from_encoders: for the masked variant, also hide the held-out image from
            the pose encoder and DINO.
        bg_color: ``'white'`` or ``'black'`` render background.

    Returns:
        ``{variant name or 'default': summary row}`` as written to the summary json.

    Raises:
        ValueError: if ``anchor_camera`` is not in ``cameras`` or a model camera is missing.
        RuntimeError: if no session has a frame in ``split``.
    """
    cameras = list(cameras)
    if anchor_camera not in cameras:
        raise ValueError(f'anchor camera {anchor_camera!r} must be one of {cameras}')
    model, config, dataset, sessions, device = _prepare_nvs(
        model_dir, dataset_path, sessions, cameras, split,
    )
    # the repeated session of the training config can put a val frame in the train split too
    train_frames = train_frames_by_session(config) if split != 'train' else {}

    output_dir = Path(output_dir)
    variants = held_out_variants(config['training'].get('training_regime', ''), held_out_tc)
    summary: dict[str, dict[str, float | int]] = {}
    for variant, mask_held_out in variants:
        variant_dir = output_dir if variant is None else output_dir / variant
        blocks = []
        for session in sessions:
            block = score_session_frames(
                model, config, dataset, model_dir, session, cameras, device, anchor_camera,
                align_cameras=align_cameras, amp=amp, context_batch=context_batch,
                intrinsics=intrinsics, anchor_source=anchor_source,
                mask_held_out=mask_held_out, variant=variant,
                hide_held_out_from_encoders=hide_held_out_from_encoders, bg_color=bg_color,
                exclude_frames=train_frames.get(session, frozenset()),
            )
            if len(block['frame_idx']) == 0:
                log_step(f'cheese3d_nvs_eval: no {split} frames for {session}', level='warning')
                continue
            save_frame_metrics(
                variant_dir / session / 'psnr_ssim_metrics.npz', [block], split, anchor_camera,
            )
            blocks.append(block)
        if not blocks:
            raise RuntimeError(f'no {split} frames in any of {sessions}')

        arrays = save_frame_metrics(
            variant_dir / 'psnr_ssim_metrics.npz', blocks, split, anchor_camera,
        )
        row = _summary_row(arrays)
        summary[variant or 'default'] = row
        log_step(
            f'cheese3d_nvs_eval [{variant or "default"}] {model_dir.name}: {row["n_frames"]} '
            f'frames, PSNR {row["psnr_mean"]:.3f} +/- {row["psnr_se"]:.3f} (SE), '
            f'SSIM {row["ssim_mean"]:.4f} +/- {row["ssim_se"]:.4f} (SE)',
            level='info',
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / _SUMMARY_NAME).write_text(
        json.dumps(
            {
                'model_dir': str(model_dir),
                'split': split,
                'camera': anchor_camera,
                'bg_color': bg_color,
                'variants': summary,
            },
            indent=2,
        ) + '\n',
    )
    return summary


def summarize_metrics(output_dirs: list[Path]) -> str:
    """Format the mean and SE of PSNR / SSIM saved by ``run_nvs_eval`` as one table.

    Args:
        output_dirs: model-level output directories, each holding ``nvs_eval_summary.json``.

    Returns:
        a text table with one row per model and variant.

    Raises:
        FileNotFoundError: if a directory has no summary json.
    """
    lines = [
        f'{"model / variant":<52}{"frames":>8}{"PSNR mean":>11}{"PSNR SE":>9}'
        f'{"SSIM mean":>11}{"SSIM SE":>9}',
    ]
    for output_dir in output_dirs:
        path = Path(output_dir) / _SUMMARY_NAME
        if not path.is_file():
            raise FileNotFoundError(f'no summary at {path}; did the eval finish?')
        info = json.loads(path.read_text())
        model_name = Path(info['model_dir']).name
        for variant, row in info['variants'].items():
            label = model_name if variant == 'default' else f'{model_name} / {variant}'
            lines.append(
                f'{label:<52}{row["n_frames"]:>8}{row["psnr_mean"]:>11.3f}{row["psnr_se"]:>9.3f}'
                f'{row["ssim_mean"]:>11.4f}{row["ssim_se"]:>9.4f}',
            )
    return '\n'.join(lines)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='model dir with config.yaml')
    parser.add_argument('--input', type=Path, help='root Cheese3D directory')
    parser.add_argument('--output', type=Path, help='model-level output directory')
    parser.add_argument(
        '--summarize',
        nargs='+',
        type=Path,
        default=None,
        help='print the mean / SE table for these finished output directories and exit',
    )
    parser.add_argument('--sessions', nargs='+', default=None)
    parser.add_argument('--split', default='val')
    parser.add_argument('--cameras', nargs='+', default=list(_EVAL_CAMERAS))
    parser.add_argument('--align-cameras', nargs='+', default=None)
    parser.add_argument('--amp', choices=('none', 'bf16'), default='none')
    parser.add_argument('--context-batch', type=int, default=1)
    parser.add_argument('--intrinsics', choices=('gt', 'pred_scaled'), default='gt')
    parser.add_argument('--anchor-camera', default='TC', help='camera that is placed and scored')
    parser.add_argument(
        '--anchor-source',
        choices=('auto', 'predicted', 'derived'),
        default='auto',
    )
    parser.add_argument('--held-out-tc', choices=('both', 'visible', 'masked'), default='masked')
    parser.add_argument('--hide-held-out-from-encoders', action='store_true')
    parser.add_argument('--bg-color', choices=tuple(_BG_COLORS), default='black')
    args = parser.parse_args()

    if args.summarize:
        print(summarize_metrics(args.summarize))
        return
    if args.model is None or args.input is None or args.output is None:
        parser.error('--model, --input and --output are required unless --summarize is given')
    run_nvs_eval(
        model_dir=args.model,
        dataset_path=args.input,
        output_dir=args.output,
        sessions=args.sessions,
        split=args.split,
        cameras=tuple(args.cameras),
        align_cameras=args.align_cameras,
        amp=args.amp,
        context_batch=args.context_batch,
        intrinsics=args.intrinsics,
        anchor_camera=args.anchor_camera,
        anchor_source=args.anchor_source,
        held_out_tc=args.held_out_tc,
        hide_held_out_from_encoders=args.hide_held_out_from_encoders,
        bg_color=args.bg_color,
    )


if __name__ == '__main__':
    main()
