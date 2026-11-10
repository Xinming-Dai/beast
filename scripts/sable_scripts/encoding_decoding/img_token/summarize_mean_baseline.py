"""Compare mean-token-baseline PSNR/SSIM with neural-decoded PSNR/SSIM, per view and pooled.

Reads, per session, the metrics written by ``step4b_finetune_decoder.sh`` (neural-decoded tokens)
and ``step4d_eval_mean_baseline.sh`` (session mean token for every frame). Both score the same
test frames with the same finetuned decoder, so the per-frame values are paired. Prints a text
table plus LaTeX rows for the rebuttal. Standard errors follow the repo convention
(``nanstd / sqrt(n)`` over all scored images) and ignore temporal correlation between frames.
"""

import argparse
from pathlib import Path

import numpy as np

DEFAULT_EIDS = (
    '4b00df29-3769-43be-bb40-128b1cba6d35',
    '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',
    '781b35fd-e1f0-4d14-b2bb-95b7263082bb',
)
DEFAULT_ROOT = Path(
    '/work/hdd/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/ibl_pretrain_restricted_sessions',
)
METRICS = ('psnr', 'ssim')


def metrics_paths(root: Path, job_id: str, eid: str, lr: str) -> tuple[Path, Path]:
    """Return the (neural-decoded, mean-baseline) metrics npz paths for one session.

    Args:
        root: ``ibl_pretrain_restricted_sessions`` directory holding the job directories.
        job_id: training job id used for the latents and the finetuned decoder.
        eid: session id.
        lr: finetuning learning-rate tag used in the ``finetuned_decoder_<lr>`` directory name.

    Returns:
        Tuple ``(neural_npz, mean_npz)``.
    """
    finetune_dir = (
        root / job_id / 'latents' / 'img_tokens_compressed' / eid
        / 'img_tokens_compressed_estimated' / eid / f'finetuned_decoder_{lr}'
    )
    return (
        finetune_dir / 'decode_saved_latents' / 'psnr_ssim_metrics.npz',
        finetune_dir / 'decode_saved_latents_meanbaseline' / 'psnr_ssim_metrics.npz',
    )


def load_sorted_metrics(path: Path) -> dict[str, np.ndarray]:
    """Load a metrics npz and sort its trials by ``neural_trial_idx``.

    Args:
        path: path to ``psnr_ssim_metrics.npz`` (``psnr``/``ssim`` have shape ``[K, T, V]``).

    Returns:
        Dict with ``psnr``, ``ssim``, ``neural_trial_idx``, ``neural_bin_idx`` and ``view_names``.
    """
    with np.load(path, allow_pickle=True) as d:
        order = np.argsort(d['neural_trial_idx'], kind='stable')
        return {
            'psnr': d['psnr'][order],
            'ssim': d['ssim'][order],
            'neural_trial_idx': d['neural_trial_idx'][order],
            'neural_bin_idx': d['neural_bin_idx'][order],
            'view_names': d['view_names'] if 'view_names' in d.files else np.array([]),
        }


def check_paired(neural: dict[str, np.ndarray], mean: dict[str, np.ndarray]) -> None:
    """Assert the two runs scored the same trials, bins and views.

    Args:
        neural: sorted metrics of the neural-decoded run.
        mean: sorted metrics of the mean-baseline run.

    Raises:
        ValueError: if the image counts or the trial/bin indices differ.
    """
    if neural['psnr'].shape != mean['psnr'].shape:
        raise ValueError(
            f"shape mismatch (K, T, V): neural {neural['psnr'].shape} vs mean "
            f"{mean['psnr'].shape}",
        )
    if not np.array_equal(neural['neural_trial_idx'], mean['neural_trial_idx']):
        raise ValueError('neural_trial_idx differs between the neural and mean-baseline runs')
    if not np.array_equal(neural['neural_bin_idx'], mean['neural_bin_idx']):
        raise ValueError('neural_bin_idx differs between the neural and mean-baseline runs')


def mean_se(x: np.ndarray) -> tuple[float, float]:
    """Return ``(nanmean, nanstd / sqrt(n))`` of ``x`` over all finite entries."""
    n = int(np.sum(~np.isnan(x)))
    if n == 0:
        return float('nan'), float('nan')
    return float(np.nanmean(x)), float(np.nanstd(x) / np.sqrt(n))


def compare(neural: np.ndarray, mean: np.ndarray) -> dict[str, float]:
    """Compute paired statistics of the neural-decoded values against the mean baseline.

    Args:
        neural: neural-decoded values for one metric, any shape.
        mean: mean-baseline values with the same shape.

    Returns:
        Dict with ``neural``/``mean`` (mean and SE), ``delta`` (neural minus mean, with the SE of
        the paired differences) and ``frac_better`` (fraction of frames where neural > mean).
    """
    neural, mean = np.ravel(neural), np.ravel(mean)
    delta = neural - mean
    valid = ~np.isnan(delta)
    out = {}
    out['neural'], out['neural_se'] = mean_se(neural)
    out['mean'], out['mean_se'] = mean_se(mean)
    out['delta'], out['delta_se'] = mean_se(delta)
    out['frac_better'] = float(np.mean(delta[valid] > 0)) if valid.any() else float('nan')
    return out


def format_row(label: str, metric: str, stats: dict[str, float]) -> str:
    """Format one plain-text table row."""
    digits = 2 if metric == 'psnr' else 3
    return (
        f"{label:<34}{metric.upper():<5}"
        f"mean {stats['mean']:.{digits}f}±{stats['mean_se']:.{digits}f}  "
        f"neural {stats['neural']:.{digits}f}±{stats['neural_se']:.{digits}f}  "
        f"Δ {stats['delta']:+.{digits}f}±{stats['delta_se']:.{digits}f}  "
        f"frames neural>mean {100 * stats['frac_better']:.1f}%"
    )


def format_latex(label: str, stats_by_metric: dict[str, dict[str, float]]) -> str:
    """Format one LaTeX table row: mean-baseline vs neural-decoded for PSNR and SSIM."""
    cells = []
    for metric, digits in (('psnr', 2), ('ssim', 3)):
        s = stats_by_metric[metric]
        cells.append(f"{s['mean']:.{digits}f} & {s['neural']:.{digits}f}")
    return f'{label} & ' + ' & '.join(cells) + r' \\'


def main(argv: list[str] | None = None) -> None:
    """Print the mean-baseline vs neural-decoded comparison for the requested sessions."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--job-id', default='21047248')
    ap.add_argument('--eids', nargs='+', default=list(DEFAULT_EIDS))
    ap.add_argument('--lr', default='1e-4')
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args(argv)

    pooled = {m: ([], []) for m in METRICS}  # all sessions, all views: (neural, mean)
    latex_rows = []
    for eid in args.eids:
        neural_path, mean_path = metrics_paths(args.root, args.job_id, eid, args.lr)
        missing = [p for p in (neural_path, mean_path) if not p.exists()]
        if missing:
            print(f'[skip] {eid}: missing {", ".join(str(p) for p in missing)}')
            continue
        neural, mean = load_sorted_metrics(neural_path), load_sorted_metrics(mean_path)
        check_paired(neural, mean)
        n_views = neural['psnr'].shape[-1]
        view_names = [str(v) for v in neural['view_names']] or [f'view{v}' for v in range(n_views)]
        print(f'\n== {eid}  (K, T, V) = {neural["psnr"].shape}')
        pooled_stats = {}
        for metric in METRICS:
            for v, name in enumerate(view_names):
                stats = compare(neural[metric][..., v], mean[metric][..., v])
                print(format_row(f'{eid[:8]} {name}', metric, stats))
            pooled_stats[metric] = compare(neural[metric], mean[metric])
            print(format_row(f'{eid[:8]} all views', metric, pooled_stats[metric]))
            if pooled_stats[metric]['delta'] <= 0:
                print(f'WARNING: neural-decoded {metric} is not above the mean baseline')
            pooled[metric][0].append(np.ravel(neural[metric]))
            pooled[metric][1].append(np.ravel(mean[metric]))
        latex_rows.append(format_latex(eid[:8], pooled_stats))

    if not latex_rows:
        raise SystemExit('no session had both metrics files; nothing to summarize')
    print('\n== all sessions pooled')
    pooled_all = {}
    for metric in METRICS:
        neural_all, mean_all = (np.concatenate(x) for x in pooled[metric])
        pooled_all[metric] = compare(neural_all, mean_all)
        print(format_row('all sessions', metric, pooled_all[metric]))
    latex_rows.append(format_latex('All', pooled_all))
    print('\n% LaTeX rows: session & PSNR mean & PSNR ours & SSIM mean & SSIM ours')
    print('\n'.join(latex_rows))


if __name__ == '__main__':
    main()
