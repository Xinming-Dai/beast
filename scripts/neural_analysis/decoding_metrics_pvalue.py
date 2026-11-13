#!/usr/bin/env python3
"""P-values for PSNR / SSIM differences between a reference method and baselines.

Companion of ``decoding_metrics_figure2.py`` (reads the same ``psnr_ssim_metrics*.npz`` files,
laid out as ``<results-dir>/<method>/<eid>/psnr_ssim_metrics*.npz``). The figure script is not
modified.

```bash
python scripts/neural_analysis/decoding_metrics_pvalue.py \
  --reference sable --baselines beast mean
```

Statistical methods (text that can be copied into a paper; the run also prints a version
filled in with the actual numbers, see :func:`build_methods_text`):

    The reference method was compared with each baseline separately for PSNR and SSIM on the
    held-out test trials. All methods reconstruct the same frames, so the comparison is paired.
    The 60 time bins and 2 camera views of one trial are strongly correlated, so frames are not
    independent observations and the trial is used as the unit of analysis: each metric is
    averaged over bins and views within a trial and the per-trial difference (reference minus
    baseline) is computed. The p-value is from a two-sided paired sign-flip permutation test on
    the mean difference (the sign of each trial difference is flipped at random, with p =
    (b + 1) / (N + 1) for b permutations at least as extreme as the observed mean in N
    permutations). A Wilcoxon signed-rank test and a paired t-test are reported as supporting
    tests, and a 95% confidence interval of the mean difference is obtained by bootstrapping
    trials within each session. Consistency across sessions is checked by repeating the test
    within each session. For comparison with the standard errors in the bar plot, which treat
    every frame as independent, the same tests are also run on all frame pairs; these p-values
    are anticonservative because of the dependence between frames. P-values are adjusted for
    the number of comparisons (baselines x metrics) with the Holm method, separately for the
    trial-level and the frame-level tests.

    Optionally (``--sample-frames K``), the per-trial value is the mean over K randomly drawn
    (bin, view) pairs instead of the mean over all of them, with identical draws for both
    methods; the procedure is repeated for several random draws and the median p-value is
    reported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy import stats

from scripts.neural_analysis.decoding_metrics_figure2 import (
    RESULTS_DIR,
    _allowed_eid_set,
    iter_eid_metrics_npys,
)
from scripts.neural_analysis.plot_helpers import EID_SET, _default_output_path_helper

logger = logging.getLogger(__name__)

METRICS: tuple[str, ...] = ('psnr', 'ssim')
LEVEL_TRIAL: str = 'trial'
LEVEL_FRAME: str = 'frame'
GROUP_ALL: str = 'all'
# upper bound on the number of entries of the random sign matrix held in memory at once
MAX_SIGN_MATRIX_SIZE: int = 2_000_000
# bootstrap resamples used for the confidence interval of the mean difference
N_BOOT: int = 10_000


@dataclass
class EidValues:
    """Per-frame metrics of one method in one EID.

    Attributes:
        trial_idx: neural trial index of each trial, shape ``(n_trials,)``.
        bin_idx: neural bin index of each frame, shape ``(n_trials, n_bins)``.
        psnr: PSNR per frame, shape ``(n_trials, n_bins, n_views)``.
        ssim: SSIM per frame, shape ``(n_trials, n_bins, n_views)``.
    """

    trial_idx: np.ndarray
    bin_idx: np.ndarray
    psnr: np.ndarray
    ssim: np.ndarray

    def metric(self, name: str) -> np.ndarray:
        """Return the ``(n_trials, n_bins, n_views)`` array of metric ``name``."""
        return getattr(self, name)


def load_eid_values(
    method_dir: Path,
    allowed_eids: frozenset[str] | None,
) -> dict[str, EidValues]:
    """Load the per-frame metrics of every matching EID of one method.

    Args:
        method_dir: folder of the method, with one subfolder per EID.
        allowed_eids: EIDs to include, or None for every EID found.

    Returns:
        Mapping from EID folder name to its metrics, empty if the method folder is missing.
    """
    values: dict[str, EidValues] = {}
    for npz in sorted(iter_eid_metrics_npys(method_dir, allowed_eids)):
        metrics = np.load(npz, allow_pickle=True)
        values[npz.parent.name] = EidValues(
            trial_idx=np.asarray(metrics['neural_trial_idx']),
            bin_idx=np.asarray(metrics['neural_bin_idx']),
            psnr=np.asarray(metrics['psnr'], dtype=np.float64),
            ssim=np.asarray(metrics['ssim'], dtype=np.float64),
        )
    return values


def check_paired(eid: str, ref: EidValues, other: EidValues) -> None:
    """Check that two methods were evaluated on the same frames of one EID.

    Args:
        eid: EID name, used in the error message.
        ref: metrics of the reference method.
        other: metrics of the baseline.

    Raises:
        ValueError: if the trial or bin indices (or the metric shapes) differ.
    """
    same_frames = (
        ref.psnr.shape == other.psnr.shape
        and np.array_equal(ref.trial_idx, other.trial_idx)
        and np.array_equal(ref.bin_idx, other.bin_idx)
    )
    if not same_frames:
        raise ValueError(f'EID {eid}: methods were not evaluated on the same frames')


def draw_frame_indices(
    rng: np.random.Generator,
    n_trials: int,
    n_frames: int,
    n_sample: int,
) -> np.ndarray:
    """Draw ``n_sample`` distinct (bin, view) pairs per trial.

    Args:
        rng: random number generator.
        n_trials: number of trials.
        n_frames: number of (bin, view) pairs per trial.
        n_sample: pairs to draw per trial.

    Returns:
        Flat (bin, view) indices of shape ``(n_trials, n_sample)``.
    """
    return np.argsort(rng.random((n_trials, n_frames)), axis=1)[:, :n_sample]


def reduce_to_trials(values: np.ndarray, idx_frame: np.ndarray | None) -> np.ndarray:
    """Reduce per-frame values to one value per trial.

    Args:
        values: metric array of shape ``(n_trials, n_bins, n_views)``.
        idx_frame: flat (bin, view) indices to average per trial, shape ``(n_trials, k)``, or
            None to average over all bins and views. Non-finite values are ignored.

    Returns:
        Array of shape ``(n_trials,)``, nan for trials without any finite value.
    """
    flat = values.reshape(values.shape[0], -1)
    if idx_frame is not None:
        flat = np.take_along_axis(flat, idx_frame, axis=1)
    finite = np.isfinite(flat)
    count = finite.sum(axis=1)
    total = np.where(finite, flat, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _drop_nan(values: np.ndarray) -> np.ndarray:
    """Return ``values`` without non-finite entries."""
    return values[np.isfinite(values)]


def trial_level_diffs(
    ref_values: dict[str, EidValues],
    other_values: dict[str, EidValues],
    sample_frames: int | None,
    rng: np.random.Generator,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-trial differences (reference minus baseline) for every EID and metric.

    With ``sample_frames`` the same random (bin, view) pairs are used for both methods.

    Args:
        ref_values: metrics of the reference method per EID.
        other_values: metrics of the baseline per EID.
        sample_frames: (bin, view) pairs to average per trial, or None to use all of them.
        rng: random number generator used for the frame draws.

    Returns:
        ``{eid: {metric: differences of shape (n_trials,)}}`` over the EIDs of both methods.

    Raises:
        ValueError: if the methods were not evaluated on the same frames of an EID.
    """
    diffs: dict[str, dict[str, np.ndarray]] = {}
    for eid in sorted(set(ref_values) & set(other_values)):
        ref, other = ref_values[eid], other_values[eid]
        check_paired(eid, ref, other)
        idx_frame = None
        if sample_frames is not None:
            n_trials, n_bins, n_views = ref.psnr.shape
            idx_frame = draw_frame_indices(rng, n_trials, n_bins * n_views, sample_frames)
        diffs[eid] = {
            metric: _drop_nan(
                reduce_to_trials(ref.metric(metric), idx_frame)
                - reduce_to_trials(other.metric(metric), idx_frame)
            )
            for metric in METRICS
        }
    return diffs


def frame_level_diffs(
    ref_values: dict[str, EidValues],
    other_values: dict[str, EidValues],
) -> dict[str, dict[str, np.ndarray]]:
    """Per-frame differences (reference minus baseline) for every EID and metric.

    Args:
        ref_values: metrics of the reference method per EID.
        other_values: metrics of the baseline per EID.

    Returns:
        ``{eid: {metric: differences of shape (n_frames,)}}``, frames with a non-finite value
        in either method are dropped.

    Raises:
        ValueError: if the methods were not evaluated on the same frames of an EID.
    """
    diffs: dict[str, dict[str, np.ndarray]] = {}
    for eid in sorted(set(ref_values) & set(other_values)):
        ref, other = ref_values[eid], other_values[eid]
        check_paired(eid, ref, other)
        diffs[eid] = {
            metric: _drop_nan((ref.metric(metric) - other.metric(metric)).reshape(-1))
            for metric in METRICS
        }
    return diffs


def sign_flip_pvalue(diffs: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Two-sided paired sign-flip permutation p-value of the mean difference.

    Args:
        diffs: paired differences.
        n_perm: number of random sign flips.
        rng: random number generator.

    Returns:
        ``(b + 1) / (n_perm + 1)`` with ``b`` the number of flips whose absolute mean is at
        least the observed one, so the smallest possible value is ``1 / (n_perm + 1)``.
    """
    n = diffs.size
    observed = abs(float(diffs.mean()))
    n_chunk = max(1, MAX_SIGN_MATRIX_SIZE // n)
    # tolerance guards against rounding differences between the observed and permuted means
    tol = 1e-12 * max(float(np.abs(diffs).max()), 1e-300)
    n_extreme = 0
    for start in range(0, n_perm, n_chunk):
        n_now = min(n_chunk, n_perm - start)
        # random bits come 8 per byte, which is much cheaper than one integer draw per sign
        bits = rng.integers(0, 256, size=(n_now, (n + 7) // 8), dtype=np.uint8)
        signs = np.unpackbits(bits, axis=1)[:, :n].astype(np.float64) * 2.0 - 1.0
        n_extreme += int((np.abs(signs @ diffs) / n >= observed - tol).sum())
    return (n_extreme + 1) / (n_perm + 1)


def bootstrap_ci(
    diffs_by_group: list[np.ndarray],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean, resampling within each group.

    Args:
        diffs_by_group: paired differences, one array per EID.
        n_boot: number of bootstrap resamples.
        rng: random number generator.

    Returns:
        Lower and upper limit of the interval.
    """
    total = sum(d.size for d in diffs_by_group)
    boot_sum = np.zeros(n_boot)
    for d in diffs_by_group:
        boot_sum += d[rng.integers(0, d.size, size=(n_boot, d.size))].sum(axis=1)
    low, high = np.percentile(boot_sum / total, [2.5, 97.5])
    return float(low), float(high)


def paired_comparison(
    diffs_by_group: list[np.ndarray],
    n_perm: int,
    rng: np.random.Generator,
    *,
    bootstrap: bool = True,
) -> dict[str, float]:
    """Paired tests of whether the mean difference is zero.

    Args:
        diffs_by_group: paired differences, one array per EID (pooled for the tests).
        n_perm: number of random sign flips of the permutation test.
        rng: random number generator.
        bootstrap: whether the interval is bootstrapped (resampling within groups); otherwise
            it is the t interval, which is much cheaper for many frames.

    Returns:
        Dict with ``n``, ``mean_diff``, ``ci_low``, ``ci_high``, ``p_perm``, ``p_wilcoxon`` and
        ``p_ttest``.
    """
    diffs = np.concatenate(diffs_by_group)
    n = diffs.size
    mean_diff = float(diffs.mean())
    sem = float(diffs.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    if bootstrap:
        ci_low, ci_high = bootstrap_ci(diffs_by_group, N_BOOT, rng)
    else:
        half_width = float(stats.t.ppf(0.975, n - 1)) * sem
        ci_low, ci_high = mean_diff - half_width, mean_diff + half_width

    if np.all(diffs == 0):
        p_wilcoxon = p_ttest = 1.0
    else:
        p_wilcoxon = float(stats.wilcoxon(diffs).pvalue)
        p_ttest = float(stats.ttest_1samp(diffs, 0.0).pvalue)
    return {
        'n': float(n),
        'mean_diff': mean_diff,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_perm': sign_flip_pvalue(diffs, n_perm, rng),
        'p_wilcoxon': p_wilcoxon,
        'p_ttest': p_ttest,
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment of p-values.

    Args:
        p_values: raw p-values.

    Returns:
        Adjusted p-values in the original order, capped at 1.
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    n = p_values.size
    scaled = (n - np.arange(n)) * p_values[order]
    adjusted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    out = np.empty(n)
    out[order] = adjusted
    return out


def _rows_for_diffs(
    baseline: str,
    level: str,
    diffs: dict[str, dict[str, np.ndarray]],
    n_perm: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Result rows (pooled over EIDs first, then one per EID) of every metric."""
    rows = []
    for metric in METRICS:
        groups = {GROUP_ALL: [d[metric] for d in diffs.values()]}
        groups.update({eid: [d[metric]] for eid, d in diffs.items()})
        for group, group_diffs in groups.items():
            result = paired_comparison(
                group_diffs, n_perm, rng, bootstrap=level == LEVEL_TRIAL
            )
            rows.append(
                {'baseline': baseline, 'metric': metric, 'level': level, 'group': group, **result}
            )
    return rows


def _median_rows(rows_per_draw: list[list[dict]]) -> list[dict]:
    """Median over draws of every numeric field of the (identically ordered) result rows."""
    numeric = ['n', 'mean_diff', 'ci_low', 'ci_high', 'p_perm', 'p_wilcoxon', 'p_ttest']
    rows = []
    for rows_same in zip(*rows_per_draw):
        row = dict(rows_same[0])
        row.update({key: float(np.median([r[key] for r in rows_same])) for key in numeric})
        rows.append(row)
    return rows


def compare_baseline(
    baseline: str,
    ref_values: dict[str, EidValues],
    other_values: dict[str, EidValues],
    n_perm: int,
    sample_frames: int | None,
    n_draws: int,
    seed: int,
) -> list[dict]:
    """Trial-level and frame-level comparison of the reference with one baseline.

    Args:
        baseline: baseline name, copied into the rows.
        ref_values: metrics of the reference method per EID.
        other_values: metrics of the baseline per EID.
        n_perm: number of random sign flips of the permutation tests.
        sample_frames: (bin, view) pairs averaged per trial, or None to average all of them.
        n_draws: number of repetitions of the frame sampling (ignored without sampling).
        seed: random seed.

    Returns:
        Result rows, dicts with ``baseline``, ``metric``, ``level``, ``group`` and the keys of
        :func:`paired_comparison` (without the Holm-adjusted p-value).
    """
    n_rep = n_draws if sample_frames is not None else 1
    rows_per_draw = []
    for draw in range(n_rep):
        rng = np.random.default_rng([seed, draw])
        diffs = trial_level_diffs(ref_values, other_values, sample_frames, rng)
        rows_per_draw.append(_rows_for_diffs(baseline, LEVEL_TRIAL, diffs, n_perm, rng))
    rows = _median_rows(rows_per_draw)

    rng = np.random.default_rng([seed, n_rep])
    diffs = frame_level_diffs(ref_values, other_values)
    return rows + _rows_for_diffs(baseline, LEVEL_FRAME, diffs, n_perm, rng)


def add_holm_adjusted(rows: list[dict]) -> None:
    """Add ``p_perm_holm`` to the pooled rows, adjusting separately for trial and frame level.

    Args:
        rows: result rows of all baselines, modified in place. Per-EID rows get nan.
    """
    for row in rows:
        row['p_perm_holm'] = np.nan
    for level in (LEVEL_TRIAL, LEVEL_FRAME):
        pooled = [r for r in rows if r['level'] == level and r['group'] == GROUP_ALL]
        if not pooled:
            continue
        adjusted = holm_adjust(np.array([r['p_perm'] for r in pooled]))
        for row, p_adj in zip(pooled, adjusted):
            row['p_perm_holm'] = float(p_adj)


def format_p(p: float, floor: float | None = None) -> str:
    """Format a p-value, as ``<floor`` when it sits at the resolution of a permutation test."""
    if not np.isfinite(p):
        return '-'
    if floor is not None and p <= floor * (1 + 1e-9):
        return f'<{floor:.0e}'
    return f'{p:.4f}' if p >= 1e-4 else f'{p:.1e}'


def format_table(rows: list[dict], n_perm: int) -> str:
    """Render the result rows as a text table.

    Args:
        rows: result rows.
        n_perm: number of permutations, which sets the smallest reportable permutation p.

    Returns:
        The table as a string.
    """
    floor = 1.0 / (n_perm + 1)
    header = (
        f"{'baseline':<10} {'metric':<5} {'level':<6} {'group':<9} {'n':>6} {'mean diff':>10} "
        f"{'95% CI':>22} {'p perm':>8} {'p holm':>8} {'p wilcox':>9} {'p t-test':>9}"
    )
    lines = [header, '-' * len(header)]
    for r in rows:
        interval = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        lines.append(
            f"{r['baseline']:<10} {r['metric']:<5} {r['level']:<6} {r['group'][:8]:<9} "
            f"{int(r['n']):>6} {r['mean_diff']:>10.4f} {interval:>22} "
            f"{format_p(r['p_perm'], floor):>8} {format_p(r['p_perm_holm'], floor):>8} "
            f"{format_p(r['p_wilcoxon']):>9} {format_p(r['p_ttest']):>9}"
        )
    return '\n'.join(lines)


def build_methods_text(
    reference: str,
    baselines: list[str],
    n_eids: int,
    n_trials: int,
    n_frames: int,
    n_perm: int,
    sample_frames: int | None,
    n_draws: int,
    seed: int,
) -> str:
    """Methods paragraph filled in with the settings and sample sizes of this run.

    Args:
        reference: reference method name.
        baselines: baselines that were compared.
        n_eids: number of EIDs (sessions) used.
        n_trials: number of test trials used (of the reference).
        n_frames: number of paired frames used (of the reference).
        n_perm: number of random sign flips of the permutation tests.
        sample_frames: (bin, view) pairs averaged per trial, or None for all of them.
        n_draws: number of repetitions of the frame sampling.
        seed: random seed.

    Returns:
        The methods text.
    """
    if sample_frames is None:
        reduction = 'each metric was averaged over time bins and camera views within a trial'
    else:
        reduction = (
            f'each metric was averaged over {sample_frames} randomly drawn (time bin, camera '
            f'view) pair(s) within a trial, with the same draws for both methods, and the '
            f'procedure was repeated for {n_draws} random draws, reporting the median p-value'
        )
    return (
        f'{reference} was compared with {", ".join(baselines)} separately for PSNR and SSIM '
        f'on the held-out test trials of {n_eids} sessions ({n_trials} trials, {n_frames} '
        f'frames). All methods reconstruct the same frames, so the comparison is paired. '
        f'Time bins and camera views within a trial are strongly correlated, so frames are '
        f'not independent observations and the trial was used as the unit of analysis: '
        f'{reduction}, and the per-trial difference between {reference} and the baseline was '
        f'computed. P-values are from a two-sided paired sign-flip permutation test on the '
        f'mean difference ({n_perm} random sign flips; p = (b + 1) / ({n_perm} + 1) for b '
        f'flips at least as extreme as the observed mean). A Wilcoxon signed-rank test and a '
        f'paired t-test are given as supporting tests, and 95% confidence intervals of the '
        f'mean difference come from bootstrapping trials within sessions ({N_BOOT} '
        f'resamples). The test was repeated within each session to check consistency. For '
        f'comparison with the error bars of the bar plot, which treat all frames as '
        f'independent, the same tests were also run on all frame pairs; these p-values are '
        f'anticonservative because of the dependence between frames. P-values were adjusted '
        f'for the number of comparisons with the Holm method, separately for the trial-level '
        f'and frame-level tests. The random seed was {seed}.'
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--results-dir', type=Path, default=RESULTS_DIR, help='Root with one folder per method')
    p.add_argument('--reference', default='sable', help='Method tested against the baselines')
    p.add_argument(
        '--baselines',
        nargs='+',
        default=['beast'],
        help='One or more method folders to compare the reference with (e.g. beast resnet mean)',
    )
    p.add_argument('--eids', nargs='*', default=None, help='EIDs to include (default: EID_SET)')
    p.add_argument('--n-perm', type=int, default=100_000, help='Random sign flips per permutation test')
    p.add_argument(
        '--sample-frames',
        type=int,
        default=None,
        help='Average K random (bin, view) pairs per trial instead of all of them',
    )
    p.add_argument(
        '--n-draws',
        type=int,
        default=100,
        help='Repetitions of the frame sampling; the median p-value is reported',
    )
    p.add_argument('--seed', type=int, default=0)
    p.add_argument(
        '-o',
        '--output-dir',
        type=Path,
        default=None,
        help='Root for the timestamped output folder (default: --results-dir)',
    )
    p.add_argument(
        '--no-save',
        action='store_true',
        help='Do not write pvalues.txt (table + methods text; written by default)',
    )
    args = p.parse_args()

    if args.n_perm < 1 or args.n_draws < 1:
        raise ValueError('--n-perm and --n-draws must be at least 1')
    if args.sample_frames is not None and args.sample_frames < 1:
        raise ValueError('--sample-frames must be at least 1')
    if args.reference in args.baselines:
        raise ValueError(f'reference {args.reference!r} is also listed in --baselines')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    results_dir = args.results_dir.expanduser().resolve()
    allowed = _allowed_eid_set(args.eids if args.eids is not None else sorted(EID_SET))

    ref_values = load_eid_values(results_dir / args.reference, allowed)
    if not ref_values:
        raise FileNotFoundError(f'no metrics found for reference {args.reference!r} in {results_dir}')

    rows: list[dict] = []
    used_baselines: list[str] = []
    for baseline in args.baselines:
        other_values = load_eid_values(results_dir / baseline, allowed)
        missing = sorted(set(ref_values) - set(other_values))
        if not other_values:
            logger.warning(f'skipping {baseline!r}: no metrics found in {results_dir / baseline}')
            continue
        if missing:
            logger.warning(f'{baseline!r} has no metrics for EIDs {missing}; they are skipped')
        logger.info(f'comparing {args.reference} with {baseline}')
        rows += compare_baseline(
            baseline,
            ref_values,
            other_values,
            args.n_perm,
            args.sample_frames,
            args.n_draws,
            args.seed,
        )
        used_baselines.append(baseline)
    if not rows:
        raise FileNotFoundError('none of the baselines could be compared')

    add_holm_adjusted(rows)
    # baselines may cover different EIDs; the methods text reports the sizes of the first one
    first = [
        r
        for r in rows
        if r['baseline'] == used_baselines[0] and r['group'] == GROUP_ALL and r['metric'] == METRICS[0]
    ]
    n_trials, n_frames = int(first[0]['n']), int(first[1]['n'])
    methods_text = build_methods_text(
        args.reference,
        used_baselines,
        len({r['group'] for r in rows if r['baseline'] == used_baselines[0]}) - 1,
        n_trials,
        n_frames,
        args.n_perm,
        args.sample_frames,
        args.n_draws,
        args.seed,
    )

    report = (
        f'mean diff = {args.reference} - baseline (positive favors {args.reference})\n\n'
        f'{format_table(rows, args.n_perm)}\n\n'
        f'Methods text:\n\n{methods_text}\n'
    )
    print(f'\n{report}')
    if not args.no_save:
        output_root = args.output_dir if args.output_dir is not None else results_dir
        save_path = _default_output_path_helper(output_root, 'decoding_pvalue', 'pvalues.txt')
        save_path.write_text(report)
        logger.info(f'saved {save_path}')


if __name__ == '__main__':
    main()
