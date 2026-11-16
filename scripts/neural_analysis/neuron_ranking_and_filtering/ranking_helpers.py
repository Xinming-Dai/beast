"""Rank the neurons of one session by BPS and filter out neurons with a flat predicted PSTH.

BPS is scored on single trials against the mean rate of the held-out set, so a prediction that only
follows the trial-to-trial gain of a neuron scores high while its trial-averaged PSTH is flat. Every
neuron is therefore also scored by the correlation of its predicted and ground-truth PSTH and by the
modulation of its predicted PSTH over time. The raw metrics are saved as a CSV, and the thresholds
are applied only when the CSV is read, so they can be tuned without recomputing.

The helpers are independent of the data pipeline: they take `gt` and `pred` arrays of shape
(trials, time, neurons) and a dict that identifies the run, e.g. session, model, K and fold.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.neural_analysis.plot_helpers import bps_per_neuron

logger = logging.getLogger(__name__)

DEFAULT_MIN_PSTH_CORR = 0.15
DEFAULT_MIN_MODULATION = 0.1
# folder of the CLS-token rankings inside the results directory
RANKING_DIRNAME = 'neuron_ranking'


def cls_ranking_path(root: Path, eid: str, method: str, encoder: str) -> Path:
    """Path of the ranking CSV of one (EID, method, encoder) of the CLS-token pipeline.

    It lives here, not in `rank_neurons_bps_corr_cls_token.py`, because the plotting scripts that
    read it are imported by that script.

    Args:
        root: folder of the rankings.
        eid: session EID.
        method: method folder name.
        encoder: encoder block, `cnn` or `rrr`.

    Returns:
        `<root>/<eid>/ranking_<method>_<encoder>.csv`.
    """
    return root / eid / f'ranking_{method}_{encoder}.csv'


def psth_metrics(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Absolute quality of the predicted PSTH of every neuron.

    Args:
        gt: ground-truth spikes, shape (trials, time, neurons).
        pred: predicted rates, same shape as `gt`.

    Returns:
        psth_corr: Pearson correlation over time between the ground-truth and predicted PSTHs,
            shape (neurons,), NaN where either PSTH is constant.
        modulation: `std / mean` over time of the predicted PSTH, shape (neurons,), NaN where the
            mean is not positive.
    """
    gt_psth = np.nanmean(np.asarray(gt, dtype=np.float64), axis=0)
    pred_psth = np.nanmean(np.asarray(pred, dtype=np.float64), axis=0)
    gt_centered = gt_psth - gt_psth.mean(axis=0)
    pred_centered = pred_psth - pred_psth.mean(axis=0)
    denom = np.sqrt((gt_centered**2).sum(axis=0) * (pred_centered**2).sum(axis=0))
    psth_corr = np.full(denom.shape, np.nan)
    valid = denom > 1e-12
    psth_corr[valid] = (gt_centered * pred_centered).sum(axis=0)[valid] / denom[valid]

    pred_mean = pred_psth.mean(axis=0)
    modulation = np.full(pred_mean.shape, np.nan)
    positive = pred_mean > 0
    modulation[positive] = pred_psth.std(axis=0)[positive] / pred_mean[positive]
    return psth_corr, modulation


def rank_neurons(gt: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Score every neuron and sort them by BPS.

    Args:
        gt: ground-truth spikes, shape (trials, time, neurons).
        pred: predicted rates, same shape as `gt`.

    Returns:
        One row per neuron, by decreasing BPS with NaN last, with columns `rank` (from 1),
        `neuron_idx`, `bps`, `psth_corr`, `modulation` and `n_spikes`.
    """
    bps = bps_per_neuron(gt, pred)
    psth_corr, modulation = psth_metrics(gt, pred)
    n_spikes = np.nansum(np.asarray(gt, dtype=np.float64), axis=(0, 1))
    order = np.argsort(-np.nan_to_num(bps, nan=-np.inf), kind='stable')
    return pd.DataFrame(
        {
            'rank': np.arange(1, order.size + 1),
            'neuron_idx': order,
            'bps': bps[order],
            'psth_corr': psth_corr[order],
            'modulation': modulation[order],
            'n_spikes': n_spikes[order],
        },
    )


def rank_by_bps(bps: np.ndarray) -> pd.DataFrame:
    """Sort neurons by a precomputed BPS, e.g. the held-out BPS averaged over folds.

    Args:
        bps: per-neuron BPS, shape (neurons,), NaN for neurons without test spikes.

    Returns:
        One row per neuron, by decreasing BPS with NaN last, with columns `rank` (from 1),
        `neuron_idx` and `bps`.
    """
    bps = np.asarray(bps, dtype=np.float64)
    order = np.argsort(-np.nan_to_num(bps, nan=-np.inf), kind='stable')
    return pd.DataFrame(
        {
            'rank': np.arange(1, order.size + 1),
            'neuron_idx': order,
            'bps': bps[order],
        },
    )


def split_by_rank(df: pd.DataFrame, frac: float) -> tuple[list[int], list[int]]:
    """Split the neurons of a ranking into the top fraction by BPS and the rest.

    Args:
        df: ranking, see `rank_neurons` or `rank_by_bps`.
        frac: fraction of the neurons with a BPS that goes to the top group, in (0, 1).

    Returns:
        Neuron indices of the top group and of the rest, both by decreasing BPS. Neurons without
        a BPS are in neither.

    Raises:
        ValueError: if `frac` is not strictly between 0 and 1.
    """
    if not 0 < frac < 1:
        raise ValueError(f'frac must be strictly between 0 and 1, got {frac}')
    idx_valid = [int(idx) for idx in df.loc[df['bps'].notna(), 'neuron_idx']]
    n_top = round(frac * len(idx_valid))
    return idx_valid[:n_top], idx_valid[n_top:]


def save_ranking(df: pd.DataFrame, path: Path, meta: dict[str, str | int]) -> None:
    """Write a ranking as a CSV whose first line describes the run it belongs to.

    Args:
        df: ranking, see `rank_neurons`.
        path: output CSV path, its folder is created.
        meta: identifies the run, e.g. `{'session': '72', 'model': 'full', 'k': 6, 'fold': 0}`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ' '.join(f'{key}={value}' for key, value in meta.items())
    with path.open('w') as f:
        f.write(f'# {header}\n')
        df.to_csv(f, index=False)


def load_ranking(
    path: Path,
    expected: dict[str, str | int],
    n_neurons: int | None = None,
) -> pd.DataFrame:
    """Read a ranking and check that it belongs to the run that asks for it.

    Args:
        path: CSV written by `save_ranking`.
        expected: keys and values the header must contain, e.g. session, model, K and fold.
        n_neurons: number of neurons of the data, the ranking must have one row per neuron.

    Returns:
        The ranking.

    Raises:
        FileNotFoundError: if the ranking has not been computed.
        ValueError: if the header does not match `expected` or the row count is not `n_neurons`.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f'no ranking at {path}, run the rank_neurons_*.py script of '
            f'neuron_ranking_and_filtering first'
        )
    with path.open() as f:
        first_line = f.readline().strip()
    if not first_line.startswith('#'):
        raise ValueError(f'{path} has no `# key=value` header, was it written by save_ranking?')
    header = dict(item.split('=', 1) for item in first_line.lstrip('# ').split())
    for key, value in expected.items():
        if header.get(key) != str(value):
            raise ValueError(
                f'{path} was made for {key}={header.get(key)}, but {key}={value} was requested'
            )
    df = pd.read_csv(path, comment='#')
    if n_neurons is not None and len(df) != n_neurons:
        raise ValueError(f'{path} ranks {len(df)} neurons, but the data has {n_neurons}')
    return df


def pass_filters(
    df: pd.DataFrame,
    *,
    min_psth_corr: float | None,
    min_modulation: float | None,
) -> pd.Series:
    """Mask of the neurons with a BPS and a predicted PSTH that is not flat.

    Args:
        df: ranking, see `rank_neurons`.
        min_psth_corr: minimum PSTH correlation, None to skip the filter.
        min_modulation: minimum PSTH modulation, None to skip the filter.

    Returns:
        Boolean series aligned with `df`, False where a metric is NaN.
    """
    keep = df['bps'].notna()
    if min_psth_corr is not None:
        keep &= df['psth_corr'] >= min_psth_corr
    if min_modulation is not None:
        keep &= df['modulation'] >= min_modulation
    return keep


def select_top_neurons(
    df: pd.DataFrame,
    num_neurons: int,
    *,
    min_psth_corr: float | None,
    min_modulation: float | None,
) -> list[int]:
    """Top-BPS neurons whose predicted PSTH is not flat.

    Args:
        df: ranking, see `rank_neurons`.
        num_neurons: number of neurons to return.
        min_psth_corr: minimum PSTH correlation, None to skip the filter.
        min_modulation: minimum PSTH modulation, None to skip the filter.

    Returns:
        Neuron indices by decreasing BPS. Shorter than `num_neurons` if too few neurons pass.
    """
    keep = pass_filters(df, min_psth_corr=min_psth_corr, min_modulation=min_modulation)
    selected = [int(idx) for idx in df.loc[keep, 'neuron_idx'].iloc[:num_neurons]]
    if len(selected) < num_neurons:
        logger.warning(
            f'only {len(selected)} of {num_neurons} requested neurons pass the PSTH filters '
            f'({int(keep.sum())} pass in total)'
        )
    return selected


def add_ranking_args(parser: argparse.ArgumentParser) -> None:
    """Add `--ranking`, `--ranking-dir` and the filter thresholds to a plotting script.

    Args:
        parser: parser of the plotting script.
    """
    parser.add_argument(
        '--ranking',
        action='store_true',
        help=(
            'take the top neurons from the saved ranking of the last model, written by '
            'neuron_ranking_and_filtering, instead of the neuron flags; the ranking uses all '
            'held-out trials, also with --trial-idx'
        ),
    )
    parser.add_argument(
        '--ranking-dir',
        type=Path,
        default=None,
        help='with --ranking: folder of the rankings (default: where the ranking script writes)',
    )
    parser.add_argument(
        '--min-psth-corr',
        type=float,
        default=DEFAULT_MIN_PSTH_CORR,
        help='with --ranking: minimum correlation of the predicted and ground-truth PSTH',
    )
    parser.add_argument(
        '--min-modulation',
        type=float,
        default=DEFAULT_MIN_MODULATION,
        help='with --ranking: minimum std / mean over time of the predicted PSTH',
    )
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='with --ranking: rank all neurons by BPS, without the PSTH filters',
    )


def thresholds_from_args(args: argparse.Namespace) -> tuple[float | None, float | None]:
    """Filter thresholds set by `add_ranking_args`, None for both with `--no-filter`."""
    if args.no_filter:
        return None, None
    return args.min_psth_corr, args.min_modulation


def neurons_from_ranking(
    args: argparse.Namespace,
    num_neurons: int,
    *,
    path: Path,
    expected: dict[str, str | int],
    n_neurons: int | None = None,
) -> list[int]:
    """Top neurons of the saved ranking of the run that is plotted.

    Args:
        args: parsed arguments of a script that called `add_ranking_args`.
        num_neurons: number of neurons to return.
        path: ranking CSV of the run, resolved by the script from its own arguments.
        expected: keys and values the ranking header must contain, see `load_ranking`.
        n_neurons: number of neurons of the data, see `load_ranking`.

    Returns:
        Neuron indices by decreasing BPS, see `select_top_neurons`.
    """
    df = load_ranking(path, expected, n_neurons)
    min_psth_corr, min_modulation = thresholds_from_args(args)
    return select_top_neurons(
        df,
        num_neurons,
        min_psth_corr=min_psth_corr,
        min_modulation=min_modulation,
    )
