"""IBL neural and behavior data extraction utilities.

Ported from E-RayZer-private's ``ibl_data_utils.py`` (itself built on the IBL ``brainbox``
pipeline). Used by ``beast.preprocess.sable.ibl.extract_sable_neural_data`` to pull spikes,
trial metadata, and continuous behaviors for one IBL session from the public IBL database
via ``ONE``, bin them into fixed-length intervals, and align spikes with behaviors for
downstream train/val/test splitting.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import uuid
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from brainbox.behavior.dlc import get_speed, likelihood_threshold
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from brainbox.population.decode import get_spike_counts_in_bins
from iblatlas.regions import BrainRegions
from iblutil.numerical import bincount2D, ismember
from one.api import ONE
from scipy.interpolate import interp1d
from tqdm import tqdm

_logger = logging.getLogger(__name__)

DYNAMIC_VARS = [
    'wheel-speed',
    'licks',
    'left-whisker-motion-energy',
    'right-whisker-motion-energy',
    'left-nose-speed',
    'right-nose-speed',
    'left-paw-speed',
    'right-paw-speed',
]

# (view for load_motion_energy, key in sess_loader.motion_energy)
# if your sessions only expose whisker on leftCamera, use ('left', 'leftCamera') for both
_WHISKER_MOTION_ENERGY = {
    'left-whisker-motion-energy': ('left', 'leftCamera'),
    'right-whisker-motion-energy': ('right', 'rightCamera'),
}

_NOSE_SPEED = {
    'left-nose-speed': ('left', 'leftCamera'),
    'right-nose-speed': ('right', 'rightCamera'),
}

_PAW_SPEED = {
    'left-paw-speed': ('left', 'leftCamera'),
    'right-paw-speed': ('right', 'rightCamera'),
}


def globalize(func: Callable) -> Callable:
    """Make a locally-defined function picklable by registering it at module scope.

    ``multiprocessing.Pool`` pickles worker functions by qualified name, which fails for
    closures defined inside another function. This wraps ``func`` under a fresh unique
    name on its own module so the pool can find it.

    Args:
        func: the (possibly closure) function to make picklable.

    Returns:
        a picklable wrapper around ``func``.
    """
    def result(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


def load_spiking_data(
    one: ONE,
    pid: str,
    compute_metrics: bool = False,
    qc: float | None = None,
    **kwargs: Any,
) -> tuple[dict | None, pd.DataFrame | None, int | None]:
    """Load and (optionally) QC-filter spike sorting output for one probe insertion.

    Args:
        one: authenticated ``ONE`` client.
        pid: probe insertion UUID.
        compute_metrics: whether to (re)compute cluster QC metrics.
        qc: minimum cluster label to keep; ``None`` skips filtering.
        **kwargs: ``eid`` and ``pname`` forwarded to ``SpikeSortingLoader``.

    Returns:
        ``(spikes, clusters, sampling_freq)``; all ``None`` if no clusters were found.
    """
    eid = kwargs.pop('eid', '')
    pname = kwargs.pop('pname', '')
    sampling_freq = 30_000
    spike_loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)

    spikes, clusters, channels = spike_loader.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=compute_metrics,
    )
    if clusters_labeled is None:
        return None, None, None
    clusters_labeled = clusters_labeled.to_df()

    if qc is None:
        return spikes, clusters_labeled, sampling_freq

    iok = clusters_labeled['label'] >= qc
    selected_clusters = clusters_labeled[iok]
    spike_idx, ib = ismember(spikes['clusters'], selected_clusters.index)
    selected_clusters.reset_index(drop=True, inplace=True)
    selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}
    selected_spikes['clusters'] = selected_clusters.index[ib].astype(np.int32)
    return selected_spikes, selected_clusters, sampling_freq


def merge_probes(
    spikes_list: list[dict],
    clusters_list: list[pd.DataFrame],
) -> tuple[dict, pd.DataFrame]:
    """Merge spikes and clusters from multiple probes into a single time-sorted session.

    Args:
        spikes_list: per-probe spike dicts (each with ``'times'``/``'clusters'`` arrays).
        clusters_list: per-probe cluster dataframes.

    Returns:
        ``(merged_spikes, merged_clusters)`` with cluster ids offset to stay unique across
        probes and spikes sorted by time.
    """
    assert len(clusters_list) == len(spikes_list), (
        'clusters_list and spikes_list must have the same length'
    )
    assert all(isinstance(s, dict) for s in spikes_list), (
        'spikes_list must contain only dictionaries'
    )
    assert all(isinstance(c, pd.DataFrame) for c in clusters_list), (
        'clusters_list must contain only pd.DataFrames'
    )

    merged_spikes = []
    merged_clusters = []
    cluster_max = 0

    for clusters, spikes in zip(clusters_list, spikes_list):
        spikes['clusters'] += cluster_max
        cluster_max = clusters.index.max() + 1
        merged_spikes.append(spikes)
        merged_clusters.append(clusters)

    merged_clusters = pd.concat(merged_clusters, ignore_index=True)
    merged_spikes = {
        k: np.concatenate([s[k] for s in merged_spikes]) for k in merged_spikes[0].keys()
    }
    sort_idx = np.argsort(merged_spikes['times'], kind='stable')
    merged_spikes = {k: v[sort_idx] for k, v in merged_spikes.items()}
    return merged_spikes, merged_clusters


def load_trials_and_mask(
    one: ONE,
    eid: str,
    min_rt: float | None = 0.0,
    max_rt: float | None = 10.0,
    nan_exclude: str | list[str] = 'default',
    min_trial_len: float | None = None,
    max_trial_len: float | None = 10,
    exclude_unbiased: bool = False,
    exclude_nochoice: bool = True,
    sess_loader: SessionLoader | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a session's trials table and a boolean mask of well-behaved trials.

    Args:
        one: authenticated ``ONE`` client.
        eid: session UUID.
        min_rt: exclude trials with reaction time below this (seconds); ``None`` skips.
        max_rt: exclude trials with reaction time above this (seconds); ``None`` skips.
        nan_exclude: columns whose NaN rows should be excluded; ``'default'`` uses a
            standard IBL column set.
        min_trial_len: exclude trials shorter than this (seconds); ``None`` skips.
        max_trial_len: exclude trials longer than this (seconds); ``None`` skips.
        exclude_unbiased: exclude trials from the unbiased (50/50) block.
        exclude_nochoice: exclude trials with no choice recorded.
        sess_loader: reuse an existing ``SessionLoader`` instead of creating one.

    Returns:
        ``(trials, mask)`` — the full trials dataframe and a boolean mask of trials
        passing all filters.
    """
    if nan_exclude == 'default':
        nan_exclude = [
            'stimOn_times',
            'choice',
            'feedback_times',
            'probabilityLeft',
            'firstMovement_times',
            'feedbackType',
        ]

    if sess_loader is None:
        sess_loader = SessionLoader(one=one, eid=eid)

    if sess_loader.trials.empty:
        sess_loader.load_trials()

    if min_rt is not None:
        query = f'(firstMovement_times - stimOn_times < {min_rt})'
    else:
        query = ''
    if max_rt is not None:
        query += f' | (firstMovement_times - stimOn_times > {max_rt})'
    if min_trial_len is not None:
        query += f' | (feedback_times - goCue_times < {min_trial_len})'
    if max_trial_len is not None:
        query += f' | (feedback_times - goCue_times > {max_trial_len})'
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    if exclude_unbiased:
        query += ' | (probabilityLeft == 0.5)'
    if exclude_nochoice:
        query += ' | (choice == 0)'
    if min_rt is None:
        query = query[3:]

    mask = ~sess_loader.trials.eval(query)
    return sess_loader.trials, mask


def list_brain_regions(neural_dict: dict, **kwargs: Any) -> tuple[list[np.ndarray], np.ndarray]:
    """List brain regions present in a session, mapped to the Beryl atlas.

    Args:
        neural_dict: dict with a ``'cluster_regions'`` array of Allen acronyms.
        **kwargs: ``single_region`` (bool) — if set, return one region-group per region
            instead of a single group containing all of them.

    Returns:
        ``(regions, beryl_reg)`` — grouped region-of-interest lists and the per-cluster
        Beryl-mapped acronym array.
    """
    brainreg = BrainRegions()
    beryl_reg = brainreg.acronym2acronym(neural_dict['cluster_regions'], mapping='Beryl')
    regions = (
        [[k] for k in np.unique(beryl_reg)] if kwargs['single_region'] else [np.unique(beryl_reg)]
    )
    _logger.info(f'use spikes from brain regions: {regions[0]}')
    return regions, beryl_reg


def select_brain_regions(
    regressors: dict,
    beryl_reg: np.ndarray,
    region: list[str],
    **kwargs: Any,
) -> np.ndarray:
    """Select cluster indices belonging to the given brain region(s).

    Args:
        regressors: unused; kept for call-site symmetry with other selection helpers.
        beryl_reg: per-cluster Beryl-mapped acronym array.
        region: acronyms to keep.
        **kwargs: unused.

    Returns:
        array of cluster indices whose region is in ``region``.
    """
    del regressors, kwargs
    reg_mask = np.isin(beryl_reg, region)
    return np.argwhere(reg_mask).flatten()


def create_intervals(start_time: float, end_time: float, interval_len: float) -> np.ndarray:
    """Create non-overlapping intervals of length ``interval_len`` spanning a time range.

    Args:
        start_time: start time of the first interval.
        end_time: end time of the last interval.
        interval_len: length of each interval.

    Returns:
        array of shape ``(n_intervals, 2)`` containing the start and end times of the
        intervals.
    """
    interval_begs = np.arange(start_time, end_time - interval_len, interval_len)
    interval_ends = np.arange(start_time + interval_len, end_time, interval_len)
    return np.c_[interval_begs, interval_ends]


def get_spike_data_per_interval(
    times: np.ndarray,
    clusters: np.ndarray,
    interval_begs: np.ndarray,
    interval_ends: np.ndarray,
    interval_len: float,
    binsize: float,
    n_workers: int = os.cpu_count(),
) -> np.ndarray:
    """Bin spike counts into sub-bins within each interval, in parallel across intervals.

    Args:
        times: spike times.
        clusters: per-spike cluster id, same length as ``times``.
        interval_begs: interval start times.
        interval_ends: interval end times.
        interval_len: length of each interval (used to size the bin grid).
        binsize: width of each sub-bin within an interval.
        n_workers: number of worker processes.

    Returns:
        array of shape ``(n_intervals, n_clusters, n_bins)`` of binned spike counts.
    """
    n_intervals = len(interval_begs)

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    cluster_ids = np.unique(clusters)
    n_clusters_in_region = len(cluster_ids)

    @globalize
    def compute_spike_count(
        interval: tuple[int, float, float],
    ) -> tuple[np.ndarray, np.ndarray, int]:
        interval_idx, t_beg, t_end = interval
        idxs_t = (times >= t_beg) & (times < t_end)
        times_curr = times[idxs_t]
        clust_curr = clusters[idxs_t]
        if times_curr.shape[0] == 0:
            # no spikes in this trial
            binned_spikes_tmp = np.zeros((n_clusters_in_region, n_bins))
            idxs_tmp = np.arange(n_clusters_in_region)
        else:
            # bin spikes
            binned_spikes_tmp, _, cluster_idxs = bincount2D(
                times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end],
            )
            # find indices of clusters that returned spikes for this trial
            _, idxs_tmp, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)
        return binned_spikes_tmp[:, :n_bins], idxs_tmp, interval_idx

    binned_spikes = np.zeros((n_intervals, n_clusters_in_region, n_bins))
    with multiprocessing.Pool(processes=n_workers) as p:
        intervals = list(zip(np.arange(n_intervals), interval_begs, interval_ends))
        with tqdm(total=len(intervals)) as pbar:
            for res in p.imap_unordered(compute_spike_count, intervals):
                pbar.update()
                binned_spikes[res[-1], res[1], :] += res[0]
        pbar.close()
    return binned_spikes


def bin_spiking_data(
    reg_clu_ids: np.ndarray,
    neural_df: dict,
    intervals: np.ndarray | None = None,
    trials_df: pd.DataFrame | None = None,
    n_workers: int = os.cpu_count(),
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin spike data for a given region of interest.

    Args:
        reg_clu_ids: array of cluster ids for the region of interest.
        neural_df: dict with ``'spike_times'``/``'spike_clusters'`` arrays.
        intervals: array of shape ``(n_intervals, 2)`` of interval start/end times; required
            when ``trials_df`` is ``None``.
        trials_df: trials dataframe; when given, intervals are derived from
            ``kwargs['align_time']``/``kwargs['time_window']`` instead.
        n_workers: number of workers to use for parallel processing.
        **kwargs: ``binsize`` (sub-bin width) and, when ``trials_df`` is given,
            ``align_time``/``time_window``.

    Returns:
        ``(binned_spikes, clusters_used_in_bins)`` — ``binned_spikes`` has shape
        ``(n_intervals, n_clusters, n_bins)``.
    """
    if trials_df is not None:
        intervals = np.vstack([
            trials_df[kwargs['align_time']] + kwargs['time_window'][0],
            trials_df[kwargs['align_time']] + kwargs['time_window'][1],
        ]).T
        chunk_len = kwargs['time_window'][1] - kwargs['time_window'][0]
    else:
        assert intervals is not None, (
            'require intervals to segment the recording into chunks including trials and '
            'non-trials'
        )
        chunk_len = intervals[0, 1] - intervals[0, 0]

    # subselect spikes for this region
    spikemask = np.isin(neural_df['spike_clusters'], reg_clu_ids)
    regspikes = neural_df['spike_times'][spikemask]
    regclu = neural_df['spike_clusters'][spikemask]
    clusters_used_in_bins = np.unique(regclu)
    binsize = kwargs.get('binsize', chunk_len)

    if chunk_len / binsize == 1.0:
        # one vector of neural activity per interval
        binned, _ = get_spike_counts_in_bins(regspikes, regclu, intervals)
        binned = binned.T  # binned is a 2D array
        binned_list = [x[None, :] for x in binned]
    else:
        binned_array = get_spike_data_per_interval(
            regspikes, regclu,
            interval_begs=intervals[:, 0],
            interval_ends=intervals[:, 1],
            interval_len=intervals[0, 1] - intervals[0, 0],
            binsize=kwargs['binsize'],
            n_workers=n_workers,
        )
        binned_list = [x.T for x in binned_array]
    return np.array(binned_list), clusters_used_in_bins


def load_target_behavior(one: ONE, eid: str, target: str) -> dict:
    """Load one continuous behavior signal for a session.

    Args:
        one: authenticated ``ONE`` client.
        eid: session UUID.
        target: behavior name; one of ``DYNAMIC_VARS``.

    Returns:
        dict with ``'times'``, ``'values'``, and ``'skip'`` (``True`` if loading failed).
    """
    sess_loader = SessionLoader(one=one, eid=eid)

    try:
        if target == 'wheel-speed':
            sess_loader.load_wheel()
            beh_dict = {
                'times': sess_loader.wheel['times'].to_numpy(),
                'values': np.abs(sess_loader.wheel['velocity'].to_numpy()),
                'skip': False,
            }
        elif target == 'licks':
            licks = one.load_object(eid, 'licks', collection='alf')
            beh_dict = {'times': licks['times'], 'values': None, 'skip': False}
        elif target in _WHISKER_MOTION_ENERGY:
            view, camera = _WHISKER_MOTION_ENERGY[target]
            sess_loader.load_motion_energy(views=[view])
            beh_dict = {
                'times': sess_loader.motion_energy[camera]['times'].to_numpy(),
                'values': sess_loader.motion_energy[camera]['whiskerMotionEnergy'].to_numpy(),
                'skip': False,
            }
        elif target in _NOSE_SPEED:
            view, camera = _NOSE_SPEED[target]
            video_features = one.load_object(eid, camera, collection='alf')
            lp = likelihood_threshold(video_features['lightningPose'], threshold=0.9)
            lp_times = video_features['times']
            values = get_speed(lp, lp_times, view, feature='nose_tip')
            beh_dict = {'times': lp_times, 'values': values, 'skip': False}
        elif target in _PAW_SPEED:
            view, camera = _PAW_SPEED[target]
            video_features = one.load_object(eid, camera, collection='alf')
            lp = likelihood_threshold(video_features['lightningPose'], threshold=0.9)
            lp_times = video_features['times']
            paw_speed_l = get_speed(lp, lp_times, view, feature='paw_l')
            paw_speed_r = get_speed(lp, lp_times, view, feature='paw_r')
            values = np.stack([paw_speed_l, paw_speed_r], axis=1)
            beh_dict = {'times': lp_times, 'values': values, 'skip': False}
        else:
            raise NotImplementedError(f'unknown behavior target {target!r}')

    except BaseException as e:
        _logger.info(f'error loading {target} data: {e}')
        beh_dict = {'times': None, 'values': None, 'skip': True}

    return beh_dict


def get_behavior_per_interval(
    target_times: np.ndarray,
    target_vals: np.ndarray,
    intervals: np.ndarray | None = None,
    trials_df: pd.DataFrame | None = None,
    allow_nans: bool = False,
    n_workers: int = os.cpu_count(),
    **kwargs: Any,
) -> tuple[list, list, np.ndarray, list]:
    """Interpolate a continuous behavior signal onto a fixed-rate bin grid per interval.

    Args:
        target_times: behavior sample times.
        target_vals: behavior sample values, same length as ``target_times``.
        intervals: array of shape ``(n_intervals, 2)`` of interval start/end times; required
            when ``trials_df`` is ``None``.
        trials_df: trials dataframe; when given, intervals are derived from
            ``kwargs['align_time']``/``kwargs['time_window']`` instead.
        allow_nans: allow NaNs in the interpolated output.
        n_workers: number of workers to use for parallel processing.
        **kwargs: ``binsize`` and ``interval_len``, and (with ``trials_df``)
            ``align_time``/``time_window``.

    Returns:
        ``(target_times_list, target_vals_list, good_interval, skip_reasons)`` — per-interval
        interpolated times/values, a boolean-ish array of which intervals succeeded, and the
        reason each failing interval was skipped.
    """
    binsize = kwargs['binsize']
    interval_len = kwargs['interval_len']

    if trials_df is not None:
        align_event = kwargs['align_time']
        align_interval = kwargs['time_window']
        interval_len = align_interval[1] - align_interval[0]
        align_times = trials_df[align_event].values
        interval_begs = align_times + align_interval[0]
        interval_ends = align_times + align_interval[1]
    else:
        assert intervals is not None, (
            'require intervals to segment the recording into chunks including trials and '
            'non-trials'
        )
        interval_begs, interval_ends = intervals.T

    n_intervals = len(interval_begs)

    if np.all(np.isnan(interval_begs)) or np.all(np.isnan(interval_ends)):
        _logger.info('interval times all nan')
        good_interval = np.nan * np.ones(interval_begs.shape[0])
        return [], [], good_interval, []

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    # split data into intervals
    idxs_beg = np.searchsorted(target_times, interval_begs, side='right')
    idxs_end = np.searchsorted(target_times, interval_ends, side='left')
    target_times_og_list = [target_times[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]
    target_vals_og_list = [target_vals[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]

    target_times_list: list = [None] * len(target_times_og_list)
    target_vals_list: list = [None] * len(target_times_og_list)
    good_interval: list = [None] * len(target_times_og_list)
    skip_reasons: list = [None] * len(target_times_og_list)

    @globalize
    def interpolate_behavior(target: tuple[int, np.ndarray, np.ndarray]) -> tuple:
        # we use interval_idx to track the interval order while working with p.imap_unordered()
        interval_idx, target_time, target_v = target

        is_good_interval, x_interp, y_interp = False, None, None

        if len(target_v) == 0:
            return interval_idx, is_good_interval, x_interp, y_interp, 'target data not present'
        if np.sum(np.isnan(target_v)) > 0 and not allow_nans:
            return interval_idx, is_good_interval, x_interp, y_interp, 'nans in target data'
        if np.isnan(interval_begs[interval_idx]) or np.isnan(interval_ends[interval_idx]):
            return interval_idx, is_good_interval, x_interp, y_interp, 'bad interval data'
        if np.abs(interval_begs[interval_idx] - target_time[0]) > binsize:
            return (
                interval_idx, is_good_interval, x_interp, y_interp,
                'target data starts too late',
            )
        if np.abs(interval_ends[interval_idx] - target_time[-1]) > binsize:
            return interval_idx, is_good_interval, x_interp, y_interp, 'target data ends too early'

        is_good_interval, skip_reason = True, None
        x_interp = np.linspace(
            interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins,
        )
        if len(target_v.shape) > 1 and target_v.shape[1] > 1:
            n_dims = target_v.shape[1]
            y_interp_tmps = [
                interp1d(
                    target_time, target_v[:, n], kind='linear', fill_value='extrapolate',
                )(x_interp)
                for n in range(n_dims)
            ]
            y_interp = np.hstack([y[:, None] for y in y_interp_tmps])
        else:
            y_interp = interp1d(
                target_time, target_v, kind='linear', fill_value='extrapolate',
            )(x_interp)
        return interval_idx, is_good_interval, x_interp, y_interp, skip_reason

    with multiprocessing.Pool(processes=n_workers) as p:
        targets = list(zip(np.arange(n_intervals), target_times_og_list, target_vals_og_list))
        with tqdm(total=n_intervals) as pbar:
            for res in p.imap_unordered(interpolate_behavior, targets):
                pbar.update()
                good_interval[res[0]] = res[1]
                target_times_list[res[0]] = res[2]
                target_vals_list[res[0]] = res[3]
                skip_reasons[res[0]] = res[-1]
        pbar.close()
    return target_times_list, target_vals_list, np.array(good_interval), skip_reasons


def load_anytime_behaviors(one: ONE, eid: str, n_workers: int = os.cpu_count()) -> dict:
    """Load all ``DYNAMIC_VARS`` behaviors for a session, in parallel.

    Args:
        one: authenticated ``ONE`` client.
        eid: session UUID.
        n_workers: number of workers to use for parallel processing.

    Returns:
        dict mapping behavior name to its ``load_target_behavior`` result.
    """
    @globalize
    def load_beh(beh: str) -> tuple[str, dict]:
        return beh, load_target_behavior(one, eid, beh)

    behave_dict = {}
    with multiprocessing.Pool(processes=n_workers) as p:
        with tqdm(total=len(DYNAMIC_VARS)) as pbar:
            for res in p.imap_unordered(load_beh, DYNAMIC_VARS):
                pbar.update()
                behave_dict[res[0]] = res[1]
        pbar.close()
    return behave_dict


def bin_behaviors(
    one: ONE,
    eid: str,
    behaviors: list[str],
    intervals: np.ndarray | None = None,
    trials_df: pd.DataFrame | None = None,
    mask: np.ndarray | None = None,
    allow_nans: bool = True,
    n_workers: int = os.cpu_count(),
    **kwargs: Any,
) -> tuple[dict, dict]:
    """Bin continuous behaviors (and, if ``trials_df`` is given, discrete trial variables).

    Args:
        one: authenticated ``ONE`` client.
        eid: session UUID.
        behaviors: behavior names to bin, e.g. ``DYNAMIC_VARS``.
        intervals: array of shape ``(n_intervals, 2)`` of interval start/end times; required
            when ``trials_df`` is ``None``.
        trials_df: trials dataframe; when given, discrete per-trial variables (choice, block,
            reward, contrast) are also added.
        mask: boolean row mask applied to ``trials_df`` before binning.
        allow_nans: allow NaNs in the binned output.
        n_workers: number of workers to use for parallel processing.
        **kwargs: forwarded to ``get_behavior_per_interval`` (``binsize``, ``interval_len``,
            and, with ``trials_df``, ``align_time``/``time_window``).

    Returns:
        ``(behave_dict, mask_dict)`` — binned behavior arrays and their per-behavior masks.
    """
    behave_dict: dict = {}
    mask_dict: dict = {}

    if mask is not None:
        trials_df = trials_df[mask]

    if trials_df is not None:
        choice = trials_df['choice'].to_numpy()
        block = trials_df['probabilityLeft'].to_numpy()
        reward = (trials_df['rewardVolume'] > 1).astype(int).to_numpy()
        contrast = np.c_[trials_df['contrastLeft'], trials_df['contrastRight']]
        contrast = (-1 * np.nan_to_num(contrast, 0)).sum(1)

        behave_dict.update(
            {'choice': choice, 'block': block, 'reward': reward, 'contrast': contrast},
        )
        behave_mask = np.ones(len(trials_df))
    else:
        assert intervals is not None, (
            'require intervals to segment the recording into chunks including trials and '
            'non-trials'
        )
        behave_mask = np.ones(len(intervals))

    for beh in behaviors:
        target_dict = load_target_behavior(one, eid, beh)

        if target_dict['skip']:
            warnings.warn(f'behavior {beh} not found; skipping.', UserWarning, stacklevel=2)
            continue

        target_times, target_vals = target_dict['times'], target_dict['values']
        if beh == 'licks':
            target_vals_list = get_spike_data_per_interval(
                target_times, np.ones_like(target_times),
                interval_begs=intervals[:, 0],
                interval_ends=intervals[:, 1],
                interval_len=intervals[0, 1] - intervals[0, 0],
                binsize=kwargs['binsize'],
                n_workers=n_workers,
            )
            target_vals_list = [x.T for x in target_vals_list]
            target_mask = np.ones(len(intervals))
        else:
            if target_vals is None:
                warnings.warn(f'behavior {beh} not found; skipping.', UserWarning, stacklevel=2)
                continue
            _, target_vals_list, target_mask, _ = get_behavior_per_interval(
                target_times, target_vals, intervals=intervals,
                trials_df=trials_df, allow_nans=allow_nans, n_workers=n_workers, **kwargs,
            )
        behave_dict[beh] = np.array(target_vals_list, dtype=object)
        mask_dict[beh] = target_mask
        behave_mask = np.logical_and(behave_mask, target_mask)

    if not allow_nans:
        for k in behave_dict:
            behave_dict[k] = behave_dict[k][behave_mask]

    return behave_dict, mask_dict


def prepare_data(
    one: ONE,
    eid: str,
    params: dict,
    n_workers: int = os.cpu_count(),
) -> tuple[dict | None, dict | None, dict | None, dict | None, pd.Series | None]:
    """Load and merge spikes, behaviors, metadata, and trial info for one session.

    Args:
        one: authenticated ``ONE`` client.
        eid: session UUID.
        params: extraction params; only ``n_workers`` for nested behavior loading is read
            directly here, the rest is forwarded implicitly via closures.
        n_workers: number of workers to use for parallel processing.

    Returns:
        ``(neural_dict, behave_dict, meta_data, trials_data, good_trials_mask)`` — all
        ``None`` if the session has no spike data on any probe.
    """
    del params
    pids, probe_names = one.eid2pid(eid)
    details = one.get_details(eid)
    _logger.info(f'merge {len(probe_names)} probes for session EID: {eid}')

    clusters_list = []
    spikes_list = []
    for pid, probe_name in zip(pids, probe_names):
        tmp_spikes, tmp_clusters, sampling_freq = load_spiking_data(
            one, pid, eid=eid, pname=probe_name,
        )
        if tmp_spikes is None:
            return None, None, None, None, None
        tmp_clusters['pid'] = pid
        spikes_list.append(tmp_spikes)
        clusters_list.append(tmp_clusters)
    spikes, clusters = merge_probes(spikes_list, clusters_list)

    _, good_trials_mask = load_trials_and_mask(one=one, eid=eid)
    trials_df, trials_mask = load_trials_and_mask(one=one, eid=eid, min_rt=0.0, max_rt=10.0)

    behave_dict = load_anytime_behaviors(one, eid, n_workers=n_workers)

    neural_dict = {
        'spike_times': spikes['times'],
        'spike_clusters': spikes['clusters'],
        'cluster_regions': clusters['acronym'].to_numpy(),
    }

    meta_data = {
        'eid': eid,
        'subject': details['subject'],
        'lab': details['lab'],
        'sampling_freq': sampling_freq,
        'cluster_channels': list(clusters['channels']),
        'cluster_regions': list(clusters['acronym']),
        'good_clusters': list((clusters['label'] >= 1).astype(int)),
        'cluster_depths': list(clusters['depths']),
        'uuids': list(clusters['uuids']),
    }

    trials_data = {'trials_df': trials_df, 'trials_mask': trials_mask}
    return neural_dict, behave_dict, meta_data, trials_data, good_trials_mask


def align_data(
    binned_spikes: np.ndarray,
    binned_behaviors: dict,
    beh_names: list[str] | None = None,
    trials_mask: pd.Series | None = None,
    nan_thresh: float = 0.3,
) -> tuple[np.ndarray, dict, list, np.ndarray]:
    """Drop trials with missing behaviors/spikes and align spikes with behaviors.

    Args:
        binned_spikes: array of shape ``(n_trials, ...)`` of binned spike counts.
        binned_behaviors: dict mapping behavior name to its per-trial binned values.
        beh_names: behaviors to keep in the output; a behavior with too many NaN trials
            (over ``nan_thresh``) is dropped from this list. Defaults to ``DYNAMIC_VARS``.
        trials_mask: optional additional per-trial boolean mask (e.g. well-behaved trials).
        nan_thresh: drop a behavior entirely if more than this fraction of its trials are
            NaN/missing.

    Returns:
        ``(aligned_binned_spikes, aligned_binned_behaviors, target_mask, bad_trial_idxs)``.
    """
    beh_names = list(DYNAMIC_VARS) if beh_names is None else list(beh_names)
    num_trials = len(binned_spikes)

    target_mask = [1] * num_trials
    for beh in list(binned_behaviors.keys()):
        beh_mask = [
            1 if (x is not None or not np.isnan(x).any()) else 0 for x in binned_behaviors[beh]
        ]
        nan_ratio = 1 - sum(beh_mask) / num_trials
        _logger.info(f'{beh} has {nan_ratio * 100.0}% NaN trials.')
        if nan_ratio >= nan_thresh:
            beh_names.remove(beh)
            _logger.info(f'remove {beh} due to too many NaN trials!')
        else:
            target_mask = target_mask and beh_mask

    if trials_mask is not None:
        trials_mask = list(trials_mask.to_numpy().astype(int))
        target_mask = target_mask and trials_mask

    bad_trial_idxs = np.argwhere(np.array(target_mask) == 0)
    aligned_binned_spikes = np.delete(binned_spikes, bad_trial_idxs, axis=0)

    num_trials = len(aligned_binned_spikes)
    aligned_binned_behaviors = {}
    for beh in beh_names:
        beh_vals = binned_behaviors[beh]
        n_tbin = beh_vals.shape[1]
        aligned_binned_behaviors[beh] = np.delete(beh_vals, bad_trial_idxs, axis=0)
        aligned_binned_behaviors[beh] = np.array(
            [y for y in aligned_binned_behaviors[beh]], dtype=float,
        ).reshape((num_trials, n_tbin, -1))
        if beh in DYNAMIC_VARS and beh != 'licks':
            beh_of_interest = aligned_binned_behaviors[beh]
            for dim in range(beh_of_interest.shape[-1]):
                top = beh_of_interest[..., dim] - np.min(beh_of_interest[..., dim])
                bottom = np.max(beh_of_interest[..., dim]) - np.min(beh_of_interest[..., dim])
                aligned_binned_behaviors[beh][..., dim] = top / bottom

    return aligned_binned_spikes, aligned_binned_behaviors, target_mask, bad_trial_idxs
