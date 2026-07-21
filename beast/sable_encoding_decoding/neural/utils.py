"""Shared utilities for neural encoding/decoding: seeding, metrics, plotting, CLI args.

note: `neg_log_likelihood` called an undefined `logger` in the source repo (no `logger`
import existed there) — a latent bug that would have raised `NameError` the first time a
zero rate prediction occurred. Fixed here by using stdlib `warnings.warn` instead.
"""

import argparse
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import gammaln
from torcheval.metrics import R2Score

_VALID_LATENT_KIND_FIXED = frozenset(('frame', 'mu_s', 'psae', 'dino', 'cat', 'mu_u'))


def _parse_latent_kind(value: str) -> str:
    """Validate the `--latent_kind` CLI argument.

    Args:
        value: raw string passed on the command line.

    Returns:
        The validated latent kind string, unchanged.

    Raises:
        argparse.ArgumentTypeError: if `value` is not one of the fixed latent kinds
            and does not start with `'img_tokens_compressed'`.
    """
    if value in _VALID_LATENT_KIND_FIXED:
        return value
    if value.startswith('img_tokens_compressed'):
        return value
    fixed = ', '.join(sorted(_VALID_LATENT_KIND_FIXED))
    raise argparse.ArgumentTypeError(
        f'invalid --latent_kind {value!r}: expected one of {{{fixed}}} '
        "or any string starting with 'img_tokens_compressed' "
        '(e.g. img_tokens_compressed_3_comp; CNN-only in run_encoding_decoding.py).',
    )


def get_encoding_decoding_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the neural encoding/decoding entry point.

    Args:
        argv: argument list to parse, e.g. `sys.argv[1:]`. If `None`, `argparse` falls
            back to reading `sys.argv` itself.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description='Neural Encoding or Decoding')
    parser.add_argument(
        '--eid', type=str, required=True, help='Session / animal id (subfolder name)',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--neural_input_dir', type=str, default='input', help='Input directory for neural data',
    )
    parser.add_argument(
        '--latent_input_dir',
        type=str,
        default='input',
        help='Input directory for video latents',
    )
    parser.add_argument(
        '--latent_kind',
        type=_parse_latent_kind,
        default=None,
        help='Latent layout under --latent_input_dir (matches src/inference.py '
        '--return-combined-*-z): '
        'frame → frame_z/<eid>/frame_z_trials.npz; '
        'mu_s → pose_mu_s_z/<eid>/pose_mu_s_z_trials.npz (--return-combined-mu-s-z); '
        'psae → psae_z/<eid>/psae_z_trials.npz full concat z '
        '(--return-combined-psae-z; requires --model_config); '
        'dino → dino_z/<eid>/dino_z_trials.npz; '
        'cat → cat_z/<eid>/cat_z_trials.npz; '
        'mu_u → same npz as psae, sliced to unsupervised tail (requires --model_config); '
        'any img_tokens_compressed* (e.g. img_tokens_compressed, img_tokens_compressed_3_comp) → '
        '<latent_kind>/<eid>/img_tokens_compressed_trials.npz (PCA outputs; '
        'run_encoding_decoding.py uses CNN Ray Tune only, no RRR). '
        'Default (omit): <latent_input_dir>/<eid>/z_trials.npz.',
    )
    parser.add_argument(
        '--model_config',
        type=str,
        default=None,
        help='Training/inference YAML. Required for --latent_kind psae or mu_u: reads '
        'num_latents and latent_partition.dim_supervised (default 6) for shape check; '
        'mu_u additionally slices z[..., dim_supervised:].',
    )
    parser.add_argument(
        '--eval_task', type=str, default='encoding', help='Evaluation task: encoding or decoding',
    )
    parser.add_argument(
        '--result_name',
        type=str,
        default=None,
        help='Basename for the saved .npy under the latent session dir (extension optional; '
        '.npy is added by numpy.save). Default: encoding_results or decoding_results from '
        '--eval_task, with _<latent_kind> appended when --latent_kind is set, or inferred '
        'from the layout (e.g. .../dino_z/<eid>/ → encoding_results_dino).',
    )
    parser.add_argument(
        '--tune_storage_path',
        type=str,
        default=None,
        help='Ray Tune experiment root; trial logs go under this directory. '
        'If --latent_kind is set and this is omitted, defaults to '
        '<latent_input_dir>/<layout subdir>: frame_z, pose_mu_s_z, psae_z, dino_z, cat_z, '
        'or for img_tokens_compressed* the same name as --latent_kind. If --latent_kind is '
        'omitted, Ray uses its usual default (e.g. ~/ray_results) when this is unset.',
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducibility.

    Args:
        seed: seed value applied to Python, numpy, and torch RNGs.
    """
    # PYTHONHASHSEED must be set before the interpreter starts to fully take effect, but
    # setting it here still helps downstream subprocesses that inherit the environment.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'seed set to {seed}')


r2_metric = R2Score()


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor, device: str = 'cpu') -> float:
    """Compute the R2 score between predictions and ground truth.

    Args:
        y_true: ground-truth tensor.
        y_pred: predicted tensor, same shape as `y_true`.
        device: device to run the metric computation on.

    Returns:
        Scalar R2 score.
    """
    r2_metric.reset()
    r2_metric.to(device)
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    r2_metric.update(y_pred, y_true)
    return r2_metric.compute().item()


def neg_log_likelihood(
    rates: np.ndarray,
    spikes: np.ndarray,
    zero_warning: bool = True,
    threshold: float = 1e-9,
) -> float:
    """Calculate the Poisson negative log likelihood given rates and spikes.

    formula: -log(e^(-r) / n! * r^n) = r - n*log(r) + log(n!)

    Args:
        rates: numpy array containing rate predictions.
        spikes: numpy array containing true spike counts, same shape as `rates`.
        zero_warning: whether to warn about zero rate predictions.
        threshold: value used to replace zero rate predictions before taking `log`.

    Returns:
        Total negative log-likelihood of the data.
    """
    assert spikes.shape == rates.shape, (
        f'neg_log_likelihood: Rates and spikes should be of the same shape. '
        f'spikes: {spikes.shape}, rates: {rates.shape}'
    )

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]

    assert not np.any(np.isnan(rates)), 'neg_log_likelihood: NaN rate predictions found'

    assert np.all(rates >= 0), 'neg_log_likelihood: Negative rate predictions found'
    if np.any(rates == 0):
        if zero_warning:
            warnings.warn(
                'neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9',
                stacklevel=2,
            )
        rates[rates == 0] = threshold
    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result)


def bits_per_spike(rates: np.ndarray, spikes: np.ndarray, threshold: float = 1e-9) -> float:
    """Compute bits per spike of rate predictions given spikes.

    Bits per spike is the difference between the log-likelihoods (in base 2) of the rate
    predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Args:
        rates: 3d numpy array containing rate predictions.
        spikes: 3d numpy array containing true spike counts.
        threshold: value used to replace zero rate predictions.

    Returns:
        Bits per spike of rate predictions.
    """
    nll_model = neg_log_likelihood(rates, spikes, threshold=threshold)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, threshold=threshold)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)


def plot_gt_pred(gt: np.ndarray, pred: np.ndarray, epoch: int = 0, modality: str = 'behavior'):
    """Plot ground truth and predictions side by side.

    Args:
        gt: ground-truth 2d array (e.g. time x neuron/behavior).
        pred: predicted 2d array, same shape as `gt`.
        epoch: training epoch, used only in the figure title.
        modality: label for the plotted signal (e.g. 'behavior' or 'spikes').

    Returns:
        The created `matplotlib.figure.Figure`.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title('Ground Truth')
    im1 = ax1.imshow(gt, aspect='auto', cmap='binary')

    ax2.set_title('Prediction')
    im2 = ax2.imshow(pred, aspect='auto', cmap='binary')

    # add colorbar
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)

    fig.suptitle(f'Epoch: {epoch}, Mod: {modality}')
    return fig


def plot_neurons_r2(
    gt: torch.Tensor,
    pred: torch.Tensor,
    epoch: int = 0,
    neuron_idx: list[int] | None = None,
    modality: str = 'behavior',
):
    """Plot ground truth vs. prediction traces per neuron, annotated with each R2 score.

    Args:
        gt: ground-truth tensor of shape `[T, N]`.
        pred: predicted tensor of shape `[T, N]`.
        epoch: training epoch, used only in the figure title.
        neuron_idx: indices of neurons to plot; defaults to an empty list.
        modality: label for the plotted signal (e.g. 'behavior' or 'spikes').

    Returns:
        The created `matplotlib.figure.Figure`.
    """
    if neuron_idx is None:
        neuron_idx = []
    fig, axes = plt.subplots(len(neuron_idx), 1, figsize=(12, 5 * len(neuron_idx)))
    r2_values = []
    for neuron in neuron_idx:
        r2 = r2_score(y_true=gt[:, neuron], y_pred=pred[:, neuron])
        r2_values.append(r2)
        ax = axes if len(neuron_idx) == 1 else axes[neuron_idx.index(neuron)]
        ax.plot(gt[:, neuron].cpu().numpy(), label='Ground Truth', color='blue')
        ax.plot(pred[:, neuron].cpu().numpy(), label='Prediction', color='red')
        ax.set_title(f'Neuron: {neuron}, R2: {r2:.4f}')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Rate')
    fig.suptitle(f'Epoch: {epoch}, Mod: {modality}, Avg R2: {np.mean(r2_values):.4f}')
    return fig


def _std(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score `arr` over its leading two axes (trial, time), keeping the last axis.

    Args:
        arr: array of shape `[K, T, N]`.

    Returns:
        A tuple `(normalized_arr, mean, std)` where `mean` and `std` have shape `[1, N]`.
    """
    mean = np.nanmean(arr, axis=(0, 1), keepdims=False).reshape(1, -1)
    std = np.nanstd(arr, axis=(0, 1), keepdims=False).reshape(1, -1)
    std = np.clip(std, 1e-8, None)
    arr = (arr - mean) / std
    return arr, mean, std
