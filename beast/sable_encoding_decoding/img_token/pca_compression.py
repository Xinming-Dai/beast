"""PCA compression / decompression of per-patch DINO image tokens.

Pipeline: normalize image tokens Z -> PCA (e.g. 3 components) on the feature dim ->
img_token_compressed -> neural (TCN/Keypoints) decoding with Ray Tune -> un-PCA + denorm -> Z_pred
-> E-RayZer image-token decoder + optional upsampling.

Tensor shapes use L for the token dimension (e.g. two-view setups may use L = 2 * N patch tokens;
single-view data can use L = N). Z has shape (K, T, L, D) with D = 768.
"""

import pickle
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------- #
# PCA and normalization
# --------------------------------------------------------------------------- #


def fit_pca(
    x: np.ndarray,
    n_components: int | None = None,
    random_state: int = 777,
) -> PCA:
    """Fit a sklearn PCA model on 2D data.

    Args:
        x: array of shape `(n_samples, n_features)`.
        n_components: number of components to keep; `None` keeps all `min(n_samples, n_features)`.
        random_state: PCA solver random seed.

    Returns:
        The fitted `PCA` instance.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(x)
    return pca


def pca_project_topk(x: np.ndarray, pca: PCA, k: int) -> np.ndarray:
    """Project `x` onto the first `k` PCA components.

    Args:
        x: array of shape `(N, P)`.
        pca: fitted PCA model.
        k: number of leading components to keep.

    Returns:
        Array of shape `(N, k)`.
    """
    scores = pca.transform(x)
    return scores[:, :k]


def plot_pca_cev(
    pca: PCA, title: str = 'PCA Explained Variance Curve', figsize: tuple[int, int] = (12, 3),
) -> None:
    """Plot the cumulative explained-variance curve for a fitted PCA model.

    Args:
        pca: fitted PCA model.
        title: plot title.
        figsize: matplotlib figure size `(width, height)`.
    """
    evr = pca.explained_variance_ratio_
    cev = np.cumsum(evr)
    plt.figure(figsize=figsize)
    plt.plot(cev, marker='.', markersize=1)
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title(title)
    plt.grid(True)
    plt.show()


def pca_encode_decode(
    x: np.ndarray, n_keep: int, *, pca: PCA | None = None,
) -> tuple[PCA, np.ndarray, np.ndarray]:
    """Encode `x` with PCA and reconstruct it using the top `n_keep` components.

    Args:
        x: array of shape `(N, P)`.
        n_keep: number of leading components to keep for reconstruction.
        pca: pre-fitted PCA model; if `None`, a new one is fit on `x` with all components.

    Returns:
        Tuple `(pca, scores_keep, x_hat)` where `scores_keep` has shape `(N, n_keep)` and `x_hat`
        `(N, P)` is the reconstruction `mean + scores_keep @ components_keep`, in the same
        centered space as `pca` (matches the normalized input when PCA is fit on normalized data).
    """
    if pca is None:
        pca = fit_pca(x, n_components=None)
    k = min(n_keep, pca.components_.shape[0], x.shape[1])
    scores = pca.transform(x)
    scores_keep = scores[:, :k]
    comps_keep = pca.components_[:k, :]
    mean = pca.mean_
    x_hat = mean + scores_keep @ comps_keep
    return pca, scores_keep, x_hat


def normalize_data(
    x: np.ndarray,
    axes: tuple[int, ...] = (0, 1),
    eps: float = 1e-6,
    chunk_axis: int = 0,
    chunk_size: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center `x` over `axes` (default trial, time) without whitening.

    Uses a chunked implementation to avoid allocating two full-size intermediate arrays
    (`x - mean` and `(x - mean) ** 2`) simultaneously with `x`. For a 38 GiB array the naive
    implementation peaks at ~115 GiB; this stays near 2x the array size (~77 GiB) by processing
    `chunk_size` slices along `chunk_axis` at a time.

    Args:
        x: input array, e.g. shape `(K, T, L, D)`.
        axes: axes to average/center over (kept via `keepdims`).
        eps: small constant added to `std` before dividing.
        chunk_axis: axis along which `x` is processed in chunks.
        chunk_size: number of slices per chunk along `chunk_axis`.

    Returns:
        Tuple `(x_norm, mean, std)`. `std` is fixed at `1.0` (centering only, no scaling) — kept
        for API compatibility with callers that expect a `(mean, std)` pair.
    """
    mean = np.mean(x, axis=axes, keepdims=True)  # tiny (1,1,L,D)

    # compute variance in chunks to avoid a full-size (x-mean)^2 intermediate.
    n = 1
    for ax in axes:
        n *= x.shape[ax]
    sq_sum = np.zeros_like(mean, dtype=np.float64)
    chunks = range(0, x.shape[chunk_axis], chunk_size)
    idx = [slice(None)] * x.ndim
    for start in chunks:
        idx[chunk_axis] = slice(start, start + chunk_size)
        diff = x[tuple(idx)].astype(np.float64) - mean
        sq_sum += (diff**2).sum(axis=axes, keepdims=True)
    # std left at 1.0 (centralize instead of standardize); sq_sum/n is computed above but unused,
    # matching the upstream behavior this was ported from.
    std = np.ones_like(mean, dtype=np.float32)

    # write normalised output in chunks — pre-allocates once, no extra temp.
    x_norm = np.empty_like(x)
    for start in chunks:
        idx[chunk_axis] = slice(start, start + chunk_size)
        sl = tuple(idx)
        x_norm[sl] = (x[sl] - mean) / (std + eps)
    return x_norm, mean.astype(np.float32), std


def apply_session_norm_fixed(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply `(x - mean) / (std + eps)` using broadcast stats from train `normalize_data`.

    Args:
        x: array to normalize.
        mean: broadcastable mean array from `normalize_data`.
        std: broadcastable std array from `normalize_data`.
        eps: small constant added to `std` before dividing.

    Returns:
        Normalized array, same shape as `x`.
    """
    return (x - mean) / (std + eps)


def feature_pca_encode_decode(
    x: np.ndarray,
    n_feat_keep: int = 3,
    random_state: int = 777,
    *,
    verbose_progress: bool | Callable[[str], None] = False,
) -> tuple[PCA, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize over trial/time, fit PCA on the feature dim, and project to `n_feat_keep` comps.

    Args:
        x: img-token array of shape `(K, T, L, D)`.
        n_feat_keep: number of PCA components to keep (capped at `D`).
        random_state: PCA solver random seed.
        verbose_progress: if `True`, print phase messages to stdout; if callable, call it with
            each message (for logging); if `False`, stay silent.

    Returns:
        Tuple `(pca, x_compressed, x_session_avg, x_session_std, x_hat)`:

        - `pca`: fitted on rows of `x_norm` reshaped to `(K*T*L, D)`, with
          `n_components = min(n_feat_keep, D)`.
        - `x_compressed`: `(K, T, L, n_feat_keep)`.
        - `x_session_avg`, `x_session_std`: broadcast stats from `normalize_data`
          (e.g. `(1, 1, L, D)` for `axes=(0, 1)`).
        - `x_hat`: `(K, T, L, D)` PCA reconstruction in normalized space (for sanity checks).
    """
    log: Callable[[str], None]
    if verbose_progress is True:
        log = lambda m: print(m, flush=True)  # noqa: E731
    elif callable(verbose_progress):
        log = verbose_progress
    else:
        log = lambda _m: None  # noqa: E731

    k, t, n_tok, d = x.shape
    n_feat_keep = min(int(n_feat_keep), d)
    n_bytes = int(x.nbytes)
    log(
        f'[PCA] input z_trials_time shape (K,T,L,D)=({k},{t},{n_tok},{d})  '
        f'~{n_bytes / (1024**3):.2f} GiB (array)',
    )
    log('[PCA] (1/4) Normalizing over trial and time axes (0, 1) …')
    x_session_norm, x_session_avg, x_session_std = normalize_data(x, axes=(0, 1))  # (K, T, L, D)

    x_flat = x_session_norm.reshape(-1, d)
    n_rows, n_feat = x_flat.shape
    log(
        f'[PCA] (2/4) Fitting sklearn PCA on {n_rows} rows x {n_feat} features, '
        f'n_components={n_feat_keep} …',
    )
    pca = fit_pca(x_flat, n_components=n_feat_keep, random_state=random_state)
    evr = pca.explained_variance_ratio_
    log(f'[PCA] (2/4) explained_variance_ratio (sum over kept) = {float(np.sum(evr)):.4f}')

    log('[PCA] (3/4) Transform + reconstruct (encode/decode in normalized space) …')
    pca, x_compressed, x_hat = pca_encode_decode(x_flat, n_feat_keep, pca=pca)

    x_compressed = x_compressed.reshape(k, t, n_tok, n_feat_keep)
    x_hat = x_hat.reshape(k, t, n_tok, d)
    log(f'[PCA] (4/4) Reshaped outputs  x_compressed {x_compressed.shape}')
    return pca, x_compressed, x_session_avg, x_session_std, x_hat


def feature_pca_fit_on_train(
    x_train: np.ndarray,
    n_feat_keep: int,
    random_state: int,
    *,
    verbose_progress: bool | Callable[[str], None] = False,
) -> tuple[PCA, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize **train** trials/time only; fit PCA on flattened normalized rows.

    Args:
        x_train: train-split img-token array of shape `(K, T, L, D)`.
        n_feat_keep: number of PCA components to keep (capped at `D`).
        random_state: PCA solver random seed.
        verbose_progress: if `True`, print phase messages to stdout; if callable, call it with
            each message (for logging); if `False`, stay silent.

    Returns:
        Tuple `(pca, x_compressed, x_session_avg, x_session_std)` where `x_compressed` has shape
        `(K, T, L, k)` with `k = min(n_feat_keep, D)`.
    """
    log: Callable[[str], None]
    if verbose_progress is True:
        log = lambda m: print(m, flush=True)  # noqa: E731
    elif callable(verbose_progress):
        log = verbose_progress
    else:
        log = lambda _m: None  # noqa: E731

    k, t, n_tok, d = x_train.shape
    n_feat_keep = min(int(n_feat_keep), d)
    log(
        f'[PCA/train] shape (K,T,L,D)=({k},{t},{n_tok},{d})  norm axes (0,1)  '
        f'n_components={n_feat_keep}',
    )
    log('[PCA/train] Normalizing train trials/time …')
    x_session_norm, x_session_avg, x_session_std = normalize_data(x_train, axes=(0, 1))

    x_flat = x_session_norm.reshape(-1, d)
    log(f'[PCA/train] Fitting PCA on {x_flat.shape[0]} rows x {d} features …')
    pca = fit_pca(x_flat, n_components=n_feat_keep, random_state=random_state)
    scores = pca.transform(x_flat)
    k_eff = min(n_feat_keep, scores.shape[1])
    scores_keep = scores[:, :k_eff]
    evr = pca.explained_variance_ratio_
    log(f'[PCA/train] explained_variance_ratio (sum over kept) = {float(np.sum(evr[:k_eff])):.4f}')
    x_compressed = scores_keep.reshape(k, t, n_tok, k_eff)
    return pca, x_compressed.astype(np.float32), x_session_avg, x_session_std


def feature_pca_project_train_normalized(
    x: np.ndarray,
    x_session_avg: np.ndarray,
    x_session_std: np.ndarray,
    pca: PCA,
    n_feat_keep: int,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize `x` with **train** mean/std, then apply the fitted PCA transform.

    Args:
        x: array of shape `(K, T, L, D)`.
        x_session_avg: train-split mean from `normalize_data` / `feature_pca_fit_on_train`.
        x_session_std: train-split std from `normalize_data` / `feature_pca_fit_on_train`.
        pca: PCA model fitted on train data.
        n_feat_keep: number of leading components to keep.
        eps: small constant added to `x_session_std` before dividing.

    Returns:
        Array of shape `(K, T, L, k)` with `k = min(n_feat_keep, D, pca.components_.shape[0])`.
    """
    k, t, n_tok, d = x.shape
    k_cap = min(int(n_feat_keep), d, pca.components_.shape[0])
    x_norm = apply_session_norm_fixed(x, x_session_avg, x_session_std, eps=eps)
    x_flat = x_norm.reshape(-1, d)
    scores = pca.transform(x_flat)
    scores_keep = scores[:, :k_cap]
    return scores_keep.reshape(k, t, n_tok, k_cap).astype(np.float32)


def pca_unproject(
    pca: PCA,
    x_compressed: np.ndarray,
    x_session_avg: np.ndarray,
    x_session_std: np.ndarray,
    *,
    n_pca_components: int | None = None,
) -> np.ndarray:
    """Map compressed PCA codes back to the latent dim `D`, then denormalize.

    If `x_compressed` was predicted by a neural decoder, this maps it back to the original video
    embedding space.

    Args:
        x_compressed: array of shape `(K, T, L, k_comp)` or `(K*T*L, k_comp)`.
        pca: PCA model used to compress `x_compressed`.
        x_session_avg: broadcast mean from `normalize_data` (same shape convention as for the
            original `(K, T, L, D)` img tokens).
        x_session_std: broadcast std from `normalize_data`.
        n_pca_components: number of PCA components to use for reconstruction; defaults to all of
            `pca.components_`.

    Returns:
        Reconstructed, denormalized array: `(K, T, L, D)` if `x_compressed` was 4D, else
        `(N, D)`.

    Raises:
        ValueError: if `x_compressed`'s last dim does not match the resolved number of PCA
            components, or if `x_compressed` is neither 2D nor 4D.
    """
    n_comp = pca.components_.shape[0] if n_pca_components is None else n_pca_components
    n_comp = min(n_comp, pca.components_.shape[0])
    if x_compressed.ndim == 4:
        k0, t0, l0, kc = x_compressed.shape
        if kc != n_comp:
            raise ValueError(f'x_compressed last dim {kc} != PCA components {n_comp}')
        x_compressed_2d = x_compressed.reshape(-1, n_comp)
        lead = (k0, t0, l0)
    elif x_compressed.ndim == 2:
        x_compressed_2d = x_compressed
        lead = None
    else:
        raise ValueError(f'Expected 2D or 4D x_compressed, got {x_compressed.shape}')
    d = pca.components_.shape[1]
    comps = pca.components_[:n_comp, :]
    x_norm_hat = pca.mean_ + x_compressed_2d @ comps  # unproject PCA
    x_norm_hat = (
        x_norm_hat.reshape(-1, d)
        if lead is None
        else x_norm_hat.reshape(lead[0], lead[1], lead[2], d)
    )
    return x_norm_hat * x_session_std + x_session_avg


# --------------------------------------------------------------------------- #
# Saving / loading PCA metadata (for reconstruction in another job)
# --------------------------------------------------------------------------- #


def save_pca_and_norm(
    path: str | Path,
    pca: PCA,
    x_compressed: np.ndarray,
    x_session_avg: np.ndarray,
    x_session_std: np.ndarray,
    extra: dict[str, np.ndarray] | None = None,
) -> None:
    """Save PCA and normalization metadata to `path`.

    Writes array copies (`pca_mean`, `pca_components`, …) for portability and a pickled sklearn
    `PCA` instance under `pca_sklearn_pickle` for a full round-trip (solver attrs, `random_state`,
    etc.). Load with `allow_pickle=True` only for trusted files.

    Args:
        path: destination `.npz` path; parent directories are created if missing.
        pca: fitted PCA model.
        x_compressed: compressed representation to save alongside the PCA metadata.
        x_session_avg: broadcast mean used for normalization before PCA.
        x_session_std: broadcast std used for normalization before PCA.
        extra: additional arrays to include in the saved archive.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _pca_pickle = np.empty(1, dtype=object)
    _pca_pickle[0] = pickle.dumps(pca, protocol=pickle.HIGHEST_PROTOCOL)
    npz: dict[str, Any] = {
        'pca_mean': pca.mean_.astype(np.float32),
        'pca_components': pca.components_.astype(np.float32),
        # needed when reloading: sklearn>=1.5 PCA.transform passes explained_variance_ to
        # get_namespace.
        'pca_explained_variance': np.asarray(pca.explained_variance_, dtype=np.float32),
        'pca_sklearn_pickle': _pca_pickle,
        'x_compressed': x_compressed.astype(np.float32),
        'x_session_avg': x_session_avg.astype(np.float32),
        'x_session_std': x_session_std.astype(np.float32),
    }
    if extra:
        for key, value in extra.items():
            npz[key] = np.asarray(value)
    np.savez(path, **npz)


_SPLIT_ORDER_DEFAULT: tuple[str, ...] = ('train', 'val', 'test')


def split_z_by_trial_split(
    z: np.ndarray,
    trial_split_labels: Sequence[str],
    *,
    split_order: tuple[str, ...] = _SPLIT_ORDER_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice `z` along the trial axis using string split labels.

    Rows are grouped in `split_order` (default train -> val -> test), matching conventions in
    `*_aligned.npz` and `trials_assembly` outputs.

    Args:
        z: array of shape `[N, T, ...]`.
        trial_split_labels: per-trial split label, length `N`.
        split_order: the three split names, in output order.

    Returns:
        Tuple of three arrays (one per entry in `split_order`), each `[N_split, T, ...]`.
    """
    labels = np.asarray([str(s).lower() for s in trial_split_labels])
    parts: list[np.ndarray] = []
    for sp in split_order:
        mask = labels == sp
        parts.append(np.asarray(z[mask], dtype=np.float32))
    return parts[0], parts[1], parts[2]
