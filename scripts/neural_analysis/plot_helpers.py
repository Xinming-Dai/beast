from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from analyses.neural_analysis.read_neural_encoding_decoding_results import (
    decoding_results_dict,
)

EID_SET: set[str] = {
    "4b00df29-3769-43be-bb40-128b1cba6d35",
    "72cb5550-43b4-4ef0-add5-e4adfdfb5e02",
    "781b35fd-e1f0-4d14-b2bb-95b7263082bb",
    "ecb5520d-1358-434c-95ec-93687ecd1396",
    "f312aaec-3b6f-44b3-86b4-3a0c119c0438",
}

METHOD_LIST: list[str] = ['Keypoints', 'PCA','BEAST', 'ResNet', 'CLS', 'Depth', 'Concat']

PANEL_LABEL_FONT_SIZE = 14
PANEL_LABEL_FONT_WEIGHT = "bold"
PANEL_LABEL_FONT_STYLE = "normal"
PANEL_LABEL_FONT_KWARGS = {
    "fontsize": PANEL_LABEL_FONT_SIZE,
    "fontweight": PANEL_LABEL_FONT_WEIGHT,
    "fontstyle": PANEL_LABEL_FONT_STYLE,
}

# bps_bar_plot_figure3.py does not override axis fonts; these are Matplotlib defaults.
AXIS_LABEL_FONT_FAMILY = "sans-serif"
AXIS_LABEL_FONT_SIZE = "medium"
AXIS_LABEL_FONT_STYLE = "normal"
AXIS_TICK_LABEL_FONT_FAMILY = "sans-serif"
AXIS_TICK_LABEL_FONT_SIZE = "medium"
AXIS_TICK_LABEL_FONT_STYLE = "normal"
AXIS_LABEL_FONT_KWARGS = {
    "fontfamily": AXIS_LABEL_FONT_FAMILY,
    "fontsize": AXIS_LABEL_FONT_SIZE,
    "fontstyle": AXIS_LABEL_FONT_STYLE,
}
AXIS_TICK_LABEL_FONT_KWARGS = {
    "fontfamily": AXIS_TICK_LABEL_FONT_FAMILY,
    "fontsize": AXIS_TICK_LABEL_FONT_SIZE,
    "fontstyle": AXIS_TICK_LABEL_FONT_STYLE,
}



def normalize_eid(s: str) -> str:
    """Lowercase/strip EID string for folder-name matching."""
    return str(s).strip().lower()


def iter_eid_encoding_npys(
    method_dir: Path,
    allowed_eids: frozenset[str] | None,
) -> Iterable[Path]:
    """Yield ``encoding_results*.npy`` paths for matching EID folders.

    If ``allowed_eids`` is set, only directories whose **name** is in that set (compared
    via :func:`normalize_eid`) are searched. Exactly one ``encoding_results*.npy`` file
    is expected under each matching EID directory. If ``allowed_eids`` is ``None``,
    every matching file under ``method_dir`` is included.
    """
    if not method_dir.is_dir():
        return
    if allowed_eids is None:
        yield from method_dir.rglob("encoding_results*.npy")
        return

    seen: set[Path] = set()
    for d in method_dir.rglob("*"):
        if not d.is_dir() or normalize_eid(d.name) not in allowed_eids:
            continue
        matches = sorted(enc for enc in d.rglob("encoding_results*.npy") if enc.is_file())
        if len(matches) > 1:
            raise ValueError(
                f"Expected one encoding_results*.npy under EID directory {d}, "
                f"found {len(matches)}: {[str(p) for p in matches]}"
            )
        if matches and matches[0] not in seen:
            seen.add(matches[0])
            yield matches[0]


def _default_output_path_helper(
    results_dir: Path, 
    save_folder_name: str,
    save_filename: str,
) -> Path:
    """``<results_dir>/<save_folder_name>_<YYYYMMDD_HHMMSS>/<save_filename>``."""
    results_dir = results_dir.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir / f"{save_folder_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / save_filename


def _resolve_eid_key(results_by_eid: dict[str, Any], eid: str | None) -> str:
    """Return the exact results-dict key for ``eid`` or the only key when omitted."""
    if not results_by_eid:
        raise ValueError("encoding_results.npy did not contain any EID entries")
    if eid is None:
        if len(results_by_eid) == 1:
            return next(iter(results_by_eid))
        raise ValueError(
            "encoding_results.npy contains multiple EIDs; pass `eid=` explicitly. "
            f"Available keys: {sorted(results_by_eid)}"
        )

    normalized = normalize_eid(eid)
    for key in results_by_eid:
        if normalize_eid(key) == normalized:
            return key
    raise KeyError(f"EID {eid!r} not found in encoding_results.npy; keys: {sorted(results_by_eid)}")


def load_encoding_session_results(
    path: str | Path,
    *,
    eid: str | None = None,
    encoder: str = "cnn",
) -> dict[str, Any]:
    """Return one encoder block from ``encoding_results.npy`` for a single session."""
    results_by_eid = decoding_results_dict(path)
    eid_key = _resolve_eid_key(results_by_eid, eid)
    session_results = results_by_eid[eid_key]
    if not isinstance(session_results, dict):
        raise TypeError(
            f"Expected dict for EID {eid_key!r}, got {type(session_results).__name__}"
        )
    if encoder not in session_results:
        raise KeyError(
            f"Encoder {encoder!r} not found under EID {eid_key!r}; "
            f"available keys: {sorted(session_results)}"
        )
    encoder_block = session_results[encoder]
    if not isinstance(encoder_block, dict):
        raise TypeError(
            f"Expected dict for encoder block {encoder!r}, got {type(encoder_block).__name__}"
        )
    return encoder_block


def load_neural_meta(path: str | Path) -> dict[str, Any]:
    """Load a session ``*_meta.pkl`` file into a plain dictionary."""
    meta_path = Path(path)
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    if not isinstance(meta, dict):
        raise TypeError(f"Expected dict inside {meta_path}, got {type(meta).__name__}")
    return meta


def bps_per_neuron(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Bits per spike for each neuron, matching the repo's single-cell analysis."""
    from analyses.neural_analysis.eval_utils import bits_per_spike

    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if gt.shape != pred.shape or gt.ndim != 3:
        raise ValueError(f"gt/pred must match 3D (K,T,N); got {gt.shape}, {pred.shape}")

    n_neurons = gt.shape[2]
    out = np.full(n_neurons, np.nan, dtype=np.float64)
    for neuron_idx in range(n_neurons):
        spikes = gt[..., neuron_idx : neuron_idx + 1]
        rates = pred[..., neuron_idx : neuron_idx + 1]
        if np.nansum(spikes) <= 0:
            continue
        bps = bits_per_spike(rates, spikes)
        if np.isinf(bps):
            bps = np.nan
        out[neuron_idx] = bps
    return out


def _acronym_to_positive_id(brain_regions: Any) -> dict[str, int]:
    """Map non-lateralized acronyms to one positive region ID."""
    out: dict[str, int] = {}
    acronyms = np.asarray(brain_regions.acronym, dtype=object)
    ids = np.asarray(brain_regions.id)
    for acronym, region_id in zip(acronyms, ids, strict=True):
        if int(region_id) <= 0:
            continue
        key = str(acronym).strip()
        if key and key.lower() != "nan":
            out.setdefault(key, int(region_id))
    return out


def map_cluster_regions_to_region_ids(
    cluster_regions: Iterable[object],
    *,
    mapping: str = "Beryl",
) -> tuple[np.ndarray, np.ndarray]:
    """Map per-neuron acronyms to target-map acronyms and numeric region IDs."""
    from iblatlas.regions import BrainRegions

    brain_regions = BrainRegions()
    source_acronyms = np.asarray(list(cluster_regions), dtype=object)
    mapped_acronyms = np.asarray(
        brain_regions.acronym2acronym(source_acronyms, mapping=mapping),
        dtype=object,
    )

    acronym_to_allen_id = _acronym_to_positive_id(brain_regions)
    remapped_ids = np.full(source_acronyms.size, np.nan, dtype=np.float64)
    valid_positions: list[int] = []
    allen_ids: list[int] = []

    for flat_idx, raw_acronym in enumerate(source_acronyms.reshape(-1)):
        key = str(raw_acronym).strip() if raw_acronym is not None else ""
        if not key or key.lower() == "nan":
            continue
        allen_id = acronym_to_allen_id.get(key)
        if allen_id is None:
            continue
        valid_positions.append(flat_idx)
        allen_ids.append(allen_id)

    if valid_positions:
        mapped_ids = np.asarray(
            brain_regions.remap(
                np.asarray(allen_ids, dtype=np.int64),
                source_map="Allen",
                target_map=mapping,
            ),
            dtype=np.float64,
        )
        remapped_ids[np.asarray(valid_positions, dtype=np.int64)] = mapped_ids

    remapped_ids = remapped_ids.reshape(source_acronyms.shape)

    invalid_mask = np.zeros(source_acronyms.shape, dtype=bool)
    for idx, mapped_acronym in np.ndenumerate(mapped_acronyms):
        key = str(mapped_acronym).strip() if mapped_acronym is not None else ""
        invalid_mask[idx] = (not key) or key.lower() in {"nan", "void"}
    remapped_ids[invalid_mask] = np.nan

    return mapped_acronyms, remapped_ids


def aggregate_region_values(
    region_ids: np.ndarray,
    neuron_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-aggregate per-neuron values into one numeric value per region ID."""
    region_ids = np.asarray(region_ids, dtype=np.float64).reshape(-1)
    neuron_values = np.asarray(neuron_values, dtype=np.float64).reshape(-1)
    if region_ids.shape != neuron_values.shape:
        raise ValueError(
            "region_ids and neuron_values must have the same shape; "
            f"got {region_ids.shape} vs {neuron_values.shape}"
        )

    valid = np.isfinite(region_ids) & np.isfinite(neuron_values) & (region_ids > 0)
    if not np.any(valid):
        raise ValueError("No valid region/value pairs were available for aggregation")

    valid_region_ids = region_ids[valid].astype(np.int64)
    valid_values = neuron_values[valid]
    unique_region_ids = np.unique(valid_region_ids)
    region_means = np.full(unique_region_ids.shape, np.nan, dtype=np.float64)

    for idx, region_id in enumerate(unique_region_ids):
        region_means[idx] = np.nanmean(valid_values[valid_region_ids == region_id])

    finite = np.isfinite(region_means)
    return unique_region_ids[finite], region_means[finite]


def encoding_results_to_region_values(
    encoding_results_path: str | Path,
    meta_path: str | Path,
    *,
    eid: str | None = None,
    encoder: str = "cnn",
    mapping: str = "Beryl",
) -> tuple[np.ndarray, np.ndarray]:
    """Return notebook-ready ``(list_region_id, list_region_value)`` arrays.

    This helper is intended for ``brain_slice_custom_colormap_test.ipynb``:

    >>> list_region_id, list_region_value = encoding_results_to_region_values(
    ...     "/path/to/encoding_results.npy",
    ...     "/path/to/session_meta.pkl",
    ...     eid="ecb5520d-1358-434c-95ec-93687ecd1396",
    ...     encoder="cnn",
    ... )
    """
    encoder_block = load_encoding_session_results(
        encoding_results_path,
        eid=eid,
        encoder=encoder,
    )
    meta = load_neural_meta(meta_path)

    gt = np.asarray(encoder_block["gt"], dtype=np.float64)
    pred = np.asarray(encoder_block["pred"], dtype=np.float64)
    neuron_bps = bps_per_neuron(gt, pred)

    if "cluster_regions" not in meta:
        raise KeyError("Expected `cluster_regions` in neural meta file")
    cluster_regions = np.asarray(meta["cluster_regions"], dtype=object).reshape(-1)
    if cluster_regions.shape[0] != neuron_bps.shape[0]:
        raise ValueError(
            "Neuron count mismatch between encoding results and meta file: "
            f"{neuron_bps.shape[0]} vs {cluster_regions.shape[0]}"
        )

    _, region_ids = map_cluster_regions_to_region_ids(cluster_regions, mapping=mapping)
    list_region_id, list_region_value = aggregate_region_values(region_ids, neuron_bps)
    return list_region_id, list_region_value