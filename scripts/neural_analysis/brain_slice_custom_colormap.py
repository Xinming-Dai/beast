from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from analyses.neural_analysis.plot_helpers import (
    encoding_results_to_region_values,
    load_neural_meta,
    map_cluster_regions_to_region_ids,
)


ba: Any | None = None
_ba_res_um: int | None = None

coord_1=(-4000+100*21)
coord_2=(-4000+100*31)
coord_3=(-4000+100*37)


def compute_top(res_um: int = 10) -> Any:
    """Return an AllenAtlas with a precomputed top surface."""
    from iblatlas.atlas import AllenAtlas

    atlas = AllenAtlas(res_um)
    axz = atlas.xyz2dims[2]
    surface = (atlas.label == 0).astype(np.int8) * 2
    l0 = np.diff(surface, axis=axz, append=np.array(2, dtype=np.int8))
    top = np.argmax(l0 == -2, axis=axz).astype(float)
    top[top == 0] = np.nan
    atlas.top = atlas.bc.i2z(top + 1)
    return atlas


def get_atlas(res_um: int = 10) -> Any:
    """Return a cached atlas used by the slice helper functions."""
    global ba, _ba_res_um
    if ba is None or _ba_res_um != res_um:
        ba = compute_top(res_um)
        _ba_res_um = res_um
    return ba


def metadata_regions(
    meta_path: str | Path,
    *,
    mapping: str = "Beryl",
) -> tuple[np.ndarray, np.ndarray]:
    """Return notebook-ready region IDs and uniform values from a neural meta file.

    The returned arrays can be passed directly to ``sag_slice_RGB`` and
    ``ctx_slice_RGB`` in ``brain_slice_custom_colormap_test.ipynb`` to visualize
    which mapped brain regions are present in a dataset.
    """
    meta = load_neural_meta(meta_path)
    cluster_regions = np.asarray(meta["cluster_regions"], dtype=object).reshape(-1)
    _, region_ids = map_cluster_regions_to_region_ids(cluster_regions, mapping=mapping)

    valid_region_ids = region_ids[np.isfinite(region_ids) & (region_ids > 0)].astype(np.int64)
    if valid_region_ids.size == 0:
        raise ValueError(f"No valid {mapping} regions found in {meta_path}")

    list_region_id = np.unique(valid_region_ids)
    list_region_value = np.ones(list_region_id.size, dtype=np.float64)
    return list_region_id, list_region_value


def _take(vol, ind: int, axis: int, mode: str = "raise"):
    """Create slices of a 3D volume."""
    if mode == "clip":
        ind = np.minimum(np.maximum(ind, 0), vol.shape[axis] - 1)
    if axis == 0:
        return vol[ind, :, :]
    if axis == 1:
        return vol[:, ind, :]
    if axis == 2:
        return vol[:, :, ind]
    raise ValueError(f"Unsupported axis: {axis}")


def _take_remap(
    vol,
    ind: int,
    axis: int,
    mapping: str = "Beryl",
    mode: str = "raise",
    atlas: Any | None = None,
):
    """Return region IDs per pixel after remapping atlas labels."""
    atlas = get_atlas() if atlas is None else atlas
    return atlas._get_mapping(mapping=mapping)[_take(vol, ind, axis, mode)]


def _color_region_values(
    im: np.ndarray,
    region_slice: np.ndarray,
    list_region_id: np.ndarray,
    list_region_value: np.ndarray,
    c_map,
    atlas: Any,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    null_color = [0.5, 0.5, 0.5]
    min_value, max_value = _resolve_value_range(list_region_value, value_range)
    value_span = max_value - min_value
    cmap_rgb_list = c_map(np.linspace(0, 1, 101))

    for i_reg in range(len(list_region_id)):
        if not np.isfinite(list_region_value[i_reg]):
            local_color = null_color
        elif value_span == 0:
            local_value = 100
            local_color = cmap_rgb_list[local_value, 0:3]
        else:
            local_value = np.ceil(100 * (list_region_value[i_reg] - min_value) / value_span)
            local_value = int(np.clip(int(local_value), 0, 100))
            if list_region_value[i_reg] < min_value:
                local_color = null_color
            else:
                local_color = cmap_rgb_list[local_value, 0:3]
        local_region_id = list_region_id[i_reg]
        local_region_indices = np.argwhere(atlas.regions.id == local_region_id)[:, 0]
        for local_region_index in local_region_indices:
            region_pixel = np.argwhere(region_slice == local_region_index)
            im[region_pixel[:, 0], region_pixel[:, 1], 0] = local_color[0]
            im[region_pixel[:, 0], region_pixel[:, 1], 1] = local_color[1]
            im[region_pixel[:, 0], region_pixel[:, 1], 2] = local_color[2]
    return im


def _resolve_value_range(
    list_region_value: np.ndarray,
    value_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    if value_range is not None:
        min_value, max_value = value_range
        if not (np.isfinite(min_value) and np.isfinite(max_value)):
            raise ValueError(f"value_range must be finite, got {value_range}")
        if max_value < min_value:
            raise ValueError(f"value_range max must be >= min, got {value_range}")
        return float(min_value), float(max_value)

    finite_values = np.asarray(list_region_value, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("No finite region values available for color normalization")
    return float(np.min(finite_values)), float(np.max(finite_values))


def shared_value_range(*list_region_values: np.ndarray) -> tuple[float, float]:
    """Return one finite ``(min, max)`` range shared by multiple region-value arrays."""
    finite_arrays = []
    for values in list_region_values:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        finite_arrays.append(values[np.isfinite(values)])

    finite_arrays = [values for values in finite_arrays if values.size > 0]
    if not finite_arrays:
        raise ValueError("No finite region values available for shared color normalization")

    all_values = np.concatenate(finite_arrays)
    return float(np.min(all_values)), float(np.max(all_values))


def _draw_boundaries(im: np.ndarray, boundary_slice: np.ndarray) -> np.ndarray:
    boundary_color = [0, 0, 0]
    boundary_pixels = np.argwhere(boundary_slice == 1)
    im[boundary_pixels[:, 0], boundary_pixels[:, 1], 0] = boundary_color[0]
    im[boundary_pixels[:, 0], boundary_pixels[:, 1], 1] = boundary_color[0]
    im[boundary_pixels[:, 0], boundary_pixels[:, 1], 2] = boundary_color[0]
    return im


def sag_slice_RGB(
    list_region_id: np.ndarray,
    list_region_value: np.ndarray,
    ML_coordinate: float,
    c_map,
    atlas: Any | None = None,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Create an RGB image of a sagittal brain slice colored by region values."""
    atlas = get_atlas() if atlas is None else atlas
    coordinate = ML_coordinate / 1e6
    index_1 = atlas.bc.x2i(np.array(coordinate), mode="raise")

    axis = 0
    mode = "raise"
    mapping = "Beryl"
    sag_slice_ind = _take_remap(
        atlas.label,
        index_1,
        atlas.xyz2dims[axis],
        mapping,
        mode,
        atlas=atlas,
    )
    sag_slice_b = atlas.slice(
        coordinate,
        axis=0,
        volume="boundary",
        mode="raise",
        region_values=None,
        mapping="Beryl",
        bc=None,
    )
    sag_slice = np.transpose(sag_slice_ind)
    sag_slice_b = np.transpose(sag_slice_b)

    im_sag_1 = np.ones((len(sag_slice[:, 0]), len(sag_slice[0, :]), 3))
    im_sag_1 = _color_region_values(
        im_sag_1,
        sag_slice,
        list_region_id,
        list_region_value,
        c_map,
        atlas,
        value_range=value_range,
    )
    return _draw_boundaries(im_sag_1, sag_slice_b)


def ctx_slice_RGB(
    list_region_id: np.ndarray,
    list_region_value: np.ndarray,
    c_map,
    atlas: Any | None = None,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Create a top-down cortex view RGB image colored by region values."""
    atlas = get_atlas() if atlas is None else atlas
    ix, iy = np.meshgrid(np.arange(atlas.bc.nx), np.arange(atlas.bc.ny))
    iz = atlas.bc.z2i(atlas.top)
    inds = atlas._lookup_inds(np.stack((ix, iy, iz), axis=-1))
    ctx_slice = atlas._get_mapping(mapping="Beryl")[atlas.label.flat[inds]]
    ctx_slice_b = atlas.compute_boundaries(ctx_slice)

    im_ctx_1 = np.ones((len(ctx_slice[:, 0]), len(ctx_slice[0, :]), 3))
    im_ctx_1 = _color_region_values(
        im_ctx_1,
        ctx_slice,
        list_region_id,
        list_region_value,
        c_map,
        atlas,
        value_range=value_range,
    )
    return _draw_boundaries(im_ctx_1, ctx_slice_b)


def sag_slice_RGB_two_encoding_results(
    encoding_results_path_1: str | Path,
    encoding_results_path_2: str | Path,
    meta_path: str | Path,
    ML_coordinate: float,
    c_map,
    *,
    eid: str | None = None,
    encoder: str = "cnn",
    mapping: str = "Beryl",
    atlas: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Render two encoding-result sagittal slices with one shared color scale.

    Returns ``(im_1, im_2, value_range)`` so the same range can also be reused for
    colorbars or additional slices.
    """
    list_region_id_1, list_region_value_1 = encoding_results_to_region_values(
        encoding_results_path_1,
        meta_path,
        eid=eid,
        encoder=encoder,
        mapping=mapping,
    )
    list_region_id_2, list_region_value_2 = encoding_results_to_region_values(
        encoding_results_path_2,
        meta_path,
        eid=eid,
        encoder=encoder,
        mapping=mapping,
    )
    value_range = shared_value_range(list_region_value_1, list_region_value_2)

    im_1 = sag_slice_RGB(
        list_region_id_1,
        list_region_value_1,
        ML_coordinate,
        c_map,
        atlas=atlas,
        value_range=value_range,
    )
    im_2 = sag_slice_RGB(
        list_region_id_2,
        list_region_value_2,
        ML_coordinate,
        c_map,
        atlas=atlas,
        value_range=value_range,
    )
    return im_1, im_2, value_range


def make_sag_plot(im_1, im_2, im_3):
    """Display three sagittal slices side-by-side without axes or borders."""
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(1, 3, figsize=(20, 6))
    for ax, im in zip(axs, [im_1, im_2, im_3]):
        ax.imshow(im)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)


def make_ctx_plot(im_1):
    """Display the cropped top cortex view without axes or borders."""
    import matplotlib.pyplot as plt

    plt.imshow(im_1[:, 0 : round(len(im_1[0, :, 0]) / 2), :])
    ax3 = plt.gca()
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    ax3.spines["left"].set_visible(False)