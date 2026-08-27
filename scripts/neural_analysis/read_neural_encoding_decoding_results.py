#!/usr/bin/env python3
"""Load neural decoding results saved as ``decoding_results.npy`` from eval output."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DECODING_RESULTS_NPY = Path("/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/ibl_ckpt/781b35fd-e1f0-4d14-b2bb-95b7263082bb_epoch/17683436/eval_results_ckpt_epoch000003_step0000000000000270/combined_latents/781b35fd-e1f0-4d14-b2bb-95b7263082bb/encoding_results.npy")


def load_decoding_results(path: str | Path, *, allow_pickle: bool = True) -> np.ndarray:
    """Load ``decoding_results.npy`` and return the array (or 0-d object array holding a dict)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=allow_pickle)


def decoding_results_dict(path: str | Path | None = None, *, allow_pickle: bool = True) -> dict[str, Any]:
    """Return the mapping stored in ``decoding_results.npy`` (session id -> results)."""
    p = Path(path) if path is not None else DEFAULT_DECODING_RESULTS_NPY
    raw = load_decoding_results(p, allow_pickle=allow_pickle)
    inner = raw.item() if raw.dtype == object and raw.shape == () else raw
    if not isinstance(inner, dict):
        raise TypeError(f"Expected dict inside .npy, got {type(inner).__name__}")
    return inner


def print_key_names(obj: Any, *, indent: str = "", max_depth: int = 12, depth: int = 0) -> None:
    """Print dict keys recursively (lists/tuples show length; first element explored if dict)."""
    if depth > max_depth:
        print(f"{indent}...")
        return
    if isinstance(obj, dict):
        for k in obj:
            print(f"{indent}{k!r}")
            print_key_names(obj[k], indent=indent + "  ", max_depth=max_depth, depth=depth + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}<{type(obj).__name__}, len={len(obj)}>")
        if obj and isinstance(obj[0], dict):
            print_key_names(obj[0], indent=indent + "  ", max_depth=max_depth, depth=depth + 1)
        elif obj:
            print(f"{indent}  [0]: {type(obj[0]).__name__}")


def _format_value(obj: Any) -> str:
    """Short string for scalars; shape + stats for numeric ndarray; repr fallback."""
    if isinstance(obj, np.ndarray):
        a = obj
        if a.size == 0:
            return f"array(shape={a.shape}, dtype={a.dtype}) (empty)"
        if a.dtype == object:
            return f"array(shape={a.shape}, dtype=object)"
        if np.issubdtype(a.dtype, np.number):
            flat = a.ravel()
            sample = flat[: min(flat.size, 10_000_000)]
            return (
                f"array(shape={a.shape}, dtype={a.dtype}) "
                f"min={sample.min():.6g} max={sample.max():.6g} mean={sample.mean():.6g}"
            )
        return f"array(shape={a.shape}, dtype={a.dtype})"
    if isinstance(obj, np.generic):
        x = obj.item()
        if isinstance(x, float):
            return f"{x:.6g}"
        return repr(x)
    if isinstance(obj, float):
        return f"{obj:.6g}"
    if isinstance(obj, (int, bool)) or obj is None:
        return repr(obj)
    if isinstance(obj, str):
        return repr(obj)
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__}(len={len(obj)}) {repr(obj)[:200]}"
    return repr(obj)[:400]


def print_key_values(obj: Any, *, indent: str = "", max_depth: int = 12, depth: int = 0) -> None:
    """Print dict keys and formatted values; recurse into nested dicts."""
    if depth > max_depth:
        print(f"{indent}...")
        return
    if isinstance(obj, dict):
        for k in obj:
            v = obj[k]
            if isinstance(v, dict):
                print(f"{indent}{k!r}")
                print_key_values(v, indent=indent + "  ", max_depth=max_depth, depth=depth + 1)
            else:
                print(f"{indent}{k!r}: {_format_value(v)}")
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}<{type(obj).__name__}, len={len(obj)}>: {_format_value(obj)}")
        if obj and isinstance(obj[0], dict):
            print_key_values(obj[0], indent=indent + "  ", max_depth=max_depth, depth=depth + 1)


def _summarize(arr: np.ndarray) -> None:
    print(f"type: {type(arr).__name__}")
    print(f"dtype: {arr.dtype}")
    print(f"shape: {arr.shape}")
    print(f"ndim: {arr.ndim}")
    if arr.dtype == object or arr.dtype.kind == "O":
        print("(object array — may hold dict/list; inspect arr.item() or arr[()])")
        if arr.ndim == 0:
            inner = arr.item()
            print(f"scalar contents type: {type(inner).__name__}")
            if isinstance(inner, dict):
                print(f"dict keys ({len(inner)}): {list(inner.keys())[:20]}{'...' if len(inner) > 20 else ''}")
        return
    if np.issubdtype(arr.dtype, np.number) and arr.size:
        flat = np.asarray(arr).ravel()
        # Avoid huge copies for very large arrays
        sample = flat[: min(flat.size, 10_000_000)]
        print(f"min: {sample.min()}, max: {sample.max()}, mean: {sample.mean()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_DECODING_RESULTS_NPY,
        help="Path to decoding_results.npy",
    )
    p.add_argument(
        "--no-pickle",
        action="store_true",
        help="Load with allow_pickle=False (fails if file needs pickle).",
    )
    args = p.parse_args()

    arr = load_decoding_results(args.path, allow_pickle=not args.no_pickle)
    print(f"Loaded: {args.path.resolve()}")
    _summarize(arr)
    try:
        d = decoding_results_dict(args.path, allow_pickle=not args.no_pickle)
    except TypeError:
        return
    print("\nKeys and values:")
    print_key_values(d)


if __name__ == "__main__":
    main()
