from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from pprint import pprint


DEFAULT_META_PATH = Path(
    "/work/nvme/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data/ecb5520d-1358-434c-95ec-93687ecd1396/ecb5520d-1358-434c-95ec-93687ecd1396_meta.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read and print a saved _meta.pkl file.")
    parser.add_argument(
        "meta_path",
        nargs="?",
        type=Path,
        default=DEFAULT_META_PATH,
        help="Path to the _meta.pkl file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_path = args.meta_path

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta pickle file not found: {meta_path}")

    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)

    print(f"Loaded metadata from: {meta_path}")
    print(f"Type: {type(meta).__name__}")
    print(f"Keys: {meta.keys()}")
    print(f"cluster_channels: {len(meta['cluster_channels'])}")
    print(f"cluster_regions: {len(set(meta['cluster_regions'])), ', '.join(set(meta['cluster_regions']))}")
    # pprint(meta)


if __name__ == "__main__":
    main()