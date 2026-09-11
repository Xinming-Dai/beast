"""CLI to merge per-view flat inference metrics npz files into one npz per EID."""

import argparse
from pathlib import Path

from beast.sable_encoding_decoding.render.metrics import combine_view_metrics_npz


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: argument list, or `None` to read from `sys.argv`.

    Returns:
        parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--left-npz', type=Path, default=None, help='left camera psnr_ssim_metrics.npz',
    )
    ap.add_argument(
        '--right-npz', type=Path, default=None, help='right camera psnr_ssim_metrics.npz',
    )
    ap.add_argument('--out-npz', type=Path, required=True, help='combined output npz path')
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Merge the left/right metrics npz files given on the CLI into one combined npz."""
    args = parse_args(argv)

    view_npz_paths = {}
    if args.left_npz is not None:
        view_npz_paths['left'] = args.left_npz
    if args.right_npz is not None:
        view_npz_paths['right'] = args.right_npz
    if not view_npz_paths:
        raise SystemExit('At least one of --left-npz/--right-npz is required.')

    arrays = combine_view_metrics_npz(view_npz_paths, args.out_npz)
    print(
        f'Wrote {args.out_npz}: average_psnr={arrays["average_psnr"]:.4f} '
        f'average_ssim={arrays["average_ssim"]:.4f} n={len(arrays["psnr"])}',
    )


if __name__ == '__main__':
    main()
