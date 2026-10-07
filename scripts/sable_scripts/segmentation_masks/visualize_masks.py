#!/usr/bin/env python3
"""Overlay a single SAM3 segmentation mask onto its source frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True, help='path to the original frame PNG')
    p.add_argument('--mask', type=Path, required=True, help='path to the mask PNG')
    p.add_argument(
        '--output',
        type=Path,
        default=Path('.'),
        help='path to write the overlay PNG (default: current directory)',
    )
    p.add_argument('--alpha', type=float, default=0.5, help='mask overlay opacity')
    p.add_argument(
        '--color',
        type=int,
        nargs=3,
        default=[0, 0, 255],
        metavar=('B', 'G', 'R'),
        help='overlay color in BGR',
    )
    return p


def main() -> None:
    """Entry point."""
    args = _build_parser().parse_args()

    image = cv2.imread(str(args.image))
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'could not read image: {args.image}')
    if mask is None:
        raise FileNotFoundError(f'could not read mask: {args.mask}')

    color = np.array(args.color, dtype=np.uint8)
    overlay = image.copy()
    binary_mask = mask > 0
    overlay[binary_mask] = (
        (1 - args.alpha) * image[binary_mask] + args.alpha * color
    ).astype(np.uint8)

    boundary = cv2.Canny(mask, 100, 200)
    overlay[boundary > 0] = color

    output_path = args.output
    if output_path.is_dir() or output_path.suffix == '':
        output_path = output_path / f'overlay_{args.mask.stem}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    print(f'wrote overlay to {output_path}')


if __name__ == '__main__':
    main()
