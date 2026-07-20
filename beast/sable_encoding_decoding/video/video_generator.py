"""Encode an ordered sequence of still images under a directory into a video file.

Uses `imageio` with the FFMPEG plugin when installed
(`pip install "beast[sable_encoding_decoding]"` pulls in `imageio[ffmpeg]`). If that plugin is
missing, falls back to `cv2.VideoWriter` (`mp4v` codec); `opencv-python-headless` is already a
base beast dependency, so this fallback path is available even without the optional extra.
"""

import argparse
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

_DEFAULT_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff')


def _natural_sort_key(path: Path) -> list[int | str]:
    """Sort key that orders numeric filename parts numerically, not lexicographically.

    Args:
        path: image path whose stem (filename without suffix) is used for sorting.

    Returns:
        List of interleaved numeric and string chunks, e.g. `frame_2` -> `['frame_', 2, '']`.
    """
    stem = path.stem
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', stem)]


def list_image_paths(
    folder: str | Path,
    *,
    extensions: tuple[str, ...] | None = None,
    recursive: bool = False,
) -> list[Path]:
    """Return image paths under `folder`, sorted by natural filename order.

    Args:
        folder: directory to scan for image files.
        extensions: allowed suffixes (lower-case, with or without leading dot). Default: png,
            jpg, jpeg, webp, tif, tiff.
        recursive: include images in subdirectories when `True`.

    Returns:
        List of matching image paths, naturally sorted (numeric filename runs sort
        numerically, e.g. `frame_2` before `frame_10`).

    Raises:
        NotADirectoryError: `folder` does not exist or is not a directory.
    """
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f'Not a directory: {folder}')

    exts = extensions or _DEFAULT_EXTENSIONS
    ext_set = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in exts}

    candidates = folder.rglob('*') if recursive else folder.iterdir()
    paths = [p for p in candidates if p.is_file() and p.suffix.lower() in ext_set]
    return sorted(paths, key=_natural_sort_key)


def _load_rgb_uint8(path: Path) -> np.ndarray:
    """Load one image as RGB uint8 `(H, W, 3)`.

    Args:
        path: image file to load.

    Returns:
        RGB uint8 array with shape `(H, W, 3)`.
    """
    frame = np.asarray(imageio.imread(path))
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)
    elif frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _write_video_cv2(paths: list[Path], output_path: Path, fps: float) -> None:
    """Encode frames with OpenCV (BGR); used when imageio-ffmpeg is not installed.

    Args:
        paths: ordered frame image paths.
        output_path: destination `.mp4` path.
        fps: frames per second.

    Raises:
        FileNotFoundError: `cv2` could not read one of `paths`.
        RuntimeError: `cv2.VideoWriter` failed to open `output_path`.
        ValueError: a later frame's size does not match the first frame's size.
    """
    import cv2

    ref_shape: tuple[int, int] | None = None
    writer: cv2.VideoWriter | None = None
    try:
        for p in paths:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f'cv2 could not read image: {p}')
            if ref_shape is None:
                ref_shape = (bgr.shape[0], bgr.shape[1])
                h, w = ref_shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f'cv2.VideoWriter failed to open {output_path}')
            elif bgr.shape[:2] != ref_shape:
                raise ValueError(
                    f'Frame size mismatch at {p}: got {bgr.shape[:2]}, expected {ref_shape}',
                )
            writer.write(bgr)
    finally:
        if writer is not None:
            writer.release()


def images_folder_to_video(
    input_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    fps: float = 24.0,
    extensions: tuple[str, ...] | None = None,
    recursive: bool = False,
    codec: str = 'libx264',
    crf: int = 23,
    macro_block_size: int | None = 1,
) -> Path:
    """Read images from `input_dir`, sort by natural filename order, and write an MP4.

    Args:
        input_dir: directory containing frames.
        output_path: destination `.mp4` path. Default: `<input_dir>/video.mp4`.
        fps: frames per second.
        extensions: allowed suffixes (lower-case, with or without leading dot). Default: png,
            jpg, jpeg, webp, tif, tiff.
        recursive: include images in subdirectories when `True`.
        codec: FFmpeg video codec passed to imageio (default `libx264`).
        crf: H.264 CRF (lower is higher quality).
        macro_block_size: passed to imageio (`1` avoids dimension alignment issues).

    Returns:
        Resolved path to the written video.

    Raises:
        FileNotFoundError: no matching images in `input_dir`.
        ValueError: frame shapes differ across the sequence.
    """
    input_dir = Path(input_dir).expanduser().resolve()
    paths = list_image_paths(input_dir, extensions=extensions, recursive=recursive)
    if not paths:
        raise FileNotFoundError(f'No image files found in {input_dir}')

    if output_path is None:
        output_path = input_dir / 'video.mp4'
    else:
        output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_params = ['-crf', str(int(crf)), '-pix_fmt', 'yuv420p']
    kwargs: dict = {
        'format': 'FFMPEG',
        'fps': fps,
        'codec': codec,
        'ffmpeg_params': ffmpeg_params,
    }
    if macro_block_size is not None:
        kwargs['macro_block_size'] = macro_block_size

    ref_shape: tuple[int, ...] | None = None
    try:
        writer = imageio.get_writer(str(output_path), **kwargs)
    except ImportError:
        _write_video_cv2(paths, output_path, fps)
        return output_path

    try:
        for p in paths:
            frame = _load_rgb_uint8(p)
            if ref_shape is None:
                ref_shape = frame.shape
            elif frame.shape != ref_shape:
                raise ValueError(
                    f'Frame shape mismatch at {p}: got {frame.shape}, expected {ref_shape}',
                )
            writer.append_data(frame)
    finally:
        writer.close()

    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the video-generation entry point.

    Args:
        argv: argument list to parse, e.g. `sys.argv[1:]`. If `None`, `argparse` falls back
            to reading `sys.argv` itself.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description='Encode images in a folder into an MP4 video.')
    parser.add_argument(
        '-i',
        '--input-dir',
        type=Path,
        required=True,
        help='Folder containing image frames',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=None,
        help='Output .mp4 path (default: <input_dir>/video.mp4)',
    )
    parser.add_argument('--fps', type=float, default=24.0, help='Frames per second (default: 24)')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Include images in subdirectories',
    )
    parser.add_argument(
        '--extensions',
        type=str,
        default=None,
        help='Comma-separated suffixes to include (default: png,jpg,jpeg,webp,tif,tiff)',
    )
    parser.add_argument('--crf', type=int, default=23, help='x264 CRF (default: 23)')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the image-folder-to-video pipeline end to end (CLI entry point).

    Args:
        argv: argument list to parse, e.g. `sys.argv[1:]`. If `None`, `argparse` falls back
            to reading `sys.argv` itself.

    Returns:
        Process exit code (always `0` on success).
    """
    args = parse_args(argv)

    extensions: tuple[str, ...] | None = None
    if args.extensions:
        extensions = tuple(x.strip() for x in args.extensions.split(',') if x.strip())

    out = images_folder_to_video(
        args.input_dir,
        args.output,
        fps=args.fps,
        extensions=extensions,
        recursive=args.recursive,
        crf=args.crf,
    )
    print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
