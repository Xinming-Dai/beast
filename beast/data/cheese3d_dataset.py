"""Cheese3D dataset for SABLE training and evaluation.

Loads multi-view Cheese3D sessions and returns SABLE-compatible batch dictionaries.
Each sample represents one frame from one session, with all camera views stacked.

Phase 1 (smoke test):
  - init_gs=false skips the merge_pcd correspondence / Kabsch branch entirely.
  - VDA runs in online mode, so depth_vda is all zeros (placeholder).
  - leftcamera_xy / rightcamera_xy / confidence are empty (confidence = 0).

Phase 2 (pseudo-correspondence):
  - Generate pseudo-points from segmentation mask bounding boxes (≥ 3 points).
  - Switch init_gs=true in the config.

Phase 3 (LP keypoints):
  - Load precomputed LitPose-style .npz correspondence bundles.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_RE = re.compile(r'^img(?P<idx>\d+)\.png$')
CAMERA_RE = re.compile(r'^img(?P<idx>\d+)\.npy$')
MASK_RE = re.compile(r'^mask(?P<idx>\d+)\.png$')


def parse_index(path: Path, pattern: re.Pattern) -> int | None:
    """Parse a numeric frame index from a file name."""
    match = pattern.match(path.name)
    if match is None:
        return None
    return int(match.group('idx'))


def collect_indexed_files(directory: Path, pattern: re.Pattern) -> dict[int, Path]:
    """Collect files in ``directory`` keyed by parsed frame index."""
    if not directory.exists():
        raise FileNotFoundError(f'Missing directory: {directory}')
    indexed: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        frame_idx = parse_index(path, pattern)
        if frame_idx is not None:
            indexed[frame_idx] = path.resolve()
    return indexed


def empty_correspondences() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return empty SABLE correspondence tensors."""
    return (
        torch.zeros(0, 2, dtype=torch.float32),
        torch.zeros(0, 2, dtype=torch.float32),
        torch.zeros(0, dtype=torch.float32),
    )


def mask_bbox_points(mask_path: Path, image_size: int, threshold: int = 0) -> torch.Tensor:
    """Convert a binary segmentation mask bbox into five pseudo keypoints.

    Points are ordered as center, top, bottom, left, right in the resized SABLE
    image coordinate system. This is a smoke-test correspondence source, not a
    semantically accurate landmark detector.
    """
    mask = Image.open(mask_path).convert('L')
    arr = np.asarray(mask)
    ys, xs = np.nonzero(arr > int(threshold))
    if xs.size == 0 or ys.size == 0:
        return torch.zeros(0, 2, dtype=torch.float32)

    x_min = float(xs.min())
    x_max = float(xs.max())
    y_min = float(ys.min())
    y_max = float(ys.max())
    x_c = 0.5 * (x_min + x_max)
    y_c = 0.5 * (y_min + y_max)
    points = np.asarray(
        [
            [x_c, y_c],
            [x_c, y_min],
            [x_c, y_max],
            [x_min, y_c],
            [x_max, y_c],
        ],
        dtype=np.float32,
    )

    width, height = mask.size
    scale = np.asarray([float(image_size) / float(width), float(image_size) / float(height)])
    points *= scale.astype(np.float32)
    points[:, 0] = np.clip(points[:, 0], 0.0, float(image_size - 1))
    points[:, 1] = np.clip(points[:, 1], 0.0, float(image_size - 1))
    return torch.from_numpy(points).float()


def resolve_mask_dir(root: Path, session_id: str, view: str) -> Path | None:
    """Resolve the segmentation mask directory for one session/view pair."""
    mask_root = root / 'segmentation_masks'
    if not mask_root.exists():
        return None
    matches = sorted(path for path in mask_root.glob(f'{session_id}_{view}_*') if path.is_dir())
    if not matches:
        return None
    return matches[0]


def frame_indices_for_view(
    *,
    session_dir: Path,
    root: Path,
    session_id: str,
    view: str,
    require_masks: bool,
) -> tuple[set[int], dict[int, Path], dict[int, Path], dict[int, Path]]:
    """Collect valid frame indices and per-frame files for a session/view."""
    view_dir = session_dir / view
    images = collect_indexed_files(view_dir, IMAGE_RE)
    cameras = collect_indexed_files(view_dir, CAMERA_RE)
    masks: dict[int, Path] = {}
    mask_dir = resolve_mask_dir(root, session_id, view)
    if mask_dir is not None:
        masks = collect_indexed_files(mask_dir / 'masks', MASK_RE)
    elif require_masks:
        raise FileNotFoundError(f'Missing mask directory for session={session_id}, view={view}')
    valid = set(images) & set(cameras)
    if require_masks:
        valid &= set(masks)
    return valid, images, cameras, masks


def load_info(dataset_dir: Path) -> dict:
    """Load the Cheese3D info.json metadata file."""
    info_path = dataset_dir / 'info.json'
    if not info_path.exists():
        raise FileNotFoundError(f'Missing Cheese3D info file: {info_path}')
    with open(info_path) as f:
        return json.load(f)


def _resolve_dataset_dir(root: Path) -> Path:
    """Find the directory containing info.json.

    Handles the Cheese3D layout where frames live under root/cheese3d_cam/cheese3d_cam/
    (double nesting) or root/cheese3d_cam/ (single nesting).
    """
    # Try direct path first (root/cheese3d_cam/cheese3d_cam/)
    candidate = root / 'cheese3d_cam' / 'cheese3d_cam'
    if (candidate / 'info.json').exists():
        return candidate
    # Try single-level (root/cheese3d_cam/)
    candidate = root / 'cheese3d_cam'
    if (candidate / 'info.json').exists():
        return candidate
    raise FileNotFoundError(
        f'Missing Cheese3D info.json under {root}; tried '
        f'{root}/cheese3d_cam/cheese3d_cam/ and {root}/cheese3d_cam/'
    )


def load_selected_frame_indices(session_dir: Path) -> set[int]:
    """Load frame indices from selected_frames.csv if present.

    Accepts rows in any of these forms:
    - bare integer: ``0``
    - image filename: ``img00000000.png``
    - CSV rows whose first cell contains either of the above
    - optional header rows (ignored)

    Returns an empty set if the file does not exist (no filtering).
    """
    csv_path = session_dir / 'selected_frames.csv'
    if not csv_path.exists():
        return set()

    indices: set[int] = set()
    with open(csv_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            first_cell = line.split(',')[0].strip().strip('"').strip("'")
            if not first_cell:
                continue

            if first_cell.isdigit():
                indices.add(int(first_cell))
                continue

            match = IMAGE_RE.match(first_cell)
            if match is not None:
                indices.add(int(match.group('idx')))
                continue

    return indices


def build_frame_index(
    root: Path,
    views: list[str],
    sessions: list[str] | None = None,
    start_frame: int = 0,
    frame_step: int = 1,
    max_frames_per_session: int | None = None,
    require_masks: bool = True,
) -> tuple[list[dict], dict]:
    """Build Cheese3D frame-level records and a summary.

    Supports two input modes:

    1. Manifest mode: ``manifest_path`` is a JSONL file — load records directly.
    2. Scan mode: ``manifest_path`` is the Cheese3D root dir — scan and build index.

    The Cheese3D data layout is::

        root/
        ├── cheese3d_cam/
        │   └── cheese3d_cam/          ← detected automatically
        │       ├── info.json
        │       ├── {session}/
        │       │   ├── {view}/img*.png, img*.npy
        │       │   └── selected_frames.csv
        │       └── ...
        └── segmentation_masks/
            └── {session}_{view}_{timestamp}/
                └── masks/mask*.png
    """
    dataset_dir = _resolve_dataset_dir(root)
    info = load_info(dataset_dir)

    if sessions:
        session_ids = sessions
    else:
        info_sessions = info.get('video_ids')
        if isinstance(info_sessions, list) and info_sessions:
            session_ids = [str(s) for s in info_sessions]
        else:
            session_ids = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())

    if not views:
        info_views = info.get('available_views')
        if isinstance(info_views, list) and info_views:
            views = [str(v) for v in info_views]
        else:
            raise ValueError(
                'Cheese3D info.json does not define available_views; '
                'pass views explicitly via the config.'
            )

    records: list[dict] = []
    per_session: dict[str, int] = {}
    skipped_sessions: dict[str, str] = {}

    for session_id in session_ids:
        session_dir = dataset_dir / session_id
        if not session_dir.exists():
            skipped_sessions[session_id] = f'missing session directory: {session_dir}'
            continue

        selected_indices = load_selected_frame_indices(session_dir)

        files_by_view: dict[str, dict] = {}
        common_indices: set | None = None
        try:
            for view in views:
                valid, images, cameras, masks = frame_indices_for_view(
                    session_dir=session_dir,
                    root=root,
                    session_id=session_id,
                    view=view,
                    require_masks=require_masks,
                )
                files_by_view[view] = {
                    'images': images,
                    'cameras': cameras,
                    'masks': masks,
                }
                common_indices = valid if common_indices is None else common_indices & valid
        except (FileNotFoundError, ValueError) as exc:
            skipped_sessions[session_id] = str(exc)
            continue

        if common_indices is None:
            common_indices = set()

        if selected_indices:
            common_indices &= selected_indices

        selected = sorted(
            idx for idx in common_indices
            if idx >= start_frame and (idx - start_frame) % frame_step == 0
        )
        if max_frames_per_session is not None:
            selected = selected[:max_frames_per_session]

        per_session[session_id] = len(selected)

        for frame_idx in selected:
            view_entries = {}
            for view in views:
                files = files_by_view[view]
                entry: dict = {
                    'image_path': str(files['images'][frame_idx]),
                    'camera_path': str(files['cameras'][frame_idx]),
                }
                if frame_idx in files['masks']:
                    entry['mask_path'] = str(files['masks'][frame_idx])
                view_entries[view] = entry

            records.append({
                'dataset': 'cheese3d',
                'session_id': session_id,
                'frame_idx': frame_idx,
                'frame_name': f'img{frame_idx:08d}',
                'view_order': views,
                'views': view_entries,
            })

    summary = {
        'root': str(root),
        'views': views,
        'sessions_requested': session_ids,
        'num_sessions_requested': len(session_ids),
        'num_sessions_with_records': sum(1 for c in per_session.values() if c > 0),
        'num_records': len(records),
        'records_per_session': per_session,
        'skipped_sessions': skipped_sessions,
        'start_frame': start_frame,
        'frame_step': frame_step,
        'max_frames_per_session': max_frames_per_session,
        'require_masks': require_masks,
    }
    return records, summary


class Cheese3DDataset(Dataset):
    """Cheese3D dataset compatible with SABLE training.

    Loads multi-view frames from Cheese3D sessions and returns SABLE-compatible
    batch dictionaries. Supports two input modes:

    1. **Manifest mode**: ``dataset_path`` is a ``.jsonl`` file generated by
       ``scripts/prepare_sable_data/prepare_cheese3d_manifest.py``.
    2. **Scan mode**: ``dataset_path`` is the Cheese3D root directory; this class
       builds the frame index at construction time (no manifest needed).

    ``__getitem__`` returns a dict with the following keys that SABLE's
    ``SableLightningModule`` and ``resolve_view_indices`` expect:

    ===================  =============================  =================================
    Key                  Shape                          Notes
    ===================  =============================  =================================
    ``image``            ``[V, 3, H, W]`` float32       RGB tensor in ``[0, 1]``
    ``context_indices``  ``[n_ctx]`` long               Phase 1: ``[0, 1]`` for L/R pair
    ``target_indices``   ``[n_tgt]`` long               Phase 1: ``[0, 1]`` for L/R pair
    ``depth_vda``        ``[V, 1, H, W]`` float32      Phase 1: all zeros (VDA online)
    ``leftcamera_xy``    ``[0, 2]`` float32            Phase 1: empty (init_gs=false)
    ``rightcamera_xy``   ``[0, 2]`` float32            Phase 1: empty (init_gs=false)
    ``confidence``       ``[0]`` float32                Phase 1: empty (init_gs=false)
    ``scene_name``       str                            ``{session_id}_frame_{idx:08d}``
    ===================  =============================  =================================

    Phase 1 smoke test requirements:

    - ``model.gaussians.init_gs: false`` — skips Kabsch / merge_pcd branch entirely.
    - ``model.vda.mode: online`` — depth is computed on the fly; ``depth_vda`` is a
      zero placeholder (the model ignores it in online mode).

    Args:
        config: Full beast config dict. Reads the following keys:

            - ``training.dataset_path``: path to manifest JSONL or Cheese3D root.
            - ``training.views``: camera view names to use (e.g. ``L`` and ``R``).
              If omitted, reads from ``info.json``.
            - ``training.sessions``: session IDs to include. If omitted, uses all.
            - ``training.image_size``: resize target (default: 320).
            - ``training.start_frame``: first frame index (default: 0).
            - ``training.frame_step``: frame stride (default: 1).
            - ``training.max_frames_per_session``: cap per session (default: None).
            - ``training.allow_missing_masks``: skip mask requirement (default: False).
            - ``training.correspondence_mode``: ``none``, ``auto``, ``cache``, or
              ``mask_bbox``. ``mask_bbox`` uses the first two configured views.
            - ``model.image_tokenizer.image_size``: used when ``image_size`` not set.

        include_splits: Which logical splits to include. Supported values:

            - ``['train']`` — include all records.
            - ``['val']`` — include all records (no built-in split; use
              ``train_split_ratio`` or provide a pre-split manifest).

            Ignored; present for API compatibility with the training loop.

        image_size: Override resize target. Defaults to
            ``config['model']['image_tokenizer']['image_size']`` (320).
    """

    def __init__(
        self,
        config: dict,
        include_splits: list[str] | None = None,
        image_size: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        training_cfg = config.get('training', {})
        model_cfg = config.get('model', {})

        # Resolve dataset_path
        dataset_path_str = training_cfg.get('dataset_path')
        if dataset_path_str is None:
            raise ValueError(
                'training.dataset_path must be set. '
                'Point it to a Cheese3D manifest (.jsonl) or the Cheese3D root directory.'
            )
        self.dataset_path = Path(dataset_path_str)

        # Resolve views
        self.views: list[str] = training_cfg.get('views', [])
        if not self.views:
            self.views = [str(v) for v in model_cfg.get('available_views', [])]
        if len(self.views) < 2:
            raise ValueError(
                f'Need at least 2 views for SABLE, got: {self.views}. '
                'Set training.views in the config (e.g. views: [L, R]).'
            )

        # Resolve sessions
        self.sessions: list[str] | None = training_cfg.get('sessions')

        # Resolve image size
        default_image_size = model_cfg.get('image_tokenizer', {}).get('image_size', 320)
        self.image_size = image_size or int(training_cfg.get('image_size', default_image_size))

        # SABLE expects data['image'] in [0, 1]; the model applies its own
        # normalization for the image tokenizer and DINO branches.
        self._image_transform = transforms.Resize((self.image_size, self.image_size))

        # Frame sampling
        self.start_frame = int(training_cfg.get('start_frame', 0))
        self.frame_step = int(training_cfg.get('frame_step', 1))
        self.max_frames_per_session = training_cfg.get('max_frames_per_session')
        if self.max_frames_per_session is not None:
            self.max_frames_per_session = int(self.max_frames_per_session)
        self.require_masks = not bool(training_cfg.get('allow_missing_masks', False))
        self.correspondence_mode = str(training_cfg.get('correspondence_mode', 'auto')).lower()
        valid_modes = {'auto', 'none', 'cache', 'mask_bbox'}
        if self.correspondence_mode not in valid_modes:
            raise ValueError(
                f'Unsupported training.correspondence_mode={self.correspondence_mode!r}; '
                f'expected one of {sorted(valid_modes)}'
            )
        self.mask_threshold = int(training_cfg.get('mask_threshold', 0))

        # Context / target view index computation
        # ibl_training_regime is read from config but for Cheese3D we always use
        # 'two_input_reconstruction': all views are both context and target.
        self.num_views = int(training_cfg.get('num_views', len(self.views)))
        self.num_input_views = int(training_cfg.get('num_input_views', self.num_views))
        self.num_target_views = int(training_cfg.get('num_target_views', self.num_views))
        self._context_indices, self._target_indices = self._build_indices()

        # Load or build the frame index
        if self.dataset_path.suffix == '.jsonl':
            self._records = self._load_manifest()
        else:
            root = self.dataset_path if self.dataset_path.name == 'cheese3d_cam' else self.dataset_path
            records, summary = build_frame_index(
                root=root,
                views=self.views,
                sessions=self.sessions,
                start_frame=self.start_frame,
                frame_step=self.frame_step,
                max_frames_per_session=self.max_frames_per_session,
                require_masks=self.require_masks,
            )
            self._records = records
            self._summary = summary
            print(f'[Cheese3DDataset] Scanned {len(records)} frames from {root}')

        if not self._records:
            raise ValueError(f'No records found in {self.dataset_path}')

        print(f'[Cheese3DDataset] Loaded {len(self._records)} frames; views={self.views}')

    def _load_manifest(self) -> list[dict]:
        """Load records from a JSONL manifest file."""
        records: list[dict] = []
        with open(self.dataset_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _build_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build context and target index tensors for the two-input reconstruction regime.

        For ``num_views == num_input_views == num_target_views`` (the Cheese3D case),
        both input and target use the full view list — i.e. [0, 1, ...] for both.
        SABLE's ``resolve_view_indices`` then selects per-batch via
        ``context_indices`` / ``target_indices`` tensors.
        """
        if self.num_input_views == self.num_views and self.num_target_views == self.num_views:
            indices = torch.arange(self.num_views, dtype=torch.long)
            return indices, indices.clone()

        if self.num_input_views + self.num_target_views != self.num_views:
            raise ValueError(
                f'Unsupported view allocation for Cheese3D: '
                f'num_input_views={self.num_input_views} + '
                f'num_target_views={self.num_target_views} != '
                f'num_views={self.num_views}. '
                f'Use num_input_views=num_target_views=num_views for the two-input regime.'
            )
        half = self.num_views // 2
        ctx = torch.arange(half, dtype=torch.long)
        tgt = torch.arange(half, self.num_views, dtype=torch.long)
        return ctx, tgt

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess a single image.

        Returns:
            Tensor of shape [3, H, W], float32, in [0, 1].
        """
        img = Image.open(path).convert('RGB')
        img = transforms.ToTensor()(img)  # [3, H, W] in [0, 1]
        img = self._image_transform(img)
        return img

    def _load_correspondences(self, record: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load LP keypoint correspondences (Phase 3) or return empty placeholders.

        Phase 1: returns empty tensors so init_gs=false can safely skip Kabsch.
        Phase 2: ``correspondence_mode=mask_bbox`` returns five mask bbox points.
        Phase 3: load from merge_pcd.correspondence_cache_root when available.

        Returns:
            Tuple of (leftcamera_xy [N, 2], rightcamera_xy [N, 2], confidence [N]).
        """
        if self.correspondence_mode == 'none':
            return empty_correspondences()

        if self.correspondence_mode == 'mask_bbox':
            return self._load_mask_bbox_correspondences(record)

        cache_root_str = self.config.get('model', {}).get('merge_pcd', {}).get(
            'correspondence_cache_root'
        )
        if cache_root_str is None:
            return empty_correspondences()

        cache_root = Path(cache_root_str)
        if not cache_root.exists():
            if self.correspondence_mode == 'cache':
                raise FileNotFoundError(f'Missing correspondence cache root: {cache_root}')
            return empty_correspondences()

        session_id = record['session_id']
        pair_dir = cache_root / session_id / f'pair_{int(record["frame_idx"]):06d}'
        npz_path = pair_dir / 'litpose_matches.npz'
        if not npz_path.exists():
            if self.correspondence_mode == 'cache':
                raise FileNotFoundError(f'Missing correspondence bundle: {npz_path}')
            return empty_correspondences()

        data_npz = np.load(npz_path)
        left_xy = np.asarray(data_npz['left_xy'], dtype=np.float32).copy()
        right_xy = np.asarray(data_npz['right_xy'], dtype=np.float32).copy()
        confidence = np.asarray(data_npz['confidence'], dtype=np.float32).copy()

        # Rescale from original camera pixel space to image_size space
        left_orig_w = float(data_npz.get('left_orig_w', self.image_size))
        left_orig_h = float(data_npz.get('left_orig_h', self.image_size))
        right_orig_w = float(data_npz.get('right_orig_w', self.image_size))
        right_orig_h = float(data_npz.get('right_orig_h', self.image_size))

        left_xy[:, 0] *= self.image_size / left_orig_w
        left_xy[:, 1] *= self.image_size / left_orig_h
        right_xy[:, 0] *= self.image_size / right_orig_w
        right_xy[:, 1] *= self.image_size / right_orig_h

        return (
            torch.from_numpy(left_xy).float(),
            torch.from_numpy(right_xy).float(),
            torch.from_numpy(confidence).float(),
        )

    def _load_mask_bbox_correspondences(
        self,
        record: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build five pseudo-correspondences from the first two view masks."""
        left_view, right_view = self.views[:2]
        left_entry = record['views'][left_view]
        right_entry = record['views'][right_view]
        if 'mask_path' not in left_entry or 'mask_path' not in right_entry:
            raise FileNotFoundError(
                'training.correspondence_mode=mask_bbox requires masks for the first '
                f'two views; got views={left_view},{right_view} scene='
                f'{record["session_id"]}_frame_{record["frame_idx"]:08d}'
            )

        left_xy = mask_bbox_points(
            Path(left_entry['mask_path']),
            self.image_size,
            threshold=self.mask_threshold,
        )
        right_xy = mask_bbox_points(
            Path(right_entry['mask_path']),
            self.image_size,
            threshold=self.mask_threshold,
        )
        n = min(int(left_xy.shape[0]), int(right_xy.shape[0]))
        left_xy = left_xy[:n]
        right_xy = right_xy[:n]
        if n < 3:
            raise ValueError(
                'mask_bbox pseudo-correspondence produced fewer than 3 points for '
                f'{record["session_id"]}_frame_{record["frame_idx"]:08d}; '
                f'left_n={left_xy.shape[0]}, right_n={right_xy.shape[0]}'
            )
        confidence = torch.ones(n, dtype=torch.float32)
        return left_xy, right_xy, confidence

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        record = self._records[idx]

        # Load images for all configured views in the correct order
        images: list[torch.Tensor] = []
        for view in self.views:
            view_entry = record['views'][view]
            img = self._load_image(Path(view_entry['image_path']))
            images.append(img)
        image_tensor = torch.stack(images, dim=0)  # [V, 3, H, W]

        # Correspondence placeholders (Phase 1: empty)
        leftcamera_xy, rightcamera_xy, confidence = self._load_correspondences(record)

        # depth_vda placeholder — zero tensor; VDA runs online in Phase 1
        # Shape: [V, 1, H, W] matching pcd resolution
        depth_vda = torch.zeros(
            len(self.views), 1, self.image_size, self.image_size, dtype=torch.float32
        )

        scene_name = f"{record['session_id']}_frame_{record['frame_idx']:08d}"

        return {
            'image': image_tensor,
            'context_indices': self._context_indices.clone(),
            'target_indices': self._target_indices.clone(),
            'depth_vda': depth_vda,
            'leftcamera_xy': leftcamera_xy,
            'rightcamera_xy': rightcamera_xy,
            'confidence': confidence,
            'scene_name': scene_name,
        }
