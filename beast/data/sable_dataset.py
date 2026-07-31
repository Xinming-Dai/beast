"""SABLE two-view dataset for SABLE model."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, default_collate

from beast.logging import log_step

_logger = logging.getLogger(__name__)

_MAX_MATCHES = 512


@dataclass(frozen=True)
class _PrecacheRecord:
    """One stereo pair loaded from a precache directory."""

    session_id: str
    pair_idx: int
    left_path: Path
    right_path: Path
    left_source_frame_index: int
    right_source_frame_index: int
    scene_name: str
    left_mask_path: Path | None = None
    right_mask_path: Path | None = None
    center_path: Path | None = None
    center_mask_path: Path | None = None
    split: str | None = None
    neural_trial_idx: int | None = None
    neural_bin_idx: int | None = None
    neural_interval_sec: tuple[float, float] | None = None


class SABLEDataset(Dataset):
    """Dataset for two-view IBL image pairs with precomputed VDA depth and correspondences.

    Supports two ``dataset_path`` layouts:

    * **Precache directory** — ``dataset_path`` is a directory containing per-session
      subdirectories, each with a ``pair_metadata.json`` that lists stereo pairs and
      their ``split`` field (``train`` / ``val`` / ``test``).  Pass ``include_splits``
      to select which splits to load.
    * **Scene JSON list** — ``dataset_path`` is a ``.txt`` file where each line is a
      path to a scene JSON file (legacy format, no split filtering).

    Camera parameters (c2w, fxfycxcy) are NOT provided — they are predicted by ERayZer.
    Precomputed VDA depth is loaded from ``model.vda.cache_root``.
    Pixel correspondences are loaded from ``model.merge_pcd.correspondence_cache_root``.
    """

    def __init__(
        self,
        config: dict,
        include_splits: list[str] | None = None,
    ) -> None:
        """Initialize.

        Args:
            config: full beast config dict. Reads keys:
                ``training.dataset_path``, ``model.merge_pcd.correspondence_cache_root``,
                ``model.vda.cache_root``, ``model.image_tokenizer.image_size``,
                ``training.training_regime``, ``training.val_split_ratio``,
                ``model.seed``.
            include_splits: for the precache directory format, only load pairs whose
                ``split`` field is in this list (e.g. ``['train']``, ``['val']``).
                For the scene JSON list format, used together with
                ``training.val_split_ratio`` to select the train or val portion of a
                deterministic random split. ``None`` loads all records.
        """
        super().__init__()
        self.config = config
        training = config['training']
        model_cfg = config['model']

        dataset_path = training.get('dataset_path')
        if not dataset_path:
            raise ValueError('training.dataset_path must be set.')
        val_split_ratio = float(training.get('val_split_ratio', 0.0))
        split_seed = int(model_cfg.get('seed', 0))
        self._records: list[_PrecacheRecord] = self._load_records(
            Path(dataset_path),
            include_splits=include_splits,
            val_split_ratio=val_split_ratio,
            split_seed=split_seed,
        )

        vda_cfg = model_cfg.get('vda', {}) or {}
        cache_root = vda_cfg.get('cache_root')
        if not cache_root:
            raise ValueError('model.vda.cache_root must be set for SABLEDataset.')
        self._vda_cache_root = Path(cache_root)

        merge_pcd_cfg = model_cfg.get('merge_pcd', {}) or {}
        corr_root = merge_pcd_cfg.get('correspondence_cache_root')
        self._corr_root: Path | None = Path(corr_root) if corr_root else None

        self._image_size: int = int(model_cfg['image_tokenizer']['image_size'])
        self._training_regime: str = str(
            training.get('training_regime', 'all_views_reconstruction')
        ).strip().lower()

    def __len__(self) -> int:
        """Return the number of scene pairs."""
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load one stereo pair.

        Args:
            idx: dataset index.

        Returns:
            dict with keys ``image``, ``context_indices``, ``target_indices``,
            ``depth_vda``, ``leftcamera_xy``, ``rightcamera_xy``, ``confidence``,
            ``scene_name``, ``split``, ``neural_trial_idx``, ``neural_bin_idx``,
            ``neural_interval_sec``.
        """
        rec = self._records[idx]

        left_img, left_orig_w, left_orig_h = self._load_image(rec.left_path)
        right_img, right_orig_w, right_orig_h = self._load_image(rec.right_path)
        vda_depths = [
            self._load_vda_depth(rec.session_id, 'left', rec.left_source_frame_index),
            self._load_vda_depth(rec.session_id, 'right', rec.right_source_frame_index),
        ]

        image_tensor = torch.stack([left_img, right_img], dim=0)  # [V, 3, H, W]
        depth_tensor = torch.stack(vda_depths, dim=0)              # [V, 1, H, W]

        context_indices, target_indices = self._resolve_view_indices()
        correspondences = self._load_correspondences(
            rec.session_id,
            rec.pair_idx,
            left_orig_size=(left_orig_w, left_orig_h),
            right_orig_size=(right_orig_w, right_orig_h),
        )

        neural_interval_sec = (
            torch.tensor(rec.neural_interval_sec, dtype=torch.float64)
            if rec.neural_interval_sec is not None
            else torch.full((2,), float('nan'), dtype=torch.float64)
        )

        return {
            'image': image_tensor,
            'context_indices': context_indices,
            'target_indices': target_indices,
            'depth_vda': depth_tensor,
            'leftcamera_xy': correspondences['leftcamera_xy'],
            'rightcamera_xy': correspondences['rightcamera_xy'],
            'confidence': correspondences['confidence'],
            'scene_name': rec.scene_name,
            'split': rec.split or '',
            'neural_trial_idx': (
                rec.neural_trial_idx if rec.neural_trial_idx is not None else -1
            ),
            'neural_bin_idx': rec.neural_bin_idx if rec.neural_bin_idx is not None else -1,
            'neural_interval_sec': neural_interval_sec,
        }

    def max_neural_bin_idx(self) -> int | None:
        """Return the neural bins-per-trial count implied by this dataset's records.

        Returns:
            ``max(neural_bin_idx) + 1`` over all records that carry neural-alignment
            metadata, or ``None`` if no record does (e.g. a regular training-layout-only
            dataset with no on-disk neural alignment).
        """
        bin_idxs = [
            rec.neural_bin_idx for rec in self._records if rec.neural_bin_idx is not None
        ]
        return max(bin_idxs) + 1 if bin_idxs else None

    def _load_records(
        self,
        dataset_path: Path,
        include_splits: list[str] | None,
        val_split_ratio: float = 0.0,
        split_seed: int = 0,
    ) -> list[_PrecacheRecord]:
        """Dispatch to directory or .txt file loader.

        Args:
            dataset_path: path to precache root directory or scene JSON list .txt file.
            include_splits: split filter; see ``__init__`` for format-specific behaviour.
            val_split_ratio: fraction of records to reserve for validation when using
                the scene JSON list format. Ignored for the precache directory format.
            split_seed: RNG seed for the deterministic train/val split.

        Returns:
            list of ``_PrecacheRecord`` instances.
        """
        if dataset_path.is_dir():
            return self._load_precache_records(dataset_path, include_splits)
        if dataset_path.is_file():
            return self._load_scene_json_records(
                dataset_path,
                include_splits=include_splits,
                val_split_ratio=val_split_ratio,
                split_seed=split_seed,
            )
        raise FileNotFoundError(f'dataset_path not found: {dataset_path}')

    def _load_precache_records(
        self,
        root: Path,
        include_splits: list[str] | None,
    ) -> list[_PrecacheRecord]:
        """Load records from a precache directory.

        Expected layout::

            root/
              {session_id}/
                pair_metadata.json   # list of pairs with split, paths, frame indices
                {left_path}          # image (relative to session dir)
                {right_path}         # image (relative to session dir)

        Args:
            root: precache root directory.
            include_splits: only include pairs whose ``split`` matches an entry in this
                list. Case-insensitive. ``None`` includes all splits.

        Returns:
            list of ``_PrecacheRecord``.
        """
        allowed = (
            {s.lower() for s in include_splits} if include_splits is not None else None
        )
        records: list[_PrecacheRecord] = []
        for session_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            metadata_path = session_dir / 'pair_metadata.json'
            if not metadata_path.exists():
                continue
            with metadata_path.open('r', encoding='utf-8') as fh:
                payload = json.load(fh)
            pairs = payload.get('pairs', [])
            for item in pairs:
                split = str(item.get('split', '')).strip().lower()
                if allowed is not None and split not in allowed:
                    continue
                left_rel = item.get('left_path')
                right_rel = item.get('right_path')
                if left_rel is None or right_rel is None:
                    continue
                left_path = session_dir / str(left_rel)
                right_path = session_dir / str(right_rel)
                if not left_path.exists() or not right_path.exists():
                    continue
                pair_idx = int(item['pair_idx'])
                records.append(_PrecacheRecord(
                    session_id=session_dir.name,
                    pair_idx=pair_idx,
                    left_path=left_path,
                    right_path=right_path,
                    left_source_frame_index=int(item['left_source_frame_index']),
                    right_source_frame_index=int(item['right_source_frame_index']),
                    scene_name=f'{session_dir.name}_pair_{pair_idx:06d}',
                ))
        if not records:
            split_hint = f' (splits={include_splits})' if include_splits else ''
            raise RuntimeError(f'No valid pairs found in {root}{split_hint}')
        return records

    def _load_scene_json_records(
        self,
        dataset_path: Path,
        include_splits: list[str] | None = None,
        val_split_ratio: float = 0.0,
        split_seed: int = 0,
    ) -> list[_PrecacheRecord]:
        """Load records from a .txt file listing scene JSON paths, one per line.

        When ``val_split_ratio > 0`` and ``include_splits`` is set, performs a
        deterministic random split: the last ``ceil(N * val_split_ratio)`` records
        (after shuffling with ``split_seed``) become the val set; the rest are train.
        If ``val_split_ratio == 0`` or ``include_splits`` is ``None``, all records
        are returned regardless of the requested split.

        Args:
            dataset_path: path to .txt file.
            include_splits: which split to return (``['train']`` or ``['val']``).
            val_split_ratio: fraction of records reserved for validation (0–1).
            split_seed: RNG seed for the deterministic shuffle.

        Returns:
            list of ``_PrecacheRecord``.
        """
        records: list[_PrecacheRecord] = []
        with dataset_path.open('r', encoding='utf-8') as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                scene_path = Path(stripped)
                if not scene_path.exists():
                    log_step(f'scene JSON not found, skipping: {scene_path}', level='warning')
                    continue
                try:
                    rec = self._parse_scene_json(scene_path)
                except Exception as exc:
                    log_step(f'failed to parse {scene_path}: {exc}', level='warning')
                    continue
                records.append(rec)
        if not records:
            raise RuntimeError(f'No valid scene JSON paths found in {dataset_path}')

        return self._split_records(
            records,
            include_splits=include_splits,
            val_split_ratio=val_split_ratio,
            split_seed=split_seed,
            source_desc=str(dataset_path),
        )

    @staticmethod
    def _split_records(
        records: list[_PrecacheRecord],
        include_splits: list[str] | None,
        val_split_ratio: float,
        split_seed: int,
        source_desc: str,
    ) -> list[_PrecacheRecord]:
        """Deterministically split records into train/val and select the requested splits.

        The last ``ceil(N * val_split_ratio)`` records (after shuffling with
        ``split_seed``) become the val set; the rest are train. If ``val_split_ratio
        == 0`` or ``include_splits`` is ``None``, all records are returned regardless
        of the requested split.

        Args:
            records: full list of records to split.
            include_splits: which splits to return (``['train']`` or ``['val']``).
            val_split_ratio: fraction of records reserved for validation (0-1).
            split_seed: RNG seed for the deterministic shuffle.
            source_desc: human-readable description of the record source, used in
                error messages.

        Returns:
            list of ``_PrecacheRecord`` for the requested splits.
        """
        if val_split_ratio <= 0.0 or include_splits is None:
            return records

        rng = np.random.default_rng(split_seed)
        idx_shuffled = rng.permutation(len(records)).tolist()
        n_val = max(1, round(len(records) * val_split_ratio))
        idx_val = set(idx_shuffled[-n_val:])
        idx_train = [i for i in idx_shuffled if i not in idx_val]

        split_map: dict[str, list[_PrecacheRecord]] = {
            'train': [records[i] for i in idx_train],
            'val': [records[i] for i in sorted(idx_val)],
        }
        selected: list[_PrecacheRecord] = []
        for split_name in include_splits:
            selected.extend(split_map.get(split_name.lower(), []))

        if not selected:
            raise RuntimeError(
                f'No records after applying val_split_ratio={val_split_ratio} '
                f'for splits={include_splits} in {source_desc}'
            )
        return selected

    @staticmethod
    def _parse_scene_json(scene_path: Path) -> _PrecacheRecord:
        """Parse a scene JSON file into a ``_PrecacheRecord``.

        Scene JSON fields used: ``session_id``, ``pair_id``, ``frames`` (each with
        ``file_path``, ``source_frame_index``, ``camera_name``).
        Image paths are resolved as ``scene_path.parent.parent / frame['file_path']``.

        Args:
            scene_path: path to scene JSON file.

        Returns:
            ``_PrecacheRecord``.
        """
        with scene_path.open('r', encoding='utf-8') as fh:
            payload = json.load(fh)
        frames = payload.get('frames', [])
        if len(frames) != 2:
            raise RuntimeError(f'Expected 2 frames, got {len(frames)}: {scene_path}')
        scene_root = scene_path.parent.parent
        by_camera = {
            str(f.get('camera_name', '')).strip().lower(): f for f in frames
        }
        left_frame = by_camera.get('left', frames[0])
        right_frame = by_camera.get('right', frames[1])
        return _PrecacheRecord(
            session_id=str(payload.get('session_id', scene_path.stem)),
            pair_idx=int(payload['pair_id']) if payload.get('pair_id') is not None else 0,
            left_path=scene_root / str(left_frame['file_path']),
            right_path=scene_root / str(right_frame['file_path']),
            left_source_frame_index=int(left_frame['source_frame_index']),
            right_source_frame_index=int(right_frame['source_frame_index']),
            scene_name=str(payload.get('scene_name', scene_path.stem)),
        )

    def _load_image(self, path: Path) -> tuple[torch.Tensor, int, int]:
        """Load a PNG/JPG image and resize to ``image_size``.

        Args:
            path: path to image file.

        Returns:
            tuple of (float32 tensor [3, H, W] in range [0, 1], original_width, original_height).
        """
        with Image.open(path) as img:
            img = img.convert('RGB')
            orig_w, orig_h = img.size
            if orig_w != self._image_size or orig_h != self._image_size:
                img = img.resize((self._image_size, self._image_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), orig_w, orig_h

    def _load_mask(self, path: Path) -> torch.Tensor:
        """Load a binary segmentation mask and resize to ``image_size x image_size``.

        Args:
            path: path to a single-channel mask PNG with values in ``{0, 255}``.

        Returns:
            float32 tensor ``[1, image_size, image_size]`` with values in ``{0, 1}``.
        """
        with Image.open(path) as img:
            arr = np.asarray(img.convert('L'), dtype=np.float32)
        mask = torch.from_numpy(arr > 0).to(torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        if mask.shape[-2] != self._image_size or mask.shape[-1] != self._image_size:
            mask = F.interpolate(
                mask,
                size=(self._image_size, self._image_size),
                mode='nearest',
            )
        return mask.squeeze(0)  # [1, S, S]

    def _load_vda_depth(
        self,
        session_id: str,
        camera_name: str,
        source_frame_index: int,
    ) -> torch.Tensor:
        """Load one precomputed VDA depth map and resize to ``image_size``.

        Expected path: ``{vda_cache_root}/{session_id}/{camera_name}/{frame_idx:06d}.npy``

        Args:
            session_id: session identifier (subdirectory name).
            camera_name: ``'left'`` or ``'right'``.
            source_frame_index: frame index used in the cache file name.

        Returns:
            float32 tensor [1, H, W] where H = W = ``image_size``.

        Raises:
            FileNotFoundError: if the depth cache file does not exist.
        """
        depth_path = (
            self._vda_cache_root
            / session_id
            / camera_name
            / f'{source_frame_index:06d}.npy'
        )
        if not depth_path.exists():
            raise FileNotFoundError(
                f'VDA depth cache not found: {depth_path}\n'
                'Make sure model.vda.cache_root points to the precomputed depth directory.'
            )
        depth_np = np.load(str(depth_path)).astype(np.float32)
        depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
        if depth_t.shape[-2] != self._image_size or depth_t.shape[-1] != self._image_size:
            depth_t = F.interpolate(
                depth_t,
                size=(self._image_size, self._image_size),
                mode='bilinear',
                align_corners=False,
            )
        return depth_t.squeeze(0)   # [1, H, W]

    def _load_correspondences(
        self,
        session_id: str,
        pair_idx: int,
        left_orig_size: tuple[int, int],
        right_orig_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Load pixel correspondences from a litpose .npz bundle and rescale to image_size.

        Expected path: ``{corr_root}/{session_id}/pair_{pair_idx:06d}/litpose_matches.npz``

        Fields in the .npz: ``left_xy [N,2]``, ``right_xy [N,2]``, ``confidence [N]``.
        Coordinates are in the original camera pixel space and are scaled to
        ``image_size × image_size`` to match the resized image tensor.
        All output tensors are zero-padded to ``_MAX_MATCHES`` entries; padding rows have
        ``confidence == 0`` so callers can derive validity with ``confidence > 0``.

        If the bundle is missing or ``model.merge_pcd.correspondence_cache_root`` is unset, returns
        all-zero tensors. The model's Kabsch step handles this gracefully by falling back
        to ICP without correspondence hints.

        Args:
            session_id: session identifier.
            pair_idx: pair index.
            left_orig_size: original (width, height) of the left image before resizing.
            right_orig_size: original (width, height) of the right image before resizing.

        Returns:
            dict with keys ``leftcamera_xy``, ``rightcamera_xy``, ``confidence``.
        """
        if self._corr_root is None:
            return self._empty_correspondences()

        bundle_path = (
            self._corr_root
            / session_id
            / f'pair_{pair_idx:06d}'
            / 'litpose_matches.npz'
        )
        if not bundle_path.exists():
            log_step(f'correspondence bundle not found, using empty: {bundle_path}', level='debug')
            return self._empty_correspondences()

        try:
            payload = np.load(str(bundle_path), allow_pickle=True)
            left_xy = np.asarray(payload['left_xy'], dtype=np.float32)
            right_xy = np.asarray(payload['right_xy'], dtype=np.float32)
            confidence = np.asarray(payload['confidence'], dtype=np.float32)
        except Exception as exc:
            log_step(f'failed to load correspondence bundle {bundle_path}: {exc}', level='warning')
            return self._empty_correspondences()

        n = min(int(len(confidence)), _MAX_MATCHES)
        padded_left = np.zeros((_MAX_MATCHES, 2), dtype=np.float32)
        padded_right = np.zeros((_MAX_MATCHES, 2), dtype=np.float32)
        padded_conf = np.zeros(_MAX_MATCHES, dtype=np.float32)

        padded_left[:n] = left_xy[:n]
        padded_right[:n] = right_xy[:n]
        padded_conf[:n] = confidence[:n]

        # scale coordinates from original camera pixel space → image_size × image_size
        lw, lh = left_orig_size
        rw, rh = right_orig_size
        padded_left[:n, 0] *= self._image_size / lw
        padded_left[:n, 1] *= self._image_size / lh
        padded_right[:n, 0] *= self._image_size / rw
        padded_right[:n, 1] *= self._image_size / rh

        return {
            'leftcamera_xy': torch.from_numpy(padded_left),
            'rightcamera_xy': torch.from_numpy(padded_right),
            'confidence': torch.from_numpy(padded_conf),
        }

    def _empty_correspondences(self) -> dict[str, torch.Tensor]:
        """Return zero-padded correspondence tensors (all confidence zero)."""
        return {
            'leftcamera_xy': torch.zeros(_MAX_MATCHES, 2, dtype=torch.float32),
            'rightcamera_xy': torch.zeros(_MAX_MATCHES, 2, dtype=torch.float32),
            'confidence': torch.zeros(_MAX_MATCHES, dtype=torch.float32),
        }

    def _resolve_view_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve context and target view indices for the training regime.

        For ``all_views_reconstruction`` all views are both context and target.
        For ``fixed_1to1`` index 0 is context, index 1 is target.

        Returns:
            tuple of (context_indices, target_indices) as long tensors.

        Raises:
            ValueError: if ``training_regime`` is not a recognised built-in value.
                Subclass and override this method to add a custom regime.
        """
        if self._training_regime == 'all_views_reconstruction':
            all_idx = torch.arange(2, dtype=torch.long)
            return all_idx, all_idx.clone()
        elif self._training_regime == 'fixed_1to1':
            return torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long)
        else:
            raise ValueError(
                f'Unsupported training_regime: {self._training_regime!r}. '
                'Built-in values: all_views_reconstruction, fixed_1to1. '
                'To use a custom regime, subclass and override _resolve_view_indices.'
            )

    @staticmethod
    def _discover_eval_split_records(
        left_dir: Path,
        right_dir: Path,
        session_id: str,
        include_splits: list[str] | None,
        segmentation_root: Path | None = None,
    ) -> list[_PrecacheRecord]:
        """Discover stereo pairs from an eval-set layout for one session.

        Shared by :class:`IBLTwoViewDataset` (raw IBL frames) and :class:`Cheese3DDataset`
        (frames extracted from Cheese3D ephys videos) — both use the same on-disk contract,
        differing only in which two camera directories they pass as ``left_dir``/``right_dir``.

        Eval frames live under a ``{train,val,test}`` split subdirectory of the
        session directory — a fixed, on-disk split (driven by ``neural_trial_idx``
        upstream) rather than the synthetic ``val_split_ratio`` split used for the
        training layout — and are named ``interval{N}timebin{M}.png`` instead of
        ``img{N:08d}.png``. Each split directory has a ``frame_index_mapping.json``
        mapping filename to its raw ``*_source_frame_index``, e.g.::

            {"interval0timebin0.png": {"left_source_frame_index": 19834, ...}}

        (the right-camera mapping file uses ``right_source_frame_index`` instead).
        Left/right frames are paired by identical filename — both cameras share the
        same filenames within a split — rather than by a reconstructed numeric
        index, and ``source_frame_index`` is looked up from the mapping instead of
        parsed from the filename.

        Each mapping entry also carries ``neural_trial_idx``, ``neural_bin_idx``, and
        ``neural_interval_sec`` (from the left-camera mapping file), used to align saved
        latents back into neural trials downstream; missing fields fall back to ``None``
        for datasets predating this metadata.

        ``pair_idx`` is a single counter incremented across all requested splits
        within the session (not reset per split), so it stays unique per session —
        required by downstream consumers that key saved artifacts on
        ``(session_id, pair_idx)`` alone.

        Args:
            left_dir: session directory for the "left" camera (role assignment, not
                necessarily a physically left-facing camera).
            right_dir: session directory for the "right" camera.
            session_id: session identifier.
            include_splits: on-disk split subdirectories to include (``train``,
                ``val``, ``test``); ``None`` includes all of them.
            segmentation_root: root directory of precomputed SAM3 masks written by
                ``beast/preprocess/sable/precompute_sam3_masks_eval.py``, i.e.
                ``{segmentation_root}/segmentation_masks/{session_id}/{left,right}/
                mask{frame_idx:08d}.png``. ``None`` disables mask loading (records get
                ``left_mask_path=right_mask_path=None``). Mask files are not required to
                exist yet — a missing file raises ``FileNotFoundError`` when loaded.

        Returns:
            list of ``_PrecacheRecord`` instances, empty if none found.
        """
        splits = include_splits if include_splits else ['train', 'val', 'test']
        records: list[_PrecacheRecord] = []
        pair_idx = 0
        for split_name in splits:
            left_split_dir = left_dir / split_name
            right_split_dir = right_dir / split_name
            left_mapping_path = left_split_dir / 'frame_index_mapping.json'
            right_mapping_path = right_split_dir / 'frame_index_mapping.json'
            if not left_mapping_path.is_file() or not right_mapping_path.is_file():
                continue

            with open(left_mapping_path) as f:
                left_mapping = json.load(f)
            with open(right_mapping_path) as f:
                right_mapping = json.load(f)

            common_filenames = sorted(set(left_mapping) & set(right_mapping))
            for filename in common_filenames:
                entry = left_mapping[filename]
                neural_trial_idx = entry.get('neural_trial_idx')
                neural_bin_idx = entry.get('neural_bin_idx')
                neural_interval_sec_raw = entry.get('neural_interval_sec')
                left_source_frame_index = int(left_mapping[filename]['left_source_frame_index'])
                right_source_frame_index = int(
                    right_mapping[filename]['right_source_frame_index']
                )
                records.append(_PrecacheRecord(
                    session_id=session_id,
                    pair_idx=pair_idx,
                    left_path=left_split_dir / filename,
                    right_path=right_split_dir / filename,
                    left_source_frame_index=left_source_frame_index,
                    right_source_frame_index=right_source_frame_index,
                    scene_name=f'{session_id}_pair_{pair_idx:06d}',
                    left_mask_path=(
                        segmentation_root
                        / 'segmentation_masks'
                        / session_id
                        / 'left'
                        / f'mask{left_source_frame_index:08d}.png'
                        if segmentation_root is not None
                        else None
                    ),
                    right_mask_path=(
                        segmentation_root
                        / 'segmentation_masks'
                        / session_id
                        / 'right'
                        / f'mask{right_source_frame_index:08d}.png'
                        if segmentation_root is not None
                        else None
                    ),
                    split=split_name,
                    neural_trial_idx=(
                        int(neural_trial_idx) if neural_trial_idx is not None else None
                    ),
                    neural_bin_idx=int(neural_bin_idx) if neural_bin_idx is not None else None,
                    neural_interval_sec=(
                        (float(neural_interval_sec_raw[0]), float(neural_interval_sec_raw[1]))
                        if neural_interval_sec_raw is not None
                        else None
                    ),
                ))
                pair_idx += 1

        return records

    @staticmethod
    def _parse_frame_indices(directory: Path) -> set[int]:
        """Parse integer frame indices from ``img*.png`` filenames in a directory.

        Args:
            directory: directory containing ``img{N:08d}.png`` image files.

        Returns:
            set of integer frame indices found.
        """
        indices: set[int] = set()
        for p in directory.glob('img*.png'):
            m = re.search(r'(\d+)\.png$', p.name)
            if m:
                indices.add(int(m.group(1)))
        return indices


class IBLTwoViewDataset(SABLEDataset):
    """Two-view IBL dataset that discovers frame pairs from the raw IBL filesystem layout.

    Like :class:`~beast.data.sable_dataset.SABLEDataset` but:

    * Records are discovered from the filesystem using ``training.dataset_path``, which
      should point to the root containing ``leftCamera.video/`` and ``rightCamera.video/``
      subdirectories (i.e. the raw IBL extracted-frames root).
    * VDA depth is **optional** — loaded from
      ``{model.vda.cache_root}/{session_id}/{camera}/depth{frame_idx:08d}.npy``.
      Returns a zero depth tensor with a debug log when the file is absent.
    * Correspondence files are **optional** — loaded from
      ``{model.merge_pcd.correspondence_cache_root}/{session_id}/
      correspondences{pair_idx:08d}.npz``.
      The bundle contains ``left_xy [K, 2]``, ``right_xy [K, 2]``, and
      ``confidence [K]`` arrays in native IBL pixel space; coordinates are rescaled
      to ``image_size × image_size`` and padded to ``_MAX_MATCHES`` at load time.
      Returns zero tensors when the bundle is absent.

    Config keys read:

    * ``training.dataset_path`` — raw IBL frames root (required)
    * ``training.session_names`` — list of session IDs to load; auto-discovers when null
    * ``model.vda.cache_root`` — precomputed VDA depth cache root
    * ``model.merge_pcd.correspondence_cache_root`` — precomputed correspondence cache root
    * ``model.image_tokenizer.image_size``
    * ``training.training_regime``
    * ``training.val_split_ratio``
    * ``model.seed``
    * ``training.use_segmentation.enabled`` / ``training.use_segmentation.cache_root`` —
      optional SAM3 mask loading for eval-layout sessions (see
      :meth:`SABLEDataset._discover_eval_split_records`)
    """

    def __init__(
        self,
        config: dict,
        include_splits: list[str] | None = None,
    ) -> None:
        """Initialize.

        Args:
            config: full beast config dict.
            include_splits: split filter; see :class:`SABLEDataset` for details.
        """
        # bypass SABLEDataset.__init__'s hard requirement on vda cache_root by
        # calling Dataset.__init__ directly and re-implementing init logic
        Dataset.__init__(self)
        self.config = config

        training = config['training']
        model_cfg = config['model']

        dataset_path = training.get('dataset_path')
        if not dataset_path:
            raise ValueError('training.dataset_path must be set.')
        val_split_ratio = float(training.get('val_split_ratio', 0.0))
        split_seed = int(model_cfg.get('seed', 0))

        seg_cfg: dict = training.get('use_segmentation') or {}
        segmentation_root: Path | None = None
        if bool(seg_cfg.get('enabled', False)):
            segmentation_root_raw = seg_cfg.get('cache_root')
            if not segmentation_root_raw:
                raise ValueError(
                    'training.use_segmentation.cache_root must be set when '
                    'training.use_segmentation.enabled is true.'
                )
            segmentation_root = Path(segmentation_root_raw)

        self._records: list[_PrecacheRecord] = self._discover_filesystem_records(
            image_root=Path(dataset_path),
            session_names=training.get('session_names'),
            include_splits=include_splits,
            val_split_ratio=val_split_ratio,
            split_seed=split_seed,
            segmentation_root=segmentation_root,
        )

        vda_cfg = model_cfg.get('vda', {}) or {}
        vda_cache_root = vda_cfg.get('cache_root')
        self._vda_cache_root: Path | None = Path(vda_cache_root) if vda_cache_root else None

        merge_pcd_cfg = model_cfg.get('merge_pcd', {}) or {}
        corr_root = merge_pcd_cfg.get('correspondence_cache_root')
        self._corr_root: Path | None = Path(corr_root) if corr_root else None

        # dataset_path is the raw frames root, not a precache dir, so the depth/
        # correspondence fallback via _dataset_root is not applicable here.
        self._dataset_root: Path | None = None

        self._image_size: int = int(model_cfg['image_tokenizer']['image_size'])
        self._training_regime: str = str(
            training.get('training_regime', 'all_views_reconstruction')
        ).strip().lower()

    # ------------------------------------------------------------------
    # filesystem discovery (no JSON required)
    # ------------------------------------------------------------------

    def _discover_filesystem_records(
        self,
        image_root: Path,
        session_names: list[str] | str | None,
        include_splits: list[str] | None,
        val_split_ratio: float,
        split_seed: int,
        segmentation_root: Path | None = None,
    ) -> list[_PrecacheRecord]:
        """Discover stereo pairs from the IBL filesystem layout without a JSON index.

        Two layouts are supported, tried in this order per session:

        1. Training layout — images directly under the session directory::

               {image_root}/leftCamera.video/_iblrig_leftCamera.downsampled.{session_id}/img{N:08d}.png
               {image_root}/rightCamera.video/_iblrig_rightCamera.downsampled.{session_id}/img{N:08d}.png

           Frame indices are parsed from filenames. The sorted position of each
           frame index within a session becomes its ``pair_idx`` (used for
           correspondence file lookup); the index value itself becomes
           ``source_frame_index`` (used for depth file lookup).

        2. Eval layout — a ``{train,val,test}`` split subdirectory between the
           session directory and the images, with ``interval{N}timebin{M}.png``
           filenames and a ``frame_index_mapping.json`` per split directory mapping
           filename to ``{left,right}_source_frame_index``. See
           :meth:`_discover_eval_split_records` for details. Used automatically
           when no ``img*.png`` files are found directly under the session
           directory.

        Args:
            image_root: base directory containing ``leftCamera.video/`` and
                ``rightCamera.video/`` subdirectories.
            session_names: explicit session IDs to use. Accepts a list of strings or
                a single string (for CLI override convenience). When ``None``,
                auto-discovers sessions by scanning ``{image_root}/leftCamera.video/``.
            include_splits: split filter passed to :meth:`_split_records`.
            val_split_ratio: fraction of records reserved for validation.
            split_seed: RNG seed for the deterministic train/val split.
            segmentation_root: root directory of precomputed SAM3 masks, passed through
                to eval-layout records only (see
                :meth:`SABLEDataset._discover_eval_split_records`); ``None`` disables
                mask loading.

        Returns:
            list of ``_PrecacheRecord`` instances.

        Raises:
            RuntimeError: if no valid stereo pairs are found.
        """
        left_video_dir = image_root / 'leftCamera.video'
        right_video_dir = image_root / 'rightCamera.video'

        if session_names is None:
            session_ids = sorted(
                p.name.split('.')[-1]
                for p in left_video_dir.iterdir()
                if p.is_dir() and p.name.startswith('_iblrig_leftCamera.downsampled.')
            )
        elif isinstance(session_names, str):
            session_ids = [session_names]
        else:
            session_ids = list(session_names)

        self.session_ids = session_ids
        self.session_id_to_idx = {session_id: idx for idx, session_id in enumerate(session_ids)}

        records: list[_PrecacheRecord] = []
        eval_records: list[_PrecacheRecord] = []
        for session_id in session_ids:
            left_dir = left_video_dir / f'_iblrig_leftCamera.downsampled.{session_id}'
            right_dir = right_video_dir / f'_iblrig_rightCamera.downsampled.{session_id}'

            if not left_dir.is_dir() or not right_dir.is_dir():
                _logger.warning(
                    'skipping session %s: image dirs not found (%s, %s)',
                    session_id,
                    left_dir,
                    right_dir,
                )
                continue

            left_indices = self._parse_frame_indices(left_dir)
            right_indices = self._parse_frame_indices(right_dir)
            common = sorted(left_indices & right_indices)

            if common:
                for pair_idx, source_frame_index in enumerate(common):
                    records.append(_PrecacheRecord(
                        session_id=session_id,
                        pair_idx=pair_idx,
                        left_path=left_dir / f'img{source_frame_index:08d}.png',
                        right_path=right_dir / f'img{source_frame_index:08d}.png',
                        left_source_frame_index=source_frame_index,
                        right_source_frame_index=source_frame_index,
                        scene_name=f'{session_id}_pair_{pair_idx:06d}',
                    ))
                continue

            session_eval_records = self._discover_eval_split_records(
                left_dir=left_dir,
                right_dir=right_dir,
                session_id=session_id,
                include_splits=include_splits,
                segmentation_root=segmentation_root,
            )
            if not session_eval_records:
                _logger.warning('skipping session %s: no common frame indices', session_id)
                continue
            eval_records.extend(session_eval_records)

        if not records and not eval_records:
            raise RuntimeError(
                f'No valid stereo pairs found under {image_root} for sessions {session_ids}'
            )

        if records:
            records = self._split_records(
                records,
                include_splits=include_splits,
                val_split_ratio=val_split_ratio,
                split_seed=split_seed,
                source_desc=str(image_root),
            )

        # eval-layout records already carry a fixed, on-disk train/val/test split
        # (see _discover_eval_split_records), so they bypass _split_records's
        # synthetic val_split_ratio-based reshuffling entirely.
        return records + eval_records

    # ------------------------------------------------------------------
    # overrides
    # ------------------------------------------------------------------

    def _resolve_view_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve context (input) and target view indices for the training regime.

        For ``all_views_reconstruction`` all views are both context and target.
        For ``fixed_1to1`` index 0 is context, index 1 is target.
        For ``pseudo_center_finetune`` all three views (left, right, pseudo center)
        are context but only left and right (indices 0, 1) are targets for loss;
        the pseudo center view has no ground truth so it is excluded from the loss.

        Returns:
            tuple of (context_indices, target_indices) as long tensors.

        Raises:
            ValueError: if ``training_regime`` is not a recognised built-in value.
                Subclass and override this method to add a custom regime.
        """
        if self._training_regime == 'all_views_reconstruction':
            all_idx = torch.arange(2, dtype=torch.long)
            return all_idx, all_idx.clone()
        elif self._training_regime == 'fixed_1to1':
            return torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long)
        elif self._training_regime == 'pseudo_center_finetune':
            return (
                torch.arange(3, dtype=torch.long),
                torch.tensor([0, 1], dtype=torch.long),
            )
        else:
            raise ValueError(
                f'Unsupported training_regime: {self._training_regime!r}. '
                'Built-in values: all_views_reconstruction, fixed_1to1, pseudo_center_finetune. '
                'To use a custom regime, subclass and override _resolve_view_indices.'
            )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load one stereo pair, or a three-view bundle for ``pseudo_center_finetune``.

        In ``pseudo_center_finetune`` mode a zero-filled pseudo center image is
        appended as the third view so the encoder sees the same 3-view structure
        as during Cheese3D center-camera-holdout pretraining.  The pseudo center
        view's tokens are masked via ``context_full_mask`` and it is excluded from
        ``target_indices``, so it contributes neither image context nor loss.

        Args:
            idx: dataset index.

        Returns:
            dict with keys ``image``, ``context_indices``, ``target_indices``,
            ``depth_vda``, ``leftcamera_xy``, ``rightcamera_xy``, ``confidence``,
            ``scene_name``, ``session_idx``, ``split``, ``neural_trial_idx``,
            ``neural_bin_idx``, ``neural_interval_sec``, and (for
            ``pseudo_center_finetune``) ``context_full_mask``. Also includes
            ``mask`` (shape ``[2, 1, H, W]``, float32, 1 = foreground) when
            ``training.use_segmentation.enabled`` is true and this record has mask
            paths (eval-layout records only).
        """
        rec = self._records[idx]

        left_img, left_orig_w, left_orig_h = self._load_image(rec.left_path)
        right_img, right_orig_w, right_orig_h = self._load_image(rec.right_path)

        vda_depths = [
            self._load_vda_depth_sable(rec.session_id, 'left', rec.left_source_frame_index),
            self._load_vda_depth_sable(rec.session_id, 'right', rec.right_source_frame_index),
        ]

        if self._training_regime == 'pseudo_center_finetune':
            pseudo_center = torch.zeros_like(left_img)   # [3, H, W], black image
            image_tensor = torch.stack([left_img, right_img, pseudo_center], dim=0)   # [3, 3, H, W]
            pseudo_depth = torch.zeros(1, self._image_size, self._image_size, dtype=torch.float32)
            depth_tensor = torch.stack([*vda_depths, pseudo_depth], dim=0)             # [3, 1, H, W]
        else:
            image_tensor = torch.stack([left_img, right_img], dim=0)   # [2, 3, H, W]
            depth_tensor = torch.stack(vda_depths, dim=0)               # [2, 1, H, W]

        context_indices, target_indices = self._resolve_view_indices()
        correspondences = self._load_correspondences_sable(
            session_id=rec.session_id,
            pair_idx=rec.pair_idx,
            left_orig_size=(left_orig_w, left_orig_h),
            right_orig_size=(right_orig_w, right_orig_h),
        )

        result = {
            'image': image_tensor,
            'context_indices': context_indices,
            'target_indices': target_indices,
            'depth_vda': depth_tensor,
            'leftcamera_xy': correspondences['leftcamera_xy'],
            'rightcamera_xy': correspondences['rightcamera_xy'],
            'confidence': correspondences['confidence'],
            'scene_name': rec.scene_name,
            'session_idx': self.session_id_to_idx[rec.session_id],
            'split': rec.split or '',
            'neural_trial_idx': (
                rec.neural_trial_idx if rec.neural_trial_idx is not None else -1
            ),
            'neural_bin_idx': rec.neural_bin_idx if rec.neural_bin_idx is not None else -1,
            'neural_interval_sec': (
                torch.tensor(rec.neural_interval_sec, dtype=torch.float64)
                if rec.neural_interval_sec is not None
                else torch.full((2,), float('nan'), dtype=torch.float64)
            ),
        }

        if self._training_regime == 'pseudo_center_finetune':
            # center view (index 2) is pseudo — zero out all its image tokens in the encoder
            result['context_full_mask'] = torch.tensor([False, False, True], dtype=torch.bool)

        if rec.left_mask_path is not None and rec.right_mask_path is not None:
            left_mask = self._load_mask(rec.left_mask_path)    # [1, H, W]
            right_mask = self._load_mask(rec.right_mask_path)  # [1, H, W]
            result['mask'] = torch.stack([left_mask, right_mask], dim=0)  # [2, 1, H, W]

        return result

    # ------------------------------------------------------------------
    # VDA depth loading
    # ------------------------------------------------------------------

    def _load_vda_depth_sable(
        self,
        session_id: str,
        camera_name: str,
        source_frame_index: int,
    ) -> torch.Tensor:
        """Load VDA depth from the configured cache root.

        Path: ``{model.vda.cache_root}/{session_id}/{camera_name}/depth{source_frame_index:08d}.npy``

        Args:
            session_id: session identifier (subdirectory name).
            camera_name: camera subdirectory name (e.g. ``'left'``, ``'right'``).
            source_frame_index: raw IBL video frame index (parsed from the image filename,
                e.g. ``img00045089.png`` → 45089); used as the depth filename suffix.

        Returns:
            float32 tensor [1, H, W] where H = W = ``image_size``.
        """
        if self._vda_cache_root is not None:
            depth_path = (
                self._vda_cache_root
                / session_id
                / camera_name
                / f'depth{source_frame_index:08d}.npy'
            )
        elif self._dataset_root is not None:
            depth_path = (
                self._dataset_root
                / session_id
                / camera_name
                / f'depth{source_frame_index:08d}.npy'
            )
        else:
            return torch.zeros(1, self._image_size, self._image_size, dtype=torch.float32)
        if not depth_path.exists():
            raise FileNotFoundError(
                f'VDA depth not found: {depth_path}. '
                'Run beast extract_sable with vda.enabled: true to precompute depth, '
                'then set model.vda.cache_root to the extraction output directory.'
            )

        depth_np = np.load(str(depth_path)).astype(np.float32)
        depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
        if depth_t.shape[-2] != self._image_size or depth_t.shape[-1] != self._image_size:
            depth_t = F.interpolate(
                depth_t,
                size=(self._image_size, self._image_size),
                mode='bilinear',
                align_corners=False,
            )
        return depth_t.squeeze(0)   # [1, H, W]

    # ------------------------------------------------------------------
    # correspondence loading
    # ------------------------------------------------------------------

    def _load_correspondences_sable(
        self,
        session_id: str,
        pair_idx: int,
        left_orig_size: tuple[int, int],
        right_orig_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Load a session-level correspondence bundle and rescale to image_size.

        Path resolution (first that applies):

        1. ``model.merge_pcd.correspondence_cache_root`` is set →
           ``{correspondence_cache_root}/{session_id}/correspondences{pair_idx:08d}.npz``
        2. ``training.dataset_path`` is set →
           ``{dataset_root}/litpose_correspondences/processed_correspondences/
           {session_id}/correspondences{pair_idx:08d}.npz``
        3. Neither set → returns empty tensors.

        The .npz bundle contains ``left_xy [K, 2]``, ``right_xy [K, 2]``, and
        ``confidence [K]`` arrays in native IBL pixel space.  Coordinates are
        rescaled to ``image_size × image_size`` and tensors are zero-padded to
        ``_MAX_MATCHES`` entries (padding rows have ``confidence == 0``).

        Returns all-zero tensors when the bundle is absent.

        Args:
            session_id: session identifier (subdirectory name).
            pair_idx: pair index used in the filename.
            left_orig_size: original (width, height) of the left image.
            right_orig_size: original (width, height) of the right image.

        Returns:
            dict with keys ``leftcamera_xy``, ``rightcamera_xy``, ``confidence``.
        """
        if self._corr_root is not None:
            bundle_path = self._corr_root / session_id / f'correspondences{pair_idx:08d}.npz'
        elif self._dataset_root is not None:
            bundle_path = (
                self._dataset_root
                / 'litpose_correspondences'
                / 'processed_correspondences'
                / session_id
                / f'correspondences{pair_idx:08d}.npz'
            )
        else:
            log_step(
                f'no correspondence cache root configured, returning empty for session {session_id} '
                f'pair {pair_idx}',
                level='debug',
            )
            return self._empty_correspondences()
        if not bundle_path.exists():
            log_step(
                f'correspondence bundle not found, using empty: {bundle_path}',
                level='debug',
            )
            return self._empty_correspondences()

        try:
            payload = np.load(str(bundle_path), allow_pickle=True)
            left_xy = np.asarray(payload['left_xy'], dtype=np.float32)
            right_xy = np.asarray(payload['right_xy'], dtype=np.float32)
            confidence = np.asarray(payload['confidence'], dtype=np.float32)
        except Exception as exc:
            log_step(
                f'failed to load correspondence bundle {bundle_path}: {exc}',
                level='warning',
            )
            return self._empty_correspondences()

        n = min(int(len(confidence)), _MAX_MATCHES)
        padded_left = np.zeros((_MAX_MATCHES, 2), dtype=np.float32)
        padded_right = np.zeros((_MAX_MATCHES, 2), dtype=np.float32)
        padded_conf = np.zeros(_MAX_MATCHES, dtype=np.float32)

        padded_left[:n] = left_xy[:n]
        padded_right[:n] = right_xy[:n]
        padded_conf[:n] = confidence[:n]

        # rescale coordinates from native pixel space → image_size × image_size
        lw, lh = left_orig_size
        rw, rh = right_orig_size
        padded_left[:n, 0] *= self._image_size / lw
        padded_left[:n, 1] *= self._image_size / lh
        padded_right[:n, 0] *= self._image_size / rw
        padded_right[:n, 1] *= self._image_size / rh

        return {
            'leftcamera_xy': torch.from_numpy(padded_left),
            'rightcamera_xy': torch.from_numpy(padded_right),
            'confidence': torch.from_numpy(padded_conf),
        }


# fixed correspondence points used by Cheese3DDataset for every scene and camera,
# in native (320x256) pixel space; rescaled to image_size x image_size at load time
_CHEESE3D_FIXED_XY = torch.tensor(
    [
        [134.8540, 210.1205],
        [113.2900, 189.3839],
        [32.9249, 69.7657],
    ],
    dtype=torch.float32,
)
_CHEESE3D_FIXED_CONFIDENCE = torch.ones(3, dtype=torch.float32)


def _npy_to_c2w_fxfycxcy(npy_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a .npy calibration file and return c2w and normalised fxfycxcy.

    Intrinsics are normalised by the original image dimensions so that after the
    model de-normalises by [target_w, target_h, target_w, target_h] the result is
    the correct pixel-space focal length / principal point for the stretched image.

    Args:
        npy_path: path to an ``img<index>.npy`` file with keys ``intrinsics``,
            ``extrinsics``, ``width``, ``height``.

    Returns:
        c2w: float32 ``[4, 4]`` camera-to-world matrix.
        fxfycxcy: float32 ``[4]`` normalised intrinsics ``[fx/W, fy/H, cx/W, cy/H]``.
    """
    cam = np.load(npy_path, allow_pickle=True).item()
    w2c = cam['extrinsics'].astype(np.float32)
    c2w = np.linalg.inv(w2c)
    K = cam['intrinsics'].astype(np.float32)
    W, H = float(cam['width']), float(cam['height'])
    fxfycxcy = np.array([K[0, 0] / W, K[1, 1] / H, K[0, 2] / W, K[1, 2] / H], dtype=np.float32)
    return c2w, fxfycxcy


class Cheese3DDataset(SABLEDataset):
    """Multi-view dataset for Cheese3D camera frames.

    Reads raw frames extracted to::

        {dataset_path}/{session_id}/{camera}/img{frame_idx:08d}.png

    For each session in ``training.session_names``, pairs frames from
    ``training.cheese3d_left_camera`` and ``training.cheese3d_right_camera`` (default
    ``'TL'``/``'TR'``) by matching frame index.  Optionally includes a third center
    camera (``training.cheese3d_center_camera``, e.g. ``'TC'``).  The accompanying
    per-frame ``.npy`` files (static camera intrinsics/extrinsics) are loaded when
    ``training.use_camera_params`` is ``true``, returning pre-calibrated ``c2w`` and
    ``fxfycxcy`` tensors that the model uses in place of its learned pose predictor.

    Unlike :class:`SABLEDataset`:

    * ``depth_vda`` is always zero — pair this dataset with ``model.vda.mode: online``
      so the SABLE model computes depth from ``data['image']`` itself.
    * Correspondences are a fixed set of 3 points (``_CHEESE3D_FIXED_XY``,
      ``_CHEESE3D_FIXED_CONFIDENCE``), identical for every scene and camera, rescaled
      from native (320x256) pixel space to ``image_size x image_size``.

    Optionally, SAM3 segmentation masks can be loaded alongside the raw frames.
    Masks are read from::

        {segmentation_root}/{session_id}_{camera}_*/masks/mask{frame_idx:08d}.png

    When ``training.use_segmentation.enabled`` is true, only frame indices with a
    mask available for all cameras are included.  The raw images are returned
    unchanged in ``data['image']`` so that VDA, the image tokeniser, and DINO all
    receive full scene context.  The masks are returned separately under
    ``data['mask']`` (shape ``[V, 1, H, W]``, float32, 1 = foreground); the model
    applies them at loss time (white background on the target) and to Gaussian opacity.

    Config keys read:

    * ``training.dataset_path`` — root Cheese3D directory.
    * ``training.session_names`` — required list of session subdirectory names.
    * ``training.cheese3d_left_camera`` (default ``'TL'``),
      ``training.cheese3d_right_camera`` (default ``'TR'``),
      ``training.cheese3d_center_camera`` (default ``None``; set to e.g. ``'TC'`` to
      enable three-view training).
    * ``training.use_segmentation.enabled`` (default ``False``),
      ``training.use_segmentation.cache_root`` (required when enabled).
    * ``training.use_camera_params`` (default ``False``) — when ``true``, load per-frame
      ``.npy`` calibration files and return ``c2w`` ``[V, 4, 4]`` and ``fxfycxcy`` ``[V, 4]``
      in the batch dict so the model can skip its learned pose predictor.
    * ``training.load_gt_camera_params_for_vis`` (default ``False``) — when ``true``, always
      load the same per-frame ``.npy`` calibration files into ``gt_c2w`` ``[V, 4, 4]`` and
      ``gt_fxfycxcy`` ``[V, 4]``, independent of ``use_camera_params``. These keys are for
      visualization only (e.g. overlaying ground-truth camera poses on predicted poses in
      ``beast.inference.save_camera_pointcloud_scene``) and are never read by the model as
      input, so they can be enabled even when the model is learning its own poses
      (``use_camera_params: false``).
    * ``training.val_split_ratio``, ``model.seed``, ``model.image_tokenizer.image_size``,
      ``training.training_regime``.
    """

    def __init__(
        self,
        config: dict,
        include_splits: list[str] | None = None,
    ) -> None:
        """Initialize.

        Args:
            config: full beast config dict.
            include_splits: split filter; see :class:`SABLEDataset` for details.
        """
        # bypass SABLEDataset.__init__'s requirements on vda cache_root and
        # correspondence cache; this dataset needs neither.
        Dataset.__init__(self)
        self.config = config

        training = config['training']
        model_cfg = config['model']

        dataset_path = training.get('dataset_path')
        if not dataset_path:
            raise ValueError('training.dataset_path must be set.')
        session_names = training.get('session_names')
        if not session_names:
            raise ValueError('training.session_names must be a non-empty list.')
        left_camera = str(training.get('cheese3d_left_camera', 'TL'))
        right_camera = str(training.get('cheese3d_right_camera', 'TR'))
        center_camera_raw = training.get('cheese3d_center_camera')
        center_camera: str | None = str(center_camera_raw) if center_camera_raw else None
        self._num_views: int = 3 if center_camera else 2

        seg_cfg: dict = training.get('use_segmentation') or {}
        use_segmentation = bool(seg_cfg.get('enabled', False))
        segmentation_root: Path | None = None
        if use_segmentation:
            segmentation_root_raw = seg_cfg.get('cache_root')
            if not segmentation_root_raw:
                raise ValueError(
                    'training.use_segmentation.cache_root must be set when '
                    'training.use_segmentation.enabled is true.'
                )
            segmentation_root = Path(segmentation_root_raw)
        self._use_segmentation = use_segmentation
        self._use_camera_params: bool = bool(training.get('use_camera_params', False))
        self._load_gt_camera_params_for_vis: bool = bool(
            training.get('load_gt_camera_params_for_vis', False),
        )

        val_split_ratio = float(training.get('val_split_ratio', 0.0))
        split_seed = int(model_cfg.get('seed', 0))
        self._records: list[_PrecacheRecord] = self._load_records(
            Path(dataset_path),
            session_names=session_names,
            left_camera=left_camera,
            right_camera=right_camera,
            center_camera=center_camera,
            segmentation_root=segmentation_root,
            include_splits=include_splits,
            val_split_ratio=val_split_ratio,
            split_seed=split_seed,
        )

        self._vda_cache_root = None  # type: ignore[assignment]
        self._corr_root = None

        self._image_size: int = int(model_cfg['image_tokenizer']['image_size'])
        self._training_regime: str = str(
            training.get('training_regime', 'all_views_reconstruction')
        ).strip().lower()

    def _load_records(  # type: ignore[override]
        self,
        dataset_path: Path,
        session_names: list[str],
        left_camera: str,
        right_camera: str,
        center_camera: str | None,
        segmentation_root: Path | None,
        include_splits: list[str] | None,
        val_split_ratio: float,
        split_seed: int,
    ) -> list[_PrecacheRecord]:
        """Build records by pairing same-index frames from two or three cameras across sessions.

        Args:
            dataset_path: root Cheese3D directory containing per-session subdirectories.
            session_names: session subdirectory names to include.
            left_camera: left camera subdirectory name (e.g. ``'TL'``).
            right_camera: right camera subdirectory name (e.g. ``'TR'``).
            center_camera: optional center camera subdirectory name (e.g. ``'TC'``).
                When set, only frames present in all three camera directories are included
                and each record stores a ``center_path``.
            segmentation_root: root directory of SAM3 segmentation masks, or ``None``
                to disable mask filtering/loading.
            include_splits: split filter; see :class:`SABLEDataset` for details.
            val_split_ratio: fraction of records reserved for validation.
            split_seed: RNG seed for the deterministic train/val split.

        Returns:
            list of ``_PrecacheRecord``.
        """
        camera_names = (
            f'{left_camera}/{right_camera}/{center_camera}'
            if center_camera
            else f'{left_camera}/{right_camera}'
        )
        records: list[_PrecacheRecord] = []
        eval_records: list[_PrecacheRecord] = []
        for session_name in session_names:
            session_dir = dataset_path / session_name
            left_dir = session_dir / left_camera
            right_dir = session_dir / right_camera
            if not left_dir.is_dir() or not right_dir.is_dir():
                log_step(
                    f'skipping session {session_name!r}: missing {left_camera!r} or '
                    f'{right_camera!r} camera directory',
                    level='warning',
                )
                continue

            # Eval layout — {camera}/{split}/interval{N}timebin{M}.png plus a
            # frame_index_mapping.json per split — used automatically when no flat
            # img*.png files sit directly under the camera directories. Mirrors
            # IBLTwoViewDataset._discover_filesystem_records's dual-layout fallback; see
            # SABLEDataset._discover_eval_split_records for the on-disk contract. Eval
            # sessions carry a fixed train/val/test split already, so they bypass
            # val_split_ratio entirely, and (like IBLTwoViewDataset) support only two
            # cameras — a center_camera is ignored for eval-layout sessions.
            if not self._frame_indices(left_dir, prefix='img', suffix='.png'):
                session_eval_records = self._discover_eval_split_records(
                    left_dir=left_dir,
                    right_dir=right_dir,
                    session_id=session_name,
                    include_splits=include_splits,
                )
                if not session_eval_records:
                    log_step(
                        f'skipping session {session_name!r}: no flat img*.png files and no '
                        f'eval-layout frame_index_mapping.json found',
                        level='warning',
                    )
                    continue
                if center_camera is not None:
                    log_step(
                        f'session {session_name!r}: eval layout does not support a center '
                        f'camera; {center_camera!r} is ignored',
                        level='warning',
                    )
                eval_records.extend(session_eval_records)
                continue

            center_dir: Path | None = None
            if center_camera is not None:
                center_dir = session_dir / center_camera
                if not center_dir.is_dir():
                    log_step(
                        f'skipping session {session_name!r}: missing {center_camera!r} '
                        f'camera directory',
                        level='warning',
                    )
                    continue

            left_indices = self._frame_indices(left_dir, prefix='img', suffix='.png')
            right_indices = self._frame_indices(right_dir, prefix='img', suffix='.png')
            common_indices = left_indices & right_indices
            if center_dir is not None:
                center_indices = self._frame_indices(center_dir, prefix='img', suffix='.png')
                common_indices &= center_indices

            left_mask_dir = right_mask_dir = center_mask_dir = None
            if segmentation_root is not None:
                left_mask_dir = self._resolve_mask_dir(segmentation_root, session_name, left_camera)
                right_mask_dir = self._resolve_mask_dir(segmentation_root, session_name, right_camera)
                left_mask_indices = self._frame_indices(left_mask_dir, prefix='mask', suffix='.png')
                right_mask_indices = self._frame_indices(right_mask_dir, prefix='mask', suffix='.png')
                common_indices &= left_mask_indices & right_mask_indices
                if center_camera is not None:
                    center_mask_dir = self._resolve_mask_dir(
                        segmentation_root, session_name, center_camera,
                    )
                    center_mask_indices = self._frame_indices(
                        center_mask_dir, prefix='mask', suffix='.png',
                    )
                    common_indices &= center_mask_indices

            for frame_idx in sorted(common_indices):
                records.append(_PrecacheRecord(
                    session_id=session_name,
                    pair_idx=len(records),
                    left_path=left_dir / f'img{frame_idx:08d}.png',
                    right_path=right_dir / f'img{frame_idx:08d}.png',
                    left_source_frame_index=frame_idx,
                    right_source_frame_index=frame_idx,
                    scene_name=f'{session_name}_frame_{frame_idx:08d}',
                    left_mask_path=(
                        left_mask_dir / f'mask{frame_idx:08d}.png' if left_mask_dir else None
                    ),
                    right_mask_path=(
                        right_mask_dir / f'mask{frame_idx:08d}.png' if right_mask_dir else None
                    ),
                    center_path=(
                        center_dir / f'img{frame_idx:08d}.png' if center_dir else None
                    ),
                    center_mask_path=(
                        center_mask_dir / f'mask{frame_idx:08d}.png'
                        if center_mask_dir
                        else None
                    ),
                ))

        if not records and not eval_records:
            raise RuntimeError(
                f'No common {camera_names} frames found for sessions '
                f'{session_names} under {dataset_path}'
            )

        if records:
            records = self._split_records(
                records,
                include_splits=include_splits,
                val_split_ratio=val_split_ratio,
                split_seed=split_seed,
                source_desc=str(dataset_path),
            )

        # eval-layout records already carry a fixed, on-disk train/val/test split, so they
        # bypass _split_records's synthetic val_split_ratio-based reshuffling entirely.
        return records + eval_records

    @staticmethod
    def _resolve_mask_dir(segmentation_root: Path, session_name: str, camera: str) -> Path:
        """Resolve the ``masks/`` directory for one session/camera.

        Expects exactly one ``{segmentation_root}/{session_name}_{camera}_*`` match,
        each containing a ``masks/`` subdirectory.

        Args:
            segmentation_root: root directory of SAM3 segmentation masks.
            session_name: session subdirectory name (e.g. ``'20231031_B6_chew_bl_000'``).
            camera: camera name (e.g. ``'TL'``).

        Returns:
            path to the ``masks/`` subdirectory.

        Raises:
            RuntimeError: if zero or more than one matching directory is found.
        """
        matches = sorted(segmentation_root.glob(f'{session_name}_{camera}_*'))
        if len(matches) != 1:
            raise RuntimeError(
                f'Expected exactly one segmentation mask directory matching '
                f'{segmentation_root}/{session_name}_{camera}_*, found {len(matches)}: '
                f'{matches}'
            )
        return matches[0] / 'masks'

    @staticmethod
    def _frame_indices(camera_dir: Path, prefix: str, suffix: str) -> set[int]:
        """Return the set of frame indices with a ``{prefix}NNNNNNNN{suffix}`` file.

        Args:
            camera_dir: directory to scan.
            prefix: filename prefix before the zero-padded frame index (e.g. ``'img'``
                or ``'mask'``).
            suffix: filename suffix after the frame index (e.g. ``'.png'``).

        Returns:
            set of frame indices parsed from matching filenames.
        """
        indices: set[int] = set()
        for path in camera_dir.glob(f'{prefix}*{suffix}'):
            name = path.name
            if name.startswith(prefix) and name.endswith(suffix):
                try:
                    indices.add(int(name[len(prefix):-len(suffix)]))
                except ValueError:
                    continue
        return indices

    def _resolve_view_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve context and target view indices for the training regime.

        For ``all_views_reconstruction`` all views are both context and target.
        For ``center_camera_holdout`` all views are both context and target, but
        ``__getitem__`` additionally returns ``context_full_mask`` so the model
        zeros all image tokens for the center camera (index 2) while still using
        its pose (Plucker ray) embeddings.
        For ``fixed_1to1`` index 0 is context, index 1 is target.

        Returns:
            tuple of (context_indices, target_indices) as long tensors.

        Raises:
            ValueError: if ``training_regime`` is not a recognised built-in value.
                Subclass and override this method to add a custom regime.
        """
        if self._training_regime in ('all_views_reconstruction', 'center_camera_holdout'):
            all_idx = torch.arange(self._num_views, dtype=torch.long)
            return all_idx, all_idx.clone()
        elif self._training_regime == 'fixed_1to1':
            return torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long)
        else:
            raise ValueError(
                f'Unsupported training_regime: {self._training_regime!r}. '
                'Built-in values: all_views_reconstruction, center_camera_holdout, fixed_1to1. '
                'To use a custom regime, subclass and override _resolve_view_indices.'
            )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load one frame tuple (two or three views).

        Raw images are always returned unchanged under ``image`` so that VDA, the
        image tokeniser, and DINO receive full scene context.  When segmentation is
        enabled, the binary masks are returned separately under ``mask`` (shape
        ``[V, 1, H, W]``, float32, 1 = foreground, 0 = background).  The model
        applies masks at loss time and to Gaussian opacity.

        Args:
            idx: dataset index.

        Returns:
            dict with keys ``image``, ``context_indices``, ``target_indices``,
            ``depth_vda``, ``leftcamera_xy``, ``rightcamera_xy``, ``confidence``,
            ``scene_name``, ``split``, ``neural_trial_idx``, ``neural_bin_idx``,
            ``neural_interval_sec``, optionally ``centercamera_xy``, optionally
            ``mask``, (when ``training.use_camera_params`` is ``true``) ``c2w``
            ``[V, 4, 4]`` and ``fxfycxcy`` ``[V, 4]``, and (when
            ``training.load_gt_camera_params_for_vis`` is ``true``) ``gt_c2w``
            ``[V, 4, 4]`` and ``gt_fxfycxcy`` ``[V, 4]``. ``split``,
            ``neural_trial_idx``, ``neural_bin_idx``, and ``neural_interval_sec``
            are only meaningful for eval-layout records (from
            :meth:`_discover_eval_split_records`); training-layout records fall back
            to ``''``, ``-1``, ``-1``, and NaN respectively.
        """
        rec = self._records[idx]

        left_img, left_orig_w, left_orig_h = self._load_image(rec.left_path)
        right_img, right_orig_w, right_orig_h = self._load_image(rec.right_path)

        if rec.center_path is not None:
            center_img, center_orig_w, center_orig_h = self._load_image(rec.center_path)
            image_tensor = torch.stack([left_img, right_img, center_img], dim=0)   # [3, 3, H, W]
            depth_tensor = torch.zeros(
                3, 1, self._image_size, self._image_size, dtype=torch.float32,
            )
            center_orig_size: tuple[int, int] | None = (center_orig_w, center_orig_h)
        else:
            image_tensor = torch.stack([left_img, right_img], dim=0)   # [2, 3, H, W]
            depth_tensor = torch.zeros(
                2, 1, self._image_size, self._image_size, dtype=torch.float32,
            )
            center_orig_size = None

        context_indices, target_indices = self._resolve_view_indices()
        correspondences = self._fixed_correspondences(
            left_orig_size=(left_orig_w, left_orig_h),
            right_orig_size=(right_orig_w, right_orig_h),
            center_orig_size=center_orig_size,
        )

        result = {
            'image': image_tensor,
            'context_indices': context_indices,
            'target_indices': target_indices,
            'depth_vda': depth_tensor,
            'leftcamera_xy': correspondences['leftcamera_xy'],
            'rightcamera_xy': correspondences['rightcamera_xy'],
            'confidence': correspondences['confidence'],
            'scene_name': rec.scene_name,
            'split': rec.split or '',
            'neural_trial_idx': (
                rec.neural_trial_idx if rec.neural_trial_idx is not None else -1
            ),
            'neural_bin_idx': rec.neural_bin_idx if rec.neural_bin_idx is not None else -1,
            'neural_interval_sec': (
                torch.tensor(rec.neural_interval_sec, dtype=torch.float64)
                if rec.neural_interval_sec is not None
                else torch.full((2,), float('nan'), dtype=torch.float64)
            ),
        }

        if 'centercamera_xy' in correspondences:
            result['centercamera_xy'] = correspondences['centercamera_xy']

        if self._training_regime == 'center_camera_holdout':
            if self._num_views != 3:
                raise ValueError(
                    f"'center_camera_holdout' requires 3 views "
                    f"(cheese3d_center_camera must be set); got {self._num_views}"
                )
            # center camera is at index 2; zero out all its image tokens in the model
            result['context_full_mask'] = torch.tensor([False, False, True], dtype=torch.bool)

        if rec.left_mask_path is not None and rec.right_mask_path is not None:
            left_mask = self._load_mask(rec.left_mask_path)    # [1, H, W]
            right_mask = self._load_mask(rec.right_mask_path)  # [1, H, W]
            masks = [left_mask, right_mask]
            if rec.center_mask_path is not None:
                masks.append(self._load_mask(rec.center_mask_path))  # [1, H, W]
            result['mask'] = torch.stack(masks, dim=0)  # [V, 1, H, W]

        if self._use_camera_params or self._load_gt_camera_params_for_vis:
            c2w_arr, fxfycxcy_arr = self._load_camera_params(rec)

        if self._use_camera_params:
            result['c2w'] = torch.from_numpy(c2w_arr)            # [V, 4, 4]
            result['fxfycxcy'] = torch.from_numpy(fxfycxcy_arr)  # [V, 4]

        if self._load_gt_camera_params_for_vis:
            result['gt_c2w'] = torch.from_numpy(c2w_arr).clone()            # [V, 4, 4]
            result['gt_fxfycxcy'] = torch.from_numpy(fxfycxcy_arr).clone()  # [V, 4]

        return result

    def _load_camera_params(self, rec: _PrecacheRecord) -> tuple[np.ndarray, np.ndarray]:
        """Load stacked GT c2w/fxfycxcy for one record's views, in view-index order.

        Args:
            rec: precache record for one sample.

        Returns:
            tuple of (c2w ``[V, 4, 4]``, fxfycxcy ``[V, 4]``) float32 arrays,
            ``V == self._num_views``.
        """
        left_c2w, left_fxfycxcy = _npy_to_c2w_fxfycxcy(rec.left_path.with_suffix('.npy'))
        right_c2w, right_fxfycxcy = _npy_to_c2w_fxfycxcy(rec.right_path.with_suffix('.npy'))
        c2w_arrays = [left_c2w, right_c2w]
        fxfycxcy_arrays = [left_fxfycxcy, right_fxfycxcy]
        if rec.center_path is not None:
            center_c2w, center_fxfycxcy = _npy_to_c2w_fxfycxcy(
                rec.center_path.with_suffix('.npy'),
            )
            c2w_arrays.append(center_c2w)
            fxfycxcy_arrays.append(center_fxfycxcy)
        return np.stack(c2w_arrays), np.stack(fxfycxcy_arrays)

    def _fixed_correspondences(
        self,
        left_orig_size: tuple[int, int],
        right_orig_size: tuple[int, int],
        center_orig_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Rescale the fixed correspondence points to ``image_size x image_size``.

        Args:
            left_orig_size: original (width, height) of the left image.
            right_orig_size: original (width, height) of the right image.
            center_orig_size: original (width, height) of the center image, or ``None``
                when no center camera is used.

        Returns:
            dict with keys ``leftcamera_xy [3, 2]``, ``rightcamera_xy [3, 2]``,
            ``confidence [3]``, and optionally ``centercamera_xy [3, 2]``.
        """
        lw, lh = left_orig_size
        rw, rh = right_orig_size
        scale_left = torch.tensor([self._image_size / lw, self._image_size / lh])
        scale_right = torch.tensor([self._image_size / rw, self._image_size / rh])
        result = {
            'leftcamera_xy': _CHEESE3D_FIXED_XY * scale_left,
            'rightcamera_xy': _CHEESE3D_FIXED_XY * scale_right,
            'confidence': _CHEESE3D_FIXED_CONFIDENCE.clone(),
        }
        if center_orig_size is not None:
            cw, ch = center_orig_size
            scale_center = torch.tensor([self._image_size / cw, self._image_size / ch])
            result['centercamera_xy'] = _CHEESE3D_FIXED_XY * scale_center
        return result


def pad_correspondence_fields_to_batch_max(batch: list[dict]) -> list[dict]:
    """Pad correspondence tensors to max length in batch so default_collate can stack.

    Args:
        batch: list of dicts, each containing correspondence tensors.

    Returns:
        list of dicts, each containing padded correspondence tensors.
    """
    if not batch:
        return batch
    keys = ('leftcamera_xy', 'rightcamera_xy', 'confidence')
    if not all(k in batch[0] for k in keys):
        return batch
    max_n = max(int(sample['leftcamera_xy'].shape[0]) for sample in batch)
    out: list[dict] = []
    for sample in batch:
        sample = dict(sample)
        n = int(sample['leftcamera_xy'].shape[0])
        if n < max_n:
            pad = max_n - n
            lt = sample['leftcamera_xy']
            rt = sample['rightcamera_xy']
            cf = sample['confidence']
            sample['leftcamera_xy'] = torch.cat(
                [lt, torch.full((pad, 2), -1.0, dtype=lt.dtype, device=lt.device)], dim=0,
            )
            sample['rightcamera_xy'] = torch.cat(
                [rt, torch.full((pad, 2), -1.0, dtype=rt.dtype, device=rt.device)], dim=0,
            )
            sample['confidence'] = torch.cat(
                [cf, torch.zeros(pad, dtype=cf.dtype, device=cf.device)], dim=0,
            )
        out.append(sample)
    return out


def normalize_optional_pose_fields(batch: list[dict]) -> list[dict]:
    """Ensure optional pose supervision keys are consistent across a batch.

    Some samples may not have pose supervision available. When any sample has
    ``pose``, inject a zero placeholder plus ``pose_valid=False`` for the rest.

    Args:
        batch: list of sample dicts.

    Returns:
        list of sample dicts with consistent pose keys.
    """
    if not batch:
        return batch
    if not all(isinstance(sample, dict) for sample in batch):
        return batch

    pose_template = None
    for sample in batch:
        pose = sample.get('pose')
        if isinstance(pose, torch.Tensor):
            pose_template = pose
            break

    if pose_template is None:
        return batch

    pose_shape = tuple(pose_template.shape)
    pose_valid_shape = pose_shape[:-1]
    out: list[dict] = []
    for sample in batch:
        sample = dict(sample)
        pose = sample.get('pose')
        if isinstance(pose, torch.Tensor):
            if 'pose_valid' not in sample:
                sample['pose_valid'] = torch.ones(
                    pose.shape[:-1],
                    dtype=torch.bool,
                    device=pose.device,
                )
        else:
            sample['pose'] = torch.zeros(
                pose_shape,
                dtype=pose_template.dtype,
                device=pose_template.device,
            )
            sample['pose_valid'] = torch.zeros(
                pose_valid_shape,
                dtype=torch.bool,
                device=pose_template.device,
            )
        out.append(sample)
    return out


def collate_with_correspondence_padding(batch: list[Any]):
    """Collate a batch with correspondence padding and optional pose normalisation."""
    batch = normalize_optional_pose_fields(batch)
    batch = pad_correspondence_fields_to_batch_max(batch)
    return default_collate(batch)
