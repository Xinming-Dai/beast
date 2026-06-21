"""SABLE two-view dataset for SABLE model."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

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
                ``training.ibl_training_regime``, ``training.val_split_ratio``,
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
            training.get('ibl_training_regime', 'all_views_reconstruction')
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
            ``scene_name``.
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

        return {
            'image': image_tensor,
            'context_indices': context_indices,
            'target_indices': target_indices,
            'depth_vda': depth_tensor,
            'leftcamera_xy': correspondences['leftcamera_xy'],
            'rightcamera_xy': correspondences['rightcamera_xy'],
            'confidence': correspondences['confidence'],
            'scene_name': rec.scene_name,
        }

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
        """
        all_idx = torch.arange(2, dtype=torch.long)
        if self._training_regime == 'all_views_reconstruction':
            return all_idx, all_idx.clone()
        return torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long)


class IBLTwoViewDataset(SABLEDataset):
    """Two-view IBL dataset reading from the reorganized ``beast extract_sable`` pipeline output.

    Like :class:`~beast.data.sable_dataset.SABLEDataset` but:

    * VDA depth is **optional** — loaded from
      ``{dataset_path}/depth_map/{session_id}/{camera}/depth{frame_idx:08d}.npy``.
      Returns a zero depth tensor with a debug log when the file is absent, allowing
      training without precomputed depth (the model handles online VDA inference via
      ``model.vda.mode: online`` in the training config).
    * Correspondence files are **optional** — a session-level ``.npz`` bundle is
      loaded from
      ``{dataset_path}/litpose_correspondences/processed_correspondences/{session_id}/
      correspondences{pair_idx:08d}.npz``.
      The bundle contains ``left_xy [K, 2]``, ``right_xy [K, 2]``, and
      ``confidence [K]`` arrays in native IBL pixel space; coordinates are rescaled
      to ``image_size × image_size`` and padded to ``_MAX_MATCHES`` at load time.
      Returns zero tensors when the bundle is absent.

    ``dataset_path`` should point to the ``dataset/`` directory that contains
    per-session subdirectories (each with ``pair_metadata.json`` and
    ``{camera}/img*.png``), as well as sibling ``depth_map/`` and
    ``litpose_correspondences/`` directories.

    Config keys read (same as :class:`SABLEDataset` except ``model.vda.cache_root``
    is not used):

    * ``training.dataset_path``
    * ``model.image_tokenizer.image_size``
    * ``training.ibl_training_regime``
    * ``training.val_split_ratio``
    * ``model.seed``
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
        self._records: list[_PrecacheRecord] = self._load_records(
            Path(dataset_path),
            include_splits=include_splits,
            val_split_ratio=val_split_ratio,
            split_seed=split_seed,
        )

        # VDA and correspondence roots are not used directly; depth and
        # correspondences are loaded from the reorganized dataset layout
        self._vda_cache_root = None  # type: ignore[assignment]
        self._corr_root = None
        self._dataset_root: Path = Path(dataset_path)

        self._image_size: int = int(model_cfg['image_tokenizer']['image_size'])
        self._training_regime: str = str(
            training.get('ibl_training_regime', 'all_views_reconstruction')
        ).strip().lower()

    # ------------------------------------------------------------------
    # overrides
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load one stereo pair.

        Args:
            idx: dataset index.

        Returns:
            dict with keys ``image``, ``context_indices``, ``target_indices``,
            ``depth_vda``, ``leftcamera_xy``, ``rightcamera_xy``, ``confidence``,
            ``scene_name``.
        """
        rec = self._records[idx]

        left_img, left_orig_w, left_orig_h = self._load_image(rec.left_path)
        right_img, right_orig_w, right_orig_h = self._load_image(rec.right_path)

        camera_left = rec.left_path.parent.name
        camera_right = rec.right_path.parent.name
        vda_depths = [
            self._load_vda_depth_sable(rec.session_id, camera_left, rec.left_source_frame_index),
            self._load_vda_depth_sable(rec.session_id, camera_right, rec.right_source_frame_index),
        ]

        image_tensor = torch.stack([left_img, right_img], dim=0)   # [V, 3, H, W]
        depth_tensor = torch.stack(vda_depths, dim=0)               # [V, 1, H, W]

        context_indices, target_indices = self._resolve_view_indices()
        correspondences = self._load_correspondences_sable(
            session_id=rec.session_id,
            pair_idx=rec.pair_idx,
            left_orig_size=(left_orig_w, left_orig_h),
            right_orig_size=(right_orig_w, right_orig_h),
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
        }

    # ------------------------------------------------------------------
    # VDA depth loading
    # ------------------------------------------------------------------

    def _load_vda_depth_sable(
        self,
        session_id: str,
        camera_name: str,
        source_frame_index: int,
    ) -> torch.Tensor:
        """Load VDA depth from the reorganized depth_map directory.

        Expected path:
            ``{dataset_root}/depth_map/{session_id}/{camera_name}/depth{source_frame_index:08d}.npy``

        Returns a zero depth tensor with a debug log when the file is absent.

        Args:
            session_id: session identifier (subdirectory name).
            camera_name: camera subdirectory name (e.g. ``'left'``, ``'right'``).
            source_frame_index: frame index used in the filename.

        Returns:
            float32 tensor [1, H, W] where H = W = ``image_size``.
        """
        depth_path = (
            self._dataset_root
            / 'depth_map'
            / session_id
            / camera_name
            / f'depth{source_frame_index:08d}.npy'
        )
        if not depth_path.exists():
            log_step(
                f'VDA depth not found (using zeros): {depth_path}. '
                'Run beast extract_sable with vda.enabled: true to precompute.',
                level='debug',
            )
            return torch.zeros(1, self._image_size, self._image_size, dtype=torch.float32)

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

        Expected path:
            ``{dataset_root}/litpose_correspondences/processed_correspondences/
            {session_id}/correspondences{pair_idx:08d}.npz``

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
        bundle_path = (
            self._dataset_root
            / 'litpose_correspondences'
            / 'processed_correspondences'
            / session_id
            / f'correspondences{pair_idx:08d}.npz'
        )
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

    For each session in ``training.cheese3d_session_names``, pairs frames from
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
    * ``training.cheese3d_session_names`` — required list of session subdirectory names.
    * ``training.cheese3d_left_camera`` (default ``'TL'``),
      ``training.cheese3d_right_camera`` (default ``'TR'``),
      ``training.cheese3d_center_camera`` (default ``None``; set to e.g. ``'TC'`` to
      enable three-view training).
    * ``training.use_segmentation.enabled`` (default ``False``),
      ``training.use_segmentation.cache_root`` (required when enabled).
    * ``training.use_camera_params`` (default ``False``) — when ``true``, load per-frame
      ``.npy`` calibration files and return ``c2w`` ``[V, 4, 4]`` and ``fxfycxcy`` ``[V, 4]``
      in the batch dict so the model can skip its learned pose predictor.
    * ``training.val_split_ratio``, ``model.seed``, ``model.image_tokenizer.image_size``,
      ``training.ibl_training_regime``.
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
        session_names = training.get('cheese3d_session_names')
        if not session_names:
            raise ValueError('training.cheese3d_session_names must be a non-empty list.')
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
            training.get('ibl_training_regime', 'all_views_reconstruction')
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

        if not records:
            raise RuntimeError(
                f'No common {camera_names} frames found for sessions '
                f'{session_names} under {dataset_path}'
            )

        return self._split_records(
            records,
            include_splits=include_splits,
            val_split_ratio=val_split_ratio,
            split_seed=split_seed,
            source_desc=str(dataset_path),
        )

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
        For ``fixed_1to1`` index 0 is context, index 1 is target.

        Returns:
            tuple of (context_indices, target_indices) as long tensors.
        """
        all_idx = torch.arange(self._num_views, dtype=torch.long)
        if self._training_regime == 'all_views_reconstruction':
            return all_idx, all_idx.clone()
        return torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long)

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
            ``scene_name``, optionally ``centercamera_xy``, optionally ``mask``,
            and (when ``training.use_camera_params`` is ``true``) ``c2w`` ``[V, 4, 4]``
            and ``fxfycxcy`` ``[V, 4]``.
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
        }

        if 'centercamera_xy' in correspondences:
            result['centercamera_xy'] = correspondences['centercamera_xy']

        if rec.left_mask_path is not None and rec.right_mask_path is not None:
            left_mask = self._load_mask(rec.left_mask_path)    # [1, H, W]
            right_mask = self._load_mask(rec.right_mask_path)  # [1, H, W]
            masks = [left_mask, right_mask]
            if rec.center_mask_path is not None:
                masks.append(self._load_mask(rec.center_mask_path))  # [1, H, W]
            result['mask'] = torch.stack(masks, dim=0)  # [V, 1, H, W]

        if self._use_camera_params:
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
            result['c2w'] = torch.from_numpy(np.stack(c2w_arrays))            # [V, 4, 4]
            result['fxfycxcy'] = torch.from_numpy(np.stack(fxfycxcy_arrays))  # [V, 4]

        return result

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
