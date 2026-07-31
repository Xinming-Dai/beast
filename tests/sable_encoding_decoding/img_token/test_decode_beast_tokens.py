import copy
import json

import numpy as np
import pytest
import torch
from PIL import Image

from beast.inference import _save_latent_batch_npz
from beast.models.vits import VisionTransformer
from beast.sable_encoding_decoding.img_token.decode_beast_tokens import (
    load_estimated_tokens_dir,
    load_ids_restore_lookup_from_sidecar,
    load_ids_restore_trials_npz,
    load_img_tokens_and_ids_restore_from_shards,
    parse_args,
    predict_frame_from_all_tokens,
    resolve_ids_restore_for_trials,
)
from beast.sable_encoding_decoding.img_token.target_frames import (
    load_frame_index_mapping,
    load_target_images_for_trials,
)


class TestPredictFrameFromAllTokens:
    """Test the function predict_frame_from_all_tokens."""

    def test_predict_frame_from_all_tokens_matches_direct_reconstruction(self, config_vit):
        config = copy.deepcopy(config_vit)
        config['model']['model_params']['random_init'] = True
        model = VisionTransformer(config)
        model.eval()

        image = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            direct = model.forward(image, return_recon=True, return_img_tokens=True)
            result = predict_frame_from_all_tokens(
                model, direct['img_tokens'], direct['ids_restore'],
            )

        assert result['render'].shape == direct['reconstructions'].shape
        torch.testing.assert_close(result['render'], direct['reconstructions'])

    def test_predict_frame_from_all_tokens_passes_through_data(self, config_vit):
        config = copy.deepcopy(config_vit)
        config['model']['model_params']['random_init'] = True
        model = VisionTransformer(config)
        model.eval()

        image = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            direct = model.forward(image, return_recon=True, return_img_tokens=True)
            result = predict_frame_from_all_tokens(
                model, direct['img_tokens'], direct['ids_restore'], data={'image': image},
            )

        assert 'render' in result
        assert result['image'] is image


class TestLoadIdsRestoreTrialsNpz:
    """Test the function load_ids_restore_trials_npz."""

    def test_load_ids_restore_trials_npz_concatenates_nonempty_splits(self, tmp_path):
        path = tmp_path / 'ids_restore.npz'
        np.savez_compressed(
            path,
            train_ids_restore=np.zeros((2, 3, 2, 4), dtype=np.float32),
            val_ids_restore=np.empty((0, 3, 2, 4), dtype=np.float32),
            test_ids_restore=np.ones((1, 3, 2, 4), dtype=np.float32),
            neural_trial_idx=np.array([0, 1, 2]),
        )

        ids_restore, meta = load_ids_restore_trials_npz(path)

        assert ids_restore.dtype == np.int64
        assert ids_restore.shape == (3, 3, 2, 4)
        assert meta['trial_split'] == ['train', 'train', 'test']

    def test_load_ids_restore_trials_npz_raises_when_all_empty(self, tmp_path):
        path = tmp_path / 'ids_restore.npz'
        np.savez_compressed(
            path,
            train_ids_restore=np.empty((0, 3, 2, 4), dtype=np.float32),
            val_ids_restore=np.empty((0, 3, 2, 4), dtype=np.float32),
            test_ids_restore=np.empty((0, 3, 2, 4), dtype=np.float32),
        )

        with pytest.raises(KeyError, match='no non-empty'):
            load_ids_restore_trials_npz(path)


class TestLoadImgTokensAndIdsRestoreFromShards:
    """Test the function load_img_tokens_and_ids_restore_from_shards."""

    def _write_shard(self, tmp_path, session_id, split, batch_idx, *, n_tok, d):
        """Write one img_tokens_batch*.npz shard: 2 rows, left/right merged into 2*n_tok tokens."""
        split_dir = tmp_path / 'img_tokens' / session_id / split
        split_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(batch_idx)
        left = rng.random((2, n_tok, d)).astype(np.float32)
        right = rng.random((2, n_tok, d)).astype(np.float32)
        z = np.concatenate([left, right], axis=1)

        left_restore = np.stack([np.arange(n_tok), np.arange(n_tok)]).astype(np.int64)
        right_restore = left_restore + 100
        ids_restore = np.concatenate([left_restore, right_restore], axis=1)

        _save_latent_batch_npz(
            split_dir / f'img_tokens_batch{batch_idx:04d}.npz',
            z=z,
            session_ids=[session_id] * 2,
            pair_idxs=[2 * batch_idx, 2 * batch_idx + 1],
            splits=[split] * 2,
            neural_trial_idx=[0, 0],
            neural_bin_idx=[2 * batch_idx, 2 * batch_idx + 1],
            neural_interval_sec=np.zeros((2, 2)),
            aux={'ids_restore': ids_restore},
        )
        return left, right, left_restore, right_restore

    def test_load_img_tokens_and_ids_restore_from_shards_unmerges_camera_axis(
        self, tmp_path,
    ) -> None:
        n_tok, d = 3, 4
        left, right, left_restore, right_restore = self._write_shard(
            tmp_path, 'sess1', 'train', 0, n_tok=n_tok, d=d,
        )

        img_tokens, ids_restore = load_img_tokens_and_ids_restore_from_shards(
            tmp_path / 'img_tokens', 'sess1', splits='train', time_bins=2,
        )

        assert img_tokens.shape == (1, 2, 2, n_tok, d)
        assert ids_restore.shape == (1, 2, 2, n_tok)
        np.testing.assert_allclose(img_tokens[0, :, 0], left, atol=1e-6)
        np.testing.assert_allclose(img_tokens[0, :, 1], right, atol=1e-6)
        np.testing.assert_array_equal(ids_restore[0, :, 0], left_restore)
        np.testing.assert_array_equal(ids_restore[0, :, 1], right_restore)

    def test_load_img_tokens_and_ids_restore_from_shards_raises_without_ids_restore(
        self, tmp_path,
    ) -> None:
        split_dir = tmp_path / 'img_tokens' / 'sess1' / 'train'
        split_dir.mkdir(parents=True)
        z = np.random.default_rng(0).random((2, 6, 4)).astype(np.float32)
        _save_latent_batch_npz(
            split_dir / 'img_tokens_batch0000.npz',
            z=z,
            session_ids=['sess1'] * 2,
            pair_idxs=[0, 1],
            splits=['train'] * 2,
            neural_trial_idx=[0, 0],
            neural_bin_idx=[0, 1],
            neural_interval_sec=np.zeros((2, 2)),
        )

        with pytest.raises(RuntimeError, match='ids_restore'):
            load_img_tokens_and_ids_restore_from_shards(
                tmp_path / 'img_tokens', 'sess1', splits='train', time_bins=2,
            )


class TestLoadIdsRestoreLookupFromSidecar:
    """Test the function load_ids_restore_lookup_from_sidecar."""

    def _write_sidecar(self, tmp_path, *, trial_split, neural_trial_idx, **restore_by_key):
        """`restore_by_key` maps a full npz key (e.g. `test_ids_restore`) to its array."""
        path = tmp_path / 'img_tokens_camera_parameters.npz'
        kw = {
            'trial_split': np.array(trial_split, dtype=object),
            'neural_trial_idx': np.asarray(neural_trial_idx, dtype=np.int64),
        }
        for key, restore in restore_by_key.items():
            kw[key] = np.asarray(restore, dtype=np.float32)
        np.savez(path, **kw)
        return path

    def test_load_ids_restore_lookup_from_sidecar_keys_by_split_and_trial(self, tmp_path):
        test_restore = np.stack(
            [np.full((2, 6), 100, dtype=np.int64), np.full((2, 6), 101, dtype=np.int64)],
        )
        path = self._write_sidecar(
            tmp_path,
            trial_split=['test', 'test'],
            neural_trial_idx=[0, 1],
            test_ids_restore=test_restore,
        )

        lookup = load_ids_restore_lookup_from_sidecar(path)

        assert set(lookup) == {('test', 0), ('test', 1)}
        np.testing.assert_array_equal(lookup[('test', 0)], test_restore[0])
        np.testing.assert_array_equal(lookup[('test', 1)], test_restore[1])

    def test_load_ids_restore_lookup_from_sidecar_raises_without_ids_restore(self, tmp_path):
        path = self._write_sidecar(tmp_path, trial_split=['test'], neural_trial_idx=[0])

        with pytest.raises(KeyError, match='ids_restore'):
            load_ids_restore_lookup_from_sidecar(path)

    def test_load_ids_restore_lookup_from_sidecar_raises_without_split_metadata(self, tmp_path):
        path = tmp_path / 'img_tokens_camera_parameters.npz'
        np.savez(path, test_ids_restore=np.zeros((1, 2, 6), dtype=np.float32))

        with pytest.raises(KeyError, match='trial_split'):
            load_ids_restore_lookup_from_sidecar(path)


class TestLoadEstimatedTokensDir:
    """Test the function load_estimated_tokens_dir."""

    def _write_estimated_npz(self, tmp_path, trial_id, split, *, t, n_tok, d):
        path = tmp_path / f'img_tokens_estimated_neuraltrial{trial_id:04d}.npz'
        z = np.random.default_rng(trial_id).random((1, t, n_tok, d)).astype(np.float32)
        np.savez_compressed(
            path,
            z=z,
            neural_trial_idx=np.int64(trial_id),
            trial_split=np.array([split], dtype=object),
        )
        return path, z[0]

    def test_load_estimated_tokens_dir_stacks_sorted_trials(self, tmp_path):
        path0, z0 = self._write_estimated_npz(tmp_path, 0, 'test', t=2, n_tok=3, d=4)
        path1, z1 = self._write_estimated_npz(tmp_path, 1, 'test', t=2, n_tok=3, d=4)

        img_tokens, split_labels, neural_trial_idx, paths = load_estimated_tokens_dir(tmp_path)

        assert img_tokens.shape == (2, 2, 3, 4)
        assert split_labels == ['test', 'test']
        np.testing.assert_array_equal(neural_trial_idx, [0, 1])
        assert paths == sorted([path0, path1])
        np.testing.assert_allclose(img_tokens[0], z0, atol=1e-6)
        np.testing.assert_allclose(img_tokens[1], z1, atol=1e-6)

    def test_load_estimated_tokens_dir_raises_when_empty(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='No img_tokens_estimated'):
            load_estimated_tokens_dir(tmp_path)


class TestResolveIdsRestoreForTrials:
    """Test the function resolve_ids_restore_for_trials."""

    def test_resolve_ids_restore_for_trials_matches_by_split_and_id(self):
        lookup = {
            ('test', 0): np.zeros((2, 6), dtype=np.int64),
            ('test', 1): np.ones((2, 6), dtype=np.int64),
        }

        out = resolve_ids_restore_for_trials(['test', 'test'], np.array([1, 0]), lookup)

        np.testing.assert_array_equal(out[0], lookup[('test', 1)])
        np.testing.assert_array_equal(out[1], lookup[('test', 0)])

    def test_resolve_ids_restore_for_trials_raises_on_missing_trial(self):
        lookup = {('test', 0): np.zeros((2, 6), dtype=np.int64)}

        with pytest.raises(KeyError, match='no matching'):
            resolve_ids_restore_for_trials(['test'], np.array([5]), lookup)


class TestParseArgsEstimatedMode:
    """Test parse_args's estimated-mode validation."""

    def test_parse_args_accepts_estimated_mode(self, tmp_path):
        args = parse_args([
            '--model-dir', str(tmp_path),
            '--estimated-dir', str(tmp_path),
            '--ids-restore-sidecar', str(tmp_path / 'img_tokens_camera_parameters.npz'),
            '--out-dir', str(tmp_path),
        ])
        assert args.estimated_dir == tmp_path

    def test_parse_args_rejects_partial_estimated_mode(self, tmp_path):
        with pytest.raises(SystemExit):
            parse_args([
                '--model-dir', str(tmp_path),
                '--estimated-dir', str(tmp_path),
                '--out-dir', str(tmp_path),
            ])

    def test_parse_args_rejects_target_frame_mapping_without_estimated_dir(self, tmp_path):
        with pytest.raises(SystemExit):
            parse_args([
                '--model-dir', str(tmp_path),
                '--input-dir', str(tmp_path),
                '--session-id', 'sess1',
                '--target-frame-mapping-left', str(tmp_path),
                '--target-frame-mapping-right', str(tmp_path),
                '--out-dir', str(tmp_path),
            ])

    def test_parse_args_rejects_no_mode(self, tmp_path):
        with pytest.raises(SystemExit):
            parse_args(['--model-dir', str(tmp_path), '--out-dir', str(tmp_path)])


class TestLoadFrameIndexMapping:
    """Test the function load_frame_index_mapping."""

    def test_load_frame_index_mapping_inverts_by_trial_and_bin(self, tmp_path):
        split_dir = tmp_path / 'test'
        split_dir.mkdir(parents=True)
        mapping = {
            'frame0.png': {'neural_trial_idx': 0, 'neural_bin_idx': 0},
            'frame1.png': {'neural_trial_idx': 0, 'neural_bin_idx': 1},
            'frame2.png': {'neural_trial_idx': 1, 'neural_bin_idx': 0},
        }
        (split_dir / 'frame_index_mapping.json').write_text(json.dumps(mapping))

        out = load_frame_index_mapping(tmp_path, 'test')

        assert out[0][0] == split_dir / 'frame0.png'
        assert out[0][1] == split_dir / 'frame1.png'
        assert out[1][0] == split_dir / 'frame2.png'

    def test_load_frame_index_mapping_returns_empty_when_missing(self, tmp_path):
        assert load_frame_index_mapping(tmp_path, 'test') == {}

    def test_load_frame_index_mapping_raises_on_duplicate_bin(self, tmp_path):
        split_dir = tmp_path / 'test'
        split_dir.mkdir(parents=True)
        mapping = {
            'frame0.png': {'neural_trial_idx': 0, 'neural_bin_idx': 0},
            'frame1.png': {'neural_trial_idx': 0, 'neural_bin_idx': 0},
        }
        (split_dir / 'frame_index_mapping.json').write_text(json.dumps(mapping))

        with pytest.raises(ValueError, match='duplicate'):
            load_frame_index_mapping(tmp_path, 'test')


class TestLoadTargetImagesForTrials:
    """Test the function load_target_images_for_trials."""

    def _write_frame(self, path, color):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (8, 8), color=color).save(path)

    def test_load_target_images_for_trials_loads_left_and_right(self, tmp_path):
        left = tmp_path / 'left0.png'
        right = tmp_path / 'right0.png'
        self._write_frame(left, (255, 0, 0))
        self._write_frame(right, (0, 255, 0))
        mapping_left = {'test': {0: {0: left}}}
        mapping_right = {'test': {0: {0: right}}}

        out = load_target_images_for_trials(
            ['test'], np.array([0]), 1, mapping_left, mapping_right, image_size=8,
        )

        assert out.shape == (1, 1, 2, 3, 8, 8)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_load_target_images_for_trials_raises_on_missing_frame(self, tmp_path):
        with pytest.raises(KeyError, match='no matching'):
            load_target_images_for_trials(['test'], np.array([0]), 1, {}, {}, image_size=8)
