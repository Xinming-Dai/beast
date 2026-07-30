import copy

import numpy as np
import pytest
import torch

from beast.inference import _save_latent_batch_npz
from beast.models.vits import VisionTransformer
from beast.sable_encoding_decoding.img_token.decode_beast_tokens import (
    load_ids_restore_trials_npz,
    load_img_tokens_and_ids_restore_from_shards,
    predict_frame_from_all_tokens,
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
