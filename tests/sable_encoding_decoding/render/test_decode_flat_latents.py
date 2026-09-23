from pathlib import Path

import pytest

from beast.sable_encoding_decoding.render.decode_flat_latents import (
    parse_flat_latent_decode_args,
)

REQUIRED = ['--model-dir', 'm', '--estimated-dir', 'e', '--out-dir', 'o']


class TestParseFlatLatentDecodeArgs:
    """Test the function parse_flat_latent_decode_args."""

    def test_parse_flat_latent_decode_args_required_only(self):
        args = parse_flat_latent_decode_args(REQUIRED, description='d')

        assert args.model_dir == Path('m')
        assert args.estimated_dir == Path('e')
        assert args.out_dir == Path('o')
        assert args.batch_size == 64
        assert args.device == 'cuda:0'
        assert args.image_size is None
        assert args.metrics_npz is None
        assert args.target_frame_mapping_left is None
        assert not args.use_segmentation_mask
        assert not args.metrics_only

    def test_parse_flat_latent_decode_args_full_metrics_config(self):
        argv = REQUIRED + [
            '--target-frame-mapping-left', 'l',
            '--target-frame-mapping-right', 'r',
            '--use-segmentation-mask',
            '--segmentation-root', 's',
            '--eid', 'abc',
            '--image-size', '320',
            '--metrics-only',
            '--neural-trial-index', '1,2',
            '--metrics-npz', 'm.npz',
        ]

        args = parse_flat_latent_decode_args(argv, description='d')

        assert args.target_frame_mapping_left == Path('l')
        assert args.target_frame_mapping_right == Path('r')
        assert args.use_segmentation_mask
        assert args.segmentation_root == Path('s')
        assert args.eid == 'abc'
        assert args.image_size == 320
        assert args.metrics_only
        assert args.neural_trial_index == frozenset({1, 2})
        assert args.metrics_npz == Path('m.npz')

    def test_parse_flat_latent_decode_args_rejects_single_target_mapping(self):
        with pytest.raises(SystemExit):
            parse_flat_latent_decode_args(
                REQUIRED + ['--target-frame-mapping-left', 'l'], description='d',
            )

    def test_parse_flat_latent_decode_args_rejects_mask_without_root_eid_targets(self):
        with pytest.raises(SystemExit):
            parse_flat_latent_decode_args(
                REQUIRED + ['--use-segmentation-mask'], description='d',
            )

    def test_parse_flat_latent_decode_args_rejects_metrics_only_without_targets(self):
        with pytest.raises(SystemExit):
            parse_flat_latent_decode_args(REQUIRED + ['--metrics-only'], description='d')
