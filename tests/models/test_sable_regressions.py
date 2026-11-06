"""SABLE forward/backward regressions without pretrained weights or GPU rendering."""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

from beast.models import sable as sable_module
from beast.models.model_utils.losses import masked_gs_reg_loss
from beast.models.model_utils.utils_vda import pseudo_pointcloud_normalized
from beast.models.sable import Sable, build_segmentation_gaussian_mask, normalize_gaussians


class _DinoStub(torch.nn.Module):
    def forward(self, images):
        means = images.mean(dim=(-2, -1))
        cls = means.repeat(1, 1, 3)[..., :8]
        return None, cls


class _GaussianStub:
    def empty(self):
        pass

    def set_data(self, xyz, features, scaling, rotation, opacity):
        self.get_xyz = xyz
        self.get_scaling = scaling.exp()
        self.get_opacity = opacity.sigmoid()
        return self


class _RendererStub(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gaussians_model = _GaussianStub()

    def forward(self, xyz, features, scaling, rotation, opacity, height, width, **kwargs):
        views = kwargs['C2W'].shape[1]
        rgb = xyz.mean(dim=1).sigmoid()[:, None, :, None, None]
        return SimpleNamespace(render=rgb.expand(-1, views, -1, height, width))


@pytest.fixture
def make_sable(monkeypatch, tmp_path):
    """Build the real SABLE encoder/decoder with tiny tensors and external stubs."""
    monkeypatch.setattr(sable_module, 'DinoV3', _DinoStub)
    monkeypatch.setattr(sable_module, 'Renderer', _RendererStub)

    def build(
        views=2, input_views=None, target_views=None, init_gs=False,
        mask_ratio=0.0, vda_mode='precomputed',
    ):
        config_path = Path(__file__).parents[2] / 'configs/sable/sable_ibl3d.yaml'
        config = yaml.safe_load(config_path.read_text())
        config['model']['image_tokenizer'].update(image_size=4, patch_size=2)
        config['model']['target_image'].update(height=4, width=4)
        config['model']['transformer'].update(
            d=8, d_head=4, encoder_n_layer=2, encoder_geom_n_layer=2,
            special_init=False,
        )
        config['model']['gaussians'].update(init_gs=init_gs, sh_degree=0)
        config['model']['patch_masking'].update(
            ratio=mask_ratio, pixel_mask=True, gaussian_mask=True,
        )
        config['model']['vda'].update(mode=vda_mode, mask_depth=True)
        config['training'].update(
            num_views=views, num_input_views=input_views or views,
            num_target_views=target_views or views, checkpoint_dir=str(tmp_path),
            perceptual_loss_weight=0.0, lpips_loss_weight=0.0,
        )
        model = Sable(config)
        monkeypatch.setattr(
            model, 'render_images_video',
            lambda *args, **kwargs: SimpleNamespace(rendered_images_video=torch.zeros(1, 3, 4, 4)),
        )
        batch_size = 2
        batch = {
            'image': torch.rand(batch_size, views, 3, 4, 4),
            'depth_vda': torch.rand(batch_size, views, 1, 4, 4) + 0.1,
            'context_indices': torch.arange(input_views or views).expand(batch_size, -1),
            'target_indices': torch.arange(target_views or views).expand(batch_size, -1),
            'confidence': torch.zeros(batch_size, 0),
            'leftcamera_xy': torch.zeros(batch_size, 0, 2),
            'rightcamera_xy': torch.zeros(batch_size, 0, 2),
            'return_frame_cls_tokens': True,
            'return_dino_cls': True,
            'return_img_tokens': True,
        }
        return model, batch

    return build


@pytest.mark.parametrize('views', [2, 3])
def test_geometry_attention_pairs_each_cls_with_its_own_patches(make_sable, views):
    model, batch = make_sable(views=views)
    model.eval()
    fused, local, final = [], [], []
    model.mlp_fuse.register_forward_hook(lambda module, args, out: fused.append(out.detach()))
    model.transformer_encoder_geom[0].register_forward_pre_hook(
        lambda module, args: local.append(args[0].detach()),
    )
    model.transformer_encoder_geom[-1].register_forward_hook(
        lambda module, args, out: final.append(out.detach()),
    )

    result = model(batch)

    torch.testing.assert_close(local[0][:, 0], result.dino_z.reshape(-1, 8))
    torch.testing.assert_close(local[0][:, 1:], fused[0].reshape(-1, 4, 8))
    final_by_view = final[0].reshape(2, views, 5, 8)
    torch.testing.assert_close(result.frame_z, final_by_view[:, :, 0])
    torch.testing.assert_close(result.img_tokens, final_by_view[:, :, 1:].reshape(2, -1, 8))


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('mask_ratio', [0.0, 0.75])
def test_heldout_pixels_cannot_change_predictions(make_sable, training, mask_ratio):
    model, batch = make_sable(views=3, mask_ratio=mask_ratio)
    model.train(training)
    batch['context_full_mask'] = torch.tensor([[False, False, True]]).expand(2, -1)
    image_inputs, cls_inputs = [], []
    model.image_tokenizer.register_forward_pre_hook(
        lambda module, args: image_inputs.append(args[0].detach().clone()),
    )
    model.transformer_encoder_geom[0].register_forward_pre_hook(
        lambda module, args: cls_inputs.append(args[0][:, 0].detach().clone()),
    )
    changed = {**batch, 'image': batch['image'].clone()}
    changed['image'][:, 2] = 1

    torch.manual_seed(123)
    first = model(batch)
    torch.manual_seed(123)
    second = model(changed)

    torch.testing.assert_close(first.render, second.render)
    torch.testing.assert_close(first.c2w_input, second.c2w_input)
    assert torch.all(image_inputs[0][:, 2:] == -1)  # also masks padded copies
    assert torch.count_nonzero(cls_inputs[0].reshape(2, 3, 8)[:, 2]) == 0
    assert torch.all(second.target_image[:, 2] == 1)  # targets stay untouched
    assert torch.all(second.pixel_mask[:, 2] == 1)
    if not training:
        assert torch.count_nonzero(second.pixel_mask[:, :2]) == 0
        assert torch.count_nonzero(second.dino_z[:, 2]) == 0


def test_masked_geometry_matches_unmasked_loss_and_gradient():
    prediction = torch.randn(2, 7, 3, requires_grad=True)
    target = torch.randn_like(prediction)
    masked = masked_gs_reg_loss(prediction, target, torch.ones(2, 7))
    unmasked = F.mse_loss(prediction, target)
    torch.testing.assert_close(masked, unmasked)
    torch.testing.assert_close(
        torch.autograd.grad(masked, prediction)[0],
        torch.autograd.grad(unmasked, prediction)[0],
    )


def test_geometry_mask_excludes_background_gradients():
    prediction = torch.ones(1, 4, 3, requires_grad=True)
    loss = masked_gs_reg_loss(
        prediction, torch.zeros_like(prediction), torch.tensor([[1, 1, 0, 0]]),
    )
    loss.backward()
    assert loss.item() == pytest.approx(1.0)
    assert torch.count_nonzero(prediction.grad[:, 2:]) == 0


@pytest.mark.parametrize('training', [True, False])
def test_pseudo_center_stays_out_of_supervised_targets(make_sable, training):
    model, batch = make_sable(views=3, target_views=2, init_gs=True)
    model.train(training)
    batch['image'][:, 2] = 0
    batch['depth_vda'][:, 2] = 0
    batch['context_full_mask'] = torch.tensor([[False, False, True]]).expand(2, -1)
    batch['mask'] = torch.ones(2, 2, 1, 4, 4)  # no GT mask for the pseudo view

    result = model(batch)
    loss, _ = model.compute_loss('train' if training else 'val', **vars(result))

    assert result.target_indices.tolist() == [[0, 1], [0, 1]]
    assert result.render.shape[1] == 2
    assert result.pixelalign_xyz.shape[1] == 2
    assert result.xyz_norm.shape == result.xyz_init_norm.shape == (2, 32, 3)
    assert result.depth_num_real_views == 2
    assert result.pixel_mask is None
    assert loss.item() > 0


@pytest.mark.parametrize('source', [0, 1])
def test_single_source_geometry_uses_source_depth_and_mask(make_sable, source):
    model, batch = make_sable(views=2, input_views=1, target_views=1, init_gs=True)
    model.eval()
    batch['context_indices'] = torch.full((2, 1), source)
    batch['target_indices'] = torch.full((2, 1), 1 - source)
    batch['mask'] = torch.zeros(2, 2, 1, 4, 4)
    batch['mask'][:, source, :, :2] = 1
    model.config['training']['mask_geom_loss'] = True

    result = model(batch)

    torch.testing.assert_close(
        result.gaussian_depth_output, batch['depth_vda'][:, source:source + 1, 0],
    )
    expected_xyz = torch.stack([
        pseudo_pointcloud_normalized(d[0], 2, 2, m[0])
        for d, m in zip(batch['depth_vda'][:, source], batch['mask'][:, source], strict=True)
    ])
    torch.testing.assert_close(result.xyz_init_norm, normalize_gaussians(expected_xyz))
    torch.testing.assert_close(
        result.gaussian_mask,
        build_segmentation_gaussian_mask(batch['mask'][:, source:source + 1], 2, 2, 2, 2),
    )
    assert result.pixel_mask is None
    assert result.depth_output is None  # source depth is not labeled as target depth
    geometry_loss = masked_gs_reg_loss(result.xyz_norm, result.xyz_init_norm, result.gaussian_mask)
    geometry_loss.backward()
    gradients = [p.grad for p in model.transformer_encoder_geom.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) for g in gradients)


def test_disabled_geometry_is_reported_and_does_not_load_vda(make_sable, monkeypatch, caplog):
    def unexpected_vda_load(config):
        pytest.fail('Disabled geometry must not load online VDA')

    monkeypatch.setattr(sable_module, 'load_frozen_video_depth_anything', unexpected_vda_load)
    with caplog.at_level(logging.WARNING, logger='beast.models.sable'):
        model, batch = make_sable(init_gs=False, vda_mode='online')
    assert 'Geometry loss is disabled' in caplog.text
    assert model.vda is None
    result = model(batch)
    assert result.xyz_norm is None and result.xyz_init_norm is None


def test_disabled_geometry_loads_legacy_vda_weights_without_relaxing_strictness(make_sable):
    model, _ = make_sable(init_gs=False, vda_mode='online')
    checkpoint = model.state_dict()
    checkpoint['vda.legacy_frozen_weight'] = torch.ones(1)

    model.load_state_dict(checkpoint, strict=True)

    checkpoint['unexpected_model_weight'] = torch.ones(1)
    with pytest.raises(RuntimeError, match='unexpected_model_weight'):
        model.load_state_dict(checkpoint, strict=True)
