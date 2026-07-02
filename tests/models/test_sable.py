"""Tests for sable model utilities."""

import pytest
import torch

from beast.models.sable import (
    Sable,
    whiten_background_for_l2_loss,
    build_segmentation_gaussian_mask,
    build_token_masks,
)


class TestBuildSegmentationGaussianMask:
    """Test the build_segmentation_gaussian_mask function."""

    def test_build_segmentation_gaussian_mask_shape(self) -> None:
        # Arrange
        b, v, hh, ww, ph, pw = 2, 3, 4, 4, 16, 16
        seg_mask = torch.ones(b, v, 1, hh * ph, ww * pw)

        # Act
        result = build_segmentation_gaussian_mask(seg_mask, hh, ww, ph, pw)

        # Assert
        assert result.shape == (b, v, hh * ww * ph * pw)

    def test_build_segmentation_gaussian_mask_all_foreground(self) -> None:
        # Arrange: all-ones mask (all foreground)
        b, v, hh, ww, ph, pw = 2, 2, 4, 4, 16, 16
        seg_mask = torch.ones(b, v, 1, hh * ph, ww * pw)

        # Act
        result = build_segmentation_gaussian_mask(seg_mask, hh, ww, ph, pw)

        # Assert: no-op; all weights remain 1
        assert result.shape == (b, v, hh * ww * ph * pw)
        assert torch.all(result == 1.0)

    def test_build_segmentation_gaussian_mask_all_background(self) -> None:
        # Arrange: all-zeros mask (all background)
        b, v, hh, ww, ph, pw = 2, 2, 4, 4, 16, 16
        seg_mask = torch.zeros(b, v, 1, hh * ph, ww * pw)

        # Act
        result = build_segmentation_gaussian_mask(seg_mask, hh, ww, ph, pw)

        # Assert: all weights zero
        assert torch.all(result == 0.0)

    def test_build_segmentation_gaussian_mask_dtype_preserved(self) -> None:
        # Arrange
        b, v, hh, ww, ph, pw = 1, 2, 2, 2, 16, 16
        seg_mask = torch.ones(b, v, 1, hh * ph, ww * pw, dtype=torch.float32)

        # Act
        result = build_segmentation_gaussian_mask(seg_mask, hh, ww, ph, pw)

        # Assert
        assert result.dtype == torch.float32

    def test_build_segmentation_gaussian_mask_matches_token_mask_shape_convention(self) -> None:
        # Arrange: build_token_masks produces [B, V, hh*ww*ph*pw]; verify same convention
        b, v, hh, ww, ph, pw = 2, 2, 4, 4, 16, 16
        n = hh * ww
        keep = torch.ones(b, v, n, dtype=torch.bool)
        _, gaussian_mask_token = build_token_masks(keep, b, v, hh, ww, ph, pw)

        seg_mask = torch.ones(b, v, 1, hh * ph, ww * pw)
        gaussian_mask_seg = build_segmentation_gaussian_mask(seg_mask, hh, ww, ph, pw)

        # Assert: shapes are compatible for elementwise combination
        assert gaussian_mask_seg.shape == gaussian_mask_token.shape

    def test_build_segmentation_gaussian_mask_partial_foreground(self) -> None:
        # Arrange: top half foreground, bottom half background
        b, v, hh, ww, ph, pw = 1, 1, 4, 4, 16, 16
        H, W = hh * ph, ww * pw
        seg_mask = torch.zeros(b, v, 1, H, W)
        seg_mask[:, :, :, : H // 2, :] = 1.0

        # Act
        result = build_segmentation_gaussian_mask(seg_mask, hh, ww, ph, pw)

        # Assert: roughly half the weights are 1, half are 0
        total = result.numel()
        assert result.sum().item() == pytest.approx(total / 2, rel=1e-5)


class TestApplyTargetMaskForL2Loss:
    """Test the apply_target_mask_for_l2_loss function."""

    def test_apply_target_mask_for_l2_loss_disabled_returns_raw_image(self) -> None:
        # Arrange: mask marks half the image as background
        target_img = torch.rand(1, 1, 3, 2, 2)
        target_mask = torch.tensor([[[[[1.0, 1.0], [0.0, 0.0]]]]])

        # Act
        result = whiten_background_for_l2_loss(target_img, target_mask, mask_l2_loss=False)

        # Assert: raw image is returned unchanged
        assert torch.equal(result, target_img)

    def test_apply_target_mask_for_l2_loss_enabled_whitens_background(self) -> None:
        # Arrange
        target_img = torch.zeros(1, 1, 3, 2, 2)
        target_mask = torch.tensor([[[[[1.0, 1.0], [0.0, 0.0]]]]])

        # Act
        result = whiten_background_for_l2_loss(target_img, target_mask, mask_l2_loss=True)

        # Assert: foreground stays raw (0), background becomes white (1)
        expected = (1.0 - target_mask).expand_as(target_img)
        assert torch.equal(result, expected)


class TestMaskedGsRegLossWithSegMask:
    """Test that gs_reg_loss is zeroed on background points when segmentation mask is applied."""

    def test_background_points_do_not_contribute_to_loss(self) -> None:
        import torch.nn.functional as F

        from beast.models.model_utils.losses import masked_gs_reg_loss

        # Arrange: predicted != target only on background points (mask=0)
        b, v, hh, ww, ph, pw = 1, 1, 2, 2, 2, 2
        n = hh * ww * ph * pw  # 16 Gaussians
        xyz_norm = torch.zeros(b, v * n, 3)
        xyz_init_norm = torch.zeros(b, v * n, 3)
        # introduce large displacement only on the second half (background)
        xyz_norm[:, n // 2 :] = 1.0

        # seg_mask: first half foreground (1), second half background (0)
        seg_mask_flat = torch.zeros(b, v, 1, hh * ph, ww * pw)
        seg_mask_flat[:, :, :, : hh * ph // 2, :] = 1.0

        gaussian_mask = build_segmentation_gaussian_mask(seg_mask_flat, hh, ww, ph, pw)
        # flatten to [B, V*N] matching xyz layout
        gaussian_mask_flat = gaussian_mask.reshape(b, -1)

        # Act
        loss_masked = masked_gs_reg_loss(xyz_norm, xyz_init_norm, gaussian_mask_flat)
        loss_unmasked = F.mse_loss(xyz_norm, xyz_init_norm)

        # Assert: masked loss is ~0 (displacement only on background), unmasked is not
        assert loss_masked.item() == pytest.approx(0.0, abs=1e-6)
        assert loss_unmasked.item() > 0.1


class TestSelectTargetGaussians:
    """Test the Sable._select_target_gaussians static method."""

    def test_select_target_gaussians_preserves_feature_dims(self) -> None:
        # Arrange: features is 4D [b, v_input*n_per_view, sh_coeffs, 3], unlike
        # the other Gaussian tensors which have a single trailing dim
        b, v_input, v_target, n_per_view, sh_coeffs = 2, 3, 2, 4, 16
        n_input = v_input * n_per_view
        xyz = torch.rand(b, n_input, 3)
        features = torch.rand(b, n_input, sh_coeffs, 3)
        scaling = torch.rand(b, n_input, 3)
        rotation = torch.rand(b, n_input, 4)
        opacity = torch.rand(b, n_input, 1)
        target_pos = torch.tensor([[0, 2], [1, 2]])

        # Act
        xyz_out, features_out, scaling_out, rotation_out, opacity_out = (
            Sable._select_target_gaussians(
                xyz=xyz,
                features=features,
                scaling=scaling,
                rotation=rotation,
                opacity=opacity,
                target_pos=target_pos,
                v_input=v_input,
                v_target=v_target,
            )
        )

        # Assert: features stays 4D instead of collapsing sh_coeffs and 3 together
        n_out = v_target * n_per_view
        assert features_out.shape == (b, n_out, sh_coeffs, 3)
        assert xyz_out.shape == (b, n_out, 3)
        assert scaling_out.shape == (b, n_out, 3)
        assert rotation_out.shape == (b, n_out, 4)
        assert opacity_out.shape == (b, n_out, 1)

    def test_select_target_gaussians_selects_correct_views(self) -> None:
        # Arrange: tag each view's Gaussians with its view idx so selection is checkable
        b, v_input, v_target, n_per_view, sh_coeffs = 2, 3, 2, 4, 16
        xyz = torch.arange(v_input, dtype=torch.float32).repeat_interleave(n_per_view)
        xyz = xyz.view(1, -1, 1).expand(b, -1, 3).contiguous()
        features = torch.rand(b, v_input * n_per_view, sh_coeffs, 3)
        scaling = torch.rand(b, v_input * n_per_view, 3)
        rotation = torch.rand(b, v_input * n_per_view, 4)
        opacity = torch.rand(b, v_input * n_per_view, 1)
        target_pos = torch.tensor([[0, 2], [1, 2]])

        # Act
        xyz_out, features_out, _, _, _ = Sable._select_target_gaussians(
            xyz=xyz,
            features=features,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
            target_pos=target_pos,
            v_input=v_input,
            v_target=v_target,
        )

        # Assert: xyz_out encodes the selected view idx per batch item
        expected_views = target_pos.repeat_interleave(n_per_view, dim=1)
        assert torch.equal(xyz_out[..., 0], expected_views.float())
        features_expected = features.view(b, v_input, n_per_view, sh_coeffs, 3)
        for b_idx in range(b):
            for out_pos, view_idx in enumerate(target_pos[b_idx].tolist()):
                start = out_pos * n_per_view
                end = start + n_per_view
                assert torch.equal(
                    features_out[b_idx, start:end], features_expected[b_idx, view_idx],
                )
