"""Sable: 3D Gaussian Splatting model with transformer encoder and pose estimation."""
import copy
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import open3d as o3d
from beast.data.datasets import _IMAGENET_MEAN, _IMAGENET_STD
from beast.models.base import BaseLightningModel
from beast.models.model_utils.losses import LossComputer
from beast.models.model_utils.utils_transformer import (
    QK_Norm_TransformerBlock,
    _init_weights,
    _init_weights_layerwise,
)
from beast.models.model_utils.camera_utils import get_interpolated_poses_many
from beast.models.model_utils.utils_pe import get_2d_sincos_pos_embed
from beast.models.model_utils.utils_rot import rot6d2mat, quat2mat
from beast.models.model_utils.utils_gaussian import get_point_range_func, Renderer
from beast.models.model_utils.utils_vis import build_stepback_c2ws
from beast.models.model_utils.data_utils import SplitData

from beast.models.model_utils.utils_vda import (
    extract_vda_depth,
    load_frozen_video_depth_anything,
    save_vda_debug_images,
    pseudo_pointcloud_normalized,
)
from beast.models.model_utils.utils_icp import (
    estimate_initial_transform,
    pixel_xy_to_pointcloud_flat_indices,
)
from beast.models.model_utils.utils_latent import latent_tensor_export_if_requested
from beast.models.model_utils.utils_dino import DinoV3

def imagenet_normalize(x):
    mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


# replace non-finite entries so losses stay valid
def sanitize(t):
    _safe = 1e6
    return torch.nan_to_num(t, nan=0.0, posinf=_safe, neginf=-_safe).clamp(-_safe, _safe)


def normalize_gaussians(xyz: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize Gaussian positions to unit variance."""
    mu = xyz.mean(dim=1, keepdim=True)
    c = xyz - mu
    # RMS of per-point Euclidean norms, averaged over points: sqrt(mean(||c_i||^2))
    rms = torch.sqrt((c.pow(2).sum(dim=-1).mean(dim=1, keepdim=True)) + eps)
    return c / rms.unsqueeze(-1)


def kabsch_rotation_batched(
    cross_cov: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Batched Wahba/Kabsch rotation from cross-covariance ``H = Xc^T Yc`` (shape ``[B, 3, 3]``).

    Runs SVD / det in float32 with autocast handled by caller. Returns proper ``SO(3)`` ``R``
    (``[B, 3, 3]``) cast to ``out_dtype``.
    """
    dev = cross_cov.device
    h32 = cross_cov.float()
    eye = torch.eye(3, device=dev, dtype=torch.float32).unsqueeze(0).expand_as(h32)
    h32 = h32 + eye * float(eps)

    u, _, vh = torch.linalg.svd(h32)
    v = vh.transpose(-2, -1)
    r = torch.bmm(v, u.transpose(-2, -1))

    # Enforce det(R) = +1 (no reflection)
    neg = torch.linalg.det(r) < 0
    if neg.any():
        vf = v.clone()
        vf[neg, :, 2] *= -1.0
        r = torch.bmm(vf, u.transpose(-2, -1))

    return r.to(out_dtype)


def build_transformer_blocks(
    num_layers: int,
    d: int,
    d_head: int,
    use_qk_norm: bool,
    special_init: bool = False,
    depth_init: bool = False,
) -> nn.ModuleList:
    """
    Initializes a list of QK_Norm_TransformerBlock layers with optional special initialization.

    Args:
        num_layers: Number of layers.
        d: Embedding dimension.
        d_head: Head dimension.
        use_qk_norm: Whether to use qk normalization.
        special_init: If True, use special weight initialization.
        depth_init: If True, use layer-depth-aware std deviation.

    Returns:
        Initialized transformer layers.
    """
    layers = [
        QK_Norm_TransformerBlock(d, d_head, use_qk_norm=use_qk_norm)
        for _ in range(num_layers)
    ]

    if special_init:
        for idx, layer in enumerate(layers):
            if depth_init:
                std = 0.02 / (2 * (idx + 1)) ** 0.5
            else:
                std = 0.02 / (2 * num_layers) ** 0.5
            layer.apply(lambda module: _init_weights_layerwise(module, std))

    return nn.ModuleList(layers)


class GaussiansUpsampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # scale and opacity initialization
        self.scaling_bias = self.config['model'].get('scaling_bias', -2.3)
        self.scaling_max = self.config['model'].get('scaling_max', -1.2)
        self.opacity_bias = self.config['model'].get('opacity_bias', -2.0)

        """
        xyz : torch.tensor of shape (n_gaussians, 3)
        features : torch.tensor of shape (n_gaussians, (sh_degree + 1) ** 2, 3)
        scaling : torch.tensor of shape (n_gaussians, 3)
        rotation : torch.tensor of shape (n_gaussians, 4)
        opacity : torch.tensor of shape (n_gaussians, 1)
        """

    def to_gs(self, gaussians):
        """
        gaussians: [b, n_gaussians, d]
        """
        xyz, features, scaling, rotation, opacity = gaussians.split(
            [3, (self.config['model']['gaussians']['sh_degree'] + 1) ** 2 * 3, 3, 4, 1], dim=2
        )

        if not self.config['model']['hard_pixelalign']:
            xyz = xyz.clamp(-500.0, 500.0)

        features = features.reshape(
            features.size(0),
            features.size(1),
            (self.config['model']['gaussians']['sh_degree'] + 1) ** 2,
            3,
        )

        scaling = (scaling + self.scaling_bias).clamp(max=self.scaling_max).clamp(min=-10.0)
        opacity = (opacity + self.opacity_bias).clamp(min=-10.0)

        return xyz, features, scaling, rotation, opacity

    def forward(self, gaussians, images):
        """
        triplane: [b, n_gaussians, d]
        images: [b, l, d]
        output: [b, n_gaussians, dd]
        """
        u = self.config['model']['gaussians']['upsampler']['upsample_factor']
        if u > 1:
            raise NotImplementedError('GaussiansUpsampler only supports u=1')

        gaussians = self.linear(self.layernorm(gaussians))

        return gaussians


class Sable(BaseLightningModel):
    def __init__(self, config: dict) -> None:
        """Initialize Sable model.

        Args:
            config: full training/model config dict.
        """
        super().__init__(config)
        self.loss_computer = LossComputer(config)
        self.d = self.config['model']['transformer']['d']
        self.d_head = self.config['model']['transformer']['d_head']
        self.hh = self.ww = self.config['model']['image_tokenizer']['image_size'] // self.config['model']['image_tokenizer']['patch_size']
        self.ph = self.pw = self.config['model']['image_tokenizer']['patch_size']
        self.inference = True

        # image tokenizer
        self.image_tokenizer = nn.Sequential(
            Rearrange(
                'b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)',
                ph=self.ph,
                pw=self.pw,
            ),
            nn.Linear(
                self.config['model']['image_tokenizer']['in_channels']
                * (self.ph * self.pw),
                self.d,
                bias=False,
            ),
        )
        self.image_tokenizer.apply(_init_weights)

        # image positional embedding embedder
        self.use_pe_embedding_layer = self.config['model'].get('input_with_pe', True)
        if self.use_pe_embedding_layer:
            self.pe_embedder = (
                nn.Sequential(
                    nn.Linear(
                        self.d,
                        self.d,
                    ),
                    nn.SiLU(),
                    nn.Linear(
                        self.d,
                        self.d,
                    ),
                )
                if self.use_pe_embedding_layer
                else nn.Identity()
            )
            self.pe_embedder.apply(_init_weights)

            self.pe_embedder_plucker = (
                nn.Sequential(
                    nn.Linear(
                        self.d,
                        self.d,
                    ),
                    nn.SiLU(),
                    nn.Linear(
                        self.d,
                        self.d,
                    ),
                )
                if self.use_pe_embedding_layer
                else nn.Identity()
            )
            self.pe_embedder_plucker.apply(_init_weights)

        # Reference: VGGT (https://github.com/facebookresearch/vggt/blob/main/vggt/models/aggregator.py)
        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.num_register_tokens = 4
        self.camera_token = nn.Parameter(torch.randn(1, 1, self.d))
        self.register_token = nn.Parameter(torch.randn(1, self.num_register_tokens, self.d))
        self.num_cls_tokens = 2

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # qk norm settings
        use_qk_norm = self.config['model']['transformer'].get('use_qk_norm', False)

        # transformer encoder and init
        self.transformer_encoder = build_transformer_blocks(
            num_layers=config['model']['transformer']['encoder_n_layer'],
            d=self.d,
            d_head=self.d_head,
            use_qk_norm=use_qk_norm,
            special_init=config['model']['transformer'].get('special_init', False),
            depth_init=config['model']['transformer'].get('depth_init', False),
        )

        self.dino_featurizer = DinoV3()

        # transformer encoder2 and init
        self.transformer_encoder_geom = build_transformer_blocks(
            num_layers=config['model']['transformer']['encoder_geom_n_layer'],
            d=self.d,
            d_head=self.d_head,
            use_qk_norm=use_qk_norm,
            special_init=config['model']['transformer'].get('special_init', False),
            depth_init=config['model']['transformer'].get('depth_init', False),
        )

        # pose predictor
        self.pose_predictor = PoseEstimator(self.config)

        # input pose tokenizer
        self.input_pose_tokenizer = nn.Sequential(
            Rearrange(
                'b v (hh ph) (ww pw) c -> (b v) (hh ww) (ph pw c)',
                ph=self.ph,
                pw=self.pw,
            ),
            nn.Linear(
                6
                * (self.ph * self.pw),
                self.d,
                bias=False,
            ),
        )
        self.input_pose_tokenizer.apply(_init_weights)

        # fuse mlp
        self.mlp_fuse = nn.Sequential(
            nn.LayerNorm(self.d * 2, bias=False),
            nn.Linear(
                self.d * 2,
                self.d,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                self.d,
                self.d,
                bias=True,
            ),
        )
        self.mlp_fuse.apply(_init_weights)

        vda_cfg = self.config['model'].get('vda', {})
        self.num_views = self.config['training']['num_views']

        gs_cfg = self.config['model']['gaussians']
        self.init_gs = gs_cfg.get('init_gs', True)
        self.mask_ratio = float(self.config['model'].get('mask_ratio', 0.0))

        # use Kabsch for view merging
        merge_pcd_cfg = self.config['model'].get('merge_pcd', {})
        self.num_icp_points = int(merge_pcd_cfg.get('num_points', 6))
        self.debug_merged_pcd = bool(merge_pcd_cfg.get('debug_merged_pcd', False))

        self.vda_mode = str(vda_cfg.get('mode', 'online')).strip().lower()
        if self.vda_mode not in {'online', 'precomputed'}:
            raise ValueError(f"Unsupported model.vda.mode: {self.vda_mode!r}")
        if self.vda_mode == 'precomputed':
            self.vda = None
        else:
            self.vda = load_frozen_video_depth_anything(vda_cfg)
        self.vda_debug_save = bool(vda_cfg.get('debug_save', False))
        self.vda_debug_dir = vda_cfg.get('debug_dir', self.config['training']['checkpoint_dir'])
        self._vda_debug_saved = False

        # 3D gaussian decoder
        self.gs_per_pixel_sqrt = self.config['model']['gs_per_pixel_sqrt']
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.d, bias=False),
            nn.Linear(
                self.d,
                (self.ph * self.pw) * (
                    3 + (self.config['model']['gaussians']['sh_degree'] + 1) ** 2 * 3 + 3 + 4 + 1
                ),
                bias=False,
            ),
        )
        self.image_token_decoder.apply(_init_weights)

        self.upsampler = GaussiansUpsampler(self.config)
        self.range_func = get_point_range_func(self.config['model']['gaussians'])
        self.renderer = Renderer(self.config)
        self.split_data = SplitData(self.config)

        # config backup
        self.config_bk = copy.deepcopy(self.config)
        self.render_interpolate = config['training'].get('render_interpolate', False)

        if config['model']['transformer'].get('fix_decoder', False):
            self.freeze_weights()

        # training settings
        if config['inference'] or config.get('evaluation', False):
            if config['training'].get('random_inputs', False):
                self.random_index = True
            else:
                self.random_index = False
        else:
            self.random_index = config['training'].get('random_split', False)

    def maybe_randomize_view_indices(self, input_idx, target_idx, device):
        """Randomize view ordering for two-view training."""
        b = input_idx.shape[0]
        swap = (torch.rand(b, device=device) < 0.5).unsqueeze(1)

        if input_idx.shape[1] == 1:
            return torch.where(swap, target_idx, input_idx), torch.where(
                swap, input_idx, target_idx
            )
        return input_idx, target_idx

    # ------------------------------------------------------------------
    # BaseLightningModel interface
    # ------------------------------------------------------------------

    def get_model_outputs(self, batch_dict: dict) -> dict:
        """Run Sable forward pass and return a plain dict.

        Args:
            batch_dict: batch from the dataloader.

        Returns:
            dict of model outputs converted from SimpleNamespace.
        """
        return vars(self(batch_dict))

    def compute_loss(
        self,
        stage: str,
        **kwargs,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Compute composite render + regularization loss.

        Args:
            stage: one of 'train', 'val', 'test'.
            **kwargs: model output fields from get_model_outputs.

        Returns:
            tuple of (total_loss, log_list).
        """
        rendering = kwargs['render']
        target = kwargs['target_image']
        xyz_norm = kwargs['xyz_norm']
        xyz_init_norm = kwargs['xyz_init_norm']
        pixel_mask = kwargs.get('pixel_mask')
        result = self.loss_computer(rendering, target, xyz_norm, xyz_init_norm, pixel_mask)
        log_list = [
            {'name': f'{stage}_l2', 'value': result.l2_loss},
            {'name': f'{stage}_psnr', 'value': result.psnr, 'prog_bar': True},
            {'name': f'{stage}_gs_reg', 'value': result.gs_reg_loss},
        ]
        if result.lpips_loss.item() != 0.0:
            log_list.append({'name': f'{stage}_lpips', 'value': result.lpips_loss})
        if result.perceptual_loss.item() != 0.0:
            log_list.append({'name': f'{stage}_perceptual', 'value': result.perceptual_loss})
        return result.loss, log_list

    def predict_step(self, batch_dict: dict, batch_idx: int) -> dict:
        """Run inference and return model outputs.

        Args:
            batch_dict: batch from the dataloader.
            batch_idx: batch index (unused).

        Returns:
            dict of model outputs.
        """
        return self.get_model_outputs(batch_dict)

    def configure_optimizers(self) -> dict:
        """Configure AdamW optimizer and cosine LR schedule from optimizer config.

        Returns:
            Lightning optimizer/scheduler config dict.
        """
        cfg = self.config['optimizer']
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=cfg['lr'],
            betas=(cfg['beta1'], cfg['beta2']),
            weight_decay=cfg['wd'],
        )
        total_steps = self.config['training']['max_fwdbwd_passes']
        warmup_steps = cfg['warmup']
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg['lr'],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
            div_factor=cfg.get('div_factor', 1.0),
            final_div_factor=cfg.get('final_div_factor', 1.0),
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
            'monitor': 'val_loss',
        }

    def forward(self, data, create_visual=False, render_video=False):

        image_all = data['image'] * 2.0 - 1.0         # [b, v_all, c, h, w], range (0,1) to (-1,1)

        b, v_real, c, h, w = image_all.shape
        device = image_all.device
        batch_idx = torch.arange(b, device=device).unsqueeze(1)

        input_idx, target_idx = self.resolve_view_indices(data, v_real, device)
        # randomize input and target view indices for two-view training
        if self.training:
            input_idx, target_idx = self.maybe_randomize_view_indices(input_idx, target_idx, device)

        v_input = input_idx.shape[1]
        v_target = target_idx.shape[1]

        # padding input to 10 views if less than 10 views (repeat the last view)
        if v_real < 10:
            pad_input = True
            v_all = 10
            pad_views = 10 - v_real
            last_view = image_all[:, -1:, ...].repeat(1, pad_views, 1, 1, 1)
            image_all = torch.cat([image_all, last_view], dim=1)     # [b, v_all, c, h, w]
        else:
            pad_input = False
            v_all = v_real

        '''se3 pose prediction for all views'''
        # tokenize images, add spatial-temporal p.e.
        img_tokens = self.image_tokenizer(image_all)     # [b*v_all, n, d]; n = number of image tokens per view (e.g. 320*320/(16*16)=400)
        _, n, d = img_tokens.shape

        # add spatial positional embedding
        if self.use_pe_embedding_layer:
            img_tokens = self.add_spatial_pe(
                img_tokens,
                b, v_all,
                self.hh,
                self.ww,
                embedder=self.pe_embedder,
            )

        # concanate all tokens together
        cam_tokens = repeat(self.camera_token, '1 one d -> bv one d', bv=b * v_all)
        register_tokens = repeat(self.register_token, '1 num_register_tokens d -> bv num_register_tokens d', bv=b * v_all)
        all_tokens = torch.cat([cam_tokens, register_tokens, img_tokens], dim=1)
        all_tokens = rearrange(all_tokens, '(b v) n d -> b (v n) d', b=b)         # [b, v_all*n, d]

        # pose estimation for all views
        all_tokens = self.run_vggt_encoder(all_tokens, b, v_all)
        all_tokens = rearrange(all_tokens, 'b (v n) d -> (b v) n d', v=v_all)
        cam_tokens, _, _ = all_tokens.split([1, self.num_register_tokens, n], dim=1)

        # get se3 poses and intrinsics
        cam_tokens = cam_tokens[:, 0]                                     # [b*v_all, d]
        cam_info = self.pose_predictor(cam_tokens, v_all)                 # [b*v_all, num_pose_element+3+4], rot, 3d trans, 4d fxfycxcy
        pred_c2w, pred_fxfycxcy = get_cam_se3(cam_info)  # [b*v_all, 4, 4], [b*v_all, 4]
        pred_c2w = rearrange(pred_c2w, '(b v) n d -> b v n d', b=b)
        pred_fxfycxcy = rearrange(pred_fxfycxcy, '(b v) d -> b v d', b=b).detach()
        normalized = True

        # dino image tokenization
        dino_image_all = data["image"]
        if pad_input:
            last_view = dino_image_all[:, -1:, ...].repeat(1, pad_views, 1, 1, 1)
            dino_image_all = torch.cat([dino_image_all, last_view], dim=1)
        dino_image_all = imagenet_normalize(dino_image_all)
        dino_img_tokens_all, dino_cls_tokens = self.dino_featurizer(dino_image_all)
        # dino_img_tokens_all = rearrange(dino_img_tokens_all, 'b v n d -> (b v) n d', b=b, v=v_all)
        # if self.use_pe_embedding_layer:
        #     dino_img_tokens_all = self.add_spatial_pe(
        #         dino_img_tokens_all,
        #         b, v_all,
        #         self.hh,
        #         self.ww,
        #         embedder=self.pe_embedder,
        #     )
        dino_cls_tokens = dino_cls_tokens[batch_idx, input_idx, ...]
        # depth_z_out is for neural encoding
        dino_cls_out = latent_tensor_export_if_requested(
            dino_cls_tokens,
            data,
            self.config,
            self.training,
            data_key='return_dino_cls',
            config_key='return_dino_cls',
        )

        # image tokens masking for input views
        img_tokens_all = rearrange(img_tokens, '(b v) n d -> b v n d', b=b, v=v_all)
        img_tokens_input = img_tokens_all[batch_idx, input_idx, ...]            # [b, v_input, n, d]

        if self.training and self.mask_ratio > 0:
            keep = (torch.rand(img_tokens_input.shape[:-1], device=img_tokens_input.device) >= self.mask_ratio)
            masked_img_tokens_input = img_tokens_input * keep.unsqueeze(-1).to(img_tokens_input.dtype)
        else:
            masked_img_tokens_input = img_tokens_input
        
        '''predict scene representation using (posed) input views'''
        c2w_input = pred_c2w[batch_idx, input_idx, ...]                                                      # [b, v_input, 4, 4]
        fxfycxcy_input = pred_fxfycxcy[batch_idx, input_idx, ...]                                            # [b, v_input, 4]
        c2w_target = pred_c2w[batch_idx, target_idx, ...]                                                    # [b, v_target, 4, 4]
        fxfycxcy_target = pred_fxfycxcy[batch_idx, target_idx, ...].clone()                                  # [b, v_target, 4]

        plucker_rays_input = cam_info_to_plucker(c2w_input, fxfycxcy_input, self.config['model']['target_image'], normalized=normalized, return_moment=True)
        plucker_rays_input = rearrange(plucker_rays_input, '(b v) c h w -> b v h w c', b=b, v=v_input)
        plucker_emb_input = self.input_pose_tokenizer(plucker_rays_input)                                     # [b*v_input, n, d]
        if self.use_pe_embedding_layer:
            plucker_emb_input = self.add_spatial_pe(
                plucker_emb_input,
                b, v_input,
                self.hh, self.ww,
                embedder=self.pe_embedder_plucker,
            )
        plucker_emb_input = rearrange(plucker_emb_input, '(b v) n d -> b (v n) d', v=v_input)                 # [b, v_input*n, d]

        # get posed image representation
        img_tokens_input = rearrange(masked_img_tokens_input, 'b v n d -> b (v n) d')
        img_tokens_input = torch.cat([img_tokens_input, plucker_emb_input], dim=-1)                           # [b, v_input*n, 2d]
        all_tokens = self.mlp_fuse(img_tokens_input)                                                          # [b, v_input*n, d]

        # cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        cls_tokens = dino_cls_tokens.clone()
        all_tokens = torch.cat([cls_tokens, all_tokens], dim=1)

        # encoder layers, predict depths and feature vectors
        all_tokens = self.run_vggt_encoder_geom(all_tokens, b, v_input)   # [b, v_input*n, d]
        frame_cls_tokens, all_tokens = all_tokens.split([self.num_cls_tokens, v_input * n], dim=1)
        # cls_3d_out is for neural encoding
        cls_3d_out = latent_tensor_export_if_requested(
            frame_cls_tokens,
            data,
            self.config,
            self.training,
            data_key='return_frame_cls_tokens',
            config_key='return_frame_cls_tokens',
        )
        # img_tokens_out is for neural decoding
        img_tokens_out = latent_tensor_export_if_requested( 
            all_tokens,
            data,
            self.config,
            self.training,
            data_key='return_img_tokens',
            config_key='return_img_tokens',
        )
        combined_z = torch.cat([frame_cls_tokens, dino_cls_tokens], dim=-1)
        # combined_z_out is for neural encoding
        combined_z_out = latent_tensor_export_if_requested(
            combined_z,
            data,
            self.config,
            self.training,
            data_key='return_combined_z',
            config_key='return_combined_z',
        )

        # decoder layers, upsample gaussians
        img_aligned_gaussians = self.image_token_decoder(all_tokens)
        img_aligned_gaussians = rearrange(
            img_aligned_gaussians,
            'b (v n) d -> b v n d',
            v=v_input,
        )[:, :v_real]
        img_aligned_gaussians = rearrange(
            img_aligned_gaussians,
            'b v n (ph pw c) -> b (v n ph pw) c',  
            ph=self.ph,
            pw=self.pw,
        )

        xyz, features, scaling, rotation, opacity = self.upsampler.to_gs(img_aligned_gaussians)
        img_aligned_xyz = rearrange(
            xyz,
            'b (v hh ww ph pw) c -> b v c (hh ph) (ww pw)',
            v=v_input,
            hh=self.hh,
            ww=self.ww,
            ph=self.ph,
            pw=self.pw,
        )

        if self.config['model']['hard_pixelalign']:
            img_aligned_xyz = img_aligned_xyz.mean(dim=2, keepdim=True)
            img_aligned_xyz = self.range_func(img_aligned_xyz)
            plucker_rays_input = cam_info_to_plucker(
                c2w_input,
                fxfycxcy_input,
                self.config['model']['target_image'],
                normalized=normalized,
                return_moment=False,
            )
            plucker_rays_input = rearrange(plucker_rays_input, '(b v) c h w -> b v c h w', b=b)
            ray_o, ray_d = plucker_rays_input.split([3, 3], dim=2)
            img_aligned_xyz = ray_o + img_aligned_xyz * ray_d
        else:
            ray_o = None
        xyz = rearrange(
            img_aligned_xyz,
            'b v c (hh ph) (ww pw) -> b (v hh ww ph pw) c',
            ph=self.ph,
            pw=self.pw,
        )

        pcd_h = int(self.config['model']['target_image']['height'])
        pcd_w = int(self.config['model']['target_image']['width'])
        # VDA depth
        depth_output = self._resolve_depth_output_for_input_views(
            data=data,
            b=b,
            v_input=v_input,
            pcd_h=pcd_h,
            pcd_w=pcd_w,
            batch_idx=batch_idx,
            input_idx=input_idx,
        ).detach()

        xyz_norm = None
        xyz_init_norm = None
        gs_reg_loss = xyz.new_zeros(())

        if self.init_gs:
            depth_for_pcd = depth_output.clone()
            depth_for_pcd = rearrange(depth_for_pcd, 'b v h w -> (b v) h w', b=b, v=v_target)
            xyz_flat = torch.stack(
                [
                    pseudo_pointcloud_normalized(d, self.ph, self.pw)
                    for d in depth_for_pcd
                ],
                dim=0,
            )
            xyz_init = rearrange(xyz_flat, '(b v) n c -> b v n c', b=b, v=v_target)
            xyz = rearrange(xyz, 'b (v n) c -> b v n c', b=b, v=v_target)
            xyz_init_for_merge = xyz_init.detach().clone().cpu().float().numpy()
            xyz_init_src_pts = xyz_init_for_merge[:, 0]
            xyz_init_tgt_pts = xyz_init_for_merge[:, 1]

            if self.debug_merged_pcd:
                self.debug_merged_pcd_num_batches = int(self.config['model']['merge_pcd'].get('debug_merged_pcd_num_batches', 1))
                n_dbg_save = min(self.debug_merged_pcd_num_batches, b)
                debug_src_corr_idx = [None] * n_dbg_save
                debug_tgt_corr_idx = [None] * n_dbg_save
                debug_icp_src_xyz = [None] * n_dbg_save
                debug_icp_tgt_xyz = [None] * n_dbg_save
                debug_corr_left_xy = [None] * n_dbg_save
                debug_corr_right_xy = [None] * n_dbg_save

            for b_i in range(b):
                corr_xy_from_pixels = None
                src_idx = []
                tgt_idx = []
                init_src_pcd = o3d.geometry.PointCloud()
                init_tgt_pcd = o3d.geometry.PointCloud()
                init_src_pcd.points = o3d.utility.Vector3dVector(xyz_init_src_pts[b_i])
                init_tgt_pcd.points = o3d.utility.Vector3dVector(xyz_init_tgt_pts[b_i])

                # use GT animal pose as correspondences for kabsch; padding entries have confidence == 0
                valid = data['confidence'][b_i] > 0
                left_xy_np = data['leftcamera_xy'][b_i][valid].cpu().numpy()
                right_xy_np = data['rightcamera_xy'][b_i][valid].cpu().numpy()
                if len(left_xy_np) >= 3:
                    src_idx = pixel_xy_to_pointcloud_flat_indices(
                        left_xy_np, pcd_h, pcd_w, self.ph, self.pw
                    )
                    tgt_idx = pixel_xy_to_pointcloud_flat_indices(
                        right_xy_np, pcd_h, pcd_w, self.ph, self.pw
                    )
                    corr_xy_from_pixels = (
                        np.ascontiguousarray(left_xy_np, dtype=np.float32),
                        np.ascontiguousarray(right_xy_np, dtype=np.float32),
                    )

                T_kabsch = estimate_initial_transform(
                    init_src_pcd, init_tgt_pcd, src_idx, tgt_idx
                )
                init_src_pcd.transform(T_kabsch)
                init_src_pts = np.asarray(init_src_pcd.points)
                init_tgt_pts = np.asarray(init_tgt_pcd.points)
                xyz_init[b_i, 0] = torch.from_numpy(init_src_pts).to(xyz.device)
                xyz_init[b_i, 1] = torch.from_numpy(init_tgt_pts).to(xyz.device)

                if self.debug_merged_pcd and b_i < n_dbg_save:
                    debug_src_corr_idx[b_i] = list(src_idx)
                    debug_tgt_corr_idx[b_i] = list(tgt_idx)
                    debug_icp_src_xyz[b_i] = np.ascontiguousarray(
                        xyz_init_src_pts[b_i]
                    )
                    debug_icp_tgt_xyz[b_i] = np.ascontiguousarray(
                        xyz_init_tgt_pts[b_i]
                    )
                    if corr_xy_from_pixels is not None:
                        debug_corr_left_xy[b_i] = corr_xy_from_pixels[0]
                        debug_corr_right_xy[b_i] = corr_xy_from_pixels[1]

            xyz_init = rearrange(xyz_init, 'b v n c -> b (v n) c', b=b, v=v_target)
            xyz = rearrange(xyz, 'b v n c -> b (v n) c', b=b, v=v_target)

            if self.debug_merged_pcd:
                xyz = xyz_init
                from beast.models.model_utils.debug_merge import save_merged_pcd
                save_merged_pcd(
                    xyz_init=xyz_init,
                    data=data,
                    target_idx=target_idx,
                    v_target=v_target,
                    pcd_h=int(self.config['model']['target_image']['height']),
                    pcd_w=int(self.config['model']['target_image']['width']),
                    ph=self.ph,
                    pw=self.pw,
                    num_batches=self.debug_merged_pcd_num_batches,
                    src_corr_idx=debug_src_corr_idx,
                    tgt_corr_idx=debug_tgt_corr_idx,
                    depth_maps=depth_output,
                    pre_kabsch_src_xyz=debug_icp_src_xyz,
                    pre_kabsch_tgt_xyz=debug_icp_tgt_xyz,
                    corr_left_xy=debug_corr_left_xy,
                    corr_right_xy=debug_corr_right_xy,
                )

            xyz_norm, xyz_init_norm = map(
                sanitize, map(
                    normalize_gaussians, 
                    (xyz, xyz_init.detach()),
                )
            )
            gs_reg_loss = torch.nn.functional.mse_loss(xyz_norm, xyz_init_norm)

        xyz, features, scaling, rotation, opacity = map(
            sanitize,
            (xyz, features, scaling, rotation, opacity),
        )

        gaussian_attrs = SimpleNamespace(
            xyz=xyz,
            features=features,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
        )

        render = None
        height, width = int(h), int(w)
        if height <= 0 or width <= 0:
            raise ValueError(
                f'Invalid image dimensions from batch: h={h}, w={w}. '
                'Check dataset preprocessing and target_image config.'
            )
        if normalized:
            intrinsics_scale = fxfycxcy_target.new_tensor([width, height, width, height])
            fxfycxcy_target = fxfycxcy_target * intrinsics_scale

        render = self.renderer(
            xyz,
            features,
            scaling,
            rotation,
            opacity,
            height,
            width,
            C2W=c2w_target,
            fxfycxcy=fxfycxcy_target,
        )
        render_results = SimpleNamespace(
            rendered_images=render.render,
        )

        vis_only_results = None
        should_render_video = (
            render_video
            or
            (not self.training)
            or self.config['inference']
            or self.config.get('evaluation', False)
            or data.get('return_render_video', False)
        )
        if should_render_video:
            with torch.no_grad():
                vis_only_results = self.render_images_video(
                    gaussian_attrs,
                    c2w_target,
                    fxfycxcy_target,
                    normalized=False,
                    step_back=self.config.get('evaluation_step_back_distance', 0),
                )
                # vis_only_results_input = self.render_images_video(gaussian_attrs, c2w_input, fxfycxcy_input, normalized=False)

        gaussians = []
        pixelalign_xyz = []
        gaussians_usage = []
        gaussians_scale = []
        gaussians_opacity = []
        for b in range(xyz.size(0)):
            self.renderer.gaussians_model.empty()
            gaussians_model = copy.deepcopy(self.renderer.gaussians_model)
            gaussians.append(
                gaussians_model.set_data(
                    xyz[b].detach().float(),
                    features[b].detach().float(),
                    scaling[b].detach().float(),
                    rotation[b].detach().float(),
                    opacity[b].detach().float(),
                )
            )

            threshold = 0.05
            usage_mask = gaussians[-1].get_opacity > threshold
            usage = usage_mask.sum() / usage_mask.numel()
            if torch.is_tensor(usage):
                usage = usage.item()
            gaussians_usage.append(usage)

            mean_scale = gaussians[-1].get_scaling.mean()
            if torch.is_tensor(mean_scale):
                mean_scale = mean_scale.item()
            gaussians_scale.append(mean_scale)

            mean_opacity = gaussians[-1].get_opacity.mean()
            if torch.is_tensor(mean_opacity):
                mean_opacity = mean_opacity.item()
            gaussians_opacity.append(mean_opacity)

            img_aligned_xyz = gaussians[-1].get_xyz
            img_aligned_xyz = rearrange(
                img_aligned_xyz,
                '(v hh ww ph pw) c -> v c (hh ph) (ww pw)',
                v=v_input,
                hh=self.hh,
                ww=self.ww,
                ph=self.ph,
                pw=self.pw,
            )
            pixelalign_xyz.append(img_aligned_xyz)
        pixelalign_xyz = torch.stack(pixelalign_xyz, dim=0)

        # return results
        result = SimpleNamespace(**{
            'ray_o': ray_o,
            'gaussians': gaussians,
            'pixelalign_xyz': pixelalign_xyz,
            'image': data['image'],
            'input_image': data['image'][batch_idx, input_idx.clamp(max=v_real - 1), ...],
            'target_image': data['image'][batch_idx, target_idx, ...],
            'render': render_results.rendered_images,
            'c2w_input': c2w_input,
            'fxfycxcy_input': fxfycxcy_input,
            'c2w_target': c2w_target,
            'fxfycxcy_target': fxfycxcy_target,
            'input_indices': input_idx,
            'target_indices': target_idx,
            'depth_output': depth_output,
            'xyz_norm': xyz_norm,
            'xyz_init_norm': xyz_init_norm,
            'gs_reg_loss': gs_reg_loss,
            'frame_z': cls_3d_out,
            'img_tokens': img_tokens_out,
            'depth_z': dino_cls_out,
            'combined_z': combined_z_out,
        })

        result.render_video = (
            vis_only_results.rendered_images_video.detach().clamp(0, 1)
            if vis_only_results is not None
            else None
        )

        return result

    def predict_frame_from_all_tokens(self, all_tokens, c2w_input, fxfycxcy_input, c2w_target, fxfycxcy_target, data):

        b, v_input, n, d = all_tokens.shape
        h = w = int(self.config['model']['image_tokenizer']['image_size'])

        all_tokens = rearrange(all_tokens, 'b v n d -> b (v n) d', v=v_input)

        # decoder layers, upsample gaussians
        img_aligned_gaussians = self.image_token_decoder(all_tokens)
        img_aligned_gaussians = rearrange(
            img_aligned_gaussians,
            'b (v n) d -> b v n d',
            v=v_input,
        )
        img_aligned_gaussians = rearrange(
            img_aligned_gaussians,
            'b v n (ph pw c) -> b (v n ph pw) c',  
            ph=self.ph,
            pw=self.pw,
        )

        if img_aligned_gaussians.shape[0] != b:
            img_aligned_gaussians = rearrange(
                img_aligned_gaussians, '(b v) n c -> b (v n) c', b=b, v=v_input
            )

        xyz, features, scaling, rotation, opacity = self.upsampler.to_gs(img_aligned_gaussians)

        img_aligned_xyz = rearrange(
            xyz,
            'b (v hh ww ph pw) c -> b v c (hh ph) (ww pw)',
            v=v_input,
            hh=self.hh,
            ww=self.ww,
            ph=self.ph,
            pw=self.pw,
        )

        normalized = True
        ray_o = None
        if self.config['model']['hard_pixelalign']:
            img_aligned_xyz = img_aligned_xyz.mean(dim=2, keepdim=True)
            img_aligned_xyz = self.range_func(img_aligned_xyz)
            plucker_rays_input = cam_info_to_plucker(
                c2w_input,
                fxfycxcy_input,
                self.config['model']['target_image'],
                normalized=normalized,
                return_moment=False,
            )
            plucker_rays_input = rearrange(plucker_rays_input, '(b v) c h w -> b v c h w', b=b)
            ray_o, ray_d = plucker_rays_input.split([3, 3], dim=2)
            img_aligned_xyz = ray_o + img_aligned_xyz * ray_d
            xyz = rearrange(
                img_aligned_xyz,
                'b v c (hh ph) (ww pw) -> b (v hh ww ph pw) c',
                ph=self.ph,
                pw=self.pw,
            )
        else:
            xyz = rearrange(
                img_aligned_xyz,
                'b v c (hh ph) (ww pw) -> b (v hh ww ph pw) c',
                ph=self.ph,
                pw=self.pw,
            )

        xyz, features, scaling, rotation, opacity = map(
            sanitize,
            (xyz, features, scaling, rotation, opacity),
        )

        gaussian_attrs = SimpleNamespace(
            xyz=xyz,
            features=features,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
        )

        render = None
        height, width = int(h), int(w)
        if height <= 0 or width <= 0:
            raise ValueError(
                f'Invalid image dimensions from batch: h={h}, w={w}. '
                'Check dataset preprocessing and target_image config.'
            )
        normalized = False
        if normalized:
            intrinsics_scale = fxfycxcy_target.new_tensor([width, height, width, height])
            fxfycxcy_target = fxfycxcy_target * intrinsics_scale

        render = self.renderer(
            xyz,
            features,
            scaling,
            rotation,
            opacity,
            height,
            width,
            C2W=c2w_target,
            fxfycxcy=fxfycxcy_target,
        )
        render_results = SimpleNamespace(
            rendered_images=render.render,
        )

        gaussians = []
        pixelalign_xyz = []
        gaussians_usage = []
        gaussians_scale = []
        gaussians_opacity = []
        for b in range(xyz.size(0)):
            self.renderer.gaussians_model.empty()
            gaussians_model = copy.deepcopy(self.renderer.gaussians_model)
            gaussians.append(
                gaussians_model.set_data(
                    xyz[b].detach().float(),
                    features[b].detach().float(),
                    scaling[b].detach().float(),
                    rotation[b].detach().float(),
                    opacity[b].detach().float(),
                )
            )
            threshold = 0.05
            usage_mask = gaussians[-1].get_opacity > threshold
            usage = usage_mask.sum() / usage_mask.numel()
            if torch.is_tensor(usage):
                usage = usage.item()
            gaussians_usage.append(usage)

            mean_scale = gaussians[-1].get_scaling.mean()
            if torch.is_tensor(mean_scale):
                mean_scale = mean_scale.item()
            gaussians_scale.append(mean_scale)

            mean_opacity = gaussians[-1].get_opacity.mean()
            if torch.is_tensor(mean_opacity):
                mean_opacity = mean_opacity.item()
            gaussians_opacity.append(mean_opacity)

            img_aligned_xyz = gaussians[-1].get_xyz
            img_aligned_xyz = rearrange(
                img_aligned_xyz,
                '(v hh ww ph pw) c -> v c (hh ph) (ww pw)',
                v=v_input,
                hh=self.hh,
                ww=self.ww,
                ph=self.ph,
                pw=self.pw,
            )
            pixelalign_xyz.append(img_aligned_xyz)
        pixelalign_xyz = torch.stack(pixelalign_xyz, dim=0)

        # return results
        result = SimpleNamespace(**{
            'ray_o': ray_o,
            'gaussians': gaussians,
            'image': data['image'],
            'pixelalign_xyz': pixelalign_xyz,
            'render': render_results.rendered_images,
        })
        return result

    def resolve_view_indices(
        self,
        data: dict[str, torch.Tensor],
        num_real_views: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve view indices for input and target views.

        Args:
            data: dictionary containing input and target view indices.
            num_real_views: number of real views.
            device: device to use for the indices.

        Returns:
            tuple of input and target view indices.
        """
        batch_size = data['image'].shape[0]

        if 'input_indices' in data and 'target_indices' in data:
            input_idx = data['input_indices'].to(device=device, dtype=torch.long)
            target_idx = data['target_indices'].to(device=device, dtype=torch.long)
            return input_idx, target_idx

        if 'context_indices' in data and 'target_indices' in data:
            input_idx = data['context_indices'].to(device=device, dtype=torch.long)
            target_idx = data['target_indices'].to(device=device, dtype=torch.long)
            return input_idx, target_idx

        can_split_training_views = (
            (not self.config['inference'])
            and (not self.config.get('evaluation', False))
            and num_real_views == self.config['training']['num_views']
        )
        if can_split_training_views:
            _, _, input_idx, target_idx = self.split_data(
                data, random_index=self.random_index
            )
            return (
                input_idx.to(device=device, dtype=torch.long),
                target_idx.to(device=device, dtype=torch.long),
            )

        if num_real_views < self.config['training'].get('num_input_views', 5):
            input_count = self.config['training'].get('num_input_views', 5)
        else:
            input_count = num_real_views

        input_idx = torch.arange(input_count, device=device, dtype=torch.long)
        target_idx = torch.arange(num_real_views, device=device, dtype=torch.long)
        input_idx = input_idx.unsqueeze(0).repeat(batch_size, 1)
        target_idx = target_idx.unsqueeze(0).repeat(batch_size, 1)
        return input_idx, target_idx

    def add_spatial_pe(self, tokens, b, v, h_tokens, w_tokens, embedder):
        """
        Add spatial (and optionally temporal) positional encoding to tokens.

        Args:
            tokens: shape [b*v, n, d]
            b: batch size
            v: number of views
            h_tokens: number of tokens along height
            w_tokens: number of tokens along width
            embedder: PE projection layer

        Returns:
            tokens with spatial PE added, shape [b*v, n, d]
        """
        bv, n, d = tokens.shape
        assert (h_tokens * w_tokens) == n, f'Token count {n} != h*w {h_tokens}x{w_tokens}'

        spatial_pe = get_2d_sincos_pos_embed(
            embed_dim=d,
            grid_size=(h_tokens, w_tokens),
            device=tokens.device,
        ).to(tokens.dtype)  # [n, d]

        spatial_pe = spatial_pe.reshape(1, 1, n, d).repeat(b, v, 1, 1)   # [b, v, n, d]
        spatial_pe = spatial_pe.reshape(bv, n, d)                       # [b*v, n, d]

        pe = embedder(spatial_pe)                                       # [b*v, n, d]
        return tokens + pe

    @staticmethod
    def slice_expand_and_flatten(token_tensor, B, S):
        """
        Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
        1) Uses the first position (index=0) for the first frame only
        2) Uses the second position (index=1) for all remaining frames (S-1 frames)
        3) Expands both to match batch size B
        4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
        followed by (S-1) second-position tokens
        5) Flattens to (B*S, X, C) for processing

        Returns:
            torch.Tensor: Processed tokens with shape (B*S, X, C)
        """

        # Slice out the "query" tokens => shape (1, 1, ...)
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
        # Slice out the "other" tokens => shape (1, S-1, ...)
        others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
        # Concatenate => shape (B, S, ...)
        combined = torch.cat([query, others], dim=1)

        # Finally flatten => shape (B*S, ...)
        combined = combined.view(B * S, *combined.shape[2:])
        return combined

    def run_vggt_encoder(self, all_tokens_encoder, b, v):
        checkpoint_every = self.config['training']['grad_checkpoint_every']
        for i in range(0, len(self.transformer_encoder), checkpoint_every):
            if i % 2 == 0:
                # frame attention
                all_tokens_encoder = rearrange(all_tokens_encoder, 'b (v n) d -> (b v) n d', v=v)
            else:
                # global attention
                all_tokens_encoder = rearrange(all_tokens_encoder, '(b v) n d -> b (v n) d', b=b)

            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self.run_layers_encoder(i, i + 1),
                all_tokens_encoder,
                use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self.run_layers_encoder(i + 1, i + checkpoint_every)(all_tokens_encoder)
        return all_tokens_encoder

    def run_vggt_encoder_geom(self, all_tokens_encoder, b, v):
        checkpoint_every = self.config['training']['grad_checkpoint_every']
        for i in range(0, len(self.transformer_encoder_geom), checkpoint_every):
            if i % 2 == 0:
                # frame attention
                all_tokens_encoder = rearrange(all_tokens_encoder, 'b (v n) d -> (b v) n d', v=v)
            else:
                # global attention
                all_tokens_encoder = rearrange(all_tokens_encoder, '(b v) n d -> b (v n) d', b=b)

            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self.run_layers_encoder_geom(i, i + 1),
                all_tokens_encoder,
                use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self.run_layers_encoder_geom(i + 1, i + checkpoint_every)(all_tokens_encoder)
        return all_tokens_encoder

    def run_layers_encoder(self, start, end):
        def custom_forward(concat_nerf_img_tokens):
            for i in range(start, min(end, len(self.transformer_encoder))):
                concat_nerf_img_tokens = self.transformer_encoder[i](concat_nerf_img_tokens)
            return concat_nerf_img_tokens
        return custom_forward

    def run_layers_encoder_geom(self, start, end):
        def custom_forward(concat_nerf_img_tokens):
            for i in range(start, min(end, len(self.transformer_encoder_geom))):
                concat_nerf_img_tokens = self.transformer_encoder_geom[i](concat_nerf_img_tokens)
            return concat_nerf_img_tokens
        return custom_forward

    def render_images_video(self, gaussian_attrs, c2w_all, fxfycxcy_all, normalized=False, step_back=0):
        '''
        points_input_all  : [b, v_input, h_rend_input, w_rend_input, 3]
        features_input_all: [b, v_input, h_rend_input, w_rend_input, c]
        c2w_all           : [b*v, 4, 4]
        fxfycxcy_all      : [b*v, 4]
        '''
        with torch.no_grad():
            xyz = gaussian_attrs.xyz.detach()
            features = gaussian_attrs.features.detach()
            scaling = gaussian_attrs.scaling.detach()
            rotation = gaussian_attrs.rotation.detach()
            opacity = gaussian_attrs.opacity.detach()
            c2w_all = c2w_all.detach()
            fxfycxcy_all = fxfycxcy_all.detach()

            b, v, _, _ = c2w_all.shape
            device = xyz.device

            all_renderings = []
            num_frames = 30
            traj_type = 'interpolate'
            order_poses = False

            for i in range(b):
                c2ws = c2w_all[i]                   # [v, 4, 4]
                fxfycxcy = fxfycxcy_all[i]          # [v, 4]
                if traj_type == 'interpolate':
                    # build Ks from fxfycxcy
                    Ks = torch.zeros((c2ws.shape[0], 3, 3), device=device)
                    Ks[:, 0, 0] = fxfycxcy[:, 0]
                    Ks[:, 1, 1] = fxfycxcy[:, 1]
                    Ks[:, 0, 2] = fxfycxcy[:, 2]
                    Ks[:, 1, 2] = fxfycxcy[:, 3]
                    c2ws, Ks = get_interpolated_poses_many(
                        c2ws[:, :3, :4], Ks, num_frames, order_poses=order_poses
                    )
                    frame_c2ws = torch.cat([c2ws.to(device), torch.tensor([[[0, 0, 0, 1]]], device=device).repeat(c2ws.shape[0], 1, 1)], dim=1)    # [v',4,4]
                    frame_fxfycxcy = torch.zeros((c2ws.shape[0], 4), device=device)
                    frame_fxfycxcy[:, 0] = Ks[:, 0, 0]      # [v',4]
                    frame_fxfycxcy[:, 1] = Ks[:, 1, 1]
                    frame_fxfycxcy[:, 2] = Ks[:, 0, 2]
                    frame_fxfycxcy[:, 3] = Ks[:, 1, 2]
                elif traj_type == 'same':
                    frame_c2ws = c2ws.clone()
                    frame_fxfycxcy = fxfycxcy.clone()
                else:
                    raise NotImplementedError

                if step_back > 0:
                    frame_c2ws = build_stepback_c2ws(frame_c2ws, step_back_distance=step_back)

                batch_size = 5
                num_views = frame_c2ws.shape[0]
                renderings = []

                for start in range(0, num_views, batch_size):
                    end = min(start + batch_size, num_views)

                    # Slice views
                    batch_c2w = frame_c2ws[start:end].unsqueeze(0)  # [1, batch_size, 4, 4]
                    batch_fx = frame_fxfycxcy[start:end].unsqueeze(0)  # [1, batch_size, 4]

                    # Call renderer on minibatch
                    rendered_images_batch = self.renderer(
                        xyz,
                        features,
                        scaling,
                        rotation,
                        opacity,
                        self.config['model']['image_tokenizer']['image_size'],
                        self.config['model']['image_tokenizer']['image_size'],
                        C2W=batch_c2w,
                        fxfycxcy=batch_fx,
                    ).render.squeeze(0)  # [batch_size, H, W, 3]

                    renderings.append(rendered_images_batch)

                # Concatenate all rendered minibatches
                rendered_images_all = torch.cat(renderings, dim=0)  # [num_views, H, W, 3]
                all_renderings.append(rendered_images_all)

            all_renderings = torch.stack(all_renderings)    # [b,v',c,h,w]

        render_results = SimpleNamespace(
            rendered_images_video=all_renderings,
        )
        return render_results

    def extract_vda_depth(self, images, num_views: int):
        """
        images: [b, num_views, 3, H, W] in [0, 1]
        returns: depth_maps [b, num_views, H_vda, W_vda] (e.g. 518x518)
        """
        if self.vda is None:
            raise RuntimeError("extract_vda_depth() was called while model.vda.mode='precomputed'.")
        return extract_vda_depth(self.vda, images, num_views)

    def _resolve_depth_output_for_input_views(
        self,
        *,
        data,
        b: int,
        v_input: int,
        pcd_h: int,
        pcd_w: int,
        batch_idx: torch.Tensor,
        input_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.vda_mode == 'precomputed':
            if 'depth_vda' not in data:
                raise KeyError(
                    "model.vda.mode='precomputed' requires batch['depth_vda'] from the dataset."
                )
            depth_vda = data['depth_vda']
            if depth_vda.ndim != 5:
                raise ValueError(
                    f'Expected depth_vda to have shape [B, V, 1, H, W], got {tuple(depth_vda.shape)}'
                )
            if int(depth_vda.shape[0]) != int(b):
                raise ValueError(
                    f'depth_vda batch size mismatch: expected {b}, got {depth_vda.shape[0]}'
                )
            if int(depth_vda.shape[1]) < int(v_input):
                raise ValueError(
                    f'depth_vda view count mismatch: need at least {v_input}, got {depth_vda.shape[1]}'
                )
            if int(depth_vda.shape[2]) != 1:
                raise ValueError(
                    f'Expected depth_vda to have a singleton channel dimension, got {depth_vda.shape[2]}'
                )
            if tuple(depth_vda.shape[-2:]) != (int(pcd_h), int(pcd_w)):
                raise ValueError(
                    f'depth_vda spatial size mismatch: expected {(pcd_h, pcd_w)}, '
                    f'got {tuple(depth_vda.shape[-2:])}'
                )
            return depth_vda[:, :v_input, 0, ...].contiguous()

        images_in = data['image'][batch_idx, input_idx, ...]
        depth_all, vda_in = self.extract_vda_depth(images_in, v_input)
        depth_all = torch.nn.functional.interpolate(
            rearrange(depth_all, 'b v h w -> (b v) 1 h w'),
            size=(pcd_h, pcd_w),
            mode='bilinear',
            align_corners=True,
        )
        depth_all = rearrange(depth_all, '(b v) 1 h w -> b v h w', b=b, v=v_input)
        self.maybe_debug_vda(images_in, vda_in, depth_all)
        return depth_all

    def maybe_debug_vda(self, vda_in_orig, vda_in, depth_maps) -> None:
        """Optionally save one VDA RGB/depth example for debugging."""

        if not getattr(self, 'vda_debug_save', False):
            return

        if getattr(self, '_vda_debug_saved', False):
            return

        try:
            save_vda_debug_images(vda_in_orig, vda_in, depth_maps, Path(self.vda_debug_dir))
            self._vda_debug_saved = True
        except Exception:
            # Never break training due to debug logging.
            self._vda_debug_saved = True


def get_cam_se3(cam_info):
    '''
    cam_info: [b,num_pose_element+3+4], rot, 3d trans, 4d fxfycxcy
    '''
    b, n = cam_info.shape

    if n == 13:
        rot_6d = cam_info[:, :6]
        R = rot6d2mat(rot_6d)  # [b,3,3]
        t = cam_info[:, 6:9].unsqueeze(-1)   # [b,3,1]
        fxfycxcy = cam_info[:, 9:]   # normalized by resolution / shift from average, [b,4]
    elif n == 11:
        rot_quat = cam_info[:, :4]
        R = quat2mat(rot_quat)
        t = cam_info[:, 4:7].unsqueeze(-1)   # [b,3,1]
        fxfycxcy = cam_info[:, 7:]   # normalized by resolution / shift from average, [b,4]
    else:
        raise NotImplementedError

    Rt = torch.cat([R, t], dim=2)  # [b,3,4]
    c2w = torch.cat([Rt,
                    torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device).view(1, 1, 4).repeat(b, 1, 1)], dim=1)  # [b,4,4]
    return c2w, fxfycxcy


def cam_info_to_plucker(c2w, fxfycxcy, target_imgs_info, normalized=True, return_moment=True):
    '''
    c2w: [b,4,4] or [b,v,4,4]
    fxfycxcy: [b,4] or [b,v,4]
    '''
    if len(c2w.shape) == 3:
        b = c2w.shape[0]
    elif len(c2w.shape) == 4:
        c2w, fxfycxcy = rearrange(c2w.clone(), 'b v n d -> (b v) n d'), \
            rearrange(fxfycxcy.clone(), 'b v d -> (b v) d')
        b = c2w.shape[0]

    device = c2w.device
    h, w = target_imgs_info['height'], target_imgs_info['width']

    fxfycxcy = fxfycxcy.clone()
    if normalized:
        fxfycxcy[:, 0] *= w
        fxfycxcy[:, 1] *= h
        fxfycxcy[:, 2] *= w
        fxfycxcy[:, 3] *= h

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    y, x = y.to(c2w), x.to(c2w)
    x = x[None, :, :].expand(b, -1, -1).reshape(b, -1)
    y = y[None, :, :].expand(b, -1, -1).reshape(b, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # [b, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b, h*w, 3]
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b, h*w, 3]

    ray_o = ray_o.reshape(b, h, w, 3).permute(0, 3, 1, 2)   # [b,3,h,w]
    ray_d = ray_d.reshape(b, h, w, 3).permute(0, 3, 1, 2)

    if return_moment:
        plucker = torch.cat(
            [
                torch.cross(ray_o, ray_d, dim=1),
                ray_d,
            ],
            dim=1,
        )
    else:
        plucker = torch.cat(
            [
                ray_o,
                ray_d,
            ],
            dim=1,
        )
    return plucker     # [b,c=6,h,w]


class CanonicalKHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config['model']['transformer']['d']

        self.layers = nn.ModuleList([
            nn.Linear(d, d, bias=True),
            nn.SiLU(),
            nn.Linear(d, 1, bias=True),
        ])

        # Apply weight initialization
        self.apply(_init_weights)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            print(f'Output after layer {i} ({layer.__class__.__name__}): {x}')
        return x


class PoseEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.canonical = self.config['model']['pose_latent'].get('canonical', 'first')
        assert self.canonical in ['first', 'middle', 'unordered'], \
            f'Unknown canonical mode: {self.canonical}'
        self.is_pairwise = self.config['model']['pose_latent'].get('mode', 'pairwise') == 'pairwise'
        self.rel_head_input = self.config['model']['transformer']['d'] * 2 if self.is_pairwise else config['model']['transformer']['d']
        self.pose_consistency_reg_weight = self.config['training'].get('pose_consistency_reg_weight', 0.0)

        self.pose_rep = self.config['model']['pose_latent'].get('representation', '6d')
        print(f'Pose representation: {self.pose_rep}')
        if self.pose_rep == '6d':
            self.num_pose_element = 6
        elif self.pose_rep == 'quat':
            self.num_pose_element = 4
        else:
            raise NotImplementedError

        self.rel_head = nn.Sequential(
            nn.Linear(
                self.rel_head_input,
                config['model']['transformer']['d'],
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                config['model']['transformer']['d'],
                self.num_pose_element + 3,
                bias=True,
            ),
        )
        self.rel_head.apply(_init_weights)

        self.canonical_k_head = nn.Sequential(
            nn.Linear(
                config['model']['transformer']['d'],
                config['model']['transformer']['d'],
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                config['model']['transformer']['d'],
                1,
                bias=False,
            ),
        )
        self.canonical_k_head.apply(_init_weights)

        self.f_bias = 1.25

    def forward(self, x, v):
        '''
        x: [b*v, d] or [b, v, d]
        '''

        if x.ndim == 2:
            x = rearrange(x, '(b v) d -> b v d', v=v)
            return_dim2 = True
        else:
            return_dim2 = False

        b = x.shape[0]

        if self.is_pairwise:
            if v == 1:
                return self.forward_canonical_single_view(x, v, self.canonical, return_dim2)
            else:
                return self.forward_canonical(x, v, self.canonical, return_dim2)
        else:
            return self.forward_global(x, v, self.canonical, return_dim2)

    def forward_global(self, x, v, canonical, return_dim2):
        '''
        Unordered: extrinsics per view, intrinsics averaged across views.
        Adds predicted offsets to canonical identity extrinsics per view.
        '''
        b = x.shape[0]

        # Identity canonical extrinsics per view
        if self.pose_rep == '6d':
            rt_canonical = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0], device=x.device).reshape(1, 1, 9).repeat(b, v, 1)
        elif self.pose_rep == 'quat':
            rt_canonical = torch.tensor([1, 0, 0, 0, 0, 0, 0], device=x.device).reshape(1, 1, 7).repeat(b, v, 1)
        else:
            raise NotImplementedError

        # Predict extrinsics offset (rotation and translation)
        extrinsics_offset = self.rel_head(x)  # [b, v, num_pose_element + 3]

        # Determine canonical index
        if self.canonical == 'first':
            cano_idx = 0
        elif self.canonical == 'middle':
            cano_idx = v // 2
        elif self.canonical == 'unordered':
            cano_idx = None  # no canonical frame; apply residuals to all
        else:
            raise ValueError(f'Unknown canonical mode: {self.canonical}')

        # Apply residuals
        if cano_idx is None:
            rt_final = rt_canonical + extrinsics_offset
        else:
            # Ordered mode: zero out residuals for canonical frame
            mask = torch.ones((1, v, 1), device=x.device)
            mask[:, cano_idx, :] = 0.0
            rt_final = rt_canonical + extrinsics_offset * mask

        # Predict fx/fy per view
        fxfy_per_view = self.canonical_k_head(x) + self.f_bias  # [b, v, 1]
        fxfy_per_view = fxfy_per_view.repeat(1, 1, 2)           # [b, v, 2]

        # Average intrinsics across views
        fxfy_avg = fxfy_per_view.mean(dim=1, keepdim=True)      # [b, 1, 2]
        fxfy_all = fxfy_avg.repeat(1, v, 1)                     # [b, v, 2]

        # Combine extrinsics and intrinsics
        info_all = torch.cat([rt_final, fxfy_all], dim=-1)  # [b, v, num_pose_element + 3 + 2]

        # Add cx, cy (fixed at 0.5)
        cxcy_all = torch.tensor([0.5, 0.5], device=info_all.device).reshape(1, 1, 2).repeat(b, v, 1)
        info_all = torch.cat([info_all, cxcy_all], dim=-1)

        return rearrange(info_all, 'b v d -> (b v) d') if return_dim2 else info_all

    def forward_canonical(self, x, v, canonical, return_dim2):
        '''
        Canonical-based modes: "first" or "middle"
        '''
        b = x.shape[0]

        if canonical == 'first':
            x_canonical = x[:, 0:1]     # [b, 1, d]
            x_rel = x[:, 1:]            # [b, v-1, d]
        elif canonical == 'middle':
            cano_idx = v // 2
            rel_indices = torch.cat([torch.arange(cano_idx), torch.arange(cano_idx + 1, v)], dim=0).to(x.device)
            x_canonical = x[:, cano_idx:cano_idx + 1]  # [b, 1, d]
            x_rel = x[:, rel_indices]                # [b, v-1, d]
        else:
            raise NotImplementedError

        # Predict canonical intrinsics
        fxfy_canonical = self.canonical_k_head(x_canonical[:, 0]) + self.f_bias  # [b, 1]
        fxfy_canonical = fxfy_canonical.unsqueeze(1).repeat(1, 1, 2)             # [b, 1, 2]

        # Identity canonical extrinsics
        if self.pose_rep == '6d':
            rt_canonical = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0], device=fxfy_canonical.device).reshape(1, 1, 9).repeat(b, 1, 1)
        elif self.pose_rep == 'quat':
            rt_canonical = torch.tensor([1, 0, 0, 0, 0, 0, 0], device=fxfy_canonical.device).reshape(1, 1, 7).repeat(b, 1, 1)

        info_canonical = torch.cat([rt_canonical, fxfy_canonical], dim=-1)  # [b, 1, num_pose_element + 3 + 2]

        # Relative extrinsics
        feat_rel = torch.cat([x_canonical.repeat(1, v - 1, 1), x_rel], dim=-1)  # [b, v-1, 2*d]
        info_rel = self.rel_head(feat_rel)                                     # [b, v-1, num_pose_element + 3]
        info_all = info_canonical.repeat(1, v, 1)                             # [b, v, num_pose_element + 3 + 2]

        if canonical == 'first':
            info_all[:, 1:, :self.num_pose_element + 3] += info_rel
        elif canonical == 'middle':
            info_all[:, rel_indices, :self.num_pose_element + 3] += info_rel

        # Add cx, cy (fixed at 0.5)
        cxcy_all = torch.tensor([0.5, 0.5], device=info_all.device).reshape(1, 1, 2).repeat(b, v, 1)
        info_all = torch.cat([info_all, cxcy_all], dim=-1)

        if self.pose_consistency_reg_weight > 0:
            info_reg = self.forward_canonical_reg(x, info_all[:, :, :self.num_pose_element + 3], canonical)
            return rearrange(info_all, 'b v d -> (b v) d') if return_dim2 else info_all, info_reg
        else:
            return rearrange(info_all, 'b v d -> (b v) d') if return_dim2 else info_all

    def forward_canonical_single_view(self, x, v, canonical, return_dim2):
        '''
        Canonical-based modes: "first" or "middle"
        '''
        b = x.shape[0]

        x_canonical = x[:, 0:1]     # [b, 1, d]

        # Predict canonical intrinsics
        fxfy_canonical = self.canonical_k_head(x_canonical[:, 0]) + self.f_bias  # [b, 1]
        fxfy_canonical = fxfy_canonical.unsqueeze(1).repeat(1, 1, 2)             # [b, 1, 2]

        # Identity canonical extrinsics
        if self.pose_rep == '6d':
            rt_canonical = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0], device=fxfy_canonical.device).reshape(1, 1, 9).repeat(b, 1, 1)
        elif self.pose_rep == 'quat':
            rt_canonical = torch.tensor([1, 0, 0, 0, 0, 0, 0], device=fxfy_canonical.device).reshape(1, 1, 7).repeat(b, 1, 1)

        info_canonical = torch.cat([rt_canonical, fxfy_canonical], dim=-1)    # [b, 1, num_pose_element + 3 + 2]                                    # [b, v-1, num_pose_element + 3]
        info_all = info_canonical.repeat(1, v, 1)                             # [b, v, num_pose_element + 3 + 2]

        # Add cx, cy (fixed at 0.5)
        cxcy_all = torch.tensor([0.5, 0.5], device=info_all.device).reshape(1, 1, 2).repeat(b, v, 1)
        info_all = torch.cat([info_all, cxcy_all], dim=-1)

        return rearrange(info_all, 'b v d -> (b v) d') if return_dim2 else info_all
