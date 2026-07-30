"""Command modules for the beast CLI."""

from beast.cli.commands import (
    combine_eval_layout_img_tokens,
    combine_eval_layout_latents,
    combine_view_latents,
    extract,
    extract_3d,
    extract_sable,
    predict,
    train,
)

# dictionary of all available commands
COMMANDS = {
    'extract': extract,              # 2D frame extraction
    'train': train,                  # model training
    'predict': predict,              # model inference on images and videos
    'extract_3d': extract_3d,        # 3D frame extraction and segmentation
    'extract_sable': extract_sable,  # SABLE IBL stereo extraction pipeline
    'combine-view-latents': combine_view_latents,  # pair per-view latents for ViT/ResNet
    'combine-eval-layout-latents': combine_eval_layout_latents,  # eval-layout ViT/ResNet trials
    'combine-eval-layout-img-tokens': combine_eval_layout_img_tokens,  # eval-layout img_tokens
}
