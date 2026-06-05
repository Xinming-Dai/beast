"""DINOv3 feature extractor for patch and CLS token extraction."""

import os
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel


def _resolve_model_path(model_name: str) -> str:
    """Resolve model identifier to a local path or HuggingFace identifier.

    Priority:
    1. If model_name is an existing local directory, use it directly.
    2. If ``HF_HOME`` is set and contains files directly (not a snapshot dir),
       use that directory directly — handles the case where weights were
       downloaded to HF_HOME/ without the snapshot subdirectory.
    3. Otherwise treat model_name as a HuggingFace model ID.
    """
    p = Path(model_name)
    if p.is_dir():
        return str(p.resolve())

    # Check HF_HOME: support both snapshot layout and flat layout
    hf_home = Path(os.environ.get('HF_HOME', ''))
    if hf_home and hf_home.is_dir():
        # Snapshot layout: HF_HOME/models--{model_name}/...
        snapshot = hf_home / f'models--{model_name.replace("/", "--")}'
        if snapshot.is_dir():
            return str(snapshot.resolve())
        # Flat layout: HF_HOME/ directly contains the model files (user downloaded here)
        if (hf_home / 'config.json').exists():
            return str(hf_home.resolve())

    return model_name


class DinoV3(nn.Module):
    """DINOv3 feature extractor returning patch and CLS tokens."""

    def __init__(
        self,
        model_name: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        freeze: bool = True,
    ):
        """Initialize DINOv3.

        Args:
            model_name: HuggingFace model identifier or path to a local directory
                containing model files (preferred when offline / gated repo).
            freeze: whether to freeze all parameters (default True).
        """
        super().__init__()

        resolved = _resolve_model_path(model_name)
        self.model = AutoModel.from_pretrained(resolved, local_files_only=Path(resolved).is_dir())

        self.embed_dim = self.model.config.hidden_size

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, images):
        """Extract patch and CLS tokens from multi-view images.

        Args:
            images: float tensor of shape (B, V, 3, H, W) in [0, 1].

        Returns:
            tuple of (patch_tokens [B, V, N, embed_dim], cls_tokens [B, V, embed_dim]).
        """
        B, V = images.shape[:2]

        x = images.view(B * V, *images.shape[2:])

        outputs = self.model(pixel_values=x)

        hidden = outputs.last_hidden_state

        cls_tokens = hidden[:, 0]
        patch_tokens = hidden[:, 5:]

        N = patch_tokens.shape[1]

        patch_tokens = patch_tokens.view(B, V, N, self.embed_dim)
        cls_tokens = cls_tokens.view(B, V, self.embed_dim)

        return patch_tokens, cls_tokens
