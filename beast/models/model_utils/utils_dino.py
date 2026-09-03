"""DINOv3 feature extractor for patch and CLS token extraction."""

import torch.nn as nn
from transformers import AutoModel


class DinoV3(nn.Module):
    """DINOv3 feature extractor returning patch and CLS tokens."""

    def __init__(
        self,
        model_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
        num_trainable_blocks=2,
    ):
        """Initialize DINOv3.

        Args:
            model_name: HuggingFace model identifier.
            num_trainable_blocks: number of final transformer blocks (plus the
                final norm) to leave trainable; the rest of the backbone is frozen.
        """
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)

        self.embed_dim = self.model.config.hidden_size

        for p in self.model.parameters():
            p.requires_grad = False

        if num_trainable_blocks > 0:
            for layer in self.model.model.layer[-num_trainable_blocks:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.model.norm.parameters():
                p.requires_grad = True

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
