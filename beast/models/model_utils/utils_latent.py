"""Utilities for conditional latent tensor export during inference/evaluation."""

import torch


def _want_latent_export(data: dict, config: dict, *, data_key: str, config_key: str) -> bool:
    return bool(data.get(data_key, False)) or bool(config['model'].get(config_key, False))


def _infer_or_eval(config: dict, training: bool) -> bool:
    return (
        (not training)
        or bool(config.get('inference', False))
        or bool(config.get('evaluation', False))
    )


def latent_tensor_export_if_requested(
    z: torch.Tensor,
    data: dict,
    config: dict,
    training: bool,
    *,
    data_key: str,
    config_key: str | None = None,
) -> torch.Tensor | None:
    """Return detached contiguous tensor when export is enabled and in inference/evaluation."""
    ck = config_key if config_key is not None else data_key
    want = _want_latent_export(data, config, data_key=data_key, config_key=ck)
    return z.detach().contiguous() if want and _infer_or_eval(config, training) else None
