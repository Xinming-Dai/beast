"""Shared tensor/parameter helpers used by both the RRR encoder and decoder models.

Private module (not part of the public API) — `rrr_encoder.py` and `rrr_decoder.py` both
duplicated these four helpers in the source repo; factored out here to avoid duplication.
"""

import numpy as np
import torch
import torch.nn as nn


def np2tensor(v: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a torch tensor (no copy).

    Args:
        v: input numpy array.

    Returns:
        Torch tensor sharing memory with `v`.
    """
    return torch.from_numpy(v)


def np2param(v: np.ndarray, grad: bool = True) -> nn.Parameter:
    """Wrap a numpy array as an `nn.Parameter`.

    Args:
        v: input numpy array.
        grad: whether the resulting parameter requires gradients.

    Returns:
        `nn.Parameter` wrapping `v`.
    """
    return nn.Parameter(np2tensor(v), requires_grad=grad)


def tensor2np(v: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor (on CPU) to a numpy array.

    Args:
        v: input torch tensor.

    Returns:
        Numpy array sharing memory with `v`.
    """
    return v.numpy()


def get_device() -> torch.device:
    """Select the compute device, preferring CUDA when available.

    Returns:
        `torch.device('cuda')` if a GPU is available, else `torch.device('cpu')`.
    """
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
        print('GPU is available')
    else:
        device = torch.device('cpu')
        print('GPU not available, CPU used')
    return device
