"""Patch TCN: facemap's `KeypointsNetwork` with optional dropout regularization.

`KeypointsNetwork` has no dropout, so `PatchTCN` adds two optional dropout layers while leaving
the architecture, initialization and forward signature of the parent untouched. With both
dropout rates at zero the model is identical to `KeypointsNetwork`.
"""

import torch
from facemap.neural_prediction.neural_model import KeypointsNetwork
from torch import nn


class PatchTCN(KeypointsNetwork):
    """`KeypointsNetwork` with dropout on the input features and before the latent layer.

    The core is `linear0 -> wavelet0 -> ReLU -> linear1 -> ReLU -> latent -> ReLU`, followed by a
    linear readout. `dropout_in` acts on the input features, and `dropout_mid` acts on the
    (ReLU-activated) `linear1` bottleneck, i.e. the input of the latent layer.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_kp: int | None = None,
        dropout_in: float = 0.0,
        dropout_mid: float = 0.0,
    ) -> None:
        """Build the network.

        Args:
            n_in: number of input features per time bin.
            n_out: number of output neurons.
            n_kp: width of the first linear layer; `None` uses `n_in`.
            dropout_in: dropout probability on the input features.
            dropout_mid: dropout probability on the `linear1` bottleneck activations.

        Raises:
            ValueError: if a dropout probability is outside [0, 1).
        """
        for name, prob in (('dropout_in', dropout_in), ('dropout_mid', dropout_mid)):
            if not 0.0 <= prob < 1.0:
                raise ValueError(f'{name} must be in [0, 1), got {prob}')
        super().__init__(n_in=n_in, n_kp=n_kp, n_out=n_out)
        self.dropout_in = dropout_in
        self.drop_in = nn.Dropout(dropout_in)
        if dropout_mid > 0.0:
            # `features[-1]` is the latent layer, which consumes the ReLU of `linear1`
            self.core.features[-1] = nn.Sequential(
                nn.Dropout(dropout_mid),
                self.core.features[-1],
            )

    def forward(
        self,
        x: torch.Tensor,
        sample_inds: torch.Tensor | None = None,
        animal_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict neural activity from features.

        Args:
            x: input features, shape (batch, time, n_in).
            sample_inds: optional indices selecting latents, as in `KeypointsNetwork`.
            animal_id: readout head index, as in `KeypointsNetwork`.

        Returns:
            Tuple `(y_pred, latents)`; `y_pred` has shape (batch, time, n_out).
        """
        if self.dropout_in > 0.0:
            x = self.drop_in(x)
        return super().forward(x, sample_inds=sample_inds, animal_id=animal_id)
