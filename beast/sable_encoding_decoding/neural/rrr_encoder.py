"""Reduced-rank regression (with gradient descent) model for neural encoding.

Predicts spike rates `y` from behavioral/latent covariates `X` via a low-rank
per-session regression: `beta = U @ V + b`, shared temporal factors `V` across sessions,
session-specific spatial factors `U` and biases `b`.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from beast.sable_encoding_decoding.neural._rrr_common import (
    get_device,
    np2param,
    np2tensor,
    tensor2np,
)


class RRRGD:
    """Reduced-rank regression model trained with gradient descent (encoder variant).

    Predicts neural rates `y: (K, T, N)` from covariates `X: (K, T, ncoef)` (last
    coefficient column is the bias term) via a shared low-rank temporal basis.

    Parameter layout (per session `eid`, stored in `self.model`, an `nn.ParameterDict`):
        `{eid}_U`: `(N, ncoef - 1, ncomp)` — session-specific spatial/neuron loadings.
        `V`: `(ncomp, T)` — temporal factors, shared across all sessions.
        `{eid}_b`: `(N, 1, T)` — session-specific per-neuron, per-timestep bias.

    The regression coefficient tensor `beta = U @ V` (concatenated with `b` along the
    coefficient axis when `withbias=True`) has shape `(N, ncoef, T)`.
    """

    def __init__(self, train_data: dict[str, Any], ncomp: int, l2: float = 0.0) -> None:
        """Initialize per-session parameters from the training data shapes.

        Args:
            train_data: mapping `eid -> {'X': [train_arr, ...], 'y': [train_arr, ...]}`
                where `train_data[eid]['X'][0]` has shape `(K, T, ncoef)` and
                `train_data[eid]['y'][0]` has shape `(K, T, N)`.
            ncomp: number of shared low-rank temporal components.
            l2: L2 regularization weight applied to `beta` in `regression_loss`.
        """
        self.l2 = l2
        self.eids = list(train_data.keys())
        self.withbias = True

        np.random.seed(0)
        self.N = 0
        self.model = {}
        for eid in train_data:
            _X = train_data[eid]['X'][0]  # (K, T, ncoef), the last coef is the bias term
            _y = train_data[eid]['y'][0]  # (K, T, N)
            K, T, ncoef = _X.shape
            K, T, N = _y.shape
            U = np.random.normal(size=(N, ncoef - 1, ncomp)) / np.sqrt(T * ncomp)
            V = np.random.normal(size=(ncomp, T)) / np.sqrt(T * ncomp)
            b = np.expand_dims(_y.mean(0).T, 1)
            b = np.ascontiguousarray(b)
            self.model[f'{eid}_U'] = np2param(U)
            self.model[f'{eid}_b'] = np2param(b)
            self.N += N
        self.model['V'] = np2param(V)  # V shared across sessions
        self.n_comp, self.T = self.model['V'].shape
        self.model = nn.ParameterDict(self.model)
        # U: model[eid+"_U"], (N, ncoef, ncomp)
        # V: model['V'], (ncomp, T)
        # b: model[eid+"_b"], (N, 1, T)

    def train(self) -> None:
        """Put the underlying parameter module into training mode."""
        self.model.train()

    def eval(self) -> None:
        """Put the underlying parameter module into eval mode."""
        self.model.eval()

    def to(self, device: torch.device | str) -> None:
        """Move all parameters to `device`.

        Args:
            device: target torch device.
        """
        self.model.to(device)

    def state_dict(self) -> dict[str, Any]:
        """Build a checkpoint dict with CPU-resident parameters and metadata.

        Returns:
            Checkpoint dict with keys `model`, `l2`, `eids`, `N`, `T`, `n_comp`.
        """
        checkpoint = {
            'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'l2': self.l2,
            'eids': self.eids,
            'N': self.N,
            'T': self.T,
            'n_comp': self.n_comp,
        }
        return checkpoint

    def load_state_dict(self, f: dict[str, Any]) -> None:
        """Load parameters from a raw `nn.ParameterDict`-compatible state dict.

        Args:
            f: state dict as produced by `self.model.state_dict()`.
        """
        self.model.load_state_dict(f)

    def compute_beta_m(
        self,
        U: torch.Tensor | np.ndarray,
        V: torch.Tensor | np.ndarray,
        b: torch.Tensor | np.ndarray,
        withbias: bool = True,
        tonp: bool = False,
    ) -> torch.Tensor | np.ndarray:
        """Compute the regression coefficient tensor `beta = U @ V` (+ bias column).

        Args:
            U: spatial factors, `(N, ncoef - 1, ncomp)`. Tensor unless `tonp=True`.
            V: temporal factors, `(ncomp, T)`. Tensor unless `tonp=True`.
            b: bias term, `(N, 1, T)`. Tensor unless `tonp=True`.
            withbias: whether to append `b` as the last coefficient row.
            tonp: if `True`, treat inputs as numpy arrays and return a numpy array.

        Returns:
            `beta` of shape `(N, ncoef, T)`, as a tensor or numpy array per `tonp`.
        """
        if tonp:
            U = np2tensor(U)
            V = np2tensor(V)
        beta = U @ V
        if withbias:
            if tonp:
                b = np2tensor(b)
            beta = torch.cat((beta, b), 1)  # (N, ncoef, T)
        else:
            # place-holder zero bias when withbias is False
            b = torch.zeros((U.shape[0], 1, V.shape[1])).to(beta.device)
            beta = torch.cat((beta, b), 1)  # (N, ncoef, T)
        if tonp:
            beta = tensor2np(beta)
        return beta

    def compute_beta(self, eid: str, withbias: bool = True) -> torch.Tensor:
        """Compute `beta` for a given session.

        Args:
            eid: session id.
            withbias: whether to append the bias column.

        Returns:
            `beta` of shape `(N, ncoef, T)` for session `eid`.
        """
        return self.compute_beta_m(
            self.model[f'{eid}_U'], self.model['V'], self.model[f'{eid}_b'], withbias=withbias,
        )

    def predict(
        self,
        beta: torch.Tensor | np.ndarray,
        X: torch.Tensor | np.ndarray,
        tonp: bool = False,
    ) -> torch.Tensor | np.ndarray:
        """Predict rates from covariates and regression coefficients.

        Args:
            beta: regression coefficients, `(N, ncoef, T)`.
            X: covariates, `(K, T, ncoef)`.
            tonp: if `True`, treat inputs as numpy arrays and return a numpy array.

        Returns:
            Predicted rates `y_pred`, `(K, T, N)`.
        """
        if tonp:
            X = np2tensor(X)
            beta = np2tensor(beta)
        y_pred = torch.einsum('ktc,nct->ktn', X, beta)
        if tonp:
            y_pred = tensor2np(y_pred)
        return y_pred

    def predict_y(
        self, data: dict[str, Any], eid: str, k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict rates for split `k` of session `eid` (normalized-space, no bias-only).

        Args:
            data: mapping `eid -> {'X': [...], 'y': [...]}`; numpy arrays.
            eid: session id.
            k: split index into `data[eid]['X']` / `data[eid]['y']`.

        Returns:
            Tuple `(X, y, ypred)` as tensors on `beta`'s device.
        """
        beta = self.compute_beta(eid, withbias=self.withbias)
        X = np2tensor(data[eid]['X'][k]).to(beta.device)
        y = np2tensor(data[eid]['y'][k]).to(beta.device)
        ypred = self.predict(beta, X)
        return X, y, ypred

    def predict_y_fr(
        self, data: dict[str, Any], eid: str, k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict rates for split `k`, de-normalized back to firing-rate units.

        Args:
            data: mapping `eid -> {'X': [...], 'y': [...], 'setup': {'mean_y_TN', 'std_y_TN'}}`.
            eid: session id.
            k: split index.

        Returns:
            Tuple `(X, y, ypred)` with `y` and `ypred` de-normalized.
        """
        X, y, ypred = self.predict_y(data, eid, k)
        mean_y = np2tensor(data[eid]['setup']['mean_y_TN']).to(y.device)
        std_y = np2tensor(data[eid]['setup']['std_y_TN']).to(y.device)
        y = y * std_y + mean_y
        ypred = ypred * std_y + mean_y
        return X, y, ypred

    def compute_MSE_RRRGD(self, data: dict[str, Any], k: int) -> dict[str, torch.Tensor]:
        """Compute per-session, per-neuron summed squared error for split `k`.

        Args:
            data: mapping `eid -> {'X': [...], 'y': [...]}`.
            k: split index.

        Returns:
            Mapping `eid -> mse` where `mse` has shape `(N,)`.
        """
        mses_all = {}
        for eid in data:
            _, y, ypred = self.predict_y(data, eid, k)
            mses_all[eid] = torch.sum((ypred - y) ** 2, axis=(0, 1))
        return mses_all

    def regression_loss(self) -> dict[str, torch.Tensor]:
        """Compute the per-session L2 regularization loss on `beta`.

        Returns:
            Mapping `eid -> scalar L2 loss`.
        """
        return {
            eid: self.l2 * torch.sum(self.compute_beta(eid, withbias=self.withbias) ** 2)
            for eid in self.eids
        }


def train_model(
    model: RRRGD,
    train_data: dict[str, Any],
    optimizer: optim.Optimizer,
    model_fname: str,
    save: bool = True,
) -> tuple[RRRGD, dict[str, Any]]:
    """Fit `model` on `train_data` (split 0) and evaluate on split 1 (validation).

    Args:
        model: `RRRGD` instance to train in place.
        train_data: mapping `eid -> {'X': [train_arr, val_arr, ...], 'y': [...]}`.
        optimizer: LBFGS (or similar closure-based) optimizer over `model.model.parameters()`.
        model_fname: path to save the checkpoint to, when `save=True`.
        save: whether to save a checkpoint after training.

    Returns:
        Tuple `(model, {'mses_val': ..., 'mse_val_mean': ...})`.
    """

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        model.train()
        total_loss = 0.0
        train_mses_all = model.compute_MSE_RRRGD(train_data, 0)
        reg_losses_all = model.regression_loss()
        for eid in train_mses_all:
            total_loss += train_mses_all[eid].sum()
            total_loss += reg_losses_all[eid]
        total_loss.backward()
        return total_loss

    optimizer.step(closure)

    model.eval()
    mses_val = model.compute_MSE_RRRGD(train_data, 1)
    best_loss = torch.sum(torch.cat([mses_val[k] for k in mses_val]))

    if save:
        print('saving model')
        checkpoint = {'RRRGD_model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, model_fname)

    return model, {'mses_val': mses_val, 'mse_val_mean': best_loss}


def train_model_main(
    train_data: dict[str, Any],
    l2: float,
    n_comp: int,
    model_fname: str,
    save: bool = True,
    lr: float = 0.01,
) -> tuple[RRRGD, dict[str, Any]]:
    """Construct an `RRRGD` model, train it with LBFGS, and return it with val MSE.

    Args:
        train_data: mapping `eid -> {'X': [...], 'y': [...]}`.
        l2: L2 regularization weight.
        n_comp: number of shared low-rank temporal components.
        model_fname: checkpoint path used when `save=True`.
        save: whether to save a checkpoint after training.
        lr: LBFGS learning rate.

    Returns:
        Tuple `(area_model, mse_val)`, matching `train_model`'s return.
    """
    area_model = RRRGD(train_data, n_comp, l2=l2)

    device = get_device()
    area_model.to(device)
    print(f'training on device: {device}')

    optimizer = optim.LBFGS(area_model.model.parameters(), lr=lr)
    _, mse_val = train_model(area_model, train_data, optimizer, model_fname=model_fname, save=save)
    return area_model, mse_val
