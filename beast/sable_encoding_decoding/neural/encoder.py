"""Neural encoding: RRR and CNN/TCN models that predict spike rates from latents.

Ray Tune orchestrates the `_with_tune` variants (hyperparameter search over `lr`/`wd`);
Ray Tune itself is only imported here, never executed as part of this module's tests.
"""

import copy
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from facemap.neural_prediction.neural_model import KeypointsNetwork
from ray import tune
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score as r2_score_sklearn
from tqdm import tqdm

from beast.sable_encoding_decoding.neural.rrr_encoder import train_model_main
from beast.sable_encoding_decoding.neural.utils import _std, bits_per_spike


class Embed_Dataset(torch.utils.data.Dataset):
    """Dataset pairing latent embeddings `X` with neural rates `y`."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store `X` and `y` as float32 arrays.

        Args:
            X: covariate array, first axis is the trial/sample axis.
            y: target array, same first-axis length as `X`.
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        assert len(self.X) == len(self.y), 'X and y should have the same trial length'

    def __len__(self) -> int:
        """Return the number of trials/samples."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the `(X, y)` pair at `idx`.

        Args:
            idx: sample index.

        Returns:
            Tuple `(X[idx], y[idx])`.
        """
        return self.X[idx], self.y[idx]


def train_rrr_encoder(
    config: dict[str, Any],
    data_dict: dict[str, Any],
    report_to_tune: bool = False,
) -> dict[str, Any] | None:
    """Train an RRR encoder (rates from latents) for each session in `data_dict`.

    Args:
        config: hyperparameters; must contain `lr`.
        data_dict: mapping `eid -> {'X': [train, val], 'y': [train, val], 'setup': {}}`;
            mutated in place (normalized, bias column appended, gaussian-smoothed).
        report_to_tune: if `True`, report `bps`/`r2` of the last session to Ray Tune
            instead of returning results.

    Returns:
        Mapping `eid -> result dict` with keys `gt`, `pred`, `norm_gt`, `norm_pred`,
        `mean_X`, `std_X`, `mean_y`, `std_y`, `bps`, `r2`, `eid`; or `None` when
        `report_to_tune=True`.
    """
    lr = config['lr']
    l2 = 100
    n_comp = 3
    smooth_w = 2  # smooth window 2 seconds
    ground_truth = {}
    for eid in data_dict:
        ground_truth[eid] = copy.deepcopy(data_dict[eid]['y'][1])
        # gaussian filter
        for i in range(len(data_dict[eid]['y'])):
            data_dict[eid]['y'][i] = gaussian_filter1d(data_dict[eid]['y'][i], smooth_w, axis=1)
        _, mean_X, std_X = _std(data_dict[eid]['X'][0])
        _, mean_y, std_y = _std(data_dict[eid]['y'][0])

        for i in range(2):
            K = data_dict[eid]['X'][i].shape[0]
            T = data_dict[eid]['X'][i].shape[1]
            data_dict[eid]['X'][i] = (data_dict[eid]['X'][i] - mean_X) / std_X
            if len(data_dict[eid]['X'][i].shape) == 2:
                data_dict[eid]['X'][i] = np.expand_dims(data_dict[eid]['X'][i], axis=0)
            # add bias
            data_dict[eid]['X'][i] = np.concatenate(
                [data_dict[eid]['X'][i], np.ones((K, T, 1))], axis=2,
            )
            data_dict[eid]['y'][i] = (data_dict[eid]['y'][i] - mean_y) / std_y
            print(
                f"X shape with bias: {data_dict[eid]['X'][i].shape}, "
                f"y shape: {data_dict[eid]['y'][i].shape}",
            )
        data_dict[eid]['setup']['mean_X_Tv'] = mean_X
        data_dict[eid]['setup']['std_X_Tv'] = std_X
        data_dict[eid]['setup']['mean_y_TN'] = mean_y
        data_dict[eid]['setup']['std_y_TN'] = std_y

    print('Training RRR')
    result = {}
    test_bps = []
    for eid in data_dict:
        _train_data = {eid: data_dict[eid]}
        model, mse_val = train_model_main(
            train_data=_train_data,
            l2=l2,
            n_comp=n_comp,
            model_fname='tmp',
            save=False,
            lr=lr,
        )
        print(f'Model {eid} trained')
        with torch.no_grad():
            _, _, pred_orig = model.predict_y_fr(data_dict, eid, 1)
        pred = pred_orig.cpu().numpy()
        threshold = 1e-3
        trial_len = 1.0
        pred = np.clip(pred, threshold, None)
        # Replace any NaN values in pred with the threshold
        if np.any(np.isnan(pred)):
            print(f'Contain NaN value, replace to {threshold}')
            pred = np.nan_to_num(pred, nan=threshold)
        num_trial, num_time, num_neuron = pred.shape
        gt_held_out = ground_truth[eid]
        mean_fr = gt_held_out.sum(1).mean(0) / trial_len
        keep_idxs = np.arange(len(mean_fr)).flatten()

        bps_result_list = []
        for n_i in tqdm(keep_idxs, desc='co-bps'):
            bps = bits_per_spike(
                pred[:, :, [n_i]],
                gt_held_out[:, :, [n_i]],
                threshold=threshold,
            )
            if np.isinf(bps):
                bps = np.nan
            bps_result_list.append(bps)
        co_bps = np.nanmean(bps_result_list)
        # calculate variance explained
        with torch.no_grad():
            _, y_norm, y_pred_norm = model.predict_y(data_dict, eid, 1)
        y_pred_norm = y_pred_norm.cpu().numpy()
        y_norm = y_norm.cpu().numpy()
        y_norm = y_norm.reshape(-1, num_neuron)
        y_pred_norm = y_pred_norm.reshape(-1, num_neuron)
        # calculate variance unexplained, r2
        try:
            r2 = r2_score_sklearn(y_norm, y_pred_norm)
        except Exception as e:
            print(e)
            r2 = -100000

        print(f'Co-BPS: {co_bps}')
        print(f'r2: {r2}')
        test_bps.append(co_bps)
        y_norm = y_norm.reshape(num_trial, num_time, num_neuron)
        y_pred_norm = y_pred_norm.reshape(num_trial, num_time, num_neuron)
        result[eid] = {
            'gt': gt_held_out,
            'pred': pred,
            'norm_gt': y_norm,
            'norm_pred': y_pred_norm,
            'mean_X': data_dict[eid]['setup']['mean_X_Tv'],
            'std_X': data_dict[eid]['setup']['std_X_Tv'],
            'mean_y': data_dict[eid]['setup']['mean_y_TN'],
            'std_y': data_dict[eid]['setup']['std_y_TN'],
            'bps': co_bps,
            'r2': r2,
            'eid': eid,
        }
    if report_to_tune:
        tune.report({'bps': co_bps, 'r2': r2})  # only report the last result eid
        return None
    return result


def train_rrr_encoder_with_tune(
    data_dict: dict[str, Any],
    num_samples: int = 10,
    tune_storage_path: str | None = None,
) -> dict[str, Any] | None:
    """Ray-Tune hyperparameter search over `lr` for `train_rrr_encoder`.

    Args:
        data_dict: mapping `eid -> {'X': [train, val, test], 'y': [...]}`.
        num_samples: number of Ray Tune trials.
        tune_storage_path: Ray Tune experiment root directory; `None` uses Ray's default.

    Returns:
        Final `train_rrr_encoder` result evaluated with the best config on the held-out
        (last) split.
    """
    train_val_dict = {}
    for eid in data_dict:
        train_val_dict[eid] = copy.deepcopy(data_dict[eid])
        # remove the last element of X and y since it is the test set
        train_val_dict[eid]['X'].pop()
        train_val_dict[eid]['y'].pop()

    search_space = {'lr': tune.loguniform(5e-2, 2)}
    analysis = tune.run(
        tune.with_parameters(train_rrr_encoder, data_dict=train_val_dict, report_to_tune=True),
        resources_per_trial={'cpu': 2, 'gpu': 1},
        config=search_space,
        num_samples=num_samples,
        log_to_file=False,
        **({'storage_path': tune_storage_path} if tune_storage_path else {}),
    )
    best_config = analysis.get_best_config(metric='bps', mode='max')
    print('Best config: ', best_config)
    # test data_dict, remove the 2nd last element of X and y since it is the validation set
    train_test_dict = {}
    for eid in data_dict:
        train_test_dict[eid] = copy.deepcopy(data_dict[eid])
        train_test_dict[eid]['X'].pop(-2)
        train_test_dict[eid]['y'].pop(-2)
    return train_rrr_encoder(config=best_config, data_dict=train_test_dict, report_to_tune=False)


def train_cnn_encoder(
    config: dict[str, Any],
    data_dict: dict[str, Any],
    report_to_tune: bool = False,
    verbose: bool = True,
) -> dict[str, Any] | None:
    """Train a `KeypointsNetwork` (facemap) CNN encoder for each session in `data_dict`.

    Args:
        config: hyperparameters; must contain `lr` and `wd`.
        data_dict: mapping `eid -> {'X': [train, test], 'y': [train, test]}`.
        report_to_tune: if `True`, report `bps`/`r2` of the last session to Ray Tune
            instead of returning results.
        verbose: whether to print periodic eval progress and LR-annealing messages.

    Returns:
        Mapping `eid -> result dict` with keys `gt`, `pred`, `norm_gt`, `norm_pred`,
        `mean_X`, `std_X`, `mean_y`, `std_y`, `bps`, `r2`, `eid`; or `None` when
        `report_to_tune=True`.
    """
    lr = config['lr']
    wd = config['wd']
    smoothing_penalty = 0.5
    epochs = 100
    annealing_steps = 2
    trial_len = 1
    anneal_epochs = epochs - 50 * np.arange(1, annealing_steps + 1)
    threshold = 1e-3
    accelerator = Accelerator()
    result = {}
    for eid in data_dict:
        train_X = data_dict[eid]['X'][0]
        train_y = data_dict[eid]['y'][0]
        test_X = data_dict[eid]['X'][1]
        test_y = data_dict[eid]['y'][1]
        # copy gt test spike
        test_y_gt = copy.deepcopy(test_y)
        # gaussian filter
        train_y = gaussian_filter1d(train_y, trial_len, axis=1)
        test_y = gaussian_filter1d(test_y, trial_len, axis=1)
        # norm
        _, mean_X, std_X = _std(train_X)
        _, mean_y, std_y = _std(train_y)
        train_X = (train_X - mean_X) / std_X
        test_X = (test_X - mean_X) / std_X
        train_y = (train_y - mean_y) / std_y
        test_y = (test_y - mean_y) / std_y
        embed_size = train_X.shape[-1]
        num_neuron = train_y.shape[-1]
        train_dataset = Embed_Dataset(train_X, train_y)
        test_dataset = Embed_Dataset(test_X, test_y)
        n_test = len(test_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        model = KeypointsNetwork(n_in=embed_size, n_out=num_neuron)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        model, optimizer, train_loader, test_loader = accelerator.prepare(
            model, optimizer, train_loader, test_loader,
        )
        for epoch in range(epochs):
            model.train()
            if epoch in anneal_epochs:
                print('annealing learning rate') if verbose else None
                optimizer.param_groups[0]['lr'] /= 10.0
            for batch in train_loader:
                X, y = batch
                y_pred = model(x=X)[0]
                loss = ((y_pred - y) ** 2).mean()
                loss += (
                    smoothing_penalty * (torch.diff(model.core.features[1].weight) ** 2).sum()
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 20 == 0 and verbose:
                model.eval()
                test_y_pred = []
                test_y = []
                with torch.no_grad():
                    for batch in test_loader:
                        X, y = batch
                        y_pred = model(x=X)[0]
                        test_y_pred.append(y_pred)
                        test_y.append(y)
                test_y_pred = torch.cat(test_y_pred, axis=0)
                test_y = torch.cat(test_y, axis=0)
                test_y_pred = test_y_pred.reshape(-1, num_neuron)
                test_y = test_y.reshape(-1, num_neuron)

        model.eval()
        test_y_pred = []
        test_y = []
        with torch.no_grad():
            for batch in test_loader:
                X, y = batch
                y_pred = model(x=X)[0]
                test_y_pred.append(y_pred)
                test_y.append(y)
        test_y_pred = torch.cat(test_y_pred, axis=0).cpu().numpy()
        test_y = torch.cat(test_y, axis=0).cpu().numpy()
        # reshape to (N * T, Neuron)
        test_y_pred = test_y_pred.reshape(-1, num_neuron)
        test_y = test_y.reshape(-1, num_neuron)
        # calculate variance unexplained, r2
        r2 = r2_score_sklearn(test_y, test_y_pred)
        # reshape to (N, T, Neuron)
        test_y_pred = test_y_pred.reshape(n_test, -1, num_neuron)
        test_y = test_y.reshape(n_test, -1, num_neuron)
        norm_test_y, norm_test_y_pred = copy.deepcopy(test_y), copy.deepcopy(test_y_pred)
        # denormalize
        test_y_pred = test_y_pred * std_y + mean_y
        test_y_pred = np.clip(test_y_pred, threshold, None)
        # Replace any NaN values in pred with the threshold
        if np.any(np.isnan(test_y_pred)):
            print(f'Contain NaN value, replace to {threshold}')
            test_y_pred = np.nan_to_num(test_y_pred, nan=threshold)
        # calculate co-bps
        bps_result_list = []
        for i in range(num_neuron):
            bps = bits_per_spike(
                test_y_pred[:, :, [i]],
                test_y_gt[:, :, [i]],  # gt spike, without gaussian filter and normalization
                threshold=threshold,
            )
            if np.isinf(bps):
                bps = np.nan
            bps_result_list.append(bps)
        co_bps = np.nanmean(bps_result_list)
        print(f'Co-BPS: {co_bps}, R2: {r2}')
        result[eid] = {
            'gt': test_y_gt,
            'pred': test_y_pred,
            'norm_gt': norm_test_y,
            'norm_pred': norm_test_y_pred,
            'mean_X': mean_X,
            'std_X': std_X,
            'mean_y': mean_y,
            'std_y': std_y,
            'bps': co_bps,
            'r2': r2,
            'eid': eid,
        }
    if report_to_tune:
        tune.report({'bps': co_bps, 'r2': r2})  # only report the last result eid
        return None
    return result


def train_cnn_encoder_with_tune(
    data_dict: dict[str, Any],
    num_samples: int = 10,
    tune_storage_path: str | None = None,
) -> dict[str, Any] | None:
    """Ray-Tune hyperparameter search over `lr`/`wd` for `train_cnn_encoder`.

    Args:
        data_dict: mapping `eid -> {'X': [train, val, test], 'y': [...]}`.
        num_samples: number of Ray Tune trials.
        tune_storage_path: Ray Tune experiment root directory; `None` uses Ray's default.

    Returns:
        Final `train_cnn_encoder` result evaluated with the best config on the held-out
        (last) split.
    """
    train_val_dict = {}
    for eid in data_dict:
        train_val_dict[eid] = copy.deepcopy(data_dict[eid])
        # remove the last element of X and y since it is the test set
        train_val_dict[eid]['X'].pop()
        train_val_dict[eid]['y'].pop()
    search_space = {'lr': tune.loguniform(1e-4, 3e-3), 'wd': 1e-4}
    analysis = tune.run(
        tune.with_parameters(
            train_cnn_encoder, data_dict=train_val_dict, report_to_tune=True, verbose=True,
        ),
        resources_per_trial={'cpu': 2, 'gpu': 1},
        config=search_space,
        num_samples=num_samples,
        log_to_file=False,
        **({'storage_path': tune_storage_path} if tune_storage_path else {}),
    )
    best_config = analysis.get_best_config(metric='bps', mode='max')
    print('Best config: ', best_config)
    # test data_dict, remove the 2nd last element of X and y since it is the validation set
    train_test_dict = {}
    for eid in data_dict:
        train_test_dict[eid] = copy.deepcopy(data_dict[eid])
        train_test_dict[eid]['X'].pop(-2)
        train_test_dict[eid]['y'].pop(-2)
    return train_cnn_encoder(
        config=best_config, data_dict=train_test_dict, report_to_tune=False, verbose=True,
    )
