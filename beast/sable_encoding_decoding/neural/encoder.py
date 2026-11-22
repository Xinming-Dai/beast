"""Neural encoding: RRR and CNN/TCN models that predict spike rates from latents.

Ray Tune orchestrates the `_with_tune` variants (hyperparameter search over `lr`/`wd` and, for
the `reg` search preset, dropout); Ray Tune itself is only imported here, never executed as part
of this module's tests.
"""

import copy
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score as r2_score_sklearn
from tqdm import tqdm

from beast.sable_encoding_decoding.neural.rrr_encoder import train_model_main
from beast.sable_encoding_decoding.neural.tcn_model import PatchTCN
from beast.sable_encoding_decoding.neural.utils import _std, bits_per_spike, set_seed

# width of the first TCN linear layer in the `reg` search preset; equals the input size of the
# 6-PC patch features, so those results stay comparable with the untuned model
TUNED_N_KP = 4800
_NUM_EPOCHS = 100
_ANNEAL_SCHEDULES = ('legacy', 'fixed')
_TCN_CONFIG_KEYS = ('lr', 'wd', 'dropout_in', 'dropout_mid', 'n_kp', 'anneal', 'lr_decay_epochs')


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


def _anneal_epochs(epochs: int, annealing_steps: int, schedule: str) -> np.ndarray:
    """Return the epochs at which the learning rate is divided by 10.

    Args:
        epochs: total number of training epochs.
        annealing_steps: number of annealing steps of the `legacy` schedule.
        schedule: `legacy` or `fixed`. `legacy` is facemap's original expression, which for
            100 epochs and 2 steps gives `[50, 0]`, so the lr is also divided at epoch 0 (the
            effective lr is `lr / 10` for the first half and `lr / 100` afterwards). `fixed`
            divides at the half and three-quarter points of training.

    Returns:
        Array of epoch indices.

    Raises:
        ValueError: if `schedule` is not one of `_ANNEAL_SCHEDULES`.
    """
    if schedule == 'legacy':
        return epochs - 50 * np.arange(1, annealing_steps + 1)
    if schedule == 'fixed':
        return np.array([epochs // 2, 3 * epochs // 4])
    raise ValueError(f'anneal must be one of {_ANNEAL_SCHEDULES}, got {schedule!r}')


def validate_tcn_config(config: dict[str, Any]) -> None:
    """Check a TCN hyperparameter config, so typos fail loudly instead of being ignored.

    Args:
        config: hyperparameters of `train_cnn_encoder`: `lr` and `wd` (required) and optionally
            `dropout_in`, `dropout_mid`, `n_kp` and one of `anneal` or `lr_decay_epochs`.

    Raises:
        ValueError: if a key is unknown or missing, a dropout rate is outside [0, 1), `anneal`
            is not one of `_ANNEAL_SCHEDULES`, `lr_decay_epochs` is not a list of epochs in
            [0, `_NUM_EPOCHS`), or both `anneal` and `lr_decay_epochs` are given.
    """
    unknown = sorted(set(config) - set(_TCN_CONFIG_KEYS))
    if unknown:
        raise ValueError(f'unknown TCN config keys {unknown}, allowed: {list(_TCN_CONFIG_KEYS)}')
    missing = [key for key in ('lr', 'wd') if key not in config]
    if missing:
        raise ValueError(f'TCN config is missing the required keys {missing}')
    for key in ('dropout_in', 'dropout_mid'):
        if not 0.0 <= config.get(key, 0.0) < 1.0:
            raise ValueError(f'{key} must be in [0, 1), got {config[key]}')
    if 'anneal' in config and 'lr_decay_epochs' in config:
        raise ValueError("give either 'anneal' or 'lr_decay_epochs', not both")
    if config.get('anneal', 'legacy') not in _ANNEAL_SCHEDULES:
        raise ValueError(f"anneal must be one of {_ANNEAL_SCHEDULES}, got {config['anneal']!r}")
    decay_epochs = config.get('lr_decay_epochs', [])
    if not isinstance(decay_epochs, list | tuple) or not all(
        isinstance(epoch, int) and 0 <= epoch < _NUM_EPOCHS for epoch in decay_epochs
    ):
        raise ValueError(
            f'lr_decay_epochs must be a list of epochs in [0, {_NUM_EPOCHS}), got {decay_epochs}',
        )


def _evaluate_cnn(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    y_gt: np.ndarray,
    mean_y: np.ndarray,
    std_y: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate a trained CNN encoder with R2 and co-bps.

    Args:
        model: network mapping `(batch, time, features)` to `(batch, time, neurons)`.
        loader: non-shuffled loader of normalized `(X, y)` pairs.
        y_gt: raw spikes, shape (trials, time, neurons), in the loader's order.
        mean_y: mean used to normalize the targets.
        std_y: std used to normalize the targets.
        threshold: lower clip of the denormalized prediction, for bits-per-spike.

    Returns:
        Mapping with `pred` (denormalized, clipped), `norm_gt`, `norm_pred`, `bps` (mean over
        neurons) and `r2`.
    """
    n_trial, _, num_neuron = y_gt.shape
    model.eval()
    y_pred_list = []
    y_list = []
    with torch.no_grad():
        for X, y in loader:
            y_pred_list.append(model(x=X)[0])
            y_list.append(y)
    y_pred = torch.cat(y_pred_list, axis=0).cpu().numpy()
    y_norm = torch.cat(y_list, axis=0).cpu().numpy()
    # reshape to (N * T, Neuron)
    y_pred = y_pred.reshape(-1, num_neuron)
    y_norm = y_norm.reshape(-1, num_neuron)
    # calculate variance unexplained, r2
    r2 = r2_score_sklearn(y_norm, y_pred)
    # reshape to (N, T, Neuron)
    norm_pred = y_pred.reshape(n_trial, -1, num_neuron)
    norm_gt = y_norm.reshape(n_trial, -1, num_neuron)
    # denormalize
    pred = norm_pred * std_y + mean_y
    pred = np.clip(pred, threshold, None)
    # Replace any NaN values in pred with the threshold
    if np.any(np.isnan(pred)):
        print(f'Contain NaN value, replace to {threshold}')
        pred = np.nan_to_num(pred, nan=threshold)
    # calculate co-bps
    bps_result_list = []
    for i in range(num_neuron):
        bps = bits_per_spike(
            pred[:, :, [i]],
            y_gt[:, :, [i]],  # gt spike, without gaussian filter and normalization
            threshold=threshold,
        )
        if np.isinf(bps):
            bps = np.nan
        bps_result_list.append(bps)
    return {
        'pred': pred,
        'norm_gt': norm_gt,
        'norm_pred': norm_pred,
        'bps': np.nanmean(bps_result_list),
        'r2': r2,
    }


def train_cnn_encoder(
    config: dict[str, Any],
    data_dict: dict[str, Any],
    report_to_tune: bool = False,
    verbose: bool = True,
    seed: int = 42,
    eval_every: int = 0,
) -> dict[str, Any] | None:
    """Train a `PatchTCN` (facemap `KeypointsNetwork`) encoder for each session in `data_dict`.

    Args:
        config: hyperparameters, checked by `validate_tcn_config`; must contain `lr` and `wd`.
            Optional keys, with defaults that reproduce the untuned model: `dropout_in` (0),
            `dropout_mid` (0), `n_kp` (the width of the first linear layer; the number of
            input features) and the lr schedule, either `anneal` (`legacy`, see `_anneal_epochs`)
            or `lr_decay_epochs` (the epochs at which the lr is divided by 10).
        data_dict: mapping `eid -> {'X': [train, test], 'y': [train, test]}`.
        report_to_tune: if `True`, report `bps`/`r2`/`epoch` of the last session to Ray Tune
            instead of returning results.
        verbose: whether to print LR-annealing messages.
        seed: RNG seed applied at the start of this call; needed because Ray Tune runs
            each trial in its own worker process, which does not inherit the driver's
            seeded RNG state, so model init and DataLoader shuffling would otherwise be
            unseeded.
        eval_every: if positive and `report_to_tune`, also report the evaluation split's
            metrics every `eval_every` epochs, which lets a Ray Tune scheduler stop trials early.

    Returns:
        Mapping `eid -> result dict` with keys `gt`, `pred`, `norm_gt`, `norm_pred`,
        `mean_X`, `std_X`, `mean_y`, `std_y`, `bps`, `r2`, `eid`; or `None` when
        `report_to_tune=True`.
    """
    validate_tcn_config(config)
    set_seed(seed)
    lr = config['lr']
    wd = config['wd']
    dropout_in = config.get('dropout_in', 0.0)
    dropout_mid = config.get('dropout_mid', 0.0)
    n_kp = config.get('n_kp')
    smoothing_penalty = 0.5
    epochs = _NUM_EPOCHS
    annealing_steps = 2
    trial_len = 1
    if 'lr_decay_epochs' in config:
        anneal_epochs = np.array(config['lr_decay_epochs'])
    else:
        anneal_epochs = _anneal_epochs(epochs, annealing_steps, config.get('anneal', 'legacy'))
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
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        model = PatchTCN(
            n_in=embed_size,
            n_out=num_neuron,
            n_kp=n_kp,
            dropout_in=dropout_in,
            dropout_mid=dropout_mid,
        )
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
            is_eval_epoch = eval_every > 0 and (epoch + 1) % eval_every == 0
            # the last epoch is reported after the loop
            if report_to_tune and is_eval_epoch and epoch + 1 < epochs:
                metrics = _evaluate_cnn(model, test_loader, test_y_gt, mean_y, std_y, threshold)
                tune.report({'bps': metrics['bps'], 'r2': metrics['r2'], 'epoch': epoch + 1})

        metrics = _evaluate_cnn(model, test_loader, test_y_gt, mean_y, std_y, threshold)
        co_bps = metrics['bps']
        r2 = metrics['r2']
        print(f'Co-BPS: {co_bps}, R2: {r2}')
        result[eid] = {
            'gt': test_y_gt,
            'pred': metrics['pred'],
            'norm_gt': metrics['norm_gt'],
            'norm_pred': metrics['norm_pred'],
            'mean_X': mean_X,
            'std_X': std_X,
            'mean_y': mean_y,
            'std_y': std_y,
            'bps': co_bps,
            'r2': r2,
            'eid': eid,
        }
    if report_to_tune:
        # only report the last result eid
        tune.report({'bps': co_bps, 'r2': r2, 'epoch': epochs})
        return None
    return result


def get_search_space(preset: str) -> dict[str, Any]:
    """Return the Ray Tune search space of a preset.

    Args:
        preset: `legacy` searches only `lr` (`wd` is the constant 1e-4); `reg` also searches
            `wd` and both dropout rates and fixes `n_kp` to `TUNED_N_KP`. Both keep the legacy
            lr-annealing schedule, whose epoch-0 division makes the effective lr 10x smaller
            than `lr`: without it, `lr` above ~2e-4 collapses to predicting the mean. The `reg`
            lr range (2e-4 to 2e-3) brackets the best untuned lr (~8e-4); above ~1.9e-3 the
            untuned model collapses.

    Returns:
        Mapping of config keys to Ray Tune domains or constants.

    Raises:
        ValueError: if `preset` is unknown.
    """
    if preset == 'legacy':
        return {'lr': tune.loguniform(1e-4, 3e-3), 'wd': 1e-4}
    if preset == 'reg':
        return {
            'lr': tune.loguniform(2e-4, 2e-3),
            # AdamW shrinks weights by ~exp(-wd * sum(lr)) ~ exp(-0.04 * wd) over training, so
            # wd below ~1 is a near no-op and above ~100 wipes the weights out
            'wd': tune.loguniform(1e-1, 1e2),
            'dropout_in': tune.uniform(0.0, 0.6),
            'dropout_mid': tune.uniform(0.0, 0.5),
            'n_kp': TUNED_N_KP,
            'anneal': 'legacy',
        }
    raise ValueError(f"preset must be 'legacy' or 'reg', got {preset!r}")


def tune_cnn_encoder(
    data_dict: dict[str, Any],
    num_samples: int = 10,
    tune_storage_path: str | None = None,
    seed: int = 42,
    preset: str = 'legacy',
) -> tuple[dict[str, Any], dict[str, float]]:
    """Ray-Tune hyperparameter search for `train_cnn_encoder` on the train/val split.

    The test split (last element of `X` and `y`) is never used. With the `reg` preset the
    search is seeded, starts from the best untuned point (`lr=8e-4`, `wd=0`, no dropout; a
    sanity check that reproduces the untuned validation bps and a baseline for whether
    regularization helps), and stops weak trials early with ASHA based on the validation
    metrics reported every 10 epochs. Only trials that trained all epochs compete
    for the best config, since early-stopped trials are scored at different epochs.

    Args:
        data_dict: mapping `eid -> {'X': [train, val, test], 'y': [...]}`.
        num_samples: number of Ray Tune trials.
        tune_storage_path: Ray Tune experiment root directory; `None` uses Ray's default.
        seed: RNG seed forwarded to every `train_cnn_encoder` trial (see its docstring).
        preset: search space, see `get_search_space`.

    Returns:
        Tuple `(best_config, best_metrics)`, the config with the highest validation co-bps and
        its `bps` and `r2`.

    Raises:
        RuntimeError: if no trial finished all epochs with a finite validation co-bps, or the
            best validation co-bps is not positive (all trials collapsed to the mean).
    """
    train_val_dict = {}
    for eid in data_dict:
        train_val_dict[eid] = copy.deepcopy(data_dict[eid])
        # remove the last element of X and y since it is the test set
        train_val_dict[eid]['X'].pop()
        train_val_dict[eid]['y'].pop()
    extra_kwargs = {}
    eval_every = 0
    if preset == 'reg':
        eval_every = 10
        extra_kwargs['search_alg'] = tune.search.BasicVariantGenerator(
            points_to_evaluate=[{'lr': 8e-4, 'wd': 0.0, 'dropout_in': 0.0, 'dropout_mid': 0.0}],
            random_state=seed,
        )
        extra_kwargs['scheduler'] = ASHAScheduler(
            time_attr='epoch',
            max_t=_NUM_EPOCHS,
            grace_period=30,
            reduction_factor=2,
        )
    analysis = tune.run(
        tune.with_parameters(
            train_cnn_encoder,
            data_dict=train_val_dict,
            report_to_tune=True,
            verbose=True,
            seed=seed,
            eval_every=eval_every,
        ),
        resources_per_trial={'cpu': 2, 'gpu': 1},
        config=get_search_space(preset),
        num_samples=num_samples,
        metric='bps',
        mode='max',
        log_to_file=False,
        **extra_kwargs,
        **({'storage_path': tune_storage_path} if tune_storage_path else {}),
    )
    trials_done = [
        trial for trial in analysis.trials
        if (trial.last_result or {}).get('epoch') == _NUM_EPOCHS
        and np.isfinite(trial.last_result['bps'])
    ]
    if not trials_done:
        raise RuntimeError('no Ray Tune trial finished all epochs with a finite validation bps')
    best_trial = max(trials_done, key=lambda trial: trial.last_result['bps'])
    if best_trial.last_result['bps'] <= 0:
        raise RuntimeError(
            f"best validation bps is {best_trial.last_result['bps']:.4f} (<= 0), i.e. every "
            'trial predicts the mean; the search space (probably lr) is out of range',
        )
    best_config = dict(best_trial.config)
    best_metrics = {key: float(best_trial.last_result[key]) for key in ('bps', 'r2')}
    print('Best config: ', best_config, 'validation metrics: ', best_metrics)
    return best_config, best_metrics


def train_cnn_encoder_with_tune(
    data_dict: dict[str, Any],
    num_samples: int = 10,
    tune_storage_path: str | None = None,
    seed: int = 42,
) -> dict[str, Any] | None:
    """Ray-Tune hyperparameter search over `lr` for `train_cnn_encoder`, then test evaluation.

    Args:
        data_dict: mapping `eid -> {'X': [train, val, test], 'y': [...]}`.
        num_samples: number of Ray Tune trials.
        tune_storage_path: Ray Tune experiment root directory; `None` uses Ray's default.
        seed: RNG seed forwarded to every `train_cnn_encoder` trial (see its docstring).

    Returns:
        Final `train_cnn_encoder` result evaluated with the best config on the held-out
        (last) split.
    """
    best_config, _ = tune_cnn_encoder(
        data_dict,
        num_samples=num_samples,
        tune_storage_path=tune_storage_path,
        seed=seed,
        preset='legacy',
    )
    # test data_dict, remove the 2nd last element of X and y since it is the validation set
    train_test_dict = {}
    for eid in data_dict:
        train_test_dict[eid] = copy.deepcopy(data_dict[eid])
        train_test_dict[eid]['X'].pop(-2)
        train_test_dict[eid]['y'].pop(-2)
    return train_cnn_encoder(
        config=best_config,
        data_dict=train_test_dict,
        report_to_tune=False,
        verbose=True,
        seed=seed,
    )
