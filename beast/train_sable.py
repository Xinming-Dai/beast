"""Sable-specific training loop using PyTorch Lightning."""

import importlib
import sys
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
import torch
import yaml

from beast.data.samplers import ResumableRandomSampler
from beast.inference import save_gaussian_pointclouds
from beast.logging import log_step
from beast.data.sable_dataset import collate_with_correspondence_padding
from beast.models.model_utils.train_vis import save_training_visuals
from beast.train import get_callbacks, pretty_print_config, reset_seeds


def _resolve_dataset_class(dataset_name: str) -> type:
    """Resolve a dotted ``module.ClassName`` path to the dataset class.

    Args:
        dataset_name: dotted path, e.g. ``'beast.data.sable_dataset.SABLEDataset'``.

    Returns:
        the resolved class.
    """
    module_name, _, class_name = dataset_name.rpartition('.')
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class LossLoggerCallback(pl.Callback):
    """Prints train and val losses to stdout at a configurable step interval.

    Args:
        max_steps: total training steps (global ceiling), used for zero-padded step counter.
        log_every: print train losses every this many steps.
        global_step_offset: added to trainer.global_step for display; set to the
            checkpoint's global_step when resuming so logs show absolute step numbers.
    """

    def __init__(self, max_steps: int, log_every: int = 1, global_step_offset: int = 0) -> None:
        """Initialize with total step count, logging interval, and step offset for resumed training."""
        super().__init__()
        self._max_steps = max_steps
        self._log_every = max(1, log_every)
        self._global_step_offset = global_step_offset

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        _outputs,
        _batch,
        batch_idx: int,
    ) -> None:
        """Print per-step train losses at the configured interval."""
        step = trainer.global_step + self._global_step_offset
        if step % self._log_every != 0:
            return
        m = trainer.callback_metrics
        width = len(str(self._max_steps))
        parts = [f'[train] step={step:0{width}d}/{self._max_steps:0{width}d}']
        for key in ('train_loss', 'train_l2', 'train_psnr', 'train_gs_reg',
                    'train_lpips', 'train_perceptual'):
            if key in m:
                short = key.replace('train_', '')
                parts.append(f'{short}={m[key].item():.6f}')
        log_step(' '.join(parts), level='info')

    @rank_zero_only
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
    ) -> None:
        """Print aggregated val losses at the end of each validation run."""
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        step = trainer.global_step + self._global_step_offset
        parts = [f'[val] step={step}']
        for key in ('val_loss', 'val_l2', 'val_psnr', 'val_gs_reg',
                    'val_lpips', 'val_perceptual'):
            if key in m:
                short = key.replace('val_', '')
                parts.append(f'{short}={m[key].item():.6f}')
        log_step(' '.join(parts), level='info')


class ValVisualizationCallback(pl.Callback):
    """Saves render-vs-target PNG grids after the first validation batch each epoch.

    Args:
        vis_dir: directory to write visualization files.
        max_samples: number of batch samples to visualize per validation run.
        max_views: number of target views to include per sample.
        save_pointclouds: whether to also save a Gaussian-center point cloud
            (as a PLY file under ``vis_dir / 'ply'``) alongside each visual.
    """

    def __init__(
        self,
        vis_dir: Path,
        max_samples: int = 1,
        max_views: int = 2,
        save_pointclouds: bool = True,
    ) -> None:
        """Initialize with output directory and visualization limits."""
        super().__init__()
        self._vis_dir = Path(vis_dir)
        self._max_samples = max_samples
        self._max_views = max_views
        self._save_pointclouds = save_pointclouds
        self._saved_this_epoch = False

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save a step-0 visual from the first val batch before any training begins."""
        if not trainer.is_global_zero:
            return
        try:
            val_loader = trainer.val_dataloaders
            if isinstance(val_loader, list):
                val_loader = val_loader[0]
            batch = next(iter(val_loader))
            batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
            pl_module.eval()
            with torch.no_grad():
                result = pl_module(batch)
            pl_module.train()
            saved_paths = save_training_visuals(
                self._vis_dir,
                result=result,
                batch=batch,
                step=0,
                max_samples=self._max_samples,
                max_views=self._max_views,
            )
            if saved_paths:
                log_step(f'Saved initial visuals: {saved_paths[0]}', level='info')
            if self._save_pointclouds:
                ply_paths = save_gaussian_pointclouds(
                    vars(result),
                    self._vis_dir,
                    batch_idx=0,
                    max_samples=self._max_samples,
                )
                if ply_paths:
                    log_step(f'Saved initial point cloud: {ply_paths[0]}', level='info')
        except Exception as exc:
            log_step(f'ValVisualizationCallback: failed to save initial visuals: {exc}', level='warning')

    def on_validation_epoch_start(self, _trainer: pl.Trainer, _pl_module: pl.LightningModule) -> None:
        """Reset per-epoch save flag at the start of each validation run."""
        self._saved_this_epoch = False

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        _outputs,
        batch: dict,
        batch_idx: int,
    ) -> None:
        """Save visuals on the first batch of each validation epoch (rank 0 only)."""
        if batch_idx != 0 or self._saved_this_epoch or not trainer.is_global_zero:
            return
        self._saved_this_epoch = True
        try:
            with torch.no_grad():
                result = pl_module(batch)
            saved_paths = save_training_visuals(
                self._vis_dir,
                result=result,
                batch=batch,
                step=trainer.global_step,
                max_samples=self._max_samples,
                max_views=self._max_views,
            )
            if saved_paths:
                log_step(f'Saved val visuals: {saved_paths[0]}', level='info')
            if self._save_pointclouds:
                ply_paths = save_gaussian_pointclouds(
                    vars(result),
                    self._vis_dir,
                    batch_idx=trainer.global_step,
                    max_samples=self._max_samples,
                )
                if ply_paths:
                    log_step(f'Saved val point cloud: {ply_paths[0]}', level='info')
        except Exception as exc:
            log_step(f'ValVisualizationCallback: failed to save visuals: {exc}', level='warning')


class StepAccumulatorCallback(pl.Callback):
    """Writes cumulative total_steps_trained into every saved checkpoint.

    Args:
        steps_before_this_job: total steps already trained across all prior jobs,
            read from the resumed checkpoint's total_steps_trained field.
    """

    def __init__(self, steps_before_this_job: int) -> None:
        """Initialize with the cumulative step count before this job."""
        super().__init__()
        self._steps_before = steps_before_this_job

    @rank_zero_only
    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> None:
        """Inject total_steps_trained = prior steps + current job's global_step."""
        checkpoint['total_steps_trained'] = trainer.global_step + self._steps_before


def train_sable(config: dict, model, output_dir: str | Path):
    """Train a Sable model on IBL data using PyTorch Lightning.

    Reads training parameters from ``config['training']``:
    ``batch_size_per_gpu``, ``max_fwdbwd_passes``, ``grad_accum_steps``,
    ``use_amp``, ``amp_dtype``, ``val_every``, ``grad_clip_norm``,
    ``checkpoint_every``, ``resume_ckpt``, ``num_workers``,
    ``val_split_ratio``, ``val_dataset_path``.

    Args:
        config: full beast config dict.
        model: Sable Lightning model instance.
        output_dir: directory to save checkpoints and logs.

    Returns:
        trained model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dest_config_file = output_dir / 'config.yaml'
    with open(dest_config_file, 'w') as file:
        yaml.dump(config, file)

    log_step('Entering train_sable()', level='debug')
    seed = int(config['model'].get('seed', 0))
    reset_seeds(seed=seed)

    pretty_print_config(config)

    training = config['training']

    log_step('Building IBL datasets (train / val splits)', level='info')
    dataset_cls = _resolve_dataset_class(
        training.get('dataset_name', 'beast.data.sable_dataset.SABLEDataset'),
    )
    train_dataset = dataset_cls(config, include_splits=['train'])
    val_dataset = dataset_cls(config, include_splits=['val'])

    if rank_zero_only.rank == 0:
        print(f'Dataset — train: {len(train_dataset)}, val: {len(val_dataset)}')

    num_workers = int(training.get('num_workers', 8))
    batch_size = int(training.get('batch_size_per_gpu', 1))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=ResumableRandomSampler(train_dataset, seed=seed),
        num_workers=num_workers,
        collate_fn=collate_with_correspondence_padding,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_correspondence_padding,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    # set up and run training

    # load checkpoint early so we can read global_step and compute remaining steps
    max_fwdbwd_passes: int = int(training.get('max_fwdbwd_passes', 4000))
    resume_ckpt: str | None = training.get('resume_ckpt') or None
    reset_training_state: bool = bool(training.get('reset_training_state', False))
    ckpt_path_for_trainer: str | None = resume_ckpt
    global_step_at_resume: int = 0

    if resume_ckpt is not None:
        raw_ckpt = torch.load(resume_ckpt, map_location='cpu', weights_only=False)
        is_lightning_ckpt = 'pytorch-lightning_version' in raw_ckpt
        if not is_lightning_ckpt or reset_training_state:
            # plain PyTorch checkpoint or explicitly resetting training state —
            # load model weights only so optimizer/scheduler start fresh
            log_step('Training state will be reset', level='warning')
            state_dict = raw_ckpt.get('state_dict', raw_ckpt.get('model', raw_ckpt))
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                log_step(f'Missing keys when loading checkpoint: {missing}', level='warning')
            if unexpected:
                log_step(f'Unexpected keys when loading checkpoint: {unexpected}', level='warning')
            ckpt_path_for_trainer = None
        else:
            global_step_at_resume = int(
                raw_ckpt.get('total_steps_trained', raw_ckpt.get('global_step', 0)),
            )

    max_steps_this_run: int = max(0, max_fwdbwd_passes - global_step_at_resume)
    if global_step_at_resume > 0:
        log_step(
            f'Resuming from global_step={global_step_at_resume}; '
            f'will train for {max_steps_this_run} more steps '
            f'(max_fwdbwd_passes={max_fwdbwd_passes})',
            level='info',
        )

    # when Lightning restores a full checkpoint it already sets its internal global_step,
    # so callbacks must not add an extra offset — only offset when Lightning starts from 0.
    callback_step_offset = 0 if ckpt_path_for_trainer is not None else global_step_at_resume

    # reuse get_callbacks from train.py for LR monitor + val-best checkpoint;
    # append a step-based periodic checkpoint on top if configured.
    print_every = int(training.get('print_every', 10))
    callbacks = [LossLoggerCallback(
        max_steps=max_fwdbwd_passes,
        log_every=print_every,
        global_step_offset=callback_step_offset,
    )]
    callbacks += get_callbacks(
        lr_monitor=True,
        checkpointing=training.get('save_val_best_checkpoint', True),
    )
    checkpoint_every = int(training.get('checkpoint_every', 0))
    if checkpoint_every > 0:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                monitor=None,
                every_n_train_steps=checkpoint_every,
                save_top_k=-1,
                filename='{step}',
            )
        )

    if training.get('save_visuals', True):
        vis_dir_cfg = training.get('vis_dir') or None
        vis_dir = Path(vis_dir_cfg) if vis_dir_cfg else output_dir / 'visuals'
        callbacks.append(
            ValVisualizationCallback(
                vis_dir=vis_dir,
                max_samples=int(training.get('vis_num_samples', 1)),
                max_views=int(training.get('vis_max_views', 2)),
                save_pointclouds=training.get('save_pointclouds', True),
            )
        )
    callbacks.append(StepAccumulatorCallback(callback_step_offset))

    precision: str | int = 32
    if training.get('use_amp', False):
        amp_dtype = str(training.get('amp_dtype', 'bf16')).lower()
        if amp_dtype == 'bf16':
            precision = 'bf16-mixed'
        elif amp_dtype in ('fp16', 'f16', '16'):
            precision = '16-mixed'

    logger = pl.loggers.TensorBoardLogger('tb_logs', name='')

    val_every = int(training.get('val_every', 10))
    val_check_interval = max(1, min(val_every, len(train_loader)))
    if val_check_interval != val_every:
        log_step(
            f'Clamping val_every from {val_every} to {val_check_interval} '
            f'(number of training batches)',
            level='info',
        )

    log_step('Creating PyTorch Lightning Trainer', level='debug')
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=int(training.get('num_gpus', 1)),
        num_nodes=int(training.get('num_nodes', 1)),
        max_steps=max_steps_this_run,
        accumulate_grad_batches=int(training.get('grad_accum_steps', 1)),
        precision=precision,
        val_check_interval=val_check_interval,
        gradient_clip_val=float(training.get('grad_clip_norm', 1.0)),
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=True,
        log_every_n_steps=int(training.get('tensorboard_log_every', 1)),
    )

    log_step('About to call trainer.fit()', level='debug')
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path_for_trainer,
    )
    log_step('trainer.fit() completed', level='debug')

    if not trainer.is_global_zero:
        sys.exit(0)

    return model
