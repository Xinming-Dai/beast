"""ERayZer-specific training loop using PyTorch Lightning."""

import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.utilities import rank_zero_only

from beast import log_step, version as beast_version
from beast.data.ibl_dataset import IBLDataset
from beast.models.model_utils.data_utils import collate_with_correspondence_padding
from beast.models.model_utils.train_vis import save_training_visuals
from beast.train import get_callbacks, pretty_print_config, reset_seeds


class ValVisualizationCallback(pl.Callback):
    """Saves render-vs-target PNG grids after the first validation batch each epoch.

    Args:
        vis_dir: directory to write visualization files.
        max_samples: number of batch samples to visualize per validation run.
        max_views: number of target views to include per sample.
    """

    def __init__(self, vis_dir: Path, max_samples: int = 1, max_views: int = 2) -> None:
        """Initialize with output directory and visualization limits."""
        super().__init__()
        self._vis_dir = Path(vis_dir)
        self._max_samples = max_samples
        self._max_views = max_views
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
            with torch.no_grad():
                result = pl_module(batch)
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
        except Exception as exc:
            log_step(f'ValVisualizationCallback: failed to save initial visuals: {exc}', level='warning')

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset per-epoch save flag at the start of each validation run."""
        self._saved_this_epoch = False

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
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
        except Exception as exc:
            log_step(f'ValVisualizationCallback: failed to save visuals: {exc}', level='warning')


def train_erayzer(config: dict, model, output_dir: str | Path):
    """Train an ERayZer model on IBL data using PyTorch Lightning.

    Reads training parameters from ``config['training']``:
    ``batch_size_per_gpu``, ``max_fwdbwd_passes``, ``grad_accum_steps``,
    ``use_amp``, ``amp_dtype``, ``val_every``, ``grad_clip_norm``,
    ``checkpoint_every``, ``resume_ckpt``, ``num_workers``,
    ``val_split_ratio``, ``val_dataset_path``.

    Args:
        config: full beast config dict.
        model: ERayZer Lightning model instance.
        output_dir: directory to save checkpoints and logs.

    Returns:
        trained model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_step('Entering train_erayzer()', level='debug')
    reset_seeds(seed=int(config['model'].get('seed', 0)))

    config['model']['beast_version'] = beast_version
    pretty_print_config(config)

    training = config['training']

    log_step('Building IBL datasets (train / val splits)', level='info')
    train_dataset = IBLDataset(config, include_splits=['train'])
    val_dataset = IBLDataset(config, include_splits=['val'])

    if rank_zero_only.rank == 0:
        print(f'Dataset — train: {len(train_dataset)}, val: {len(val_dataset)}')

    num_workers = int(training.get('num_workers', 8))
    batch_size = int(training.get('batch_size_per_gpu', 1))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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

    # ----------------------------------------------------------------------------------
    # Save configuration in output directory
    # ----------------------------------------------------------------------------------

    log_step(f'Saving config to {output_dir}', level='debug')
    dest_config_file = output_dir / 'config.yaml'
    with open(dest_config_file, 'w') as fh:
        yaml.dump(config, fh)
    log_step('Config saved', level='debug')

    # ----------------------------------------------------------------------------------
    # Set up and run training
    # ----------------------------------------------------------------------------------

    # reuse get_callbacks from train.py for LR monitor + val-best checkpoint;
    # append a step-based periodic checkpoint on top if configured.
    callbacks = get_callbacks(
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
            )
        )

    # precision
    precision: str | int = 32
    if training.get('use_amp', False):
        amp_dtype = str(training.get('amp_dtype', 'bf16')).lower()
        if amp_dtype == 'bf16':
            precision = 'bf16-mixed'
        elif amp_dtype in ('fp16', 'f16', '16'):
            precision = '16-mixed'

    logger = pl.loggers.TensorBoardLogger('tb_logs', name='')

    log_step('Creating PyTorch Lightning Trainer', level='debug')
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=int(training.get('num_gpus', 1)),
        num_nodes=int(training.get('num_nodes', 1)),
        max_steps=int(training.get('max_fwdbwd_passes', 4000)),
        accumulate_grad_batches=int(training.get('grad_accum_steps', 1)),
        precision=precision,
        val_check_interval=int(training.get('val_every', 10)),
        gradient_clip_val=float(training.get('grad_clip_norm', 1.0)),
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=True,
        log_every_n_steps=int(training.get('tensorboard_log_every', 1)),
    )

    resume_ckpt: str | None = training.get('resume_ckpt') or None
    reset_training_state: bool = bool(training.get('reset_training_state', False))
    ckpt_path_for_trainer: str | None = resume_ckpt

    if resume_ckpt is not None:
        raw_ckpt = torch.load(resume_ckpt, map_location='cpu', weights_only=False)
        is_lightning_ckpt = 'pytorch-lightning_version' in raw_ckpt
        if not is_lightning_ckpt or reset_training_state:
            # plain PyTorch checkpoint or explicitly resetting training state —
            # load model weights only so optimizer/scheduler start fresh
            state_dict = raw_ckpt.get('state_dict', raw_ckpt.get('model', raw_ckpt))
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                log_step(f'Missing keys when loading checkpoint: {missing}', level='warning')
            if unexpected:
                log_step(f'Unexpected keys when loading checkpoint: {unexpected}', level='warning')
            ckpt_path_for_trainer = None

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
