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
from beast.train import get_callbacks, pretty_print_config, reset_seeds


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
