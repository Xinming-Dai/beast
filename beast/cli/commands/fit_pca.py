"""Command to fit a linear PCA model on training images for pca-model initialization."""

import argparse
import logging
from typing import Any

import numpy as np
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader

from beast.cli.types import valid_dir
from beast.data.datasets import BaseDataset
from beast.models.pca import GPU_PCA, save_pca_model

_logger = logging.getLogger('BEAST.CLI.FIT_PCA')


def register_parser(subparsers: Any) -> None:
    """Register the fit-pca command parser."""

    parser = subparsers.add_parser(
        'fit-pca',
        description=(
            'Fit a linear PCA model on a directory of training images and save it to a '
            'pickle file usable as model.model_params.pca_pickle_path for a pca model.'
        ),
        usage='beast fit-pca --data-dir <path> --output <pickle_path> [options]',
    )

    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--data-dir', '-d',
        type=valid_dir,
        required=True,
        help='Directory of training images (searched recursively for .png files)',
    )
    required.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to write the fitted PCA pickle file',
    )

    # Optional arguments
    optional = parser.add_argument_group('options')
    optional.add_argument(
        '--session-names',
        nargs='+',
        metavar='SESSION',
        help='session IDs to restrict images to (default: use every image under --data-dir)',
    )
    optional.add_argument(
        '--n-components', '-n',
        type=int,
        default=50,
        help='number of principal components to keep (default: 50)',
    )
    optional.add_argument(
        '--num-channels',
        type=int,
        default=3,
        help='number of image channels (default: 3)',
    )
    optional.add_argument(
        '--batch-size', '-b',
        type=int,
        default=128,
        help='batch size for the incremental PCA fit (default: 128)',
    )


def handle(args: argparse.Namespace) -> None:
    """Handle the fit-pca command execution."""

    dataset = BaseDataset(
        data_dir=args.data_dir,
        imgaug_pipeline=None,
        num_channels=args.num_channels,
        session_names=args.session_names,
    )
    if len(dataset) < args.n_components:
        raise ValueError(
            f'Cannot fit {args.n_components} components from only {len(dataset)} images; '
            'lower --n-components or point --data-dir at more images.'
        )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    _logger.info(f'Fitting incremental PCA ({args.n_components} components) on {args.data_dir}')
    pca = IncrementalPCA(n_components=args.n_components)
    for batch_dict in dataloader:
        images = batch_dict['image']
        flat = images.reshape(images.shape[0], -1).numpy()
        pca.partial_fit(flat)

    gpu_pca = GPU_PCA(
        components=pca.components_,
        mean=pca.mean_,
        explained_variance=np.asarray(pca.explained_variance_),
    )
    save_pca_model(gpu_pca, args.output)
    _logger.info(f'Saved fitted PCA model to {args.output}')
