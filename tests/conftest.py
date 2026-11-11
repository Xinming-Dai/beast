import gc
import io
import json
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
import requests
import torch

from beast.api.model import Model
from beast.data.augmentations import expand_imgaug_str_to_dict, imgaug_pipeline
from beast.data.datamodules import BaseDataModule
from beast.data.datasets import BaseDataset
from beast.io import load_config

ROOT = Path(__file__).parent.parent

_CALIBRATION_TOML = """\
[cam_0]
name = "BC"
size = [ 640, 512,]
matrix = [ [ 1000.0, 0.0, 319.5,], [ 0.0, 1000.0, 255.5,], [ 0.0, 0.0, 1.0,],]
distortions = [ -0.15, 0.0, 0.0, 0.0, 0.0,]
rotation = [ 0.0, 0.0, 0.0,]
translation = [ 0.0, 0.0, 0.0,]

[cam_1]
name = "TL"
size = [ 640, 512,]
matrix = [ [ 800.0, 0.0, 319.5,], [ 0.0, 800.0, 255.5,], [ 0.0, 0.0, 1.0,],]
distortions = [ -0.15, 0.0, 0.0, 0.0, 0.0,]
rotation = [ 0.0, 0.0, 0.0,]
translation = [ -10.0, 0.0, 0.0,]

[cam_2]
name = "TR"
size = [ 640, 512,]
matrix = [ [ 800.0, 0.0, 319.5,], [ 0.0, 800.0, 255.5,], [ 0.0, 0.0, 1.0,],]
distortions = [ -0.15, 0.0, 0.0, 0.0, 0.0,]
rotation = [ 0.0, 0.0, 1.5707963267948966,]
translation = [ 0.0, -20.0, 0.0,]
"""


@pytest.fixture
def calibration_dir(tmp_path: Path) -> Path:
    """Directory with a 3-camera (BC, TL, TR) toml rig as video_calibration and session_a."""
    calib_dir = tmp_path / 'calibrations'
    calib_dir.mkdir()
    (calib_dir / 'video_calibration.toml').write_text(_CALIBRATION_TOML)
    (calib_dir / 'session_a.toml').write_text(_CALIBRATION_TOML)
    return calib_dir


# ---------------------------------------------
# functions for loading test data from figshare
# ---------------------------------------------

def _load_dataset_metadata(dst_dir: Path) -> dict:
    """Load metadata from dataset directory."""
    metadata_file = dst_dir / '.dataset_metadata.json'
    if metadata_file.exists():
        with open(metadata_file) as f:
            return json.load(f)
    return {}


def _save_dataset_metadata(dst_dir: Path, url: str, dataset_name: str) -> None:
    """Save metadata to dataset directory."""
    metadata = {
        'url': url,
        'dataset_name': dataset_name
    }
    metadata_file = dst_dir / '.dataset_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def fetch_test_data_if_needed(save_dir: str | Path, dataset_name: str = 'testing_data') -> None:
    """
    Fetch test data from figshare if needed.

    Downloads data if:
    1. Dataset directory doesn't exist
    2. URL in function differs from cached URL (version change)

    Args:
        save_dir: Directory to save the dataset
        dataset_name: Name of the dataset to download
    """
    datasets_url_dict = {
        # 'testing_data': 'https://figshare.com/ndownloader/articles/29207330/versions/1',
        'testing_data': 'https://github.com/paninski-lab/beast-test-fixtures/releases/download/v1/test_models.zip',  # noqa
    }

    dst_dir = Path(save_dir) / dataset_name
    url = datasets_url_dict[dataset_name]

    # Check if data exists and URL matches
    if dst_dir.exists():
        metadata = _load_dataset_metadata(dst_dir)
        cached_url = metadata.get('url')

        if cached_url == url:
            print(f'Dataset {dataset_name} is up to date')
            return
        else:
            print(f'URL changed from {cached_url} to {url}, updating dataset')
            # Remove old data
            shutil.rmtree(dst_dir)

    # Download data
    print(f'Fetching {dataset_name} from {url}')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Check for download errors
        with zipfile.ZipFile(io.BytesIO(r.raw.read())) as z:
            z.extractall(dst_dir)

    # Save metadata with current URL
    _save_dataset_metadata(dst_dir, url, dataset_name)

    print('Done')


fetch_test_data_if_needed(save_dir=Path(__file__).parent)

# ---------------------------------------------
# pytest fixtures
# ---------------------------------------------


@pytest.fixture
def data_dir() -> Path:
    return ROOT.joinpath('tests/testing_data/data_dir')


@pytest.fixture
def video_file() -> Path:
    return ROOT.joinpath('tests/testing_data/test_vid.mp4')


@pytest.fixture
def config_ae_path() -> Path:
    return ROOT.joinpath('configs/resnet_ae.yaml')


@pytest.fixture
def config_ae(config_ae_path, data_dir) -> dict:
    config = load_config(config_ae_path)
    config['data']['data_dir'] = data_dir
    config['training']['train_batch_size'] = 32
    config['training']['val_batch_size'] = 32
    config['training']['test_batch_size'] = 32
    return config


@pytest.fixture
def config_vit_path() -> Path:
    return ROOT.joinpath('configs/vit.yaml')


@pytest.fixture
def config_vit(config_vit_path, data_dir) -> dict:
    config = load_config(config_vit_path)
    config['data']['data_dir'] = data_dir
    config['training']['train_batch_size'] = 4
    config['training']['val_batch_size'] = 4
    config['training']['test_batch_size'] = 4
    config['model']['model_params']['use_infoNCE'] = False
    return config


@pytest.fixture
def aug_pipeline() -> Callable:
    params_dict = expand_imgaug_str_to_dict('default')
    pipeline = imgaug_pipeline(params_dict)
    return pipeline


@pytest.fixture
def base_dataset(data_dir, aug_pipeline) -> BaseDataset:
    dataset = BaseDataset(data_dir=data_dir, imgaug_pipeline=aug_pipeline)
    return dataset


@pytest.fixture
def base_datamodule(base_dataset) -> BaseDataModule:
    datamodule = BaseDataModule(
        dataset=base_dataset,
        train_batch_size=10,
        val_batch_size=10,
        test_batch_size=10,
        train_probability=0.8,
        val_probability=0.1,
        test_probability=0.1,
    )
    datamodule.setup()
    return datamodule


@pytest.fixture
def base_datamodule_contrastive(base_dataset) -> BaseDataModule:
    """Fixture for testing contrastive learning functionality."""
    datamodule = BaseDataModule(
        dataset=base_dataset,
        train_batch_size=8,  # Even batch size for contrastive pairs
        val_batch_size=8,
        test_batch_size=8,
        train_probability=0.9,
        val_probability=0.05,
        test_probability=0.05,
        use_sampler=True,  # Enable contrastive sampler
    )
    datamodule.setup()
    return datamodule


@pytest.fixture
def run_model_test(tmp_path, data_dir) -> Callable:

    def _run_model_test(config):
        """Helper function to simplify unit tests which run different models."""

        # build model
        config['training']['num_epochs'] = 2
        config['training']['log_every_n_steps'] = 1
        config['training']['check_val_every_n_epoch'] = 1
        model = Model.from_config(config)

        try:
            # train model for a couple epochs
            model.train(tmp_path)
            # run inference on labeled data
            model.predict_images(image_dir=data_dir)
            # ensure model checkpoint saved
            assert model.model_dir is not None
            assert len(list(model.model_dir.rglob('*.ckpt'))) == 1
        finally:
            # remove tensors from gpu
            del model
            gc.collect()
            torch.cuda.empty_cache()

    return _run_model_test
