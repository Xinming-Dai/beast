"""Tests for the Sable config schemas."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from beast.io import load_config
from beast.models.sable.sable_config import SableBeastConfig

_CONFIG_PATHS = sorted((Path(__file__).parents[3] / 'configs' / 'sable').glob('*.yaml'))

_MINIMAL_SABLE = {
    'model': {'model_class': 'sable', 'transformer': {'d': 8}},
    'training': {'dataset_name': 'beast.data.sable_dataset.SABLEDataset', 'max_fwdbwd_passes': 5},
    'optimizer': {'lr': 1e-4, 'warmup': 2},
}


class TestSableBeastConfig:
    """Test the SableBeastConfig model."""

    def test_valid_config(self) -> None:
        cfg = SableBeastConfig.model_validate(_MINIMAL_SABLE)
        assert cfg.model.model_class == 'sable'

    def test_defaults_filled_in(self) -> None:
        dumped = SableBeastConfig.model_validate(_MINIMAL_SABLE).model_dump()
        assert dumped['model']['seed'] == 0
        assert dumped['training']['batch_size_per_gpu'] == 1
        assert dumped['training']['num_workers'] == 8
        assert dumped['optimizer']['type'] == 'AdamW'

    def test_extra_keys_pass_through(self) -> None:
        dumped = SableBeastConfig.model_validate(_MINIMAL_SABLE).model_dump()
        assert dumped['model']['transformer'] == {'d': 8}
        assert dumped['training']['max_fwdbwd_passes'] == 5
        assert dumped['optimizer']['warmup'] == 2

    def test_data_section_is_optional(self) -> None:
        assert SableBeastConfig.model_validate(_MINIMAL_SABLE).data is None

    def test_wrong_model_class_raises(self) -> None:
        raw = {**_MINIMAL_SABLE, 'model': {'model_class': 'vit'}}
        with pytest.raises(ValidationError):
            SableBeastConfig.model_validate(raw)

    def test_missing_dataset_name_raises(self) -> None:
        raw = {**_MINIMAL_SABLE, 'training': {'max_fwdbwd_passes': 5}}
        with pytest.raises(ValidationError):
            SableBeastConfig.model_validate(raw)

    def test_missing_lr_raises(self) -> None:
        raw = {**_MINIMAL_SABLE, 'optimizer': {'warmup': 2}}
        with pytest.raises(ValidationError):
            SableBeastConfig.model_validate(raw)


@pytest.mark.parametrize('config_path', _CONFIG_PATHS, ids=lambda p: p.name)
class TestShippedSableConfigs:
    """Test that every shipped sable yaml loads through ``load_config``."""

    def test_loads_as_sable(self, config_path: Path) -> None:
        cfg = load_config(config_path)
        assert cfg['model']['model_class'] == 'sable'

    def test_keeps_every_training_key(self, config_path: Path) -> None:
        raw = yaml.safe_load(config_path.read_text())
        cfg = load_config(config_path)
        assert set(raw['training']) <= set(cfg['training'])
