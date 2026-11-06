"""Regression coverage for SABLE session discovery and split isolation."""

import pytest

from beast.data.sable_dataset import Cheese3DDataset, IBLTwoViewDataset


@pytest.mark.parametrize('dataset_type', [Cheese3DDataset, IBLTwoViewDataset])
def test_repeated_session_cannot_put_the_same_frame_in_train_and_validation(
    tmp_path, dataset_type,
):
    for session in ('first', 'second'):
        directories = (
            [tmp_path / session / camera for camera in ('TL', 'TR')]
            if dataset_type is Cheese3DDataset
            else [
                tmp_path / f'{camera}Camera.video'
                / f'_iblrig_{camera}Camera.downsampled.{session}'
                for camera in ('left', 'right')
            ]
        )
        for directory in directories:
            directory.mkdir(parents=True)
            for frame in range(100):
                (directory / f'img{frame:08d}.png').touch()
    config = {
        'model': {'seed': 0, 'image_tokenizer': {'image_size': 4}},
        'training': {
            'dataset_path': str(tmp_path), 'session_names': ['first', 'first', 'second'],
            'val_split_ratio': 0.1,
        },
    }

    train = dataset_type(config, include_splits=['train'])
    val = dataset_type(config, include_splits=['val'])

    train_paths = {rec.left_path for rec in train._records}
    val_paths = {rec.left_path for rec in val._records}
    assert len(train) == len(train_paths) == 180
    assert len(val) == len(val_paths) == 20
    assert train_paths.isdisjoint(val_paths)
    if dataset_type is IBLTwoViewDataset:
        assert train.session_id_to_idx == val.session_id_to_idx == {'first': 0, 'second': 1}
