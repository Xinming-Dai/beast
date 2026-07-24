import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import trimesh
import yaml
from PIL import Image

from beast.inference import (
    ImagePredictionHandler,
    VideoPredictionHandler,
    _batch_output_path,
    _batch_session_ids,
    _flatten_mask_for_points,
    _is_valid_batch_npz,
    _num_batches_for,
    _parse_scene_name,
    _resume_batch_start,
    _save_latent_batch_npz,
    extract_sable_latents,
    predict_images,
    predict_video,
    save_camera_pointcloud_scene,
    save_gaussian_pointclouds,
)


class TestImagePredictionHandler:
    """Test suite for PredictionHandler class."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary source and output directories."""
        temp_source = Path(tempfile.mkdtemp())
        temp_output = Path(tempfile.mkdtemp())

        # Create test directory structure
        (temp_source / "video1").mkdir(parents=True)
        (temp_source / "video2").mkdir(parents=True)

        # Create dummy image files
        for video in ["video1", "video2"]:
            for frame in ["frame001.png", "frame002.png"]:
                dummy_img = Image.new('RGB', (64, 64), color='red')
                dummy_img.save(temp_source / video / frame)

        yield temp_source, temp_output

        # Cleanup
        shutil.rmtree(temp_source)
        shutil.rmtree(temp_output)

    @pytest.fixture
    def handler(self, temp_dirs):
        """Create PredictionHandler instance with temp directories."""
        source_dir, output_dir = temp_dirs
        return ImagePredictionHandler(output_dir, source_dir)

    @pytest.fixture
    def sample_tensor(self):
        """Create sample tensor for testing."""
        return torch.rand(3, 64, 64)  # (C, H, W) format

    @pytest.fixture
    def sample_batch_tensor(self):
        """Create sample batch tensor for testing."""
        return torch.rand(2, 3, 64, 64)  # (B, C, H, W) format

    @pytest.fixture
    def sample_latents(self):
        """Create sample latent tensor."""
        return torch.rand(2, 128)  # (B, latent_dim)

    @pytest.fixture
    def sample_metadata(self, temp_dirs):
        """Create sample batch metadata."""
        source_dir, _ = temp_dirs
        return {
            'video': ['video1', 'video1'],
            'idx': [torch.tensor(0), torch.tensor(1)],
            'image_paths': [
                source_dir / "video1" / "frame001.png",
                source_dir / "video1" / "frame002.png"
            ]
        }

    def test_init(self, temp_dirs):
        """Test PredictionHandler initialization."""
        source_dir, output_dir = temp_dirs
        handler = ImagePredictionHandler(output_dir, source_dir)
        assert handler.output_dir == Path(output_dir)
        assert handler.source_dir == Path(source_dir)
        assert handler.output_dir.exists()
        assert handler.metadata == []

    def test_tensor_to_image_3d(self, handler, sample_tensor):
        """Test tensor to image conversion with 3D tensor (C, H, W)."""
        image = handler.tensor_to_image(sample_tensor)
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert image.size == (64, 64)

    def test_tensor_to_image_4d(self, handler, sample_batch_tensor):
        """Test tensor to image conversion with 4D tensor (B, C, H, W)."""
        image = handler.tensor_to_image(sample_batch_tensor)
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert image.size == (64, 64)

    # def test_tensor_to_image_grayscale(self, handler):
    #     """Test tensor to image conversion with grayscale (1 channel)."""
    #     tensor_gray = torch.rand(1, 32, 32)
    #     image = handler.tensor_to_image(tensor_gray)
    #     assert isinstance(image, Image.Image)
    #     assert image.mode == 'RGB'  # always return RGB
    #     assert image.size == (32, 32, 3)

    def test_tensor_to_image_scaling(self, handler):
        """Test tensor value scaling from [0,1] to [0,255]."""
        # Create tensor with known values
        tensor = torch.ones(3, 2, 2) * 0.5  # All values = 0.5
        image = handler.tensor_to_image(tensor)
        # Check that values were scaled (0.5 * 255 = 127.5 → 127)
        np_array = np.array(image)
        assert np_array.max() <= 255
        assert np_array.min() >= 1

    def test_save_reconstruction(self, handler, sample_tensor, temp_dirs):
        """Test saving reconstruction image."""
        source_dir, output_dir = temp_dirs
        original_path = source_dir / 'video1' / 'frame001.png'

        saved_path = handler.save_reconstruction(
            sample_tensor, 'video1', 0, original_path
        )

        expected_path = output_dir / 'video1' / 'frame001.png'
        assert saved_path == expected_path
        assert saved_path.exists()
        assert saved_path.is_file()
        assert (output_dir / 'video1').exists()

    def test_save_latents(self, handler, temp_dirs):
        """Test saving latent representations."""
        source_dir, output_dir = temp_dirs
        original_path = source_dir / 'video1' / 'frame001.png'
        latents = torch.rand(128)

        saved_path = handler.save_latents(latents, 'video1', 0, original_path)

        expected_path = output_dir / 'latents' / 'video1' / 'frame001.npy'
        assert saved_path == expected_path
        assert saved_path.exists()
        assert saved_path.is_file()
        assert (output_dir / 'latents' / 'video1').exists()

        # Check that saved data matches
        loaded_latents = np.load(saved_path)
        np.testing.assert_array_almost_equal(loaded_latents, latents.detach().cpu().numpy())

    def test_process_batch_predictions_reconstructions_only(
        self, handler, sample_batch_tensor, sample_latents, sample_metadata,
    ):
        """Test processing batch with reconstructions only."""
        predictions = {
            'reconstructions': sample_batch_tensor,
            'latents': sample_latents
        }

        result = handler.process_batch_predictions(
            predictions,
            sample_metadata,
            save_reconstructions=True,
            save_latents=False
        )

        # Check results structure
        assert 'reconstructions' in result
        assert 'latents' in result
        assert 'metadata' in result

        # Should have saved reconstructions
        assert len(result['reconstructions']) == 2
        assert len(result['latents']) == 0  # No latents saved
        assert len(result['metadata']) == 2

        # Check metadata entries
        for _i, metadata in enumerate(result['metadata']):
            assert 'original_path' in metadata
            assert 'video' in metadata
            assert 'idx' in metadata
            assert 'reconstruction_path' in metadata
            assert 'latents_path' not in metadata  # Latents not saved

    def test_process_batch_predictions_latents_only(
        self, handler, sample_batch_tensor, sample_latents, sample_metadata
    ):
        """Test processing batch with latents only."""
        predictions = {
            'reconstructions': sample_batch_tensor,
            'latents': sample_latents
        }

        result = handler.process_batch_predictions(
            predictions,
            sample_metadata,
            save_reconstructions=False,
            save_latents=True
        )

        # Should have saved latents only
        assert len(result['reconstructions']) == 0
        assert len(result['latents']) == 2
        assert len(result['metadata']) == 2

        # Check metadata entries
        for metadata in result['metadata']:
            assert 'reconstruction_path' not in metadata
            assert 'latents_path' in metadata

    def test_process_batch_predictions_both(
        self, handler, sample_batch_tensor, sample_latents, sample_metadata,
    ):
        """Test processing batch with both reconstructions and latents."""
        predictions = {
            'reconstructions': sample_batch_tensor,
            'latents': sample_latents
        }

        result = handler.process_batch_predictions(
            predictions,
            sample_metadata,
            save_reconstructions=True,
            save_latents=True
        )

        # Should have saved both
        assert len(result['reconstructions']) == 2
        assert len(result['latents']) == 2
        assert len(result['metadata']) == 2

        # Check metadata entries have both paths
        for metadata in result['metadata']:
            assert 'reconstruction_path' in metadata
            assert 'latents_path' in metadata

    def test_save_metadata_summary(self, handler, temp_dirs):
        """Test saving metadata summary to YAML."""
        # Add some test metadata
        test_metadata = [
            {
                'original_path': '/path/to/original.png',
                'reconstruction_path': '/path/to/recon.png',
                'video': 'video1',
                'idx': 0
            }
        ]
        handler.metadata = test_metadata

        metadata_path = handler.save_metadata_summary()

        _, output_dir = temp_dirs
        expected_path = output_dir / 'prediction_metadata.yaml'

        assert metadata_path == expected_path
        assert metadata_path.exists()

        # Check YAML content
        with open(metadata_path) as f:
            loaded_metadata = yaml.safe_load(f)

        assert loaded_metadata == test_metadata

    def test_process_predictions_full_workflow(
        self, handler, sample_batch_tensor, sample_latents, sample_metadata
    ):
        """Test full workflow with process_predictions method."""
        # Create mock predictions (list of batches)
        predictions = [
            {
                'reconstructions': sample_batch_tensor,
                'latents': sample_latents,
                'metadata': sample_metadata
            }
        ]

        result = handler.process_predictions(
            predictions,
            save_reconstructions=True,
            save_latents=True
        )

        # Check results structure
        assert 'output_dir' in result
        assert 'num_images_processed' in result
        assert 'metadata_file' in result
        assert 'reconstructions_saved' in result
        assert 'latents_saved' in result
        assert 'reconstructions_dir' in result
        assert 'latents_dir' in result

        # Check counts
        assert result['num_images_processed'] == 2
        assert result['reconstructions_saved'] == 2
        assert result['latents_saved'] == 2

        # Check metadata file was created
        metadata_path = Path(result['metadata_file'])
        assert metadata_path.exists()

    def test_process_predictions_empty_list(self, handler):
        """Test process_predictions with empty prediction list."""
        result = handler.process_predictions([], save_reconstructions=True)

        assert result['num_images_processed'] == 0
        assert result['reconstructions_saved'] == 0

    def test_directory_creation(self, handler, sample_tensor, temp_dirs):
        """Test that subdirectories are created properly."""
        source_dir, output_dir = temp_dirs
        original_path = source_dir / 'new_video' / 'frame001.png'

        # This should create the new_video directory
        saved_path = handler.save_reconstruction(
            sample_tensor, 'new_video', 0, original_path
        )

        assert (output_dir / 'new_video').exists()
        assert saved_path.parent == output_dir / 'new_video'

    @pytest.mark.parametrize('save_recons,save_latents', [
        (True, False),
        (False, True),
        (True, True),
        (False, False)
    ])
    def test_process_predictions_save_options(
            self, handler, sample_batch_tensor, sample_latents, sample_metadata,
            save_recons, save_latents
    ):
        """Test different combinations of save options."""
        predictions = [
            {
                'reconstructions': sample_batch_tensor,
                'latents': sample_latents,
                'metadata': sample_metadata
            }
        ]

        result = handler.process_predictions(
            predictions,
            save_reconstructions=save_recons,
            save_latents=save_latents
        )

        if save_recons:
            assert 'reconstructions_saved' in result
            assert result['reconstructions_saved'] == 2
        else:
            assert result.get('reconstructions_saved', 0) == 0

        if save_latents:
            assert 'latents_saved' in result
            assert result['latents_saved'] == 2
        else:
            assert result.get('latents_saved', 0) == 0


class TestVideoPredictionHandler:
    """Test suite for VideoPredictionHandler class."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary source and output directories."""
        temp_source = Path(tempfile.mkdtemp())
        temp_output = Path(tempfile.mkdtemp())

        # create test video file
        video_file = temp_source / 'video1.mp4'
        # create a simple test video with 10 frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
        out = cv2.VideoWriter(str(video_file), fourcc, 10.0, (64, 64))
        for i in range(10):
            # create frames with different colors
            frame = np.full((64, 64, 3), (i * 25, i * 25, i * 25), dtype=np.uint8)
            out.write(frame)

        out.release()

        yield temp_source, temp_output, video_file

        # cleanup
        shutil.rmtree(temp_source)
        shutil.rmtree(temp_output)

    @pytest.fixture
    def handler(self, temp_dirs):
        """Create VideoPredictionHandler instance with temp directories."""
        source_dir, output_dir, video_file = temp_dirs
        return VideoPredictionHandler(output_dir, video_file)

    @pytest.fixture
    def sample_tensor(self):
        """Create sample tensor for testing."""
        return torch.rand(3, 64, 64)  # (C, H, W) format

    @pytest.fixture
    def sample_batch_tensor(self):
        """Create sample batch tensor for testing."""
        return torch.rand(2, 3, 64, 64)  # (B, C, H, W) format

    @pytest.fixture
    def sample_latents(self):
        """Create sample latent tensor."""
        return torch.rand(2, 128)  # (B, latent_dim)

    @pytest.fixture
    def sample_metadata(self, temp_dirs):
        """Create sample batch metadata for video."""
        source_dir, _, video_paths = temp_dirs
        return {
            'video_path': [str(video_paths[0]), str(video_paths[0])],
            'frame_idx': [torch.tensor(0), torch.tensor(1)],
            'batch_start_idx': torch.tensor(0)
        }

    def test_init(self, temp_dirs):
        """Test VideoPredictionHandler initialization."""
        _, output_dir, video_file = temp_dirs
        handler = VideoPredictionHandler(output_dir, video_file)
        assert handler.output_dir == Path(output_dir)
        assert handler.output_dir.exists()
        assert handler.metadata == {
            'video_file': str(video_file),
            'output_dir': str(output_dir),
            'fps': 10,
            'width': 64,
            'height': 64,
            'total_frames': 10,
        }

    def test_tensor_to_image_3d(self, handler, sample_tensor):
        """Test tensor to image conversion with 3D tensor (C, H, W)."""
        image = handler.tensor_to_numpy_bgr(sample_tensor)
        assert isinstance(image, np.ndarray)
        assert image.shape == (64, 64, 3)

    def test_tensor_to_image_4d(self, handler, sample_batch_tensor):
        """Test tensor to image conversion with 4D tensor (B, C, H, W)."""
        image = handler.tensor_to_numpy_bgr(sample_batch_tensor)
        assert isinstance(image, np.ndarray)
        assert image.shape == (64, 64, 3)

    def test_tensor_to_image_grayscale(self, handler):
        """Test tensor to image conversion with grayscale (1 channel)."""
        tensor_gray = torch.rand(1, 32, 32)
        image = handler.tensor_to_numpy_bgr(tensor_gray)
        assert isinstance(image, np.ndarray)
        assert image.shape == (32, 32, 3)

    def test_tensor_to_image_scaling(self, handler):
        """Test tensor value scaling from [0,1] to [0,255]."""
        # Create tensor with known values
        tensor = torch.ones(3, 2, 2) * 0.5  # All values = 0.5
        image = handler.tensor_to_numpy_bgr(tensor)
        # Check that values were scaled (0.5 * 255 = 127.5 → 127)
        np_array = np.array(image)
        assert np_array.max() <= 255
        assert np_array.min() >= 1

    def test_process_predictions_empty_list(self, handler):
        """Test process_predictions with empty prediction list."""
        result = handler.process_predictions([], save_reconstructions=True)

        assert result['frames_processed'] == 0
        assert result['reconstruction_video'] is None

    @pytest.mark.parametrize('save_recons,save_latents', [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ])
    def test_process_predictions_save_options(
        self, handler, sample_batch_tensor, sample_latents, save_recons, save_latents,
    ):
        """Test different combinations of save options."""

        predictions = [
            {
                'reconstructions': sample_batch_tensor,
                'latents': sample_latents,
            },
            {
                'reconstructions': sample_batch_tensor,
                'latents': sample_latents,
            },
        ]

        result = handler.process_predictions(
            predictions,
            save_reconstructions=save_recons,
            save_latents=save_latents,
        )

        if save_recons:
            assert Path(result['reconstruction_video']).is_file()
        else:
            assert result['reconstruction_video'] is None

        if save_latents:
            assert Path(result['latents_file']).is_file()
            assert result['latents_shape'] == (4, 128)
        else:
            assert result['latents_file'] is None


class TestVideoPredictionHandlerInitVideoWriter:
    """Test the _init_video_writer error path."""

    def test_init_video_writer_bad_path_raises(self, tmp_path, video_file) -> None:
        # Arrange — handler with a bad output path so the writer fails to open
        handler = VideoPredictionHandler(tmp_path / 'out', video_file)
        mock_writer = Mock()
        mock_writer.isOpened.return_value = False
        with patch('beast.inference.cv2.VideoWriter', return_value=mock_writer):
            # Act / Assert
            with pytest.raises(ValueError, match='Failed to open video writer'):
                handler._init_video_writer()


class TestPredictImages:
    """Test the predict_images standalone function."""

    def test_predict_images_none_predictions_raises(self, data_dir, tmp_path) -> None:
        # Arrange — trainer returns None instead of a prediction list
        mock_model = Mock()
        with patch('beast.inference.pl.Trainer') as MockTrainer:
            MockTrainer.return_value.predict.return_value = None
            # Act / Assert
            with pytest.raises(RuntimeError, match="trainer.predict\\(\\) returned None"):
                predict_images(
                    model=mock_model,
                    output_dir=tmp_path,
                    source_dir=data_dir,
                )

    def test_predict_images_returns_results(self, data_dir, tmp_path) -> None:
        # Arrange — mock trainer to return a minimal prediction list
        mock_model = Mock()
        mock_predictions = [
            {
                'reconstructions': torch.rand(2, 3, 224, 224),
                'latents': torch.rand(2, 128),
                'metadata': {
                    'video': ['vid', 'vid'],
                    'idx': [torch.tensor(0), torch.tensor(1)],
                    'image_paths': [
                        next(data_dir.rglob('*.png')),
                        next(data_dir.rglob('*.png')),
                    ],
                },
            }
        ]
        with patch('beast.inference.pl.Trainer') as MockTrainer:
            MockTrainer.return_value.predict.return_value = mock_predictions
            # Act
            result = predict_images(
                model=mock_model,
                output_dir=tmp_path,
                source_dir=data_dir,
                save_reconstructions=False,
                save_latents=False,
            )
        # Assert
        assert 'num_images_processed' in result
        assert result['num_images_processed'] == 2


class TestPredictVideo:
    """Test the predict_video standalone function."""

    def test_predict_video_none_predictions_raises(self, video_file, tmp_path) -> None:
        # Arrange — trainer returns None
        mock_model = Mock()
        with patch('beast.inference.pl.Trainer') as MockTrainer:
            MockTrainer.return_value.predict.return_value = None
            # Act / Assert
            with pytest.raises(RuntimeError, match="trainer.predict\\(\\) returned None"):
                predict_video(
                    model=mock_model,
                    output_dir=tmp_path,
                    video_file=video_file,
                )

    def test_predict_video_returns_results(self, video_file, tmp_path) -> None:
        # Arrange — mock trainer to return a minimal prediction list
        mock_model = Mock()
        mock_predictions = [
            {
                'reconstructions': torch.rand(2, 3, 224, 224),
                'latents': torch.rand(2, 128),
            }
        ]
        with patch('beast.inference.pl.Trainer') as MockTrainer:
            MockTrainer.return_value.predict.return_value = mock_predictions
            # Act
            result = predict_video(
                model=mock_model,
                output_dir=tmp_path,
                video_file=video_file,
                save_reconstructions=False,
                save_latents=False,
            )
        # Assert
        assert 'frames_processed' in result
        assert result['frames_processed'] == 2


def _read_ply_colors(path: Path) -> np.ndarray:
    """Read per-point RGB colors (in [0, 1]) back from a saved PLY file."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.colors)


class TestFlattenMaskForPoints:
    """Test the _flatten_mask_for_points function."""

    def test_flatten_mask_for_points_spatial(self) -> None:
        # Arrange: [B, V, 1, H, W], first view foreground, second view background
        mask = torch.zeros(1, 2, 1, 2, 2)
        mask[:, 0] = 1.0

        # Act
        result = _flatten_mask_for_points(mask, sample_idx=0, num_points=8)

        # Assert
        assert result is not None
        assert result.shape == (8, 1)
        assert np.all(result[:4] == 1.0)
        assert np.all(result[4:] == 0.0)

    def test_flatten_mask_for_points_gaussian(self) -> None:
        # Arrange: [B, V, N], first view foreground, second view background
        mask = torch.zeros(1, 2, 4)
        mask[:, 0] = 1.0

        # Act
        result = _flatten_mask_for_points(mask, sample_idx=0, num_points=8)

        # Assert
        assert result is not None
        assert result.shape == (8, 1)
        assert np.all(result[:4] == 1.0)
        assert np.all(result[4:] == 0.0)

    def test_flatten_mask_for_points_none_when_mask_missing(self) -> None:
        assert _flatten_mask_for_points(None, sample_idx=0, num_points=8) is None

    def test_flatten_mask_for_points_none_on_size_mismatch(self) -> None:
        # Arrange
        mask = torch.ones(1, 2, 1, 2, 2)

        # Act / Assert
        assert _flatten_mask_for_points(mask, sample_idx=0, num_points=999) is None


class TestSaveGaussianPointclouds:
    """Test the save_gaussian_pointclouds function."""

    @pytest.fixture
    def temp_output_dir(self):
        temp_output = Path(tempfile.mkdtemp())
        yield temp_output
        shutil.rmtree(temp_output)

    def test_save_gaussian_pointclouds_whitens_background(self, temp_output_dir) -> None:
        # Arrange: 1 view, 2x2 pixel grid; left column foreground, right column background
        gs = Mock()
        result = {
            'gaussians': [gs],
            'pixelalign_xyz': torch.zeros(1, 1, 3, 2, 2),
            'image': torch.zeros(1, 1, 3, 2, 2),
            'target_mask': torch.tensor([[[[[1.0, 0.0], [1.0, 0.0]]]]]),
        }

        # Act
        saved = save_gaussian_pointclouds(result, temp_output_dir, batch_idx=0)

        # Assert
        assert len(saved) == 1
        rgb = _read_ply_colors(saved[0])
        # right column (flattened indices 1, 3) is background -> white
        assert np.allclose(rgb[[1, 3]], 1.0)
        assert np.allclose(rgb[[0, 2]], 0.0)

    def test_save_gaussian_pointclouds_no_mask_keeps_colors(self, temp_output_dir) -> None:
        # Arrange: no mask present -> colors pass through unchanged
        gs = Mock()
        result = {
            'gaussians': [gs],
            'pixelalign_xyz': torch.zeros(1, 1, 3, 2, 2),
            'image': torch.full((1, 1, 3, 2, 2), 0.5),
        }

        # Act
        saved = save_gaussian_pointclouds(result, temp_output_dir, batch_idx=0)

        # Assert
        assert len(saved) == 1
        rgb = _read_ply_colors(saved[0])
        assert np.allclose(rgb, 0.5, atol=1.0 / 255)


class TestSaveCameraPointcloudScene:
    """Test the save_camera_pointcloud_scene function."""

    @pytest.fixture
    def temp_output_dir(self):
        temp_output = Path(tempfile.mkdtemp())
        yield temp_output
        shutil.rmtree(temp_output)

    def test_save_camera_pointcloud_scene_writes_expected_geometries(
        self,
        temp_output_dir,
    ) -> None:
        # Arrange: 1 sample, 2 input views (idx 0, 1) + 1 disjoint target view (idx 2)
        gs = Mock()
        num_input_views = 2
        num_target_views = 1
        result = {
            'gaussians': [gs],
            'pixelalign_xyz': torch.zeros(1, num_input_views, 3, 2, 2),
            'image': torch.full((1, num_input_views, 3, 2, 2), 0.5),
            'c2w_input': torch.eye(4).unsqueeze(0).repeat(1, num_input_views, 1, 1),
            'c2w_target': torch.eye(4).unsqueeze(0).repeat(1, num_target_views, 1, 1),
            'input_indices': torch.tensor([[0, 1]]),
            'target_indices': torch.tensor([[2]]),
        }

        # Act
        saved = save_camera_pointcloud_scene(result, temp_output_dir, batch_idx=0)

        # Assert
        assert len(saved) == 1
        assert saved[0].name == 'scene_batch0000_sample00.glb'
        scene = trimesh.load(saved[0])
        # 1 point cloud + (num_input_views + num_target_views) disjoint camera frustums
        assert len(scene.geometry) == 1 + num_input_views + num_target_views

    def test_save_camera_pointcloud_scene_dedupes_overlapping_views(
        self,
        temp_output_dir,
    ) -> None:
        # Arrange: input and target index sets are identical (e.g. full-context
        # inference), so c2w_input and c2w_target hold the same 2 camera poses
        gs = Mock()
        num_views = 2
        result = {
            'gaussians': [gs],
            'pixelalign_xyz': torch.zeros(1, num_views, 3, 2, 2),
            'image': torch.full((1, num_views, 3, 2, 2), 0.5),
            'c2w_input': torch.eye(4).unsqueeze(0).repeat(1, num_views, 1, 1),
            'c2w_target': torch.eye(4).unsqueeze(0).repeat(1, num_views, 1, 1),
            'input_indices': torch.tensor([[0, 1]]),
            'target_indices': torch.tensor([[0, 1]]),
        }

        # Act
        saved = save_camera_pointcloud_scene(result, temp_output_dir, batch_idx=0)

        # Assert: only 2 unique cameras drawn, not 4
        assert len(saved) == 1
        scene = trimesh.load(saved[0])
        assert len(scene.geometry) == 1 + num_views

    def test_save_camera_pointcloud_scene_missing_c2w_returns_empty(self, temp_output_dir) -> None:
        # Arrange: no camera poses present
        gs = Mock()
        result = {
            'gaussians': [gs],
            'pixelalign_xyz': torch.zeros(1, 1, 3, 2, 2),
            'image': torch.zeros(1, 1, 3, 2, 2),
        }

        # Act
        saved = save_camera_pointcloud_scene(result, temp_output_dir, batch_idx=0)

        # Assert
        assert saved == []

    def test_save_camera_pointcloud_scene_overlays_gt_cameras_when_aligned(
        self,
        temp_output_dir,
    ) -> None:
        # Arrange: 2 predicted cameras with distinct poses; gt_c2w is a known
        # scale+rotation+translation of the predicted poses, so the alignment
        # should recover it and draw both predicted and GT frustums.
        from scipy.spatial.transform import Rotation

        from beast.models.model_utils.utils_icp import apply_similarity_transform_to_poses

        gs = Mock()
        num_views = 2
        pred_c2w = torch.eye(4).unsqueeze(0).repeat(num_views, 1, 1)
        pred_c2w[0, :3, :3] = torch.from_numpy(Rotation.from_euler('y', 20, degrees=True).as_matrix()).float()
        pred_c2w[0, :3, 3] = torch.tensor([0.0, 0.0, 0.0])
        pred_c2w[1, :3, :3] = torch.from_numpy(Rotation.from_euler('y', -20, degrees=True).as_matrix()).float()
        pred_c2w[1, :3, 3] = torch.tensor([1.0, 0.0, 0.0])

        known_transform = np.eye(4)
        known_transform[:3, :3] = 2.0 * Rotation.from_euler('z', 30, degrees=True).as_matrix()
        known_transform[:3, 3] = [5.0, -1.0, 2.0]
        gt_c2w_np = apply_similarity_transform_to_poses(known_transform, pred_c2w.numpy())

        result = {
            'gaussians': [gs],
            'pixelalign_xyz': torch.zeros(1, num_views, 3, 2, 2),
            'image': torch.full((1, num_views, 3, 2, 2), 0.5),
            'c2w_input': pred_c2w.unsqueeze(0),
            'c2w_target': pred_c2w.unsqueeze(0),
            'input_indices': torch.tensor([[0, 1]]),
            'target_indices': torch.tensor([[0, 1]]),
            'gt_c2w': torch.from_numpy(gt_c2w_np).float().unsqueeze(0),
        }

        # Act
        saved = save_camera_pointcloud_scene(result, temp_output_dir, batch_idx=0)

        # Assert: 1 point cloud + num_views predicted frustums + num_views GT frustums
        assert len(saved) == 1
        scene = trimesh.load(saved[0])
        assert len(scene.geometry) == 1 + num_views + num_views

    def test_save_camera_pointcloud_scene_skips_gt_overlay_without_view_index_overlap(
        self,
        temp_output_dir,
    ) -> None:
        # Arrange: predicted view indices (5, 6) have no overlap with gt_c2w's row
        # indices (0, 1), so alignment should be skipped and only predicted
        # geometry drawn, exactly as when gt_c2w is absent.
        gs = Mock()
        num_views = 2
        result = {
            'gaussians': [gs],
            'pixelalign_xyz': torch.zeros(1, num_views, 3, 2, 2),
            'image': torch.full((1, num_views, 3, 2, 2), 0.5),
            'c2w_input': torch.eye(4).unsqueeze(0).repeat(1, num_views, 1, 1),
            'c2w_target': torch.eye(4).unsqueeze(0).repeat(1, num_views, 1, 1),
            'input_indices': torch.tensor([[5, 6]]),
            'target_indices': torch.tensor([[5, 6]]),
            'gt_c2w': torch.eye(4).unsqueeze(0).repeat(1, num_views, 1, 1),
        }

        # Act
        saved = save_camera_pointcloud_scene(result, temp_output_dir, batch_idx=0)

        # Assert: only the predicted frustums are drawn, no GT frustums
        assert len(saved) == 1
        scene = trimesh.load(saved[0])
        assert len(scene.geometry) == 1 + num_views


class TestParseSceneName:
    """Test the _parse_scene_name function."""

    def test_parse_scene_name_splits_session_id_and_pair_idx(self) -> None:
        assert _parse_scene_name('sess_a_pair_000123') == ('sess_a', 123)

    def test_parse_scene_name_handles_underscores_in_session_id(self) -> None:
        assert _parse_scene_name('sub_01_2024_01_01_pair_000007') == ('sub_01_2024_01_01', 7)

    def test_parse_scene_name_raises_on_unexpected_format(self) -> None:
        with pytest.raises(ValueError):
            _parse_scene_name('not_a_valid_scene_name')


class TestBatchOutputPath:
    """Test the _batch_output_path function."""

    def test_batch_output_path_includes_session_and_split_subdirs(self) -> None:
        path = _batch_output_path(Path('/out'), 'frame_z', 'sess0', 'train', 3)
        assert path == Path('/out/frame_z/sess0/train/frame_z_batch0003.npz')

    def test_batch_output_path_sanitizes_session_id(self) -> None:
        path = _batch_output_path(Path('/out'), 'frame_z', 'a/b', 'train', 0)
        assert path == Path('/out/frame_z/a_b/train/frame_z_batch0000.npz')


class TestNumBatchesFor:
    """Test the _num_batches_for function."""

    def test_num_batches_for_divides_evenly(self) -> None:
        assert _num_batches_for(4, 2) == 2

    def test_num_batches_for_rounds_up(self) -> None:
        assert _num_batches_for(5, 2) == 3

    def test_num_batches_for_zero_records(self) -> None:
        assert _num_batches_for(0, 2) == 0


class _Rec:
    """Minimal stand-in for a `_PrecacheRecord`, exposing only `session_id`."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


class TestBatchSessionIds:
    """Test the _batch_session_ids function."""

    def test_batch_session_ids_single_session_batch(self) -> None:
        records = [_Rec('sess0'), _Rec('sess0'), _Rec('sess1'), _Rec('sess1')]
        assert _batch_session_ids(records, batch_idx=0, batch_size=2) == ['sess0']
        assert _batch_session_ids(records, batch_idx=1, batch_size=2) == ['sess1']

    def test_batch_session_ids_boundary_batch_spans_two_sessions(self) -> None:
        records = [_Rec('sess0'), _Rec('sess0'), _Rec('sess0'), _Rec('sess1'), _Rec('sess1')]
        # batch_idx 1 (rows 2,3) straddles sess0 -> sess1
        assert _batch_session_ids(records, batch_idx=1, batch_size=2) == ['sess0', 'sess1']

    def test_batch_session_ids_last_batch_may_be_partial(self) -> None:
        records = [_Rec('sess0')] * 3
        assert _batch_session_ids(records, batch_idx=1, batch_size=2) == ['sess0']


class TestResumeBatchStart:
    """Test the _resume_batch_start function."""

    @pytest.fixture
    def temp_output_dir(self):
        temp_output = Path(tempfile.mkdtemp())
        yield temp_output
        shutil.rmtree(temp_output)

    @staticmethod
    def _touch_batch(output_dir: Path, latent_type: str, session_id: str, split: str, idx: int):
        path = _batch_output_path(output_dir, latent_type, session_id, split, idx)
        _save_latent_batch_npz(
            path,
            z=np.zeros((1, 3, 4), dtype=np.float32),
            session_ids=[session_id],
            pair_idxs=[0],
            splits=[split],
            neural_trial_idx=[0],
            neural_bin_idx=[0],
            neural_interval_sec=np.zeros((1, 2), dtype=np.float64),
        )
        return path

    def test_resume_batch_start_zero_when_nothing_saved(self, temp_output_dir) -> None:
        sessions_by_batch = [['sess0'], ['sess0']]
        assert _resume_batch_start(temp_output_dir, ['frame_z'], sessions_by_batch, 'train') == 0

    def test_resume_batch_start_finds_contiguous_prefix(self, temp_output_dir) -> None:
        sessions_by_batch = [['sess0'], ['sess0'], ['sess0']]
        self._touch_batch(temp_output_dir, 'frame_z', 'sess0', 'train', 0)
        self._touch_batch(temp_output_dir, 'frame_z', 'sess0', 'train', 1)
        assert _resume_batch_start(temp_output_dir, ['frame_z'], sessions_by_batch, 'train') == 2

    def test_resume_batch_start_requires_every_latent_type(self, temp_output_dir) -> None:
        sessions_by_batch = [['sess0']]
        self._touch_batch(temp_output_dir, 'frame_z', 'sess0', 'train', 0)
        # dino_z batch 0 missing -> batch 0 is not complete
        assert _resume_batch_start(
            temp_output_dir, ['frame_z', 'dino_z'], sessions_by_batch, 'train',
        ) == 0

    def test_resume_batch_start_revalidates_boundary_batch(self, temp_output_dir) -> None:
        sessions_by_batch = [['sess0'], ['sess0']]
        self._touch_batch(temp_output_dir, 'frame_z', 'sess0', 'train', 0)
        path1 = _batch_output_path(temp_output_dir, 'frame_z', 'sess0', 'train', 1)
        path1.parent.mkdir(parents=True, exist_ok=True)
        path1.write_bytes(b'not a valid npz')
        assert _resume_batch_start(temp_output_dir, ['frame_z'], sessions_by_batch, 'train') == 1

    def test_resume_batch_start_treats_valid_combined_trials_as_all_done(
        self, temp_output_dir,
    ) -> None:
        # batches deleted after a prior successful combine -> only the trials npz remains
        session_dir = temp_output_dir / 'frame_z' / 'sess0'
        session_dir.mkdir(parents=True)
        np.savez(
            session_dir / 'frame_z_trials.npz',
            train_z_trials_time=np.zeros((1, 1, 3, 4), dtype=np.float32),
        )
        sessions_by_batch = [['sess0'], ['sess0']]
        assert _resume_batch_start(temp_output_dir, ['frame_z'], sessions_by_batch, 'train') == 2


class TestSaveLatentBatchNpz:
    """Test the _save_latent_batch_npz and _is_valid_batch_npz functions."""

    @pytest.fixture
    def temp_output_dir(self):
        temp_output = Path(tempfile.mkdtemp())
        yield temp_output
        shutil.rmtree(temp_output)

    @staticmethod
    def _save_kwargs(batch_size: int = 2):
        return {
            'z': np.arange(batch_size * 3 * 4, dtype=np.float32).reshape(batch_size, 3, 4),
            'session_ids': [f'sess{i}' for i in range(batch_size)],
            'pair_idxs': list(range(batch_size)),
            'splits': ['train'] * batch_size,
            'neural_trial_idx': list(range(batch_size)),
            'neural_bin_idx': [0] * batch_size,
            'neural_interval_sec': np.zeros((batch_size, 2), dtype=np.float64),
        }

    def test_save_latent_batch_npz_round_trips(self, temp_output_dir) -> None:
        path = temp_output_dir / 'frame_z' / 'train' / 'frame_z_batch0000.npz'
        kwargs = self._save_kwargs()

        _save_latent_batch_npz(path, **kwargs)

        assert path.is_file()
        assert list(temp_output_dir.rglob('*.tmp-*')) == []
        data = np.load(path, allow_pickle=True)
        np.testing.assert_array_equal(data['z'], kwargs['z'])
        assert list(data['session_id']) == kwargs['session_ids']
        assert _is_valid_batch_npz(path)

    def test_save_latent_batch_npz_leaves_no_partial_file_on_crash(self, temp_output_dir) -> None:
        path = temp_output_dir / 'frame_z' / 'train' / 'frame_z_batch0000.npz'

        with patch('numpy.savez', side_effect=OSError('disk full')):
            with pytest.raises(OSError):
                _save_latent_batch_npz(path, **self._save_kwargs())

        assert not path.is_file()

    def test_is_valid_batch_npz_false_for_missing_file(self, temp_output_dir) -> None:
        assert not _is_valid_batch_npz(temp_output_dir / 'does_not_exist.npz')

    def test_is_valid_batch_npz_false_for_missing_keys(self, temp_output_dir) -> None:
        path = temp_output_dir / 'incomplete.npz'
        np.savez(path, z=np.zeros((2, 3, 4), dtype=np.float32))
        assert not _is_valid_batch_npz(path)


class _FakeDataset:
    """Minimal stand-in for a SABLEDataset, supporting only what extract_sable_latents needs."""

    def __init__(self, records: list[_Rec], max_bin_idx: int | None) -> None:
        self._records = records
        self._max_bin_idx = max_bin_idx

    def __len__(self) -> int:
        return len(self._records)

    def max_neural_bin_idx(self) -> int | None:
        return self._max_bin_idx


@contextmanager
def _patched_extract_deps(dataset, batches: list[dict]):
    """Patch dataset/loader construction so `extract_sable_latents` serves `batches` in order.

    `_build_sable_dataset` always returns `dataset`; `_build_sable_inference_loader` returns
    `batches` sliced from `start_row // batch_size` onward, mimicking a `Subset`-skipped loader
    without needing a real `DataLoader`.
    """
    def fake_loader(config, include_splits=None, batch_size=None, start_row=0, dataset=None):
        start_idx = start_row // batch_size
        return dataset, batches[start_idx:]

    with (
        patch('beast.inference._build_sable_dataset', return_value=dataset),
        patch('beast.inference._build_sable_inference_loader', side_effect=fake_loader),
    ):
        yield


class TestExtractSableLatents:
    """Test the extract_sable_latents function."""

    @pytest.fixture
    def temp_output_dir(self):
        temp_output = Path(tempfile.mkdtemp())
        yield temp_output
        shutil.rmtree(temp_output)

    @staticmethod
    def _fake_batches(
        num_batches: int = 2, batch_size: int = 2, split_name: str = 'train',
    ) -> list[dict]:
        """Fake collated batches; batch `i`'s rows all belong to session `sess{i}`."""
        return [
            {
                'scene_name': [
                    f'sess{batch_idx}_pair_{row_idx:06d}' for row_idx in range(batch_size)
                ],
                'split': [split_name] * batch_size,
                'neural_trial_idx': torch.arange(
                    batch_idx * batch_size, batch_idx * batch_size + batch_size, dtype=torch.int64,
                ),
                'neural_bin_idx': torch.zeros(batch_size, dtype=torch.int64),
                'neural_interval_sec': torch.zeros(batch_size, 2, dtype=torch.float64),
            }
            for batch_idx in range(num_batches)
        ]

    @staticmethod
    def _records_for(batches: list[dict]) -> list[_Rec]:
        """Build the `_records` list implied by `batches`' `scene_name`s, in row order."""
        records = []
        for batch in batches:
            for scene_name in batch['scene_name']:
                session_id, _ = _parse_scene_name(scene_name)
                records.append(_Rec(session_id))
        return records

    @staticmethod
    def _fake_model(batch_size: int = 2, view_dim: int = 3, feat_dim: int = 4):
        model = Mock()
        model.parameters.return_value = iter([torch.zeros(1)])
        # get_model_outputs returns `vars(self(batch_dict))` in production -> a plain dict
        model.get_model_outputs.return_value = vars(SimpleNamespace(
            frame_z=torch.arange(batch_size * view_dim * feat_dim, dtype=torch.float32).reshape(
                batch_size, view_dim, feat_dim,
            ),
            dino_z=torch.zeros(batch_size, view_dim, feat_dim),
            combined_z=torch.ones(batch_size, view_dim, 2 * feat_dim),
            img_tokens=torch.full((batch_size, view_dim, feat_dim), 2.0),
        ))
        return model

    def test_extract_sable_latents_saves_requested_types_with_correct_filenames(
        self, temp_output_dir,
    ) -> None:
        model = self._fake_model()
        batches = self._fake_batches()
        dataset = _FakeDataset(self._records_for(batches), max_bin_idx=None)

        with _patched_extract_deps(dataset, batches):
            result = extract_sable_latents(
                config={'training': {}},
                model=model,
                output_dir=temp_output_dir,
                latent_types=['frame_z', 'img_tokens'],
                include_splits=['train'],
                resume=True,
                batch_size=2,
            )

        assert result['num_batches'] == 2
        assert result['num_batches_skipped'] == 0
        expected = temp_output_dir / 'frame_z' / 'sess0' / 'train' / 'frame_z_batch0000.npz'
        assert expected.is_file()
        # batch 1 is session sess1
        expected_sess1 = temp_output_dir / 'frame_z' / 'sess1' / 'train' / 'frame_z_batch0001.npz'
        assert expected_sess1.is_file()
        assert not (temp_output_dir / 'dino_z').exists()
        assert len(result['saved_files']['frame_z']) == 2
        assert len(result['saved_files']['img_tokens']) == 2
        # no neural-alignment metadata (max_bin_idx=None) -> combine step is skipped
        assert result['combined_trials_files'] == []

    def test_extract_sable_latents_splits_a_session_boundary_batch(self, temp_output_dir) -> None:
        # one batch of 2 rows straddling two sessions
        batches = [
            {
                'scene_name': ['sessA_pair_000000', 'sessB_pair_000000'],
                'split': ['train', 'train'],
                'neural_trial_idx': torch.tensor([0, 0], dtype=torch.int64),
                'neural_bin_idx': torch.zeros(2, dtype=torch.int64),
                'neural_interval_sec': torch.zeros(2, 2, dtype=torch.float64),
            },
        ]
        dataset = _FakeDataset(self._records_for(batches), max_bin_idx=None)
        model = self._fake_model()

        with _patched_extract_deps(dataset, batches):
            result = extract_sable_latents(
                config={'training': {}},
                model=model,
                output_dir=temp_output_dir,
                latent_types=['frame_z'],
                include_splits=['train'],
                resume=True,
                batch_size=2,
            )

        path_a = temp_output_dir / 'frame_z' / 'sessA' / 'train' / 'frame_z_batch0000.npz'
        path_b = temp_output_dir / 'frame_z' / 'sessB' / 'train' / 'frame_z_batch0000.npz'
        assert path_a.is_file()
        assert path_b.is_file()
        # each session's file only carries its own row
        assert list(np.load(path_a, allow_pickle=True)['session_id']) == ['sessA']
        assert list(np.load(path_b, allow_pickle=True)['session_id']) == ['sessB']
        assert sorted(result['saved_files']['frame_z']) == sorted([path_a, path_b])

    def test_extract_sable_latents_resume_skips_completed_batches(self, temp_output_dir) -> None:
        model = self._fake_model()
        batches = self._fake_batches()
        dataset = _FakeDataset(self._records_for(batches), max_bin_idx=None)

        # pre-create batch 0's frame_z output (session sess0)
        path = _batch_output_path(temp_output_dir, 'frame_z', 'sess0', 'train', 0)
        _save_latent_batch_npz(
            path,
            z=np.zeros((2, 3, 4), dtype=np.float32),
            session_ids=['sess0', 'sess0'],
            pair_idxs=[0, 1],
            splits=['train', 'train'],
            neural_trial_idx=[0, 1],
            neural_bin_idx=[0, 0],
            neural_interval_sec=np.zeros((2, 2), dtype=np.float64),
        )

        with _patched_extract_deps(dataset, batches):
            result = extract_sable_latents(
                config={'training': {}},
                model=model,
                output_dir=temp_output_dir,
                latent_types=['frame_z'],
                include_splits=['train'],
                resume=True,
                batch_size=2,
            )

        assert result['num_batches_skipped'] == 1
        assert result['num_batches'] == 1
        assert model.get_model_outputs.call_count == 1

    def test_extract_sable_latents_no_resume_overwrites(self, temp_output_dir) -> None:
        model = self._fake_model()
        batches = self._fake_batches()
        dataset = _FakeDataset(self._records_for(batches), max_bin_idx=None)
        path = _batch_output_path(temp_output_dir, 'frame_z', 'sess0', 'train', 0)
        _save_latent_batch_npz(
            path,
            z=np.zeros((2, 3, 4), dtype=np.float32),
            session_ids=['sess0', 'sess0'],
            pair_idxs=[0, 1],
            splits=['train', 'train'],
            neural_trial_idx=[0, 1],
            neural_bin_idx=[0, 0],
            neural_interval_sec=np.zeros((2, 2), dtype=np.float64),
        )

        with _patched_extract_deps(dataset, batches):
            result = extract_sable_latents(
                config={'training': {}},
                model=model,
                output_dir=temp_output_dir,
                latent_types=['frame_z'],
                include_splits=['train'],
                resume=False,
                batch_size=2,
            )

        assert result['num_batches_skipped'] == 0
        assert result['num_batches'] == 2
        assert model.get_model_outputs.call_count == 2

    def test_extract_sable_latents_invalid_latent_type_raises(self, temp_output_dir) -> None:
        with pytest.raises(ValueError):
            extract_sable_latents(
                config={'training': {}},
                model=self._fake_model(),
                output_dir=temp_output_dir,
                latent_types=['not_a_real_latent'],
            )

    def test_extract_sable_latents_resume_skips_already_combined_session(
        self, temp_output_dir,
    ) -> None:
        # a valid combined trials npz already on disk for sess0 (its batches may already be
        # deleted by a prior successful combine)
        session_dir = temp_output_dir / 'frame_z' / 'sess0'
        session_dir.mkdir(parents=True)
        trials_path = session_dir / 'frame_z_trials.npz'
        np.savez(trials_path, train_z_trials_time=np.zeros((1, 1, 3, 4), dtype=np.float32))

        batches = self._fake_batches(num_batches=1, batch_size=2)  # sess0 only
        dataset = _FakeDataset(self._records_for(batches), max_bin_idx=1)
        model = self._fake_model()

        with _patched_extract_deps(dataset, batches):
            result = extract_sable_latents(
                config={'training': {}},
                model=model,
                output_dir=temp_output_dir,
                latent_types=['frame_z'],
                include_splits=['train'],
                resume=True,
                batch_size=2,
            )

        model.get_model_outputs.assert_not_called()
        assert result['num_batches'] == 0
        assert result['combined_trials_files'] == [trials_path]

    def test_extract_sable_latents_combines_and_deletes_batches_per_session(
        self, temp_output_dir,
    ) -> None:
        model = self._fake_model()
        batches = self._fake_batches(num_batches=1, batch_size=2)
        dataset = _FakeDataset(self._records_for(batches), max_bin_idx=1)

        with _patched_extract_deps(dataset, batches):
            result = extract_sable_latents(
                config={'training': {}},
                model=model,
                output_dir=temp_output_dir,
                latent_types=['frame_z'],
                include_splits=['train'],
                resume=False,
                batch_size=2,
            )

        combined_path = temp_output_dir / 'frame_z' / 'sess0' / 'frame_z_trials.npz'
        assert result['combined_trials_files'] == [combined_path]
        assert combined_path.is_file()
        data = np.load(combined_path, allow_pickle=True)
        assert 'train_z_trials_time' in data.files
        # 2 rows, distinct neural_trial_idx, single bin each -> 2 complete trials
        assert data['train_z_trials_time'].shape[0] == 2
        assert list(data['session_id']) == ['sess0', 'sess0']
        # batch file deleted (and its now-empty split dir) after a successful combine
        assert not _batch_output_path(temp_output_dir, 'frame_z', 'sess0', 'train', 0).is_file()
        assert not (temp_output_dir / 'frame_z' / 'sess0' / 'train').exists()

    def test_extract_sable_latents_never_combines_or_deletes_img_tokens(
        self, temp_output_dir,
    ) -> None:
        model = self._fake_model()
        batches = self._fake_batches(num_batches=1, batch_size=2)
        dataset = _FakeDataset(self._records_for(batches), max_bin_idx=1)

        with _patched_extract_deps(dataset, batches):
            result = extract_sable_latents(
                config={'training': {}},
                model=model,
                output_dir=temp_output_dir,
                latent_types=['frame_z', 'img_tokens'],
                include_splits=['train'],
                resume=False,
                batch_size=2,
            )

        assert not (temp_output_dir / 'img_tokens' / 'sess0' / 'img_tokens_trials.npz').is_file()
        assert result['combined_trials_files'] == [
            temp_output_dir / 'frame_z' / 'sess0' / 'frame_z_trials.npz',
        ]
        # img_tokens batch file untouched
        assert _batch_output_path(temp_output_dir, 'img_tokens', 'sess0', 'train', 0).is_file()
