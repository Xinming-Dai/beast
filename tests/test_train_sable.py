from beast.data.samplers import ResumableRandomSampler
from beast.train_sable import SamplerStateCallback


class TestSamplerStateCallback:
    """Test the class SamplerStateCallback."""

    def test_on_save_checkpoint_writes_sampler_state(self):
        # Arrange
        sampler = ResumableRandomSampler(range(20), seed=3)
        sampler.load_state_dict({'epoch': 2, 'pos': 5, 'seed': 3})
        callback = SamplerStateCallback(sampler)
        checkpoint = {}

        # Act
        callback.on_save_checkpoint(None, None, checkpoint)

        # Assert
        assert checkpoint['sampler_state'] == sampler.state_dict()

    def test_on_load_checkpoint_restores_sampler_state(self):
        # Arrange
        sampler = ResumableRandomSampler(range(20), seed=3)
        callback = SamplerStateCallback(sampler)
        checkpoint = {'sampler_state': {'epoch': 2, 'pos': 5, 'seed': 3}}

        # Act
        callback.on_load_checkpoint(None, None, checkpoint)

        # Assert
        assert sampler.state_dict() == checkpoint['sampler_state']

    def test_on_load_checkpoint_noop_without_sampler_state(self):
        # Arrange
        sampler = ResumableRandomSampler(range(20), seed=3)
        original_state = sampler.state_dict()
        callback = SamplerStateCallback(sampler)

        # Act
        callback.on_load_checkpoint(None, None, {})

        # Assert
        assert sampler.state_dict() == original_state
