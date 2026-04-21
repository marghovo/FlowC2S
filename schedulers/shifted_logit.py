import torch
import random


class ShiftedLogitNormalTimestepSampler:
    """
    Samples timesteps from a shifted logit-normal distribution,
    where the shift is determined by the sequence length.
    """

    def __init__(self, std: float = 1.0, per_token: bool = False, conditioning_p: float = 0.1):
        super().__init__()
        self.std = std
        self.per_token = per_token
        self.conditioning_p = conditioning_p

    def sample(self, batch_size: int, seq_length: int,
               latent_num_frames: int,
               device: torch.device = None) -> torch.Tensor:
        """Sample timesteps for a batch from a shifted logit-normal distribution.

        Args:
            batch_size: Number of timesteps to sample
            seq_length: Length of the sequence being processed, used to determine the shift
            device: Device to place the samples on

        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by seq_length
        """
        shift = self._get_shift_for_sequence_length(seq_length)
        if self.per_token and self.conditioning_p and random.random() < self.conditioning_p:
            normal_samples = torch.randn((batch_size, seq_length), device=device) * self.std + shift
        else:
            normal_samples = torch.randn((batch_size, 1), device=device) * self.std + shift
        timesteps = torch.sigmoid(normal_samples)
        return timesteps

    def sample_for(self, batch: torch.Tensor, latent_num_frames: int) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)

        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by the sequence length
            of the input batch

        Raises:
            ValueError: If the input batch does not have 3 dimensions
        """
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, seq_length, latent_num_frames=latent_num_frames, device=batch.device)

    @staticmethod
    def _get_shift_for_sequence_length(
            seq_length: int,
            min_tokens: int = 1024,
            max_tokens: int = 4096,
            min_shift: float = 0.95,
            max_shift: float = 2.05,
    ) -> float:
        # Calculate the shift value for a given sequence length using linear interpolation
        # between min_shift and max_shift based on sequence length.
        m = (max_shift - min_shift) / (max_tokens - min_tokens)  # Calculate slope
        b = min_shift - m * min_tokens  # Calculate y-intercept
        shift = m * seq_length + b  # Apply linear equation y = mx + b
        return shift
