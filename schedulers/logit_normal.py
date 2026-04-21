import torch
import random

class WANLogitNormalTimestepSampler:
    """
    Samples timesteps from a shifted logit-normal distribution,
    where the shift is determined by the sequence length.
    """

    def __init__(self, per_token:bool = False, conditioning_p: float = 0.1):
        super().__init__()
        self.per_token = per_token
        self.conditioning_p = conditioning_p

    def sample(self, batch_shape: tuple,
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

        batch_size, _, latent_num_frames, height, width = batch_shape
        if self.per_token and self.conditioning_p and random.random() < self.conditioning_p:
            normal_samples = torch.randn((batch_size, 1, latent_num_frames, height, width), device=device)
        else:
            normal_samples = torch.randn((batch_size, 1, 1, 1, 1), device=device)
        timesteps = torch.sigmoid(normal_samples)
        return timesteps

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
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
        if batch.ndim != 5:
            raise ValueError(f"Batch should have 5 dimensions, got {batch.ndim}")
        return self.sample(batch.shape, device=batch.device)