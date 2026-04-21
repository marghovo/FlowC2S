import torch
import random
from schedulers.shifted_logit import ShiftedLogitNormalTimestepSampler

class FrameWindowTimeStepSampler(ShiftedLogitNormalTimestepSampler):
    def __init__(self, std: float = 1.0, per_token:bool = False,
                 conditioning_p: float = 0.1, latent_window_size: int = 2):
        super().__init__(std, per_token, conditioning_p)
        self.latent_window_size = latent_window_size

    def sample(self, batch_size: int, seq_length: int,
               latent_num_frames: int,
               device: torch.device = None) -> torch.Tensor:
        """
        Samples sigmas such that ...
        :param batch_size:
        :param seq_length:
        :param device:
        :return:
        """
        shift = self._get_shift_for_sequence_length(seq_length)
        if self.per_token and self.conditioning_p and random.random() < self.conditioning_p:
            assert latent_num_frames % self.latent_window_size == 0

            num_tokens_per_frame = seq_length // latent_num_frames
            if self.latent_window_size == latent_num_frames:
                normal_samples = torch.randn((batch_size, latent_num_frames, ), device=device)
                # sort in increasing order
                normal_samples, _ = torch.sort(normal_samples, dim=1, descending=False)
                # to ensure all tokens in a frame have the same sigma
                normal_samples = torch.repeat_interleave(normal_samples, num_tokens_per_frame, dim=1)
                # shift and scale
                normal_samples = normal_samples * self.std + shift
            else:
                # per window of frames will have the same sigmas
                k = latent_num_frames // self.latent_window_size
                normal_samples = torch.randn((batch_size, k,), device=device)
                # sort in increasing order
                normal_samples, _ = torch.sort(normal_samples, dim=1, descending=False)
                # repeat so as each window of frames has the same sigma
                normal_samples = normal_samples.repeat((1, self.latent_window_size))
                normal_samples = torch.repeat_interleave(normal_samples, num_tokens_per_frame, dim=1)
                normal_samples = normal_samples * self.std + shift
        else:
            normal_samples = torch.randn((batch_size, 1), device=device) * self.std + shift
        timesteps = torch.sigmoid(normal_samples)
        return timesteps