import torch
from diffusers.utils.torch_utils import randn_tensor

@torch.no_grad()
def decode_standalone(self,
                      latents,
                      latent_num_frames,
                      latent_height,
                      latent_width,
                      device,
                      decode_timestep,
                      batch_size, decode_noise_scale, output_type, generator,
                      dtype):
    if len(latents.shape) == 3:
        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
    latents = self._denormalize_latents(
        latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
    )
    latents = latents.to(dtype)

    if not self.vae.config.timestep_conditioning:
        timestep = None
    else:
        noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
        if not isinstance(decode_timestep, list):
            decode_timestep = [decode_timestep] * batch_size
        if decode_noise_scale is None:
            decode_noise_scale = decode_timestep
        elif not isinstance(decode_noise_scale, list):
            decode_noise_scale = [decode_noise_scale] * batch_size

        timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
        decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                             :, None, None, None, None
                             ]
        latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

    video = self.vae.decode(latents, timestep, return_dict=False)[0]
    video = self.video_processor.postprocess_video(video, output_type=output_type)
    return video


@torch.no_grad()
def decode_standalone_wan(self, latents, output_type):
    latents = latents.to(self.vae.dtype)
    latents_mean = (
        torch.tensor(self.vae.config.latents_mean)
        .view(1, self.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    video = self.vae.decode(latents, return_dict=False)[0]
    video = self.video_processor.postprocess_video(video, output_type=output_type)
    return video
