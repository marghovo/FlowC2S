import torch


# Adapted from https://github.com/Lightricks/LTX-Video-Trainer
def _normalize_latents(
        latents: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
) -> torch.Tensor:
    """Normalizes latents using mean and standard deviation across the channel dimension."""
    mean = mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std = std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - mean) / std
    return latents


def _normalize_latents_wan(latents: torch.Tensor,
                           mean: torch.Tensor,
                           std: torch.Tensor,
                           z_dim) -> torch.Tensor:
    mean = torch.tensor(mean).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    std = 1.0 / torch.tensor(std).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - mean) * std

    return latents


# Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._denormalize_latents
def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    # Denormalize latents across the channel dimension [B, C, F, H, W]
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents * latents_std / scaling_factor + latents_mean
    return latents


# From https://github.com/Lightricks/LTX-Video-Trainer
def _pack_latents(latents: torch.Tensor, spatial_patch_size: int = 1, temporal_patch_size: int = 1) -> torch.Tensor:
    """Reshapes latents [B,C,F,H,W] into patches and flattens to sequence form [B,L,D].

    Args:
        latents: Input latent tensor
        spatial_patch_size: Size of spatial patches
        temporal_patch_size: Size of temporal patches

    Returns:
        Flattened sequence of patches
    """
    b, c, f, h, w = latents.shape
    latents = latents.reshape(
        b,
        -1,
        f // temporal_patch_size,
        temporal_patch_size,
        h // spatial_patch_size,
        spatial_patch_size,
        w // spatial_patch_size,
        spatial_patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


# From diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._unpack_latents
def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def prepare_video_ids(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
    device: torch.device = None,
) -> torch.Tensor:
    latent_sample_coords = torch.meshgrid(
        torch.arange(0, num_frames, patch_size_t, device=device),
        torch.arange(0, height, patch_size, device=device),
        torch.arange(0, width, patch_size, device=device),
        indexing="ij",
    )
    latent_sample_coords = torch.stack(latent_sample_coords, dim=0)
    latent_coords = latent_sample_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    latent_coords = latent_coords.reshape(batch_size, -1, num_frames * height * width)

    return latent_coords

def scale_video_ids(
    video_ids: torch.Tensor,
    scale_factor: int = 32,
    scale_factor_t: int = 8,
    frame_index: int = 0,
) -> torch.Tensor:
    scaled_latent_coords = (
        video_ids
        * torch.tensor([scale_factor_t, scale_factor, scale_factor], device=video_ids.device)[None, :, None]
    )
    scaled_latent_coords[:, 0] = (scaled_latent_coords[:, 0] + 1 - scale_factor_t).clamp(min=0)
    scaled_latent_coords[:, 0] += frame_index

    return scaled_latent_coords
