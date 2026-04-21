import torch
from PIL import Image
from typing import List
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_tensor, to_pil_image

def make_video_grid(videos: List[List[Image.Image]], nrow: int) -> List[Image.Image]:
    """
    Create a grid of videos
    :param videos: Videos to concatenate
    :param nrow: number of rows
    :return:
    """

    video_tensors = []
    for video in videos:
        video_tensor = [to_tensor(frame)[None] for frame in video]
        video_tensor = torch.cat(video_tensor, dim=0)
        video_tensors.append(video_tensor)

    video_tensors = [video[None] for video in video_tensors]
    all_videos = torch.cat(video_tensors, dim=0)
    grid = []
    for frame_idx in range(all_videos.shape[1]):
        frame_grid = make_grid(all_videos[:, frame_idx], nrow=nrow)
        grid.append(frame_grid[None])
    grid_tensor = torch.cat(grid, dim=0)
    grid = [to_pil_image(frame) for frame in grid_tensor]
    return grid