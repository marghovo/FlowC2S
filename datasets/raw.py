# based on https://github.com/a-r-r-o-w/finetrainers/blob/main/finetrainers/dataset.py and
# https://github.com/Lightricks/LTX-Video-Trainer/blob/main/src/ltxv_trainer/datasets.py

import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import torchvision.transforms as TT
from torch.utils.data import Dataset, Sampler
from typing import Optional, List, Tuple, Union
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

try:
    import decord
except ImportError:
    raise ImportError(
        "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
    )

PRECOMPUTED_DIR_NAME = ".precomputed"
PRECOMPUTED_CONDITIONS_DIR_NAME = "conditions"
PRECOMPUTED_LATENTS_DIR_NAME = "latents"
PRECOMPUTED_INVERTED_LATENTS_DIR_NAME = "inverted_latents"
PRECOMPUTED_DECODEDS_DIR_NAME = "decoded_videos"


# copied from a-r-r-o-w/finetrainers/dataset.py
class VideoDataset(Dataset):
    def __init__(
            self,
            dataset_path: str,
            video_dataset_root: str,
            resolution_buckets: List[Tuple[int, int, int]],
            caption_column: str = "text",
            video_column: str = "video",
            video_reshape_mode: str = "center",
            fps: int = 25,
            max_num_frames: int = 257,
            skip_frames_start: int = 0,
            skip_frames_end: int = 0,
            cache_dir: Optional[str] = None,
            id_token: Optional[str] = None,
            shuffle_df: bool = True,
            validation: bool = False,
            only_caption: bool = False,
    ) -> None:
        super().__init__()

        decord.bridge.set_bridge("torch")

        self.dataset_path = dataset_path
        self.video_dataset_root = video_dataset_root
        self.caption_column = caption_column
        self.video_column = video_column
        self.video_reshape_mode = video_reshape_mode
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.cache_dir = cache_dir
        self.id_token = id_token or ""
        self.resolution_buckets = resolution_buckets
        self.validation = validation
        self.only_caption = only_caption
        min_resolution = min(bucket[0] for bucket in self.resolution_buckets)

        if "csv" in self.dataset_path:
            if self.validation:
                self.df = pd.read_csv(self.dataset_path)[:5]
            else:
                self.df = pd.read_csv(self.dataset_path)
        elif "parquet" in self.dataset_path:
            self.df = pd.read_parquet(self.dataset_path)
        else:
            raise NotImplementedError("Dataset extension not supported!")

        if not self.validation:
            if "frame" in self.df:
                self.df = self.df[self.df["frame"] >= min_resolution]
                self.df = self.df.reset_index(drop=True)

        if self.validation:
            self.df = self.df.sample(frac=1.0)

        if shuffle_df:
            self.df = self.df.sample(
                frac=1.0)  # random shuffle, I think this is needed as we run the experiments with batch size of 1

        self.df = self.df.reset_index(drop=True)

        self.instance_prompts = self.df[self.caption_column]
        if not self.only_caption:
            self.instance_video_paths = self.df[self.video_column]
        self.num_instance_videos = len(self.instance_prompts)

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, index):
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            # TODO if there is time improve this, I can directly have the number of frames without
            # TODO loading the video form the csv in __init__
            return index
        if not self.only_caption:
            video = self._preprocess_video(index)
            output_dict = {
                "instance_prompt": self.id_token + self.instance_prompts.iloc[index],
                "instance_video": video,
                "instance_path": f"{self.video_dataset_root}/{self.instance_video_paths.iloc[index]}",
                "video_metadata": {
                    "num_frames": video.shape[0],
                    "height": video.shape[2],
                    "width": video.shape[3],
                },
            }
        else:
            output_dict = {
                "instance_prompt": self.id_token + self.instance_prompts.iloc[index],
            }
        return output_dict

    # copied from a-r-r-o-w/finetrainers/dataset.py
    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    # copied and modified from a-r-r-o-w/finetrainers/dataset.py
    def _preprocess_video(self, index) -> torch.Tensor:
        filename = Path(f"{self.video_dataset_root}/{self.instance_video_paths.iloc[index]}")
        video_reader = decord.VideoReader(uri=filename.as_posix())
        video_num_frames = len(video_reader)

        nearest_frame_bucket = min(
            [bucket for bucket in self.resolution_buckets if bucket[0] <= video_num_frames],
            key=lambda x: abs(x[0] - min(video_num_frames, self.max_num_frames)),
            default=(1, self.resolution_buckets[0][1], self.resolution_buckets[0][1]),
        )[0]

        frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

        frames = video_reader.get_batch(frame_indices)

        frames = frames[:nearest_frame_bucket].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()

        nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
        frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
        # frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)
        frames = (frames_resized / 127.5) - 1.0

        if frames.ndim == 3:
            frames = frames[None]

        return frames

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolution_buckets, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


class BucketSampler(Sampler):
    r"""
    PyTorch Sampler that groups 3D data by height, width and frames.

    Args:
        data_source (`VideoDataset`):
            A PyTorch dataset object that is an instance of `VideoDataset`.
        batch_size (`int`, defaults to `8`):
            The batch size to use for training.
        shuffle (`bool`, defaults to `True`):
            Whether or not to shuffle the data in each batch before dispatching to dataloader.
        drop_last (`bool`, defaults to `False`):
            Whether or not to drop incomplete buckets of data after completely iterating over all data
            in the dataset. If set to True, only batches that have `batch_size` number of entries will
            be yielded. If set to False, it is guaranteed that all data in the dataset will be processed
            and batches that do not have `batch_size` number of entries will also be yielded.
    """

    def __init__(
            self, data_source: VideoDataset, batch_size: int = 8,
            shuffle: bool = True, drop_last: bool = False) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = {resolution: [] for resolution in data_source.resolution_buckets}
        self._raised_warning_for_drop_last = False

    def __len__(self):
        if self.drop_last and not self._raised_warning_for_drop_last:
            self._raised_warning_for_drop_last = True
            logger.warning(
                "Calculating the length for bucket sampler is not possible when `drop_last` is set to True. This may cause problems when setting the number of epochs used for training."
            )
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for index, data in enumerate(self.data_source):
            video_metadata = data["video_metadata"]
            f, h, w = video_metadata["num_frames"], video_metadata["height"], video_metadata["width"]

            self.buckets[(f, h, w)].append(data)
            if len(self.buckets[(f, h, w)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(f, h, w)])
                yield self.buckets[(f, h, w)]
                del self.buckets[(f, h, w)]
                self.buckets[(f, h, w)] = []

        if self.drop_last:
            return

        for fhw, bucket in list(self.buckets.items()):
            if len(bucket) == 0:
                continue
            if self.shuffle:
                random.shuffle(bucket)
                yield bucket
                del self.buckets[fhw]
                self.buckets[fhw] = []