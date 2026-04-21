# based on https://github.com/a-r-r-o-w/finetrainers/blob/main/finetrainers/dataset.py and
# https://github.com/Lightricks/LTX-Video-Trainer/blob/main/src/ltxv_trainer/datasets.py

import re
import random
import torch
import os.path
import pandas as pd
from pathlib import Path
from typing import  Sized
from torch.utils.data import Dataset, Sampler
from typing import List, Union

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


class PrecomputedDataset(Dataset):
    def __init__(self, data_root: str) -> None:
        super().__init__()

        self.data_root = Path(data_root)

        # If the given path is the dataset root, use the precomputed sub-directory.
        if (self.data_root / PRECOMPUTED_DIR_NAME).exists():
            self.data_root = self.data_root / PRECOMPUTED_DIR_NAME

        self.latents_path = self.data_root / PRECOMPUTED_LATENTS_DIR_NAME
        self.conditions_path = self.data_root / PRECOMPUTED_CONDITIONS_DIR_NAME

        # Verify that the required directories exist
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {self.data_root}")

        if not self.latents_path.exists():
            raise FileNotFoundError(
                f"Precomputed latents directory does not exist: {self.latents_path}. "
                f"Make sure you've run the preprocessing step.",
            )

        if not self.conditions_path.exists():
            raise FileNotFoundError(
                f"Precomputed conditions directory does not exist: {self.conditions_path}. "
                f"Make sure you've run the preprocessing step.",
            )

        # Check if directories are empty
        if not list(self.latents_path.iterdir()):
            raise ValueError(f"Precomputed latents directory is empty: {self.latents_path}")

        if not list(self.conditions_path.iterdir()):
            raise ValueError(f"Precomputed conditions directory is empty: {self.conditions_path}")

        self.latent_conditions = sorted([p.name for p in self.latents_path.iterdir()])
        self.text_conditions = sorted([p.name for p in self.conditions_path.iterdir()])

        assert len(self.latent_conditions) == len(self.text_conditions), "Number of captions and videos do not match"

    def __len__(self) -> int:
        return len(self.latent_conditions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        conditions = {}
        latent_path = self.latents_path / self.latent_conditions[index]
        condition_path = self.conditions_path / self.text_conditions[index]
        conditions["latent_conditions"] = torch.load(latent_path, map_location="cpu", weights_only=True)
        conditions["text_conditions"] = torch.load(condition_path, map_location="cpu", weights_only=True)
        return conditions

class PrecomputedDatasetSampleable(Dataset):
    """
    Precomputed Dataset
    Sampleablity refers to the concept that __getitem__ returns a dictionary of distributions (e.g. in the VAE space)
    """

    def __init__(self,
                 data_root1: Union[str, List[str]],
                 data_root2: Union[str, List[str]],
                 shuffle_init: bool,
                 load_inverted_latents: bool = False,
                 text_regularization_prob: float = 1.0) -> None:
        """

        :param data_root1: path(s) to the initial distribution (precomputed) to be used in flow matching
        :param data_root2: path(s) to the target distribution (precomputed) to be used in flow matching
        :param shuffle_init: Weather to shuffle the initial distribution (for training without inherent optimal couplings)
        :param load_inverted_latents: Weather to load the inverted latents of the target (for training with target inversion)
        :param text_regularization_prob: Text drop-out probability to apply during the training
        """
        super().__init__()

        self.load_inverted_latents = load_inverted_latents

        if type(data_root1) == str:
            data_root1 = [data_root1]

        if type(data_root2) == str:
            data_root2 = [data_root2]

        self.data_root1 = [Path(p) for p in data_root1]
        self.dist_params_path_p_init = [p / PRECOMPUTED_LATENTS_DIR_NAME for p in self.data_root1]
        self.conditions_path = [p / PRECOMPUTED_CONDITIONS_DIR_NAME for p in self.data_root1]

        self.data_root2 = [Path(p) for p in data_root2]
        self.dist_params_path_p_data = [p / PRECOMPUTED_LATENTS_DIR_NAME for p in self.data_root2]
        if self.load_inverted_latents:
            self.inverted_latents_path = [p / PRECOMPUTED_INVERTED_LATENTS_DIR_NAME for p in self.data_root2]

        self.text_regularization_prob = text_regularization_prob

        # Verify that the required directories exist
        for p in self.data_root1:
            if not p.exists():
                raise FileNotFoundError(f"Data root directory does not exist: {p}")

        for p in self.data_root2:
            if not p.exists():
                raise FileNotFoundError(f"Data root directory does not exist: {p}")

        for p in self.dist_params_path_p_init:
            if not p.exists():
                raise FileNotFoundError(
                    f"Precomputed latents directory does not exist: {p}. "
                    f"Make sure you've run the preprocessing step.",
                )

        for p in self.dist_params_path_p_data:
            if not p.exists():
                raise FileNotFoundError(
                    f"Precomputed latents directory does not exist: {p}. "
                    f"Make sure you've run the preprocessing step.",
                )

        for p in self.conditions_path:
            if not p.exists():
                raise FileNotFoundError(
                    f"Precomputed conditions directory does not exist: {p}. "
                    f"Make sure you've run the preprocessing step.",
                )

        # Check if directories are empty
        for p in self.dist_params_path_p_init:
            if not list(p.iterdir()):
                raise ValueError(f"Precomputed latents directory is empty: {p}")

        for p in self.dist_params_path_p_data:
            if not list(p.iterdir()):
                raise ValueError(f"Precomputed latents directory is empty: {p}")

        for p in self.conditions_path:
            if not list(p.iterdir()):
                raise ValueError(f"Precomputed conditions directory is empty: {p}")

        # self.dist_params_p_init = sorted([p.name for p in self.dist_params_path_p_init.iterdir()])

        self.dist_params_p_init = []
        for p_init in self.dist_params_path_p_init:
            if shuffle_init:
                self.dist_params_p_init.extend(
                    sorted([p for p in random.shuffle(p_init.iterdir())])
                )
            else:
                self.dist_params_p_init.extend(
                    sorted([p for p in p_init.iterdir()])
                )

        self.dist_params_p_data = []
        for p_data in self.dist_params_path_p_data:
            self.dist_params_p_data.extend(
                sorted([p for p in p_data.iterdir()])
            )

        self.text_conditions = []
        for condition_path in self.conditions_path:
            self.text_conditions.extend(
                sorted([p for p in condition_path.iterdir()])
            )

        if self.load_inverted_latents:
            self.p_data_inverted_latents = []
            for inverted_latent_path in self.inverted_latents_path:
                self.p_data_inverted_latents.extend(
                    sorted([p for p in inverted_latent_path.iterdir()])
                )

        assert len(self.dist_params_p_init) == len(self.text_conditions) and len(self.dist_params_p_data) == len(
            self.dist_params_p_init), "Number of captions, init or data paths do not match"

        self.length_groups = {}
        for idx, path in enumerate(data_root1):
            # match = re.search(r'\.precomputed_(\d+)FRAMES', path)
            match = re.search(r'\.precomputed_.*?(\d+)FRAMES', path)
            length = int(match.group(1))
            start_ = 0 if idx == 0 else len(list(self.dist_params_path_p_init[idx - 1].iterdir()))
            self.length_groups[length] = torch.tensor(
                list(range(start_, start_ + len(list(self.dist_params_path_p_init[idx].iterdir())))))

    def __len__(self) -> int:
        return len(self.dist_params_p_init)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        conditions = {}
        dist_params_path_p_init = self.dist_params_p_init[index]
        dist_params_path_p_data = self.dist_params_p_data[index]
        condition_path = self.text_conditions[index]

        conditions["dist_params_p_init"] = torch.load(dist_params_path_p_init, map_location="cpu", weights_only=True)
        conditions["dist_params_p_data"] = torch.load(dist_params_path_p_data, map_location="cpu", weights_only=True)

        if self.load_inverted_latents:
            p_data_inverted_path = self.p_data_inverted_latents[index]
            conditions["p_data_inverted"] = torch.load(p_data_inverted_path, map_location="cpu", weights_only=True)

        if self.text_regularization_prob == 1.0:
            conditions["text_conditions"] = torch.load(condition_path, map_location="cpu", weights_only=True)
        else:
            random_random = random.random()
            if random_random < self.text_regularization_prob:
                conditions["text_conditions"] = torch.load(condition_path, map_location="cpu", weights_only=True)
            else:
                conditions["text_conditions"] = torch.load(f"{self.data_root1}/empty_text.pt", map_location="cpu",
                                                           weights_only=True)
        return conditions

class PrecomputedDatasetSampleableVal(PrecomputedDatasetSampleable):
    """
    Precomputed Dataset for Validation
    Sampleablity refers to the concept that __getitem__ returns a dictionary of distributions (e.g. in the VAE space)
    """

    def __init__(self, data_root1: list[str], data_root2: list[str],
                 dataset_file: str, video_column: str, caption_column: str,
                 load_inverted_latents: bool = False) -> None:
        """

        :param data_root1: Data roots of the initial distribution (used in flow matching)
        :param data_root2:  Data roots of the target distribution (used in flow matching)
        :param dataset_file: Path to the dataset file (e.g. .csv, .json)
        :param video_column: Video column in dataframe
        :param caption_column: Prompt column in dataframe
        :param load_inverted_latents: Weather to load inverted latents of the target
        """
        super().__init__(data_root1, data_root2, False,
                         load_inverted_latents=load_inverted_latents,
                         text_regularization_prob=1.0)

        if dataset_file.endswith(".csv"):
            df = pd.read_csv(dataset_file)
        elif dataset_file.endswith(".json"):
            df = pd.read_json(dataset_file)
        else:
            raise ValueError(
                "Expected `--dataset_file` to be a path to a CSV or JSON file.",
            )

        self.video_paths = df[video_column]
        self.video_names = [os.path.basename(path).split(".mp4")[0] for path in self.video_paths]
        self.prompts = df[caption_column]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        conditions = super().__getitem__(index)
        video_name = str(self.dist_params_p_init[index]).split("latent_first_")[-1].split(".pt")[0]
        conditions["video_name"] = video_name
        try:
            conditions["prompts"] = self.prompts[self.video_names.index(video_name)]
        except:
            conditions["prompts"] = ""

        return conditions

class FixedLengthBatchSampler(Sampler):
    """
    Batch Sampler for fixed-length precomputed videos.
    """

    def __init__(self, dataset: Sized,
                 batch_size: int,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 random_state=None):
        """

        :param dataset: A Sized Dataset object.
        :param batch_size: Batch Size
        :param shuffle: Whether to shuffle the batch
        :param drop_last: Whether to drop the last batch
        :param random_state: Random state for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = random_state

        self.batches = []
        for length, idxs in self.dataset.length_groups.items():
            if self.shuffle:
                generator = torch.Generator()
                generator.manual_seed(self.seed + length)
                rnd_idxs = torch.randperm(len(idxs), generator=generator).tolist()
                idxs = idxs[rnd_idxs]

            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i:i + self.batch_size]
                if self.drop_last:
                    if len(batch) == self.batch_size:
                        self.batches.append(batch)
                else:
                    self.batches.append(batch)

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            self.batches = [self.batches[i].tolist() for i in torch.randperm(len(self.batches), generator=g)]

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        yield from self.batches
