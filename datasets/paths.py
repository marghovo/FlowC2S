import torch
import os.path
import pandas as pd
from typing import Optional


class DatasetVal:
    """
    Validation dataset for OpenVid dataset
    """

    def __init__(self, dataset_file: str, video_column: str, caption_column: str,
                 preprocess: bool = False, sample_n: Optional[int] = None) -> None:
        """

        :param dataset_file:  Path to the dataset file (e.g. .csv, .json)
        :param video_column: Video column in dataframe
        :param caption_column: Prompt column in dataframe
        :param preprocess: Weather to preprocess the data (not implemented)
        :param sample_n: The number of samples to use
        """
        if dataset_file.endswith(".csv"):
            df = pd.read_csv(dataset_file)
        elif dataset_file.endswith(".json"):
            df = pd.read_json(dataset_file)
        else:
            raise ValueError(
                "Expected `--dataset_file` to be a path to a CSV or JSON file.",
            )

        if sample_n is not None:
            # df = df.sample(n=sample_n, replace=False, random_state=991221)
            # df = df.sample(n=sample_n, replace=False, random_state=231)
            df = df[df["frame"] >= 246]
            df = df[1:]
            df.reset_index(drop=True, inplace=True)

        self.video_paths = df[video_column]
        self.video_names = [os.path.basename(path).split(".mp4")[0] for path in self.video_paths]
        self.prompts = df[caption_column]
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        conditions = {}
        video_name = self.video_paths.iloc[index]
        conditions["video_paths"] = video_name
        conditions["prompts"] = self.prompts.iloc[index]

        if self.preprocess:
            pass

        return conditions


class DatasetNuscencesVal:
    """
    Validation dataset for Nuscences dataset
    """

    def __init__(self, dataset_file: str, video_column: str, caption_column: str,
                 preprocess: bool = False, sample_n: Optional[int] = None) -> None:

        """

        :param dataset_file:  Path to the dataset file (e.g. .csv, .json)
        :param video_column: Video column in dataframe
        :param caption_column: Prompt column in dataframe
        :param preprocess: Weather to preprocess the data (not implemented)
        :param sample_n: The number of samples to use
        """

        if dataset_file.endswith(".csv"):
            df = pd.read_csv(dataset_file)
        elif dataset_file.endswith(".json"):
            df = pd.read_json(dataset_file)
        else:
            raise ValueError(
                "Expected `--dataset_file` to be a path to a CSV or JSON file.",
            )

        if sample_n is not None:
            df = df[df["frame"] >= 246]
            df = df[1:]
            df.reset_index(drop=True, inplace=True)

        self.video_frame_paths = df[video_column]
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.video_frame_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        conditions = {}
        video_frame_paths_list = self.video_frame_paths.iloc[index]
        conditions["video_paths"] = video_frame_paths_list
        conditions["prompts"] = ''

        if self.preprocess:
            raise NotImplementedError("Preprocessing not implemented for Nuscences dataset.")

        return conditions
