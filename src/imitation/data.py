"""Dataset utilities for Push-T."""
from __future__ import annotations

import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

import robomimic.utils.file_utils as FileUtils
from robomimic import DATASET_REGISTRY
# the dataset registry can be found at robomimic/__init__.py

PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"

ZARR_RELATIVE_PATH = Path("pusht") / "pusht_cchi_v7_replay.zarr"

TASK_PATHS = {
    "pusht": Path("pusht") / "pusht_cchi_v7_replay.zarr",
    "robomimic/lift": Path("robomimic") / "low_dim_lift.hdf5",
    "robomimic/can": Path("robomimic") / "low_dim_can.hdf5",
    "robomimic/square": Path("robomimic") / "low_dim_square.hdf5",
    "robomimic/transport": Path("robomimic") / "low_dim_transport.hdf5",
}

TASK_NAMES = list(TASK_PATHS.keys())


@dataclass(frozen=True)
class Normalizer:
    """Feature-wise normalizer for states and actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.maximum(std, eps)

    @classmethod
    def from_data(cls, states: np.ndarray, actions: np.ndarray) -> "Normalizer":
        state_mean = states.mean(axis=0)
        state_std = cls._safe_std(states.std(axis=0))
        action_mean = actions.mean(axis=0)
        action_std = cls._safe_std(actions.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean


def download_dataset(task_name: str, dataset_dir: Path) -> Path:
    """Download a dataset if needed.

    Returns the path to the extracted Zarr/HDF5 dataset.
    """
    assert task_name in TASK_NAMES, (
        f"Unknown task {task_name}; available tasks are {TASK_NAMES}"
    )

    dataset_path = dataset_dir / TASK_PATHS[task_name]
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.exists():
        return dataset_path

    if task_name == "pusht":
        zip_path = dataset_dir / "pusht.zip"
        if not zip_path.exists():
            print("Downloading PushT dataset...")
            urllib.request.urlretrieve(PUSHT_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)
        return dataset_path

    # Robomimic dataset
    robomimic_task = task_name.split("/")[1]
    dataset_type = "ph"  # proficient human
    hdf5_type = "low_dim"
    url = DATASET_REGISTRY[robomimic_task][dataset_type][hdf5_type]["url"]
    filename = url.split("/")[-1]
    FileUtils.download_url(url=url, download_dir=str(dataset_path.parent))
    os.rename(dataset_dir / TASK_PATHS[task_name].parent / filename, dataset_path)
    return dataset_path


def load_pusht_zarr(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = zarr.open(zarr_path, mode="r")
    states = np.asarray(root["data"]["state"][:], dtype=np.float32)
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    return states, actions, episode_ends


def build_valid_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    starts = np.concatenate(([0], episode_ends[:-1]))
    indices: list[int] = []
    for start, end in zip(starts, episode_ends, strict=True):
        last_start = end - chunk_size
        if last_start < start:
            continue
        indices.extend(range(start, last_start + 1))
    return np.asarray(indices, dtype=np.int64)


class PushtChunkDataset(Dataset):
    """Dataset of (state, action_chunk) pairs using a sliding window."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        chunk_size: int,
        normalizer: Normalizer | None = None,
    ) -> None:
        self.states = states
        self.actions = actions
        self.chunk_size = chunk_size
        self.normalizer = normalizer
        self.indices = build_valid_indices(episode_ends, chunk_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = int(self.indices[idx])
        state = self.states[t]
        action_chunk = self.actions[t : t + self.chunk_size]

        if self.normalizer is not None:
            state = self.normalizer.normalize_state(state)
            action_chunk = self.normalizer.normalize_action(action_chunk)

        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(action_chunk).float(),
        )
