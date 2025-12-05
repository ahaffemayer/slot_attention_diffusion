import os
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


device = "cuda" if torch.cuda.is_available() else "cpu"

class IndirectMotionDatasetWithObstaclesWithDrawerCspace(Dataset):
    def __init__(self, data_dir: Path, device="cuda"):
        """
        Dataset that returns robot motion data with associated cspace features from scene images.

        Args:
            data_dir (str or Path): Directory containing data.
            device (str): Device for tensor conversion.
        """
        super().__init__()
        self.device = device
        self.data_dir = Path(data_dir)

        self.image_dir = (
            self.data_dir / "generated_scenes" / "drawer_for_diffusion"
        )

        raw_file = self.data_dir / "trajectories" / "trajectories_data_drawer.json"
        cache_dir = self.data_dir / "processed_datasets"
        cache_dir.mkdir(exist_ok=True)

        file_samples = (
            cache_dir
            / "indirect_motion_with_obstacles_cache_samples_with_drawer.npy"
        )
        file_q0 = (
            cache_dir
            / "indirect_motion_with_obstacles_cache_q0_with_drawer.npy"
        )
        file_goal = (
            cache_dir
            / "indirect_motion_with_obstacles_cache_goal_with_drawer.npy"
        )
        file_scenes = (
            cache_dir
            / "indirect_motion_with_obstacles_cache_scene_ids_with_drawer.npy"
        )
        file_cspaces = (
            cache_dir
            / "indirect_motion_with_obstacles_cache_with_drawer_cspace.npy"
        )

        # Load or parse data
        if all(f.exists() for f in [file_samples, file_q0, file_goal, file_scenes]):
            self.samples = torch.tensor(np.load(file_samples), dtype=torch.float32)
            self.q0 = torch.tensor(np.load(file_q0), dtype=torch.float32)
            self.goal = torch.tensor(np.load(file_goal), dtype=torch.float32)
            self.scene_ids = np.load(file_scenes)
            self.full_trajs = None  # not needed at runtime
        else:
            with open(raw_file, "r") as f:
                data = json.load(f)

            self.samples = torch.tensor(
                np.asarray([res["trajectory"][1:] for res in data]), dtype=torch.float32
            )
            self.q0 = torch.tensor(
                np.asarray([res["trajectory"][0] for res in data]), dtype=torch.float32
            )
            self.goal = torch.tensor(
                np.asarray([res["target"] for res in data]), dtype=torch.float32
            )
            self.scene_ids = np.asarray(
                [
                    int(res["scene"].split("_")[1])
                    for res in data  # e.g., "scene_0021" -> 21
                ],
                dtype=np.int64,
            )
            np.save(file_samples, self.samples.numpy())
            np.save(file_q0, self.q0.numpy())
            np.save(file_goal, self.goal.numpy())
            np.save(file_scenes, self.scene_ids)

        if os.path.exists(file_cspaces):
            self.cspace = torch.tensor(np.load(file_cspaces), dtype=torch.float32)
        else:
            raise FileNotFoundError(
                f"File {file_cspaces} not found. Please generate the C-space data first."
            )

        # Shuffle
        idx = torch.randperm(self.samples.shape[0])
        self.samples = self.samples[idx]
        self.q0 = self.q0[idx]
        self.goal = self.goal[idx]
        self.scene_ids = self.scene_ids[idx.numpy()]

        # Stats
        self.samples_mean = self.samples.mean(dim=(0, 1), keepdim=True)
        self.samples_std = self.samples.std(dim=(0, 1), keepdim=True)
        self.q0_mean = self.q0.mean(dim=0, keepdim=True)
        self.q0_std = self.q0.std(dim=0, keepdim=True)
        self.goal_mean = self.goal.mean(dim=0, keepdim=True)
        self.goal_std = self.goal.std(dim=0, keepdim=True)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        scene_id = int(self.scene_ids[idx])  # make sure it's an integer index
        cspace_feat = self.cspace[scene_id]
        return self.samples[idx], {
            "q0": self.q0[idx],
            "goal": self.goal[idx],
            "cspace": cspace_feat,
        }


class IndirectDataModuleWithObstacleWithDrawerCspace(pl.LightningDataModule):
    """Defines the data used for training the diffusion process."""

    def __init__(self, data_dir:Path = None) -> None:
        super().__init__()

        self.dataset = IndirectMotionDatasetWithObstaclesWithDrawerCspace(
            data_dir=data_dir, device=device
        )

        _, self.seq_length, self.configuration_size = self.dataset.samples.shape

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=256,
            num_workers=4,
            shuffle=True,  # optionally enable shuffling here
        )
