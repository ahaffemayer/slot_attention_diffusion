import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


device = "cuda" if torch.cuda.is_available() else "cpu"

class IndirectMotionDatasetWithObstaclesWithoutCond(Dataset):
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

        raw_file = self.data_dir / "trajectories" / "trajectories_data_shelf.json"
        cache_dir = self.data_dir / "processed_datasets"
        cache_dir.mkdir(exist_ok=True)

        file_samples = (
            cache_dir
            / "indirect_motion_with_obstacles_cache_samples_with_shelf.npy"
        )
    

        # Load or parse data
        if all(f.exists() for f in [file_samples]):
            self.samples = torch.tensor(np.load(file_samples), dtype=torch.float32)
        else:
            with open(raw_file, "r") as f:
                data = json.load(f)

            self.samples = torch.tensor(
                np.asarray([res["trajectory"][1:] for res in data]), dtype=torch.float32
            )
            np.save(file_samples, self.samples.numpy())


        # Shuffle
        idx = torch.randperm(self.samples.shape[0])
        self.samples = self.samples[idx]


        # Stats
        self.samples_mean = self.samples.mean(dim=(0, 1), keepdim=True)
        self.samples_std = self.samples.std(dim=(0, 1), keepdim=True)


    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx]


class IndirectDataModuleWithObstacleWithoutCond(pl.LightningDataModule):
    """Defines the data used for training the diffusion process."""

    def __init__(self, data_dir:Path = None) -> None:
        super().__init__()

        self.dataset = IndirectMotionDatasetWithObstaclesWithoutCond(
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
