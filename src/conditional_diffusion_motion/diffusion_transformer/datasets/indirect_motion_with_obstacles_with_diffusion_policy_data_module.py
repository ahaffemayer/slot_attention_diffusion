import os
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision.transforms import ToTensor
from PIL import Image
import pickle
device = "cuda" if torch.cuda.is_available() else "cpu"


class IndirectMotionDatasetWithObstaclesWithShelfDiffusionPolicy(Dataset):
    def __init__(self, data_dir: Path, device="cuda"):
        super().__init__()
        self.device = device
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "generated_scenes" / "shelf_for_diffusion"

        raw_file = self.data_dir / "trajectories" / "trajectories_data_shelf.json"
        cache_dir = self.data_dir / "processed_datasets"
        cache_dir.mkdir(exist_ok=True)

        file_samples = cache_dir / "indirect_motion_with_obstacles_cache_samples_with_shelf.npy"
        file_q0 = cache_dir / "indirect_motion_with_obstacles_cache_q0_with_shelf.npy"
        file_goal = cache_dir / "indirect_motion_with_obstacles_cache_goal_with_shelf.npy"
        file_scenes = cache_dir / "indirect_motion_with_obstacles_cache_scene_ids_with_shelf.npy"
        file_images = cache_dir / "indirect_motion_with_obstacles_cache_images_with_shelf.npy"

        if all(f.exists() for f in [file_samples, file_q0, file_goal, file_scenes]):
            self.samples = torch.tensor(np.load(file_samples), dtype=torch.float32)
            self.q0 = torch.tensor(np.load(file_q0), dtype=torch.float32)
            self.goal = torch.tensor(np.load(file_goal), dtype=torch.float32)
            self.scene_ids = np.load(file_scenes)
        else:
            with open(raw_file, "r") as f:
                data = json.load(f)

            self.samples = torch.tensor(np.asarray([res["trajectory"][1:] for res in data]), dtype=torch.float32)
            self.q0 = torch.tensor(np.asarray([res["trajectory"][0] for res in data]), dtype=torch.float32)
            self.goal = torch.tensor(np.asarray([res["target"] for res in data]), dtype=torch.float32)
            self.scene_ids = np.asarray(
                [int(res["scene"].split("_")[1]) for res in data],  # e.g., "scene_0021" -> 21
                dtype=np.int64,
            )

            np.save(file_samples, self.samples.numpy())
            np.save(file_q0, self.q0.numpy())
            np.save(file_goal, self.goal.numpy())
            np.save(file_scenes, self.scene_ids)

        if file_images.exists():
            self.image_tensors = np.load(file_images, mmap_mode="r")  # will be (N, 3, H, W)
            with open(cache_dir / "scene_id_to_index.pkl", "rb") as f:
                self.scene_id_to_index = pickle.load(f)
            
        else:
            print("Caching images...")

            images_name_list = [f for f in os.listdir(self.image_dir) if f.endswith(".png")]

            scene_ids_in_images = sorted([int(fname.split("_")[1].split(".")[0]) for fname in images_name_list])
            self.scene_id_to_index = {sid: i for i, sid in enumerate(scene_ids_in_images)}

            # Load in sorted order
            image_tensors = []
            for sid in scene_ids_in_images:
                image_path = self.image_dir / f"scene_{sid:04d}.png"
                image = Image.open(image_path).convert("RGB")
                image_tensor = ToTensor()(image).numpy()
                image_tensors.append(image_tensor)

            self.image_tensors = np.stack(image_tensors)  # shape: (N, 3, H, W)
            np.save(file_images, self.image_tensors)
            with open(cache_dir / "scene_id_to_index.pkl", "wb") as f:
                pickle.dump(self.scene_id_to_index, f)


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
        scene_id = int(self.scene_ids[idx])
        image_idx = self.scene_id_to_index[scene_id]
        image_tensor = self.image_tensors[image_idx]  # shape: (3, H, W)

        return self.samples[idx], {
            "q0": self.q0[idx],
            "goal": self.goal[idx],
            "image": image_tensor,
        }


class IndirectDataModuleWithObstacleWithShelfDiffusionPolicy(pl.LightningDataModule):
    """Defines the data used for training the diffusion process."""

    def __init__(self, data_dir: Path = None, device="cuda") -> None:
        super().__init__()

        self.dataset = IndirectMotionDatasetWithObstaclesWithShelfDiffusionPolicy(data_dir=data_dir, device=device)

        _, self.seq_length, self.configuration_size = self.dataset.samples.shape

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=32,
            num_workers=0,
            shuffle=True,  # optionally enable shuffling here
        )
