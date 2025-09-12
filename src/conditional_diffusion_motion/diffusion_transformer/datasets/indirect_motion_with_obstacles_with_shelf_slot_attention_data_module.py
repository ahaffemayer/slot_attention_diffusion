from collections import defaultdict
import os
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
import pickle
from conditional_diffusion_motion.slot_attention.slot_attention_wrapper import SlotEncoderWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"


class IndirectMotionDatasetWithObstaclesWithShelfSlotAttention(Dataset):
    def __init__(self, slot_encoder, data_dir: Path, device="cuda", target_total_trajectories = None):
        """
        Dataset that returns robot motion data with associated slot features from scene images.

        Args:
            slot_encoder (SlotEncoderWrapper): Callable that returns slot features from a PIL image.
            data_dir (str or Path): Directory containing data.
            device (str): Device for tensor conversion.
        """
        super().__init__()
        self.device = device
        self.slot_encoder = slot_encoder
        self.data_dir = Path(data_dir)

        self.image_dir = self.data_dir / "generated_scenes" / "shelf_for_diffusion"

        raw_file = self.data_dir / "trajectories" / "trajectories_data_shelf.json"
        cache_dir = self.data_dir / "processed_datasets"
        cache_dir.mkdir(exist_ok=True)

        file_samples = cache_dir / "indirect_motion_with_obstacles_cache_samples_with_shelf.npy"
        file_q0 = cache_dir / "indirect_motion_with_obstacles_cache_q0_with_shelf.npy"
        file_goal = cache_dir / "indirect_motion_with_obstacles_cache_goal_with_shel.npy"
        file_scenes = cache_dir / "indirect_motion_with_obstacles_cache_scene_ids_with_shelf.npy"
        file_slots = cache_dir / "indirect_motion_with_obstacles_cache_slots_with_shelf_slot_attention.npy"

        # Load or parse data
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

        # Optional: precompute slots
        self.slots = None
        if file_slots.exists():
            self.slots = torch.tensor(np.load(file_slots), dtype=torch.float32)
            with open(cache_dir / "scene_id_to_index.pkl", "rb") as f:
                self.scene_id_to_index = pickle.load(f)
        else:
            print("Computing slots...")
            images_name_list = [f for f in os.listdir(self.image_dir) if f.endswith(".png")]
            scene_ids_in_images = sorted([int(fname.split("_")[1].split(".")[0]) for fname in images_name_list])
            self.scene_id_to_index = {sid: i for i, sid in enumerate(scene_ids_in_images)}
            self.slots = []
            for sid in scene_ids_in_images:
                image_path = self.image_dir / f"scene_{sid:04d}.png"
                image = Image.open(image_path).convert("RGB")
                slots = self.slot_encoder(image)["slots"]
                self.slots.append(slots.unsqueeze(0))  # shape (1, num_slots, slot_dim)
            self.slots = torch.cat(self.slots, dim=0)  # shape (N, num_slots, slot_dim)
            np.save(file_slots, self.slots.numpy())
            
            with open(cache_dir / "scene_id_to_index.pkl", "wb") as f:
                pickle.dump(self.scene_id_to_index, f)


        if target_total_trajectories is not None:
            print(f"Subsampling to {target_total_trajectories} trajectories...")
            unique_scene_ids = np.unique(self.scene_ids)
            samples_per_scene = target_total_trajectories // len(unique_scene_ids)

            # Group indices by scene
            scene_to_indices = defaultdict(list)
            for idx, sid in enumerate(self.scene_ids):
                scene_to_indices[sid].append(idx)

            # Collect subsampled indices
            selected_indices = []
            for sid in unique_scene_ids:
                indices = scene_to_indices[sid]
                if len(indices) >= samples_per_scene:
                    selected = np.random.choice(indices, samples_per_scene, replace=False)
                else:
                    selected = np.random.choice(indices, samples_per_scene, replace=True)
                selected_indices.extend(selected)

            # Shuffle once after subsampling
            selected_indices = np.random.permutation(selected_indices)

            # Filter the data tensors
            self.samples = self.samples[selected_indices]
            self.q0 = self.q0[selected_indices]
            self.goal = self.goal[selected_indices]
            self.scene_ids = self.scene_ids[selected_indices]
        else:
            # Shuffle only if we're not subsampling
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
        if self.slots is not None:
            scene_id = int(self.scene_ids[idx])
            image_idx = self.scene_id_to_index[scene_id]
            slot_feat = self.slots[image_idx]
        else:
            scene = self.scene_ids[idx]
            image_path = self.image_dir / f"{scene}.png"
            image = Image.open(image_path).convert("RGB")
            slot_feat = self.slot_encoder(image)["slots"]

        return self.samples[idx], {
            "q0": self.q0[idx],
            "goal": self.goal[idx],
            "slots": slot_feat,
        }


class IndirectDataModuleWithObstacleWithShelfSlotAttention(pl.LightningDataModule):
    """Defines the data used for training the diffusion process."""

    def __init__(self, data_dir: Path = None, slot_attention_model_dir: Path = None, target_total_trajectories = None) -> None:
        super().__init__()

        slot_encoder = SlotEncoderWrapper(
            device=device,
            image_size=(128, 128),
            num_slots=6,
            model_dir=slot_attention_model_dir,
        )

        self.dataset = IndirectMotionDatasetWithObstaclesWithShelfSlotAttention(
            slot_encoder=slot_encoder, data_dir=data_dir, device=device, target_total_trajectories=target_total_trajectories
        )

        _, self.seq_length, self.configuration_size = self.dataset.samples.shape

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=256,
            num_workers=4,
            shuffle=True,  # optionally enable shuffling here
        )
