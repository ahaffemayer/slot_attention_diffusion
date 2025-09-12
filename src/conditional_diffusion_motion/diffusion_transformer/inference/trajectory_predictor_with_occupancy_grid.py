import torch
import numpy as np

from conditional_diffusion_motion.diffusion_transformer.models.wrapper_models.model_occupancy_grid import (
    ModelOccupancyGrid,
)


class TrajectoryPredictorWithOccupancyGrid:
    def __init__(
        self,
        diffusion_ckpt_path,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load diffusion model
        self.model = ModelOccupancyGrid.load_from_checkpoint(diffusion_ckpt_path).to(self.device)
        self.model.eval()

    def predict(
        self,
        initial_pose,  # shape: (nq,), torch.Tensor or np.ndarray
        goal,  # shape: (3,), torch.Tensor or np.ndarray
        occupancy_grid: torch.Tensor = None,  # Optional occupancy grid tensor
        seq_length=15,
        configuration_size=7,
        bs=1,
        diffusion_steps: int = None,
        projection_fn=None,  # Optional projection function
    ):
        # Ensure inputs are torch tensors on correct device
        if isinstance(initial_pose, np.ndarray):
            initial_pose = torch.tensor(initial_pose, dtype=torch.float32)
        if isinstance(goal, np.ndarray):
            goal = torch.tensor(goal, dtype=torch.float32)

        initial_pose = initial_pose.to(self.device)
        goal = goal.to(self.device)

        print(f"Occupancy grid shape: {occupancy_grid.shape}, dtype: {occupancy_grid.dtype}")
        # Prepare conditioning
        cond = {
            "q0": initial_pose.unsqueeze(0).repeat(bs, 1),  # (bs, nq)
            "goal": goal.unsqueeze(0).repeat(bs, 1),  # (bs, nq)
            "occupancy_grid": occupancy_grid.unsqueeze(0).repeat(
                bs, 1, 1, 1
            ),  # (bs, [grid_size_x, grid_size_y, grid_size_z])
        }

        # Sample trajectory
        with torch.no_grad():
            sample = self.model.sample(
                bs=bs,
                seq_length=seq_length,
                configuration_size=configuration_size,
                cond=cond,
                diffusion_steps=diffusion_steps,
                projection_fn=projection_fn,
            )

        return sample.cpu().numpy()  # shape: (seq_length, configuration_size)
