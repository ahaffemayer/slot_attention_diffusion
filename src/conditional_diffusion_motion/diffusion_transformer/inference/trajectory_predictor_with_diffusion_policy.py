import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from conditional_diffusion_motion.diffusion_transformer.models.wrapper_models.model_diffusion_policy import (
    ModelDiffusionPolicy,
)


class TrajectoryPredictorWithDiffusionPolicy:
    def __init__(
        self,
        diffusion_ckpt_path,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load diffusion model
        self.model = ModelDiffusionPolicy.load_from_checkpoint(diffusion_ckpt_path).to(self.device)
        self.model.eval()
    def predict(
        self,
        initial_pose,
        goal,
        img,
        seq_length=15,
        configuration_size=7,
        bs=1,
        diffusion_steps: int = None,
        projection_fn=None,
    ):
        # Ensure tensors
        if isinstance(initial_pose, np.ndarray):
            initial_pose = torch.tensor(initial_pose, dtype=torch.float32)
        if isinstance(goal, np.ndarray):
            goal = torch.tensor(goal, dtype=torch.float32)

        # Move state to device
        initial_pose = initial_pose.to(self.device)
        goal = goal.to(self.device)

        # Make sure img is a 3-channel PIL image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if isinstance(img, Image.Image):
            img = img.convert("RGB")

        # Preprocess to CHW float32 in [0,1]
        preproc = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        x = preproc(img)                               # (3, 128, 128) on CPU
        x = x.to(self.device, dtype=torch.float32, non_blocking=True)

        # Build conditioning on the correct device
        cond = {
            "q0":   initial_pose.unsqueeze(0).repeat(bs, 1),      # (bs, nq)
            "goal": goal.unsqueeze(0).repeat(bs, 1),              # (bs, 3)
            "image":  x.unsqueeze(0).repeat(bs, 1, 1, 1),           # (bs, 3, H, W)  NOTE key name
        }

        with torch.no_grad():
            sample = self.model.sample(
                bs=bs,
                seq_length=seq_length,
                configuration_size=configuration_size,
                cond=cond,
                diffusion_steps=diffusion_steps,
                projection_fn=projection_fn,
            )

        return sample.detach().cpu().numpy()
