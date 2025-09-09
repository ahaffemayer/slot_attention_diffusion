import torch
import numpy as np
from conditional_diffusion_motion.diffusion_transformer.models.wrapper_models.model_without_cond import ModelWithoutCond

class TrajectoryPredictorWithoutCond:
    def __init__(
        self,
        diffusion_ckpt_path,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load diffusion model
        self.model = ModelWithoutCond.load_from_checkpoint(diffusion_ckpt_path).to(self.device)
        self.model.eval()


    def predict(
        self,
        seq_length=15,
        configuration_size=7,
        bs=1,
        projection_fn=None,  # Optional projection function
        diffusion_steps: int | None = None,  # Optional number of diffusion steps
    ):
        # Sample trajectory
        with torch.no_grad():
            sample = self.model.sample(
                bs=bs,
                seq_length=seq_length,
                configuration_size=configuration_size,
                projection_fn=projection_fn,
                diffusion_steps=diffusion_steps,
            )

        return sample.cpu().numpy()  # shape: (seq_length, configuration_size)
