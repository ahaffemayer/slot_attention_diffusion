import torch
from pytorch_lightning.cli import LightningCLI
from conditional_diffusion_motion.diffusion_transformer.datasets.indirect_motion_with_obstacles_with_diffusion_policy_data_module import IndirectDataModuleWithObstacleWithShelfDiffusionPolicy
from conditional_diffusion_motion.diffusion_transformer.models.wrapper_models.model_diffusion_policy import ModelDiffusionPolicy


if __name__ == "__main__":
    cli = LightningCLI(
        ModelDiffusionPolicy,
        datamodule_class=IndirectDataModuleWithObstacleWithShelfDiffusionPolicy,
        trainer_defaults={
            "log_every_n_steps": 1,
            "max_epochs": 3,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
        },
    )