import torch
from pytorch_lightning.cli import LightningCLI
from conditional_diffusion_motion.diffusion_transformer.datasets.indirect_motion_with_obstacles_without_cond import IndirectDataModuleWithObstacleWithoutCond
from conditional_diffusion_motion.diffusion_transformer.models.wrapper_models.model_without_cond import ModelWithoutCond


if __name__ == "__main__":
    cli = LightningCLI(
        ModelWithoutCond,
        datamodule_class=IndirectDataModuleWithObstacleWithoutCond,
        trainer_defaults={
            "log_every_n_steps": 1,
            "max_epochs": 3,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
        },
    )