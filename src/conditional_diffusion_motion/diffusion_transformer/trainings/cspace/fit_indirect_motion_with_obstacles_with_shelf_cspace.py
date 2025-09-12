import torch
from pytorch_lightning.cli import LightningCLI
from conditional_diffusion_motion.diffusion_transformer.datasets.indirect_motion_with_obstacles_with_cspace_data_module import IndirectDataModuleWithObstacleWithShelfCspace
from conditional_diffusion_motion.diffusion_transformer.models.wrapper_models.model_cspace import ModelCSpace


if __name__ == "__main__":
    cli = LightningCLI(
        ModelCSpace,
        datamodule_class=IndirectDataModuleWithObstacleWithShelfCspace,
        trainer_defaults={
            "log_every_n_steps": 1,
            "max_epochs": 3,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
        },
    )