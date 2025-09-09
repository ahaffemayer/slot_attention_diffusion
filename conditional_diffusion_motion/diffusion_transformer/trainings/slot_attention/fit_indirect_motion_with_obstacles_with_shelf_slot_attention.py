import torch
from pytorch_lightning.cli import LightningCLI
from conditional_diffusion_motion.diffusion_transformer.datasets.indirect_motion_with_obstacles_with_shelf_slot_attention_data_module import IndirectDataModuleWithObstacleWithShelfSlotAttention
from conditional_diffusion_motion.diffusion_transformer.models.wrapper_models.model_slot_attention import ModelSlotAttention


if __name__ == "__main__":
    cli = LightningCLI(
        ModelSlotAttention,
        datamodule_class=IndirectDataModuleWithObstacleWithShelfSlotAttention,
        trainer_defaults={
            "log_every_n_steps": 1,
            "max_epochs": 3,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
        },
    )
    
    
    