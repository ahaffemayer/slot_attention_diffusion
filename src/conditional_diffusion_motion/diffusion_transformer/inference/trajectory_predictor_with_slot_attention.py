import torch
import numpy as np
from conditional_diffusion_motion.slot_attention.slot_attention_wrapper import SlotEncoderWrapper
from conditional_diffusion_motion.diffusion_transformer.models.wrapper_models.model_slot_attention import ModelSlotAttention

class TrajectoryPredictorWithSlotAttention:
    def __init__(
        self,
        diffusion_ckpt_path,
        slot_attention_ckpt_path,
        num_slots=6,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load diffusion model
        self.model = ModelSlotAttention.load_from_checkpoint(diffusion_ckpt_path).to(self.device)
        self.model.eval()

        # Load slot encoder
        self.slot_encoder = SlotEncoderWrapper(model_dir=str(slot_attention_ckpt_path), num_slots=num_slots)

    def predict(
        self,
        image,  # RGB image as PIL.Image or np.ndarray (H, W, 3)
        initial_pose,  # shape: (nq,), torch.Tensor or np.ndarray
        goal,          # shape: (3,), torch.Tensor or np.ndarray
        seq_length=15,
        configuration_size=7,
        bs=1,
        diffusion_steps:int = None,
        projection_fn=None,  # Optional projection function
    ):
        # Ensure inputs are torch tensors on correct device
        if isinstance(initial_pose, np.ndarray):
            initial_pose = torch.tensor(initial_pose, dtype=torch.float32)
        if isinstance(goal, np.ndarray):
            goal = torch.tensor(goal, dtype=torch.float32)

        initial_pose = initial_pose.to(self.device)
        goal = goal.to(self.device)

        # Encode image using slot attention
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                import PIL.Image
                image = PIL.Image.fromarray(image)
            slots = self.slot_encoder(image)["slots"].to(self.device)

        # Prepare conditioning
        cond = {
            "q0": initial_pose.unsqueeze(0).repeat(bs, 1),  # (bs, nq)
            "goal": goal.unsqueeze(0).repeat(bs, 1),        # (bs, nq)
            "slots": slots.unsqueeze(0).repeat(bs, 1, 1),   # (bs, n_slots, slot_size)
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
