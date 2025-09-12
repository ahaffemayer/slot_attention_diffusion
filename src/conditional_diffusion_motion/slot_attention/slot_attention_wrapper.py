import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

from conditional_diffusion_motion.slot_attention.model_slot_attention import SlotAttentionEncodeOnly  # Adjust path if needed 

class SlotEncoderWrapper:
    def __init__(self, device="cpu", image_size=(128, 128), num_slots=4, model_dir=None):
        """
        Wraps a slot encoder model for standardized image input.

        Args:
            device (str): Device to run the encoder on.
            image_size (tuple): Target (H, W) resolution for images.
        """
        self.device = device
        model_dir = './slot_attention_shelf.ckpt' if model_dir is None else model_dir
        checkpoint = torch.load(model_dir, map_location=self.device)
        state_dict = checkpoint['model_state_dict']

        self.model = SlotAttentionEncodeOnly(
            input_shape=image_size,
            num_slots=num_slots,
            num_iters=3,
            num_channels=3,
        ).to(self.device)

        encoder_only_dict = {k: v for k, v in state_dict.items() if not k.startswith("decoder")}
        self.model.load_state_dict(encoder_only_dict, strict=False)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # converts to [0, 1]
        ])

    def __call__(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        """
        Args:
            image (PIL.Image or torch.Tensor): Input RGB image of shape [C, H, W].

        Returns:
            dict: {"slots": (num_slots, slot_dim)}
        """
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = TF.to_pil_image(image.cpu())
            else:
                raise ValueError(f"Expected a 3D tensor [C, H, W], got shape {image.shape}")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image or torch.Tensor, got {type(image)}")

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # shape: (1, C, H, W)

        with torch.no_grad():
            output = self.model(image_tensor)  # output should contain "slots"

        return {"slots": output["slots"].squeeze(0).cpu()}  # shape: (num_slots, slot_dim)
