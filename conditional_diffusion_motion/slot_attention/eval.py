from pathlib import Path
import torch
from conditional_diffusion_motion.slot_attention.dataset_slot_attention import PARTNET
from conditional_diffusion_motion.slot_attention.model_slot_attention import SlotAttention, SlotAttentionEncodeOnly
import matplotlib.pyplot as plt
import numpy as np
import random

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load dataset
ressource_dir = Path(__file__).parent.parent.parent.parent / 'ressources' / 'shelf_example'
image_dir = ressource_dir / 'generated_scenes' / 'shelf_for_training_slot_attention'
dataset = PARTNET(split='train', image_dir=image_dir)  # Should return (B, C, H, W)
# img_index = random.randint(0, len(dataset) - 1)
img_index = 334
sample = dataset[img_index]
image = sample['image'].unsqueeze(0).to(device)  # Add batch dim
print(f"img index: {img_index}")
# Load model
resolution = (128, 128)
num_slots = 6
num_iterations = 3
hid_dim = 64
model_dir = ressource_dir / "slot_attention" / "model_slot_attention" / 'slot_attention_shelf.ckpt'

model = SlotAttention(
    input_shape=resolution,
    num_slots= num_slots, # opt.num_slots,,
    # slot_size=opt.hid_dim,
    # hidden_dim=opt.hid_dim * 8,
    num_iters=3,     # opt.num_iterations,
    num_channels=3,
).to(device)


checkpoint = torch.load(model_dir,  map_location=device)
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)
model.eval()

encoder_only_model = SlotAttentionEncodeOnly(
    input_shape=resolution,
    num_slots= num_slots, # opt.num_slots,,
    # slot_size=opt.hid_dim,
    # hidden_dim=opt.hid_dim * 8,
    num_iters=3,     # opt.num_iterations,
    num_channels=3,
).to(device)


# You might filter out the decoder-related parameters:
encoder_only_dict = {k: v for k, v in state_dict.items() if not k.startswith("decoder")}
encoder_only_model.load_state_dict(encoder_only_dict, strict=False)
encoder_only_model.eval()
# Forward pass (no grad)
import time
start = time.time()
with torch.no_grad():
    dict_results = model(image)
end = time.time()
print(f"Time taken for the encoder / decoder forward pass: {end - start:.4f} seconds")

# Forward pass
start = time.time()

with torch.no_grad():
    output = encoder_only_model(image)

# Extract slot latent vectors
slots = output['slots']  # Shape: [B, num_slots, slot_size]
print(f"slots : {slots.shape}")
end = time.time()
print(f"Time taken for the encoder forward pass: {end - start:.4f} seconds")
print("Slots shape:", slots.shape)


recon_combined, recons, masks, slots = dict_results['recons_full'], dict_results['recons'], dict_results['masks_dec'], dict_results['slots']
# Prepare visuals
input_img = image.squeeze().cpu().numpy().transpose(1, 2, 0)  # [C,H,W] → [H,W,C]
recon_img = recon_combined.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

# Per-slot reconstructions
slot_imgs = recons.squeeze().detach().cpu().numpy()  # [num_slots, H, W, C]
slot_masks = masks.squeeze().detach().cpu().numpy()  # [num_slots, H, W, 1]

fig, axs = plt.subplots(2, num_slots + 2, figsize=(3 * (num_slots + 2), 6))
cmap = plt.get_cmap('tab10')
input_img = image.squeeze().cpu().numpy().transpose(1, 2, 0)
input_img = np.clip(input_img, 0, 1)
recon_img = recon_combined.squeeze().cpu().numpy().transpose(1, 2, 0)

# Image originale
axs[0, 0].imshow(input_img)
axs[0, 0].set_title("Original Image")
axs[0, 0].axis('off')

# Reconstruction complète
axs[0, 1].imshow(recon_img)
axs[0, 1].set_title("Reconstruction")
axs[0, 1].axis('off')

# Par-slot
for i in range(num_slots):
    # Slot image + mask intensifié
    slot_img = slot_imgs[i]
    mask = slot_masks[i].squeeze()  # [H, W]

    # Normalisation + intensification pour visualisation
    enhanced = slot_img * (mask * 2.5)  # Contraste boosté
    enhanced = np.clip(enhanced, 0, 1)
    enhanced = np.transpose(enhanced, (1, 2, 0))  # [H,W,C]

    axs[0, i + 2].imshow(enhanced)
    axs[0, i + 2].set_title(f"Slot {i+1}")
    axs[0, i + 2].axis('off')

    # Overlay du masque sur l'image originale
    color = np.array(cmap(i % 10)[:3])  # RGB
    overlay = input_img.copy()
    alpha = 0.5

    mask_thresh = (mask > 0.3).astype(np.float32)
    for c in range(3):
        overlay[:, :, c] = (
            alpha * overlay[:, :, c] +
            (1 - alpha) * color[c] * mask_thresh
        )


    axs[1, i + 2].imshow(overlay)
    axs[1, i + 2].set_title(f"Mask {i+1} (overlay)")
    axs[1, i + 2].axis('off')

# Masques inutilisés
axs[1, 0].axis('off')
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()


# Define which masks to display
# selected_slots = [0, 1, 4]  # example: show only masks for slots 0, 2, and 4

# # Prepare figure: 1 row (original + masks), columns = 1 + len(selected_slots)
# fig, axs = plt.subplots(1, len(selected_slots) + 1, figsize=(3 * (len(selected_slots) + 1), 3))

# # Original image
# axs[0].imshow(input_img)
# # axs[0].set_title("Original Image")
# axs[0].axis('off')

# # Plot selected masks
# for idx, i in enumerate(selected_slots):
#     axs[idx + 1].imshow(slot_masks[i].squeeze(), cmap='gray')
#     # axs[idx + 1].set_title(f"Mask {i}")
#     axs[idx + 1].axis('off')

# plt.tight_layout()
# plt.show()