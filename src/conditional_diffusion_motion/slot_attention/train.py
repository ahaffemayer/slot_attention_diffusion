import os
import argparse
from conditional_diffusion_motion.slot_attention.dataset_slot_attention import *  # Assumes your PARTNET dataset is in here
from conditional_diffusion_motion.slot_attention.model_slot_attention import  SlotAttention  # Assumes SlotAttention is defined in here
from tqdm import tqdm
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------------
# Argument parser
# -----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./tmp/model1000.ckpt', type=str, help='where to save models')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_slots', default=6, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=1, type=int, help='number of epochs to train')

opt = parser.parse_args()

# -----------------------------------
# Setup
# -----------------------------------
torch.manual_seed(opt.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resolution = (128, 128)

# -----------------------------------
# Dataset
# -----------------------------------
image_dir = Path(__file__).parent.parent.parent.parent / 'ressources' / 'panda_slot_attention_example' / 'generated_scenes' / 'shelf'
train_set = PARTNET('train', image_dir=image_dir)  # Should return (B, C, H, W)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.num_workers
)
# -----------------------------------
# Model
# -----------------------------------
model = SlotAttention(
    input_shape=resolution,
    num_slots= 4, # opt.num_slots,,
    # slot_size=opt.hid_dim,
    # hidden_dim=opt.hid_dim * 8,
    num_iters=3,     # opt.num_iterations,
    num_channels=3,
).to(device)

# Uncomment to resume training
# model.load_state_dict(torch.load(opt.model_dir)['model_state_dict'])

optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

# -----------------------------------
# Training Loop
# -----------------------------------
start = time.time()
global_step = 0

for epoch in range(opt.num_epochs):
    model.train()
    total_loss = 0.0

    for sample in tqdm(train_loader, desc=f"Epoch {epoch}"):
        global_step += 1

        # Learning rate scheduling
        if global_step < opt.warmup_steps:
            lr = opt.learning_rate * (global_step / opt.warmup_steps)
        else:
            lr = opt.learning_rate
        lr *= opt.decay_rate ** (global_step / opt.decay_steps)
        optimizer.param_groups[0]['lr'] = lr

        # Input image
        image = sample['image'].to(device)  # shape: (B, C, H, W)

        # Forward pass
        out_dict = model(image, train=True)
        recon_combined = out_dict['recons_full']
        loss = out_dict['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    elapsed = str(datetime.timedelta(seconds=time.time() - start))
    print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Time: {elapsed}")

    # Save model
    if epoch % 10 == 0:
        os.makedirs(os.path.dirname(opt.model_dir), exist_ok=True)
        torch.save({'model_state_dict': model.state_dict()}, opt.model_dir)