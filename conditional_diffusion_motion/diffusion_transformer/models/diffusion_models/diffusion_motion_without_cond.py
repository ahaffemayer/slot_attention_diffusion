from __future__ import annotations

import tqdm
import torch
from torch import Tensor
import pytorch_lightning as pl
from diffusers import DDPMScheduler


class DiffusionMotionWithoutCond(pl.LightningModule):
    def __init__(
        self, model=None, noise_scheduler=None, default_diffusion_steps=1000
    ) -> None:
        super().__init__()
        self.model = model
        if noise_scheduler is None:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=default_diffusion_steps,
                beta_schedule="linear",
                clip_sample=False,
            )
        else:
            self.noise_scheduler = noise_scheduler

        self.diffusion_noising_steps = len(self.noise_scheduler.timesteps)

    def single_denoise_step(self, xt: Tensor, t: Tensor | int) -> Tensor:
        """Perform a single de-noising step on a batch of motions. It runs model to get
        x0 and noise it back by diffusion to get x_{t-1}.
        If t-1 is zero, no diffusion is applied.
        Args:
            xt: batch of motions (bs, seq_length, config_size)
            t: current noising timestep of the motion (bs,) or int if same timestep
               should be used for all batches

        Returns:
            Tensor: batch of motions at noising timestep t-1
        """
        if isinstance(t, int):
            t = t * torch.ones((xt.shape[0],), dtype=torch.int, device=self.device)
        x0 = self.model(xt, t)
        xt_minus_1 = self.generate_x_t(x0, torch.clip(t - 1, min=0))
        mask = t <= 1  # i.e. t-1 <= 0; no diffusion just copy from x0
        xt_minus_1[mask] = x0[mask]
        return xt_minus_1

    def sample(
        self, bs=1, seq_length=32, configuration_size=2, projection_fn=None, diffusion_steps: int | None = None
    ) -> Tensor:
        """
        Sample a new motion randomly by de-noising gaussian noise.
        Args:
            bs: number of samples to create
            seq_length: length of the sequence
            configuration_size: size of the configuration
            projection_fn: projection function that is used to map xt after each
                de-noising step; it should be in form x_projected = f(x, t), t has
                values from T-1 to 0, where T is self.diffusion_noising_steps.

        Returns:
        """
        with torch.no_grad():
            xt = torch.randn((bs, seq_length, configuration_size), device=self.device)
            steps = diffusion_steps or self.diffusion_noising_steps
            for t in tqdm.trange(steps- 1, 0, -1):
                xt = self.single_denoise_step(xt=xt, t=t)
                if projection_fn is not None:
                    xt = projection_fn(xt, t - 1)
            return xt

    def generate_x_t(self, x0, t):
        """Generate x_t (bs, seq_len, conf_len) from given x_0 (bs, seq_len, conf_len) and
        noising time steps t (bs). I.e. apply forward process of the diffusion."""
        assert t.shape == x0.shape[:1]
        return self.noise_scheduler.add_noise(x0, torch.randn_like(x0), t)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

    def training_step(self, train_batch, batch_idx):
        bs = train_batch.shape[0]
        t = torch.randint(0, self.diffusion_noising_steps, (bs,), device=self.device)
        pred = self.model(self.generate_x_t(train_batch, t), t)
        loss = (pred - train_batch).square().sum()
        self.log("loss", loss)
        return loss