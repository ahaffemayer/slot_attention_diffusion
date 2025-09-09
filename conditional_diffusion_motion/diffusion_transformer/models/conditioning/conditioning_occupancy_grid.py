import torch
import torch.nn as nn
from torch import Tensor
from conditional_diffusion_motion.diffusion_transformer.models.positional_encoding import PositionalEncoding


class ConditioningEncoder3DOccupancyGrid(nn.Module):
    def __init__(
        self,
        condition_shapes: dict[str, int | tuple[int, int, int]],
        position_encoding_size: int,
        encoder_embedding_size: int,
    ) -> None:
        super().__init__()

        self.position_encoding_size = position_encoding_size
        self.encoder_embedding_size = encoder_embedding_size

        self.conditioners = nn.ModuleDict()
        self.grid_keys = []

        for k, v in condition_shapes.items():
            if isinstance(v, tuple) and len(v) == 3:
                self.grid_keys.append(k)
                # 3D CNN will handle encoding, not a linear layer
            else:
                self.conditioners[k] = nn.Linear(v, encoder_embedding_size)

        self.num_conditions = len(self.conditioners) + len(self.grid_keys)

        self.token_type_embedding = nn.Embedding(
            num_embeddings=self.num_conditions + 1,  # +1 for timestep token
            embedding_dim=position_encoding_size,
        )

        # 3D CNN Encoder shared across all grid keys
        self.backbone = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, encoder_embedding_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Timestep embedding
        self.noising_time_steps_embedding = nn.Sequential(
            nn.Linear(position_encoding_size, position_encoding_size),
            nn.SiLU(),
            nn.Linear(position_encoding_size, encoder_embedding_size),
        )

        self.noising_position_encoding = PositionalEncoding(position_encoding_size)

    def forward(self, cond: dict[str, Tensor], noising_time_steps: Tensor) -> Tensor:
        if noising_time_steps.dim() == 0:
            noising_time_steps = noising_time_steps.unsqueeze(0)
        if noising_time_steps.dim() != 1:
            raise ValueError(f"Invalid shape for noising_time_steps: {noising_time_steps.shape}")

        batch_size = noising_time_steps.shape[0]
        cond_embs = []
        token_indices = []
        token_type_id = 0

        # Process standard (non-grid) conditions
        for key, emb_layer in self.conditioners.items():
            token = cond[key]  # (B, D)
            token_emb = emb_layer(token).unsqueeze(1)  # (B, 1, D)
            cond_embs.append(token_emb)
            token_indices.append(torch.full((batch_size, 1), token_type_id, device=token.device))
            token_type_id += 1

        # Process 3D grid keys
        for key in self.grid_keys:
            grid = cond[key]  # (B, 1, X, Y, Z)
            grid = cond[key].unsqueeze(1)
            cnn_out = self.backbone(grid)  # (B, D, sx, sy, sz)
            cnn_out = cnn_out.flatten(2).transpose(1, 2)  # (B, N, D)
            cond_embs.append(cnn_out)
            token_indices.append(torch.full((batch_size, cnn_out.shape[1]), token_type_id, device=grid.device))
            token_type_id += 1

        # Combine all condition tokens
        cond_tokens = torch.cat(cond_embs, dim=1)  # (B, T_cond, D)
        token_type_ids = torch.cat(token_indices, dim=1)  # (B, T_cond)
        pos_emb = self.token_type_embedding(token_type_ids)  # (B, T_cond, pos_dim)
        encoder_input = torch.cat([cond_tokens, pos_emb], dim=-1)  # (B, T_cond, D+pos_dim)

        # Timestep token
        t_indices = torch.clamp(noising_time_steps, max=self.noising_position_encoding.pe.shape[1] - 1)
        t_pe = self.noising_position_encoding.pe[0, t_indices, :]  # (B, pos_dim)
        t_emb = self.noising_time_steps_embedding(t_pe).unsqueeze(1)  # (B, 1, D)
        t_type = torch.full((batch_size, 1), token_type_id, device=t_emb.device)
        t_pos = self.token_type_embedding(t_type)  # (B, 1, pos_dim)
        t_token = torch.cat([t_emb, t_pos], dim=-1)  # (B, 1, D + pos_dim)

        return torch.cat([encoder_input, t_token], dim=1)  # (B, T_total, D + pos_dim)
