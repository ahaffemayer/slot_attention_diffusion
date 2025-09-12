import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet18
from conditional_diffusion_motion.diffusion_transformer.models.positional_encoding import PositionalEncoding


class ConditioningEncoderDiffusionPolicy(nn.Module):
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
        self.image_key = None  # the key corresponding to the image input

        for k, v in condition_shapes.items():
            if isinstance(v, tuple) and len(v) == 3:  # (C, H, W)
                if self.image_key is not None:
                    raise ValueError("Only one image conditioning input is supported.")
                self.image_key = k
            else:
                self.conditioners[k] = nn.Linear(v, encoder_embedding_size)

        self.num_conditions = len(self.conditioners) + (1 if self.image_key else 0)

        self.token_type_embedding = nn.Embedding(
            num_embeddings=self.num_conditions + 1,  # +1 for timestep
            embedding_dim=position_encoding_size,
        )

        # ResNet18 encoder for image input
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # remove classification head
        self.image_proj = nn.Linear(512, encoder_embedding_size)  # project ResNet output to embedding size

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

        # Process standard (non-image) conditions
        for key, emb_layer in self.conditioners.items():
            token = cond[key]  # (B, D)
            token_emb = emb_layer(token).unsqueeze(1)  # (B, 1, D)
            cond_embs.append(token_emb)
            token_indices.append(torch.full((batch_size, 1), token_type_id, device=token.device))
            token_type_id += 1

        # Process image condition
        if self.image_key is not None:
            img = cond[self.image_key]  # (B, C, H, W)
            img_feat = self.resnet(img)  # (B, 512)
            img_emb = self.image_proj(img_feat).unsqueeze(1)  # (B, 1, D)
            cond_embs.append(img_emb)
            token_indices.append(torch.full((batch_size, 1), token_type_id, device=img.device))
            token_type_id += 1

        # Combine condition tokens
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
