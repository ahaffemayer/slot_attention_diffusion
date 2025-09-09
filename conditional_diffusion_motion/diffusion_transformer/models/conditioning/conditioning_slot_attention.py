import torch
import torch.nn as nn
from torch import Tensor
from conditional_diffusion_motion.diffusion_transformer.models.positional_encoding import PositionalEncoding


class ConditioningEncoderSlotAttention(nn.Module):
    def __init__(
        self,
        condition_shapes: dict[str, int],
        position_encoding_size: int,
        encoder_embedding_size: int,
    ) -> None:
        super().__init__()

        self.position_encoding_size = position_encoding_size
        self.encoder_embedding_size = encoder_embedding_size

        self.conditioners = nn.ModuleDict({
            k: nn.Linear(v, encoder_embedding_size) for k, v in condition_shapes.items()
        })

        self.num_conditions = len(self.conditioners)

        self.token_type_embedding = nn.Embedding(
            num_embeddings=self.num_conditions + 1,
            embedding_dim=position_encoding_size,
        )

        self.noising_time_steps_embedding = nn.Sequential(
            nn.Linear(position_encoding_size, position_encoding_size),
            nn.SiLU(),
            nn.Linear(position_encoding_size, encoder_embedding_size),
        )

        self.noising_position_encoding = PositionalEncoding(position_encoding_size)

    def forward(self, cond: dict[str, Tensor], noising_time_steps: Tensor) -> Tensor:
        # Validate timestep shape
        if noising_time_steps.dim() == 0:
            noising_time_steps = noising_time_steps.unsqueeze(0)
        if noising_time_steps.dim() == 1:
            batch_size = next(iter(cond.values())).shape[0]
            if noising_time_steps.shape[0] != batch_size:
                raise ValueError(
                    f"Expected noising_time_steps batch size {batch_size}, but got {noising_time_steps.shape[0]}"
                )
        else:
            raise ValueError(f"Invalid shape for noising_time_steps: {noising_time_steps.shape}")

        bs = noising_time_steps.shape[0]
        cond_embs = []
        token_indices = []
        token_type_id = 0

        # Process each conditioning token
        for key, emb_layer in self.conditioners.items():
            token = cond[key]
            if key == "slots":
                bs, n_slots, slot_dim = token.shape
                token_flat = token.view(-1, slot_dim)
                token_emb = emb_layer(token_flat).view(bs, n_slots, self.encoder_embedding_size)
                cond_embs.append(token_emb)
                token_indices.append(torch.full((bs, n_slots), token_type_id, device=token.device))
            else:
                token_emb = emb_layer(token).unsqueeze(1)  # (bs, 1, emb_dim)
                cond_embs.append(token_emb)
                token_indices.append(torch.full((bs, 1), token_type_id, device=token.device))
            token_type_id += 1

        cond_tokens = torch.cat(cond_embs, dim=1)  # (bs, num_tokens, emb_dim)
        token_indices = torch.cat(token_indices, dim=1)  # (bs, num_tokens)
        encoder_pos = self.token_type_embedding(token_indices)  # (bs, num_tokens, pos_dim)
        encoder_input = torch.cat([cond_tokens, encoder_pos], dim=-1)  # (bs, num_tokens, emb+pos)

        # Add timestep token
        t_indices = torch.clamp(noising_time_steps, max=self.noising_position_encoding.pe.shape[1] - 1)
        t_pe = self.noising_position_encoding.pe[0, t_indices, :]  # (bs, pos_dim)
        t_emb = self.noising_time_steps_embedding(t_pe).unsqueeze(1)  # (bs, 1, emb_dim)
        t_index = torch.full((bs, 1), token_type_id, device=t_emb.device)
        t_pos = self.token_type_embedding(t_index)  # (bs, 1, pos_dim)
        t_token = torch.cat([t_emb, t_pos], dim=-1)  # (bs, 1, emb+pos)

        # Final output: (bs, num_tokens + 1, emb + pos)
        return torch.cat([encoder_input, t_token], dim=1)
