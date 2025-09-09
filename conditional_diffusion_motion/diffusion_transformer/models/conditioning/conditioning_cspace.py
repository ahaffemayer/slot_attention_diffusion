import torch
import torch.nn as nn
from torch import Tensor
from conditional_diffusion_motion.diffusion_transformer.models.positional_encoding import PositionalEncoding


class ConditioningEncoderCSpace(nn.Module):
    def __init__(
        self,
        condition_shapes: dict[str, int],  # e.g., {"q0": 7, "goal": 3, "cspace": 1024}
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
            num_embeddings=self.num_conditions + 1,  # +1 for the timestep token
            embedding_dim=position_encoding_size,
        )

        self.noising_time_steps_embedding = nn.Sequential(
            nn.Linear(position_encoding_size, position_encoding_size),
            nn.SiLU(),
            nn.Linear(position_encoding_size, encoder_embedding_size),
        )

        self.noising_position_encoding = PositionalEncoding(position_encoding_size)

    def forward(self, cond: dict[str, Tensor], noising_time_steps: Tensor) -> Tensor:
        bs = noising_time_steps.shape[0]
        sz_pe = self.position_encoding_size
        sz_enc_emb = self.encoder_embedding_size

        cond_embs = []
        token_indices = []

        token_type_id = 0
        for key, emb_layer in self.conditioners.items():
            token = cond[key]
            if token.ndim == 3:
                bs, n, d = token.shape
                token = token.view(-1, d)
                token_emb = emb_layer(token).view(bs, n, sz_enc_emb)
                cond_embs.append(token_emb)
                token_indices.append(torch.full((bs, n), token_type_id, device=token.device))
            else:  # Assume (bs, d)
                token_emb = emb_layer(token).unsqueeze(1)  # (bs, 1, emb)
                cond_embs.append(token_emb)
                token_indices.append(torch.full((bs, 1), token_type_id, device=token.device))
            token_type_id += 1

        cond_tokens = torch.cat(cond_embs, dim=1)
        token_indices = torch.cat(token_indices, dim=1)
        encoder_pos = self.token_type_embedding(token_indices)
        encoder_input_pe = torch.cat([cond_tokens, encoder_pos], dim=-1)

        # Add timestep token
        t_pe = self.noising_position_encoding.pe[0, noising_time_steps, :]  # (bs, pe_size)
        t_emb = self.noising_time_steps_embedding(t_pe).unsqueeze(1)  # (bs, 1, emb)
        t_index = torch.full((bs, 1), token_type_id, device=t_emb.device)
        t_pos = self.token_type_embedding(t_index)
        t_input = torch.cat([t_emb, t_pos], dim=-1)

        return torch.cat([encoder_input_pe, t_input], dim=1)  # (bs, num_tokens + 1, emb + pos)
