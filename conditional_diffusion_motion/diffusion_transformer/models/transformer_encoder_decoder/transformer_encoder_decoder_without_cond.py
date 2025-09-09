from __future__ import annotations

from torch import nn, Tensor, cat
from conditional_diffusion_motion.diffusion_transformer.models.positional_encoding import PositionalEncoding


class TransformerForDiffusionWithoutCond(nn.Module):
    """Motion diffusion model is basically a transformer that takes sequence and
    predict a new sequence."""

    def __init__(
        self,
        position_encoding_size: int = 16,
        configuration_size: int = 2,
        noising_time_step_embedding_size: int | None = None,
        ff_size: int = 1024,
        num_heads: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_layers: int = 8,
    ) -> None:
        super().__init__()

        self.position_encoding = PositionalEncoding(position_encoding_size)
        if num_heads is None:
            num_heads = position_encoding_size + configuration_size
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=position_encoding_size + configuration_size,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=num_layers
        )
        if noising_time_step_embedding_size is None:
            noising_time_step_embedding_size = position_encoding_size
        self.noising_time_steps_embedding = nn.Sequential(
            nn.Linear(position_encoding_size, noising_time_step_embedding_size),
            nn.SiLU(),
            nn.Linear(
                noising_time_step_embedding_size,
                configuration_size,
            ),
        )

    def forward(self, x: Tensor, noising_time_steps: Tensor):
        """
        :param x: [batch_size, seq_length, configuration_size]
        :param noising_time_steps: [batch_size], int
        :return: tensor of size [bs, seq_length, configuration_size]
        """
        configuration_size = x.shape[-1]
        noising_time_step_emb = self.noising_time_steps_embedding(
            self.position_encoding.pe[0, noising_time_steps, :]
        ).unsqueeze(1)
        x = cat((noising_time_step_emb, x), dim=1)
        x = self.position_encoding(x)
        x = self.transformer_encoder(x)
        return x[..., 1:, :configuration_size]