from conditional_diffusion_motion.diffusion_transformer.models.diffusion_models.diffusion_motion import (
    DiffusionMotion,
)
from conditional_diffusion_motion.diffusion_transformer.models.transformer_encoder_decoder.transformer_encoder_decoder import (
    TransformerDiffusionEncoderDecoder,
)
from conditional_diffusion_motion.diffusion_transformer.models.conditioning.conditioning_diffusion_policy import (
    ConditioningEncoderDiffusionPolicy,
)


class ModelDiffusionPolicy(DiffusionMotion):
    def __init__(self) -> None:
        """
        Model that uses transformer inside diffusion.
        """
        # Conditioning encoder
        conditioning_encoder = ConditioningEncoderDiffusionPolicy(
            condition_shapes={
                "q0": 7,
                "goal": 3,
                "image": (3, 128, 128),
            },
            encoder_embedding_size=16,
            position_encoding_size=16,
        )

        model = TransformerDiffusionEncoderDecoder(
            configuration_size=7,
            configuration_size_embedding=16,
            encoder_embedding_size=16,
            position_encoding_size=16,
            ff_size=4084,
            dropout=0.01,
            num_decoder_layers=4,
            num_encoder_layers=4,
            num_heads=4,
            conditioning_encoder=conditioning_encoder,
        )
        super().__init__(model, default_diffusion_steps=100)
