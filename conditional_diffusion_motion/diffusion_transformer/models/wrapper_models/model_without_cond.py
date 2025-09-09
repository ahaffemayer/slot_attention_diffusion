from conditional_diffusion_motion.diffusion_transformer.models.diffusion_models.diffusion_motion_without_cond import DiffusionMotionWithoutCond
from conditional_diffusion_motion.diffusion_transformer.models.transformer_encoder_decoder.transformer_encoder_decoder_without_cond import TransformerForDiffusionWithoutCond

class ModelWithoutCond(DiffusionMotionWithoutCond):
    def __init__(self) -> None:
        """
        Model that uses transformer inside diffusion.
        """

        model = TransformerForDiffusionWithoutCond(
            configuration_size=7,
            position_encoding_size=32,
            ff_size=4084,
            dropout=0.01,
            num_layers=4,
        )
        super().__init__(model, default_diffusion_steps=25)