import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional, Iterable, Tuple, Dict
import einops

DEFAULT_WEIGHT_INIT = "default"

def _infer_common_length(fail_on_missing_length=True, **kwargs) -> int:
    """Given kwargs of scalars and lists, checks that all lists have the same length and returns it.

    Optionally fails if no length was provided.
    """
    length = None
    name = None
    for cur_name, arg in kwargs.items():
        if isinstance(arg, (tuple, list)):
            cur_length = len(arg)
            if length is None:
                length = cur_length
                name = cur_name
            elif cur_length != length:
                raise ValueError(
                    f"Inconsistent lengths: {cur_name} has length {cur_length}, "
                    f"but {name} has length {length}"
                )

    if fail_on_missing_length and length is None:
        names = ", ".join(f"`{key}`" for key in kwargs.keys())
        raise ValueError(f"Need to specify a list for at least one of {names}.")

    return length

def _maybe_expand_list(arg: Union[int, List[int]], length: int) -> list:
    if not isinstance(arg, (tuple, list)):
        return [arg] * length

    return list(arg)

def init_parameters(layers: Union[nn.Module, Iterable[nn.Module]], weight_init: str = "default"):
    assert weight_init in ("default", "he_uniform", "he_normal", "xavier_uniform", "xavier_normal")
    if isinstance(layers, nn.Module):
        layers = [layers]

    for idx, layer in enumerate(layers):
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.zeros_(layer.bias)

        if hasattr(layer, "weight") and layer.weight is not None:
            gain = 1.0
            if isinstance(layers, nn.Sequential):
                if idx < len(layers) - 1:
                    next = layers[idx + 1]
                    if isinstance(next, nn.ReLU):
                        gain = 2**0.5

            if weight_init == "he_uniform":
                torch.nn.init.kaiming_uniform_(layer.weight, gain)
            elif weight_init == "he_normal":
                torch.nn.init.kaiming_normal_(layer.weight, gain)
            elif weight_init == "xavier_uniform":
                torch.nn.init.xavier_uniform_(layer.weight, gain)
            elif weight_init == "xavier_normal":
                torch.nn.init.xavier_normal_(layer.weight, gain)

class CNNEncoder(nn.Sequential):
    """Simple convolutional encoder.

    For `features`, `kernel_sizes`, `strides`, scalars can be used to avoid repeating arguments,
    but at least one list needs to be provided to specify the number of layers.
    """

    def __init__(
        self,
        inp_dim: int,
        features: Union[int, List[int]],
        kernel_sizes: Union[int, List[int]],
        strides: Union[int, List[int]] = 1,
        outp_dim: Optional[int] = None,
        weight_init: str = "default",
    ):
        length = _infer_common_length(features=features, kernel_sizes=kernel_sizes, strides=strides)
        features = _maybe_expand_list(features, length)
        kernel_sizes = _maybe_expand_list(kernel_sizes, length)
        strides = _maybe_expand_list(strides, length)

        layers = []
        cur_dim = inp_dim
        for dim, kernel_size, stride in zip(features, kernel_sizes, strides):
            layers.append(
                nn.Conv2d(
                    cur_dim,
                    dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=self.get_same_padding(kernel_size, stride),
                )
            )
            layers.append(nn.ReLU(inplace=True))
            cur_dim = dim

        if outp_dim is not None:
            layers.append(nn.Conv1d(cur_dim, outp_dim, kernel_size=1, stride=1))

        super().__init__(*layers)
        init_parameters(self, weight_init)

    @staticmethod
    def get_same_padding(kernel_size: int, stride: int) -> Union[str, int]:
        """Try to infer same padding for convolutions."""
        # This method is very lazily implemented, but oh well..
        if stride == 1:
            return "same"
        if kernel_size == 3:
            if stride == 2:
                return 1
        elif kernel_size == 5:
            if stride == 2:
                return 2

        raise ValueError(f"Don't know 'same' padding for kernel {kernel_size}, stride {stride}")

def make_slot_attention_encoder(
    inp_dim: int,
    feature_multiplier: float = 1,
    downsamplings: int = 0,
    weight_init: str = DEFAULT_WEIGHT_INIT,
) -> CNNEncoder:
    """CNN encoder as used in Slot Attention paper.

    By default, 4 layers with 64 channels each, keeping the spatial input resolution the same.

    This encoder is also used by SAVi, in the following configurations:

    - for image resolution 64: feature_multiplier=0.5, downsamplings=0
    - for image resolution 128: feature_multiplier=1, downsamplings=1

    and STEVE, in the following configurations:

    - for image resolution 64: feature_multiplier=1, downsamplings=0
    - for image resolution 128: feature_multiplier=1, downsamplings=1
    """
    assert 0 <= downsamplings <= 4
    channels = int(64 * feature_multiplier)
    strides = [2] * downsamplings + [1] * (4 - downsamplings)
    return CNNEncoder(
        inp_dim,
        features=[channels, channels, channels, channels],
        kernel_sizes=[5, 5, 5, 5],
        strides=strides,
        weight_init=weight_init,
    )

class MLP(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        hidden_dims,
        initial_layer_norm: bool = False,
        final_activation = False,
        residual: bool = False,
        weight_init: str = "default",
    ):
        super().__init__()
        self.residual = residual
        if residual:
            assert inp_dim == outp_dim

        layers = []
        if initial_layer_norm:
            layers.append(nn.LayerNorm(inp_dim))

        cur_dim = inp_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(cur_dim, dim))
            layers.append(nn.ReLU())
            cur_dim = dim

        layers.append(nn.Linear(cur_dim, outp_dim))
        if final_activation:
            if isinstance(final_activation, bool):
                final_activation = "relu"
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        init_parameters(self.layers, weight_init)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        outp = self.layers(inp)

        if self.residual:
            return inp + outp
        else:
            return outp

class SlotAttentionLayer(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        slot_dim: int,
        kvq_dim= None,
        hidden_dim = None,
        n_iters: int = 3,
        eps: float = 1e-8,
        use_gru: bool = True,
        use_mlp: bool = True,
    ):
        super().__init__()
        assert n_iters >= 1

        if kvq_dim is None:
            kvq_dim = slot_dim
        self.to_k = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_v = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_q = nn.Linear(slot_dim, kvq_dim, bias=False)

        if use_gru:
            self.gru = nn.GRUCell(input_size=kvq_dim, hidden_size=slot_dim)
        else:
            assert kvq_dim == slot_dim
            self.gru = None

        if hidden_dim is None:
            hidden_dim = 4 * slot_dim

        if use_mlp:
            self.mlp = MLP(
                slot_dim, slot_dim, [hidden_dim], initial_layer_norm=True, residual=True
            )
        else:
            self.mlp = None

        self.norm_features = nn.LayerNorm(inp_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.n_iters = n_iters
        self.eps = eps
        self.scale = kvq_dim**-0.5

    def step(
        self, slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attn_mask = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one iteration of slot attention."""
        slots = self.norm_slots(slots)
        queries = self.to_q(slots)

        dots = torch.einsum("bsd, bfd -> bsf", queries, keys) * self.scale
        if attn_mask is not None:
            dots = dots.masked_fill(attn_mask, float('-inf'))
        pre_norm_attn = torch.softmax(dots, dim=1)
        attn = pre_norm_attn + self.eps
        attn = attn / attn.sum(-1, keepdim=True)

        updates = torch.einsum("bsf, bfd -> bsd", attn, values)

        if self.gru:
            updated_slots = self.gru(updates.flatten(0, 1), slots.flatten(0, 1))
            slots = updated_slots.unflatten(0, slots.shape[:2])
        else:
            slots = slots + updates

        if self.mlp is not None:
            slots = self.mlp(slots)

        return slots, pre_norm_attn

    def forward(self, slots: torch.Tensor, features: torch.Tensor, n_iters = None, attn_mask = None):
        features = self.norm_features(features)
        keys = self.to_k(features)
        values = self.to_v(features)

        if n_iters is None:
            n_iters = self.n_iters

        for _ in range(n_iters):
            slots, pre_norm_attn = self.step(slots, keys, values, attn_mask=attn_mask)

        return {"slots": slots, "masks": pre_norm_attn}

class CNNDecoder(nn.Sequential):
    """Simple convolutional decoder.

    For `features`, `kernel_sizes`, `strides`, scalars can be used to avoid repeating arguments,
    but at least one list needs to be provided to specify the number of layers.
    """

    def __init__(
        self,
        inp_dim: int,
        features: Union[int, List[int]],
        kernel_sizes: Union[int, List[int]],
        strides: Union[int, List[int]] = 1,
        outp_dim: Optional[int] = None,
        weight_init: str = DEFAULT_WEIGHT_INIT,
    ):
        length = _infer_common_length(features=features, kernel_sizes=kernel_sizes, strides=strides)
        features = _maybe_expand_list(features, length)
        kernel_sizes = _maybe_expand_list(kernel_sizes, length)
        strides = _maybe_expand_list(strides, length)

        layers = []
        cur_dim = inp_dim
        for dim, kernel_size, stride in zip(features, kernel_sizes, strides):
            padding, output_padding = self.get_same_padding(kernel_size, stride)
            layers.append(
                nn.ConvTranspose2d(
                    cur_dim,
                    dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            cur_dim = dim

        if outp_dim is not None:
            layers.append(nn.Conv1d(cur_dim, outp_dim, kernel_size=1, stride=1))

        super().__init__(*layers)
        init_parameters(self, weight_init)

    @staticmethod
    def get_same_padding(kernel_size: int, stride: int) -> Tuple[int, int]:
        """Try to infer same padding for transposed convolutions."""
        # This method is very lazily implemented, but oh well..
        if kernel_size == 3:
            if stride == 1:
                return 1, 0
            if stride == 2:
                return 1, 1
        elif kernel_size == 5:
            if stride == 1:
                return 2, 0
            if stride == 2:
                return 2, 1

        raise ValueError(f"Don't know 'same' padding for kernel {kernel_size}, stride {stride}")

def make_savi_decoder(
    inp_dim: int,
    feature_multiplier: float = 1,
    upsamplings: int = 4,
    weight_init: str = DEFAULT_WEIGHT_INIT,
) -> CNNDecoder:
    """CNN encoder as used in SAVi paper.

    By default, 4 layers with 64 channels each, upscaling from a 8x8 feature map to 128x128.
    """
    assert 0 <= upsamplings <= 4
    channels = int(64 * feature_multiplier)
    strides = [2] * upsamplings + [1] * (4 - upsamplings)
    return CNNDecoder(
        inp_dim,
        features=[channels, channels, channels, channels],
        kernel_sizes=[5, 5, 5, 5],
        strides=strides,
        weight_init=weight_init,
    )

class SpatialBroadcastDecoder(nn.Module):
    """Decoder that reconstructs a spatial map independently per slot."""

    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        backbone: nn.Module,
        initial_size: Union[int, Tuple[int, int]] = 8,
        backbone_dim: Optional[int] = None,
        pos_embed: Optional[nn.Module] = None,
        output_transform: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.outp_dim = outp_dim
        if isinstance(initial_size, int):
            self.initial_size = (initial_size, initial_size)
        else:
            self.initial_size = initial_size

        if pos_embed is None:
            self.pos_embed = CoordinatePositionEmbed(inp_dim, initial_size)
        else:
            self.pos_embed = pos_embed

        self.backbone = backbone

        if output_transform is None:
            if backbone_dim is None:
                raise ValueError("Need to provide backbone dim if output_transform is unspecified")
            self.output_transform = nn.Conv2d(backbone_dim, outp_dim + 1, 1, 1)
        else:
            self.output_transform = output_transform

        self.init_parameters()

    def init_parameters(self):
        if isinstance(self.output_transform, nn.Conv2d):
            init_parameters(self.output_transform)



    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        bs, n_slots, _ = slots.shape
        slots = einops.repeat(
            slots, "b s d -> (b s) d h w", h=self.initial_size[0], w=self.initial_size[1]
        )
        slots = self.pos_embed(slots)
        features = self.backbone(slots)
        outputs = self.output_transform(features)
        outputs = einops.rearrange(outputs, "(b s) ... -> b s ...", b=bs, s=n_slots)
        recons, alpha = einops.unpack(outputs, [[self.outp_dim], [1]], "b s * h w")

        masks = torch.softmax(alpha, dim=1)
        recon_combined = torch.sum(masks*recons, dim=1)

        return {"reconstruction": recons, "recon_combined": recon_combined, "masks": masks.squeeze(2)}

class CoordinatePositionEmbed(nn.Module):
    """Coordinate positional embedding as in Slot Attention."""

    def __init__(self, dim: int, size: Tuple[int, int], proj_dim: int = None):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.register_buffer("grid", self.build_grid(size))
        self.proj = nn.Conv2d(self.grid.shape[0], dim, kernel_size=1, bias=True)
        init_parameters(self.proj, "xavier_uniform")
        if proj_dim is not None:
            self.proj_dim = proj_dim
            self.proj_linear = nn.Linear(dim, proj_dim)

    @staticmethod
    def build_grid(
        size: Tuple[int, int],
        bounds: Tuple[float, float] = (-1.0, 1.0),
        add_inverse: bool = False,
    ) -> torch.Tensor:
        ranges = [torch.linspace(*bounds, steps=res) for res in size]
        grid = torch.meshgrid(*ranges, indexing="ij")

        if add_inverse:
            grid = torch.stack((grid[0], grid[1], 1.0 - grid[0], 1.0 - grid[1]), axis=0)
        else:
            grid = torch.stack((grid[0], grid[1]), axis=0)

        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.ndim == 4
        ), f"Expected input shape (batch, channels, height, width), but got {x.shape}"
        x = x + self.proj(self.grid)
        if hasattr(self, "proj_dim"):
            x = x.flatten(-2, -1)
            x = x.permute(0, 2, 1)
            x = self.proj_linear(x)
        return x

class SlotAttention(nn.Module):
    def __init__(
        self,
        input_shape,
        num_slots=8,
        slot_size=64,
        hidden_dim=512,
        num_iters=2,
        num_channels=3,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.resolution = input_shape
        self.num_iters = num_iters
        self.num_slots = num_slots
        self.slot_size = slot_size
        if self.resolution[0] == 128:
            self.visual_resolution = tuple(i // 2 for i in self.resolution)
            feature_multiplier = 1
            downsample = 1
        elif self.resolution[0] == 64:
            self.visual_resolution = self.resolution
            feature_multiplier = 0.5
            downsample = 0
        else:
            raise ValueError(f"Invalid resolution: {self.resolution} needed 128x128 or 64x64")

        self.init_latents = nn.Parameter(
                    nn.init.normal_(torch.empty(1, self.num_slots, self.slot_size)))

        ## ENCODER :Classical CNN
        self.encoder = make_slot_attention_encoder(
            inp_dim=self.num_channels,
            feature_multiplier=feature_multiplier,
            downsamplings=downsample,
        )
        self.visual_channels = int(64 * feature_multiplier)
        self.projection_layer = CoordinatePositionEmbed(self.visual_channels, self.visual_resolution, proj_dim=slot_size)

        ## OBJECT-CENTRIC MODULE : Slot-Attention for Video == SA + Transformer
        self.grouping = SlotAttentionLayer(
            inp_dim=slot_size,
            slot_dim=self.slot_size,
            kvq_dim=self.slot_size*2,
            n_iters=self.num_iters,
            hidden_dim=hidden_dim,
            use_gru=True,
            use_mlp=True,
        )

        ## DECODER: Spatial-Broadcast Decoder (SBD)
        self.dec_resolution = (8, 8) # TODO: make this a parameter
        savi_decoder = make_savi_decoder(
            inp_dim=self.slot_size,
            feature_multiplier=feature_multiplier,
            upsamplings=4 if downsample else 3 
        )
        self.decoder = SpatialBroadcastDecoder(
            inp_dim=self.slot_size,
            outp_dim=self.num_channels,
            backbone=savi_decoder,
            backbone_dim=int(64*feature_multiplier),
            initial_size=self.dec_resolution,
            pos_embed=CoordinatePositionEmbed(self.slot_size, self.dec_resolution),
        )

    def encode(self, img):
        B, C, H, W = img.shape

        h = self.encoder(img)
        h = self.projection_layer(h)

        # Extract slots
        prev_slots = self.init_latents.repeat(B, 1, 1)
        out_dict = self.grouping(prev_slots, h, n_iters=self.num_iters)
        slots = out_dict["slots"]  # [B, num_slots, slot_size]
        masks = out_dict["masks"] # [B, num_slots, 1, H, W]
        return slots, masks

    def decode(self, slots):
        """Decode from slots to reconstructed images and masks."""
        out_dict = self.decoder(slots)
        recons = out_dict['reconstruction']
        masks = out_dict['masks']
        
        masks = F.softmax(masks, dim=1)  # [B, num_slots, 1, H, W]
        if "recon_combined" not in out_dict:
            recon_combined = torch.sum(recons * masks, dim=1) # [B, 3, H, W]
        else:
            recon_combined = out_dict["recon_combined"]
        return recon_combined, recons, masks, slots
    
    def forward(self, img, train=True):
        is_video = img.ndim == 5  # (B, T, C, H, W)
        
        if is_video:
            B, T, C, H, W = img.shape
            img = img.flatten(0, 1)  # → (B*T, C, H, W)
        else:
            B, T = img.shape[0], 1  # treat static images as 1-frame videos

        slots, masks_enc = self.encode(img)
        out_dict = {
            'slots': slots,
            'masks_enc': masks_enc,
            'video': img,
        }

        if train:
            recons_full, recons, masks_dec, slots = self.decode(slots)

            if is_video:
                loss = self.loss_function(img, recons_full)
                # reshape everything back to (B, T, ...)
                recon_combined = recons_full.unflatten(0, (B, T))
                recons = recons.unflatten(0, (B, T))
                masks_dec = masks_dec.unflatten(0, (B, T))
            else:
                loss = self.loss_function(img, recons_full)
                recon_combined = recons_full

            out_dict['masks_dec'] = masks_dec
            out_dict['recons'] = recons
            out_dict['recons_full'] = recon_combined
            out_dict['loss'] = loss
            out_dict['mse_loss'] = loss

        return out_dict

    def loss_function(self, img, recon_combined):
        """Compute the loss function."""
        loss = F.mse_loss(recon_combined, img, reduction='mean')
        return loss

    def output_shape(self):
        return self.output_shape
    
    

class SlotAttentionEncodeOnly(nn.Module):
    def __init__(
        self,
        input_shape,
        num_slots=8,
        slot_size=64,
        hidden_dim=512,
        num_iters=2,
        num_channels=3,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.resolution = input_shape
        self.num_iters = num_iters
        self.num_slots = num_slots
        self.slot_size = slot_size
        if self.resolution[0] == 128:
            self.visual_resolution = tuple(i // 2 for i in self.resolution)
            feature_multiplier = 1
            downsample = 1
        elif self.resolution[0] == 64:
            self.visual_resolution = self.resolution
            feature_multiplier = 0.5
            downsample = 0
        else:
            raise ValueError(f"Invalid resolution: {self.resolution} needed 128x128 or 64x64")

        self.init_latents = nn.Parameter(
                    nn.init.normal_(torch.empty(1, self.num_slots, self.slot_size)))

        ## ENCODER :Classical CNN
        self.encoder = make_slot_attention_encoder(
            inp_dim=self.num_channels,
            feature_multiplier=feature_multiplier,
            downsamplings=downsample,
        )
        self.visual_channels = int(64 * feature_multiplier)
        self.projection_layer = CoordinatePositionEmbed(self.visual_channels, self.visual_resolution, proj_dim=slot_size)

        ## OBJECT-CENTRIC MODULE : Slot-Attention for Video == SA + Transformer
        self.grouping = SlotAttentionLayer(
            inp_dim=slot_size,
            slot_dim=self.slot_size,
            kvq_dim=self.slot_size*2,
            n_iters=self.num_iters,
            hidden_dim=hidden_dim,
            use_gru=True,
            use_mlp=True,
        )


    def encode(self, img):
        B, C, H, W = img.shape

        h = self.encoder(img)
        h = self.projection_layer(h)

        # Extract slots
        prev_slots = self.init_latents.repeat(B, 1, 1)
        out_dict = self.grouping(prev_slots, h, n_iters=self.num_iters)
        slots = out_dict["slots"]  # [B, num_slots, slot_size]
        masks = out_dict["masks"] # [B, num_slots, 1, H, W]
        return slots, masks
    
    def forward(self, img, train=True):
        is_video = img.ndim == 5  # (B, T, C, H, W)
        
        if is_video:
            B, T, C, H, W = img.shape
            img = img.flatten(0, 1)  # → (B*T, C, H, W)
        else:
            B, T = img.shape[0], 1  # treat static images as 1-frame videos

        slots, masks_enc = self.encode(img)
        out_dict = {
            'slots': slots,
            'masks_enc': masks_enc,
            'video': img,
        }

        return out_dict

    def output_shape(self):
        return self.output_shape