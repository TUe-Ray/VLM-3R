import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from typing import Union, Optional, Tuple
import os
from llava.utils import rank0_print
from einops import rearrange
import sys

# Add pi3x package path
pi3x_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pi3x'))
if pi3x_path not in sys.path:
    sys.path.append(pi3x_path)
from pi3x.models.pi3 import Pi3


class Pi3xSpatialConfig(PretrainedConfig):
    model_type = "pi3x_spatial_model"

    def __init__(
        self,
        weights_path: str = "",
        input_size: int = 518,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weights_path = weights_path
        self.input_size = input_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get("model_type") == "pi3x_spatial_model":
            pass
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}.")
        return cls.from_dict(config_dict, **kwargs)


class Pi3xSpatialPreTrainedModel(PreTrainedModel):
    config_class = Pi3xSpatialConfig
    base_model_prefix = "pi3x"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        pass


def prepare_input(pixel_values, input_size=518):
    """Prepare input for Pi3X encoder.

    Args:
        pixel_values: (F, C, H, W) tensor in [-1, 1] range (from SigLIP preprocessing).
        input_size: Target size, must be a multiple of 14 (Pi3X patch size).
    Returns:
        Tensor of shape (1, F, C, input_size, input_size) in [0, 1] range.
    """
    # Resize to target size (must be multiple of 14 for DINOv2 patch size)
    pixel_values = nn.functional.interpolate(
        pixel_values, size=(input_size, input_size), mode='bilinear', align_corners=False
    )
    # Convert from [-1, 1] to [0, 1] (Pi3 forward handles ImageNet normalization)
    pixel_values = pixel_values * 0.5 + 0.5
    # Add batch dimension: (F, C, H, W) -> (1, F, C, H, W)
    pixel_values = pixel_values.unsqueeze(0)
    return pixel_values


class Pi3xEncoder(nn.Module):
    def __init__(self, config: Pi3xSpatialConfig, **kwargs):
        super().__init__()
        rank0_print(f"Loading Pi3X from: {config.weights_path}")
        self.pi3 = Pi3.from_pretrained(config.weights_path)
        self.pi3.eval()
        self.config = config
        for param in self.pi3.parameters():
            param.requires_grad = False

    def forward(
        self,
        pixel_values,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Pi3X encoder.

        Args:
            pixel_values: (F, C, H, W) frames in [-1, 1] range.
        Returns:
            Tuple of (camera_tokens, patch_tokens):
                camera_tokens: (F, 1, 2048) - first register token per frame
                patch_tokens: (F, num_patches, 2048) - spatial patch features
        """
        # Prepare input: resize, normalize to [0,1], add batch dim
        # DEBUG: check input stats
        # rank0_print(f"[Pi3X DEBUG] input pixel_values: shape={pixel_values.shape}, dtype={pixel_values.dtype}, "
        #              f"min={pixel_values.min().item():.4f}, max={pixel_values.max().item():.4f}, "
        #              f"mean={pixel_values.mean().item():.4f}, has_nan={pixel_values.isnan().any().item()}, "
        #              f"has_inf={pixel_values.isinf().any().item()}")

        imgs = prepare_input(pixel_values, input_size=self.config.input_size)

        # DEBUG: check after prepare_input (should be [0,1] range)
        # rank0_print(f"[Pi3X DEBUG] after prepare_input: shape={imgs.shape}, "
        #              f"min={imgs.min().item():.4f}, max={imgs.max().item():.4f}, "
        #              f"mean={imgs.mean().item():.4f}")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=pixel_values.dtype):
                # Pi3 forward: normalize with ImageNet stats, encode, decode
                # imgs shape: (B=1, N=num_frames, C, H, W)
                imgs_norm = (imgs - self.pi3.image_mean) / self.pi3.image_std
                B, N, _, H, W = imgs_norm.shape

                # DEBUG: check after ImageNet normalization
                # rank0_print(f"[Pi3X DEBUG] after imagenet norm: shape={imgs_norm.shape}, "
                #              f"min={imgs_norm.min().item():.4f}, max={imgs_norm.max().item():.4f}")

                # Encode frames through DINOv2
                imgs_flat = imgs_norm.reshape(B * N, _, H, W)
                hidden = self.pi3.encoder(imgs_flat, is_training=True)
                if isinstance(hidden, dict):
                    hidden = hidden["x_norm_patchtokens"]

                # DEBUG: check encoder output
                # rank0_print(f"[Pi3X DEBUG] encoder hidden: shape={hidden.shape}, "
                #              f"min={hidden.min().item():.4f}, max={hidden.max().item():.4f}, "
                #              f"mean={hidden.mean().item():.4f}, has_nan={hidden.isnan().any().item()}, "
                #              f"has_inf={hidden.isinf().any().item()}")

                # Decode: apply register tokens + transformer decoder
                # Output shape: (B*N, num_patches + num_register, 2 * dec_embed_dim)
                features, pos = self.pi3.decode(hidden, N, H, W)

        # features shape: (B*N, num_register + num_patches, 2048)
        features = features.to(pixel_values.dtype)

        # DEBUG: check decoder output
        # rank0_print(f"[Pi3X DEBUG] decoder features: shape={features.shape}, dtype={features.dtype}, "
        #              f"min={features.min().item():.4f}, max={features.max().item():.4f}, "
        #              f"mean={features.mean().item():.4f}, std={features.std().item():.4f}, "
        #              f"has_nan={features.isnan().any().item()}, has_inf={features.isinf().any().item()}")

        # Split into register tokens (camera) and patch tokens
        ps_idx = self.pi3.patch_start_idx  # 5 (number of register tokens)
        camera_tokens = features[:, 0:1, :]       # (N, 1, 2048) - first register token
        patch_tokens = features[:, ps_idx:, :]     # (N, num_patches, 2048)

        # DEBUG: check final output tokens
        # rank0_print(f"[Pi3X DEBUG] camera_tokens: shape={camera_tokens.shape}, "
        #              f"min={camera_tokens.min().item():.4f}, max={camera_tokens.max().item():.4f}, "
        #              f"mean={camera_tokens.mean().item():.4f}, has_nan={camera_tokens.isnan().any().item()}")
        # rank0_print(f"[Pi3X DEBUG] patch_tokens: shape={patch_tokens.shape}, "
        #              f"min={patch_tokens.min().item():.4f}, max={patch_tokens.max().item():.4f}, "
        #              f"mean={patch_tokens.mean().item():.4f}, has_nan={patch_tokens.isnan().any().item()}")

        return (camera_tokens, patch_tokens)


class Pi3xSpatialTransformer(nn.Module):
    def __init__(self, config: Pi3xSpatialConfig, **kwargs):
        super().__init__()
        self.config = config
        self.encoder = Pi3xEncoder(config=config, **kwargs)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Pi3xSpatialModel(Pi3xSpatialPreTrainedModel):
    config_class = Pi3xSpatialConfig
    main_input_name = "pixel_values"

    def __init__(self, config: Pi3xSpatialConfig, **kwargs):
        super().__init__(config)
        self.spatial_model = Pi3xSpatialTransformer(config, **kwargs)
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], config: Pi3xSpatialConfig, **kwargs):
        model = cls(config=config, **kwargs)
        return model

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.spatial_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Pi3xSpatialTower(nn.Module):
    def __init__(self, spatial_tower, spatial_tower_cfg, delay_load=True):
        super().__init__()

        self.is_loaded = False

        # Default weights path
        default_weights_path = "/leonardo_scratch/fast/EUHPC_D32_006/hf_models/Pi3X"
        weights_path = getattr(spatial_tower_cfg, 'pi3x_weights_path', default_weights_path)

        self.config = Pi3xSpatialConfig(
            weights_path=weights_path,
            input_size=getattr(spatial_tower_cfg, 'pi3x_input_size', 518),
        )

        self.spatial_tower_name = spatial_tower
        mm_tunable_parts = getattr(spatial_tower_cfg, "mm_tunable_parts", "") or ""

        if not delay_load:
            rank0_print(f"Loading spatial tower: {spatial_tower} using weights from {self.config.weights_path}")
            self.load_model()
        elif getattr(spatial_tower_cfg, "unfreeze_mm_spatial_tower", False):
            rank0_print(f"The checkpoint seems to contain `spatial_tower` weights: `unfreeze_mm_spatial_tower`: True.")
            self.load_model()
        elif "mm_spatial_tower" in mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `spatial_tower` weights: `mm_tunable_parts` contains `mm_spatial_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.spatial_tower_name))
            return

        rank0_print(f"Instantiating Pi3xSpatialModel with weights from: {self.config.weights_path}")
        self.spatial_tower = Pi3xSpatialModel.from_pretrained(
            pretrained_model_name_or_path=self.spatial_tower_name,
            config=self.config,
            device_map=device_map,
        )
        self.spatial_tower.requires_grad_(False)
        self.is_loaded = True

    def forward(self, images):
        if not self.is_loaded or not hasattr(self, "spatial_tower"):
            self.load_model()

        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.spatial_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_features.append(image_forward_out)
        else:
            image_features = self.spatial_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.spatial_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.spatial_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return 2048  # Pi3X: 2 * dec_embed_dim = 2 * 1024
