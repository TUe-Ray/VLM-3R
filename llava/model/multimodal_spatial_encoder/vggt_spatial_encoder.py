import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from typing import Union, Optional, Tuple
import os
from llava.utils import rank0_print
import sys


def _resolve_vggt_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vlm_3r_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    candidates = [
        os.path.join(vlm_3r_root, 'third_party', 'VGGT'),
        os.path.join(vlm_3r_root, 'vggt'),
        os.path.join(vlm_3r_root, 'VGGT'),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


_VGGT_ROOT = _resolve_vggt_root()
# Ensure the resolved VGGT root wins import precedence over stale root-level checkouts.
for existing_path in list(sys.path):
    norm_path = os.path.normpath(existing_path)
    if os.path.basename(norm_path) in {"VGGT", "vggt"} and norm_path != os.path.normpath(_VGGT_ROOT):
        sys.path.remove(existing_path)
if _VGGT_ROOT in sys.path:
    sys.path.remove(_VGGT_ROOT)
sys.path.insert(0, _VGGT_ROOT)

try:
    from vggt.models.vggt import VGGT
except ImportError as vggt_err:
    raise ImportError(
        "Unable to import VGGT. Initialize the VGGT submodule with "
        "`git submodule update --init --recursive third_party/VGGT`."
    ) from vggt_err


def _load_vggt_model(weights_path):
    if os.path.isdir(weights_path):
        for filename in ("model.safetensors", "model.pt", "pytorch_model.bin"):
            candidate = os.path.join(weights_path, filename)
            if os.path.isfile(candidate):
                weights_path = candidate
                break
        else:
            return VGGT.from_pretrained(weights_path)

    if os.path.isfile(weights_path):
        model = VGGT()
        if weights_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        return model

    return VGGT.from_pretrained(weights_path)

class VGGTSpatialConfig(PretrainedConfig):
    model_type = "vggt_spatial_model"

    def __init__(
        self,
        weights_path: str = "facebook/VGGT-1B",
        image_size: int = 518,
        patch_size: int = 14,
        hidden_size: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weights_path = weights_path
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the spatial config dict if we are loading from Cut3rSpatialConfig
        if config_dict.get("model_type") == "vggt_spatial_model": # Or check if it's a base VGGT config?
            # Assuming a base VGGT config might not have 'spatial_config'. Adjust logic if needed.
            # If loading from a parent config that wraps this, you might need:
            # config_dict = config_dict["spatial_config"]
            pass # Keep config_dict as is if it's already the correct type

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type " f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.")

        return cls.from_dict(config_dict, **kwargs)

class VGGTSpatialPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VGGTSpatialConfig
    base_model_prefix = "vggt"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass

def prepare_input(pixel_values, image_size=518):
    # Accept:
    # 3D: (C, H, W)
    # 4D: (F, C, H, W)
    # 5D: (B, F, C, H, W)
    if not isinstance(pixel_values, torch.Tensor):
        raise ValueError(f"Expected pixel_values to be a torch.Tensor, got {type(pixel_values)}")

    if pixel_values.dim() == 3:
        pixel_values = pixel_values.unsqueeze(0)

    if pixel_values.dim() == 4:
        pixel_values = nn.functional.interpolate(
            pixel_values, size=(image_size, image_size), mode='bilinear', align_corners=False
        )
        pixel_values = pixel_values.unsqueeze(0)
    elif pixel_values.dim() == 5:
        B, F, C, H, W = pixel_values.shape
        pixel_values = pixel_values.reshape(B * F, C, H, W)
        pixel_values = nn.functional.interpolate(
            pixel_values, size=(image_size, image_size), mode='bilinear', align_corners=False
        )
        pixel_values = pixel_values.reshape(B, F, C, image_size, image_size)
    else:
        raise ValueError(f"Expected pixel_values to be 3D, 4D, or 5D, got shape {tuple(pixel_values.shape)}")

    # Convert from SigLIP-style [-1, 1] values to VGGT's expected [0, 1] range.
    pixel_values = pixel_values * 0.5 + 0.5
    return pixel_values

class VGGT_Encoder(nn.Module):
    def __init__(self, config: VGGTSpatialConfig, **kwargs):
        super().__init__()
        self.config = config
        # Load model using the path from the config
        rank0_print(f"Loading VGGT from: {config.weights_path}")
        self.vggt = _load_vggt_model(config.weights_path)
        self.vggt.eval()
        for param in self.vggt.parameters():
            param.requires_grad = False

    def forward(
        self,
        pixel_values,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        point_cloud_output_paths: Optional[list[str]] = None # Add for API consistency if needed, though not used here
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        views = prepare_input(pixel_values=pixel_values, image_size=self.config.image_size)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=views.dtype):
                aggregated_tokens_list, ps_idx = self.vggt.aggregator(views)

        # Last layer shape: (B, F, camera/register/patch tokens, 2048).
        # Return one token sequence per frame/image to match the fusion path.
        spatial_feat = aggregated_tokens_list[-1]
        spatial_feat = spatial_feat.to(pixel_values.dtype)
        B, F, num_tokens, hidden_size = spatial_feat.shape
        spatial_feat = spatial_feat.reshape(B * F, num_tokens, hidden_size)
        camera_token = spatial_feat[:, 0:1, :]
        patch_tokens = spatial_feat[:, ps_idx:, :]

        # for debug(visualize point cloud)
        # pts3d_pred, pts3d_conf = self.vggt.point_head(
        #                 aggregated_tokens_list, images=views, patch_start_idx=ps_idx
        #             )
        
        return (camera_token, patch_tokens)


class VGGT_SpatialTransformer(nn.Module):
    def __init__(self, config: VGGTSpatialConfig, **kwargs):
        super().__init__()
        self.config = config
        self.encoder = VGGT_Encoder(config=config, **kwargs)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return encoder_outputs

class VGGT_SpatialModel(VGGTSpatialPreTrainedModel):
    config_class = VGGTSpatialConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["VGGTSpatialEncoderLayer"]

    def __init__(self, config: VGGTSpatialConfig, **kwargs):
        super().__init__(config)

        self.spatial_model = VGGT_SpatialTransformer(config, **kwargs)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], config: VGGTSpatialConfig, **kwargs):
        # Use the provided config object
        model = cls(config=config, **kwargs)
        # Potentially load state dict here if pretrained_model_name_or_path points to a checkpoint
        # containing the wrapper model's state, not just the base VGGT model.
        # For now, it assumes the config dictates the base model loading within VGGT_SpatialTransformer.
        return model

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        point_cloud_output_paths: Optional[list[str]] = None # Add for API consistency
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.spatial_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # Pass point_cloud_output_paths if the underlying model needs it, though VGGT_SpatialTransformer doesn't use it currently
            # point_cloud_output_paths=point_cloud_output_paths
        )

class VGGTSpatialTower(nn.Module):
    def __init__(self, spatial_tower, spatial_tower_cfg, delay_load=True):
        super().__init__()

        self.is_loaded = False

        default_weights_path = "facebook/VGGT-1B"
        weights_path = (
            getattr(spatial_tower_cfg, 'vggt_weights_path', None)
            or getattr(spatial_tower_cfg, 'weights_path', None)
            or default_weights_path
        )
        image_size = getattr(spatial_tower_cfg, 'vggt_image_size', None) or getattr(spatial_tower_cfg, 'image_size', None) or 518
        patch_size = getattr(spatial_tower_cfg, 'vggt_patch_size', None) or getattr(spatial_tower_cfg, 'patch_size', None) or 14

        self.config = VGGTSpatialConfig(
            weights_path=weights_path,
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=2048,
        )

        self.spatial_tower_name = spatial_tower # Keep track of the logical name/identifier
        mm_tunable_parts = getattr(spatial_tower_cfg, "mm_tunable_parts", "") or ""

        if not delay_load:
            rank0_print(f"Loading spatial tower: {spatial_tower} using weights from {self.config.weights_path}")
            self.load_model()
        elif getattr(spatial_tower_cfg, "unfreeze_mm_spatial_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `spatial_tower` weights: `unfreeze_mm_spatial_tower`: True.")
            self.load_model()
        elif "mm_spatial_tower" in mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `spatial_tower` weights: `mm_tunable_parts` contains `mm_spatial_tower`.")
            self.load_model()
        else:
            # Store the config even if not loading immediately
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.spatial_tower_name))
            return

        # Load the VGGT_SpatialModel using the config which contains the weights_path
        # The spatial_tower_name might be used here if it points to a checkpoint containing the *entire* VGGTSpatialModel state,
        # rather than just the base VGGT weights. Adjust logic based on how checkpoints are saved/loaded.
        # Assuming spatial_tower_name is mainly an identifier and config.weights_path points to base VGGT weights.
        rank0_print(f"Instantiating VGGT_SpatialModel with config pointing to: {self.config.weights_path}")
        self.spatial_tower = VGGT_SpatialModel.from_pretrained(
            pretrained_model_name_or_path=self.spatial_tower_name, # Or potentially self.config.weights_path if appropriate
            config=self.config,
            device_map=device_map
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
                    image.to(device=self.device, dtype=self.dtype),
                    output_hidden_states=True,
                )
                image_features.append(image_forward_out)
        else:
            image_features = self.spatial_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)

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
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size
