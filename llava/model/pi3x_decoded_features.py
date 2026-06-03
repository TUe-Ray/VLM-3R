"""
Pi3XDecodedFeatures — accessor for pre-extracted Pi3X decoded feature .pt files
produced by scripts/extraction/extract_spatial_features_pi3x_decoded_schema.py.

Stored .pt schema
-----------------
{
  "frames": {
      "decoded_features": Tensor[F, T, C],   # main decoder output (C = 2048)
      "frame_idx":        Tensor[F],
  },
  "meta": {
      "decoded_pos_template": Tensor[T, 2],  # positional encoding, shared across frames
      "num_frames":       int,
      "input_size":       int,
      "patch_size":       int,
      "patch_start_idx":  int,               # = num_register_tokens = 5
  },
}

Token layout within T
---------------------
  [0 : patch_start_idx]   register / special tokens  (dim C = 2048)
  [patch_start_idx : ]    spatial patch tokens        (dim C = 2048)

Camera tokens (semantically correct)
-------------------------------------
Pi3 does NOT use a single register token as its camera representation.
The camera branch is a dedicated TransformerDecoder head:

    camera_hidden = pi3.camera_decoder(decoded_features, xpos=pos)  # (F, T, D)

where D = 512 (camera_decoder out_dim).  The register-token slice of this output:

    camera_tokens = camera_hidden[:, patch_start_idx:, :]             # (F, num_patches, D)

is the correct "camera token" representation — matching Pi3's camera_head which also
receives camera_hidden[:, patch_start_idx:] as input to produce camera poses.

Because the camera_decoder is a lightweight head (5 blocks, no image encoding),
it is run at training time from the stored decoded_features via:

    sf.compute_camera_tokens(pi3.camera_decoder, device=..., dtype=...)

This must be called before accessing sf.camera_tokens.

Legacy schema compatibility
---------------------------
Legacy flat schema is intentionally unsupported in this code path.
Any payload that stores pre-sliced "camera_tokens" directly is rejected,
so camera tokens can only come from camera_decoder(decoded_features, xpos).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class Pi3XDecodedFeatures:
    """
    Accessor around a loaded Pi3X decoded-feature dict.

    Construct via:
      - ``Pi3XDecodedFeatures(data_dict)``       (dict already in memory)
      - ``Pi3XDecodedFeatures.from_file(path)``  (loads from .pt file)
      - ``Pi3XDecodedFeatures.from_loaded(data)`` (dict or already-wrapped instance)
    """

    _NEW_SCHEMA_KEY = "frames"
    _LEGACY_CAM_KEY = "camera_tokens"

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, data: dict) -> None:
        if self._NEW_SCHEMA_KEY in data:
            self._mode = "new"
            self._frames = data["frames"]
            self._meta   = data["meta"]
            # Cache for camera_decoder output, populated by compute_camera_tokens()
            self._camera_decoder_cache: Optional[torch.Tensor] = None
            self._camera_pose_cache: Optional[torch.Tensor] = None
        elif self._LEGACY_CAM_KEY in data:
            raise ValueError(
                "Legacy Pi3X feature schema with pre-sliced camera_tokens is no longer supported. "
                "Please regenerate .pt features with scripts/extraction/extract_spatial_features_pi3x_decoded_schema.py "
                "so camera tokens are computed via camera_decoder at runtime."
            )
        else:
            raise ValueError(
                "Unrecognised Pi3X feature dict. Expected 'frames' key (new schema) "
                f"with decoded features metadata. Got: {list(data.keys())}"
            )

    @classmethod
    def from_file(cls, path: str, map_location: str = "cpu") -> "Pi3XDecodedFeatures":
        """Load a .pt file and return a Pi3XDecodedFeatures instance."""
        data = torch.load(path, map_location=map_location, weights_only=False)
        return cls(data)

    @classmethod
    def from_loaded(cls, data) -> "Pi3XDecodedFeatures":
        """
        Accept a raw dict (from torch.load) or an already-wrapped instance (no-op).
        Use this in llava_arch.py where the type depends on which script produced the file.
        """
        if isinstance(data, cls):
            return data
        return cls(data)

    # ------------------------------------------------------------------ #
    #  Core tensors                                                        #
    # ------------------------------------------------------------------ #

    @property
    def decoded_features(self) -> torch.Tensor:
        """Full main decoder output.  Shape: (F, T, C=2048).  New schema only."""
        return self._frames["decoded_features"]

    @property
    def frame_idx(self) -> Optional[torch.Tensor]:
        """Original video frame indices.  Shape: (F,)."""
        return self._frames["frame_idx"]

    @property
    def decoded_pos_template(self) -> Optional[torch.Tensor]:
        """
        Token positional-encoding template shared across all frames.
        Shape: (T, 2).
        """
        return self._meta["decoded_pos_template"]

    # ------------------------------------------------------------------ #
    #  Meta                                                                #
    # ------------------------------------------------------------------ #

    @property
    def patch_start_idx(self) -> int:
        """Number of register tokens = index where patch tokens start (= 5)."""
        return int(self._meta["patch_start_idx"])

    @property
    def num_frames(self) -> int:
        return int(self._meta["num_frames"])

    @property
    def input_size(self) -> Optional[int]:
        return int(self._meta["input_size"])

    @property
    def patch_size(self) -> Optional[int]:
        return int(self._meta["patch_size"])

    # ------------------------------------------------------------------ #
    #  Camera tokens — requires compute_camera_tokens() in new schema     #
    # ------------------------------------------------------------------ #

    def compute_camera_tokens(
        self,
        camera_decoder: nn.Module,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """
        Run pi3.camera_decoder on the stored decoded_features and cache the result.

        Must be called before accessing .camera_tokens in new schema mode.

        Args:
            camera_decoder: pi3.camera_decoder (TransformerDecoder, out_dim=512).
                            Typically obtained via spatial_tower.camera_decoder.
            device: target device for computation (e.g. 'cuda:0').
            dtype:  target dtype (e.g. torch.bfloat16).

        Returns:
            camera_decoder_features  Tensor[F, T, 512]
        """
        feats = self.decoded_features          # (F, T, 2048)
        pos   = self.decoded_pos_template      # (T, 2)
        F     = feats.shape[0]
        pos_expanded = pos.unsqueeze(0).expand(F, -1, -1)  # (F, T, 2)

        if device is not None:
            feats        = feats.to(device=device)
            pos_expanded = pos_expanded.to(device=device)
        if dtype is not None:
            feats = feats.to(dtype=dtype)

        with torch.no_grad():
            cam_feats = camera_decoder(feats, xpos=pos_expanded)  # (F, T, 512)

        self._camera_decoder_cache = cam_feats
        return cam_feats

    def compute_camera_pose(
        self,
        camera_head,          # pi3.camera_head (CameraHead module)
        patch_h: int,
        patch_w: int,
        device=None,
    ) -> torch.Tensor:
        """
        Run pi3.camera_head on camera_tokens to produce a compact 12-value pose
        representation (first 3 rows of the 4×4 pose matrix, flattened).

        Must be called AFTER compute_camera_tokens().

        Args:
            camera_head: pi3.camera_head — CameraHead(dim=512).
            patch_h:     number of patch rows  (= input_size // patch_size).
            patch_w:     number of patch cols  (= input_size // patch_size).
            device:      target device for the camera_head forward pass.

        Returns:
            Tensor of shape (F, 12).
        """
        cam_tokens = self.camera_tokens  # (F, num_patches, 512) – requires compute_camera_tokens first
        if device is not None:
            cam_tokens = cam_tokens.to(device=device)

        head_param = next(camera_head.parameters(), None)
        if head_param is not None and cam_tokens.dtype != head_param.dtype:
            cam_tokens = cam_tokens.to(dtype=head_param.dtype)

        with torch.no_grad():
            # Keep input dtype aligned with camera_head weights to avoid matmul dtype mismatch.
            pose_4x4 = camera_head(cam_tokens, patch_h, patch_w)   # (F, 4, 4)

        # Take first 3 rows (drop the fixed [0,0,0,1] row) and flatten → (F, 12)
        pose_12 = pose_4x4[:, :3, :].reshape(pose_4x4.shape[0], 12)
        self._camera_pose_cache = pose_12
        return pose_12

    @property
    def camera_tokens(self) -> torch.Tensor:
        """
        Register-token slice of the camera_decoder branch output.
        Shape: (F, num_patches, D=512), computed from camera_decoder output.

        Call compute_camera_tokens(pi3.camera_decoder) first.
        """
        if self._camera_decoder_cache is None:
            raise RuntimeError(
                "camera_tokens requires compute_camera_tokens(pi3.camera_decoder) "
                "to be called first. In llava_arch.py call "
                "_sf.compute_camera_tokens(spatial_tower.camera_decoder, ...)."
            )
        return self._camera_decoder_cache[:, self.patch_start_idx :, :]

    @property
    def camera_decoder_features(self) -> torch.Tensor:
        """
        Full camera_decoder branch output.  Shape: (F, T, D=512).
        Available only after compute_camera_tokens() has been called.
        """
        if self._camera_decoder_cache is None:
            raise RuntimeError(
                "camera_decoder_features requires compute_camera_tokens() to be called first."
            )
        return self._camera_decoder_cache

    @property
    def camera_pose(self) -> torch.Tensor:
        """
        12-value pose representation (first 3 rows of the 4×4 camera pose matrix).
        Shape: (F, 12).  Requires compute_camera_pose() to be called first.
        """
        if self._camera_pose_cache is None:
            raise RuntimeError(
                "camera_pose requires compute_camera_pose(camera_head, patch_h, patch_w) "
                "to be called first."
            )
        return self._camera_pose_cache

    # ------------------------------------------------------------------ #
    #  Patch tokens                                                        #
    # ------------------------------------------------------------------ #

    @property
    def register_tokens(self) -> torch.Tensor:
        """
        All register/special tokens from the main decoder (indices 0..patch_start_idx-1).
        Shape: (F, patch_start_idx, C=2048).  New schema only.
        """
        return self.decoded_features[:, : self.patch_start_idx, :]

    @property
    def patch_tokens(self) -> torch.Tensor:
        """
        Spatial patch tokens from the main decoder.
        Shape: (F, num_patches, C=2048).
        """
        return self.decoded_features[:, self.patch_start_idx :, :]

    @property
    def patch_pos(self) -> Optional[torch.Tensor]:
        """Positional encoding for patch tokens only.  Shape: (num_patches, 2)."""
        return self.decoded_pos_template[self.patch_start_idx :]

    @property
    def all_tokens(self) -> torch.Tensor:
        """
        All main-decoder tokens (register + patch), dim C=2048.
        Note: camera_decoder_features (dim D=512) are separate and accessed via
        camera_tokens / camera_decoder_features after compute_camera_tokens().
        """
        return self.decoded_features

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def to(self, device=None, dtype=None) -> "Pi3XDecodedFeatures":
        """Move/cast all stored tensors in-place and return self."""
        def _cast(t):
            if t is None:
                return t
            if device is not None and dtype is not None:
                return t.to(device=device, dtype=dtype)
            if device is not None:
                return t.to(device=device)
            if dtype is not None:
                return t.to(dtype=dtype)
            return t

        self._frames["decoded_features"] = _cast(self._frames["decoded_features"])
        if self._frames.get("frame_idx") is not None:
            self._frames["frame_idx"] = _cast(self._frames["frame_idx"])
        self._meta["decoded_pos_template"] = _cast(self._meta["decoded_pos_template"])
        if self._camera_decoder_cache is not None:
            self._camera_decoder_cache = _cast(self._camera_decoder_cache)
        if self._camera_pose_cache is not None:
            self._camera_pose_cache = _cast(self._camera_pose_cache)
        return self

    def is_new_schema(self) -> bool:
        return True

    def __repr__(self) -> str:
        f, t, c = self.decoded_features.shape
        cam_ready = self._camera_decoder_cache is not None
        return (
            f"Pi3XDecodedFeatures(schema=new, frames={f}, tokens={t}, "
            f"patch_start_idx={self.patch_start_idx}, C={c}, "
            f"camera_ready={cam_ready})"
        )
