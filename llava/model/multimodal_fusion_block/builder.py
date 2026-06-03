import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from .cross_attention_transformers import MultiLayerCrossAttentionFusion
from .cross_attention_mlp import CrossAttentionFusionWithMLP
from .video_3d_llm_block import video_3d_llm_fusion_block

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_clip, d_spatial_encoder, d_attn, num_heads):
        super(CrossAttentionFusion, self).__init__()
        
        # pre-norm
        self.clip_norm = nn.LayerNorm(d_clip)
        self.spatial_encoder_norm = nn.LayerNorm(d_spatial_encoder)
        
        # projection
        self.clip_query_proj = nn.Linear(d_clip, d_attn)
        self.spatial_encoder_key_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.spatial_encoder_value_proj = nn.Linear(d_spatial_encoder, d_attn)
        
        # cross attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)
        
        # post-norm
        self.out_norm = nn.LayerNorm(d_attn)
        
        # projection
        self.out_proj = nn.Linear(d_attn, d_clip)
        
        # dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, clip_features, spatial_encoder_features):
        """
        Args:
            clip_features: [B, N, D_clip]
            spatial_encoder_features: [B, N, D_spatial_encoder]
        Returns:
            fused_features: [B, N, D_clip]
        """
        target_device = clip_features.device
        param = next(self.parameters(), None)
        if param is not None and param.device != target_device:
            self.to(device=target_device)
        if spatial_encoder_features.device != target_device:
            spatial_encoder_features = spatial_encoder_features.to(device=target_device)

        # pre-norm
        clip_features_norm = self.clip_norm(clip_features)  # [B, N, D_clip]
        spatial_encoder_features_norm = self.spatial_encoder_norm(spatial_encoder_features)  # [B, N, D_spatial_encoder]
        
        # projection to D_attn dimension
        clip_query_proj = self.clip_query_proj(clip_features_norm)  # [B, N, D_attn]
        spatial_encoder_key_proj = self.spatial_encoder_key_proj(spatial_encoder_features_norm)  # [B, N, D_attn]
        spatial_encoder_value_proj = self.spatial_encoder_value_proj(spatial_encoder_features_norm)  # [B, N, D_attn]
        
        # cross attention
        fused_features, attn_weights = self.cross_attention(
            query=clip_query_proj,
            key=spatial_encoder_key_proj,
            value=spatial_encoder_value_proj
        )
        
        # projection to D_clip dimension
        fused_features = self.out_proj(fused_features)   # [B, N_clip, D_clip]
        
        # residual connection and dropout
        fused_features = self.out_norm(fused_features)
        fused_features = fused_features + clip_features  # [B, N_clip, D_clip]
        # print(f'status_of_fused_features: max:{fused_features.max():.2f}, min:{fused_features.min():.2f}, mean:{fused_features.mean():.2f}, std:{fused_features.std():.2f}')
        # print(f'status_of_clip_features: max:{clip_features.max():.2f}, min:{clip_features.min():.2f}, mean:{clip_features.mean():.2f}, std:{clip_features.std():.2f}')
        fused_features = self.dropout(fused_features)
        
        return fused_features, attn_weights


class GeoRoPEFusionRotary(nn.Module):
    def __init__(self, head_dim, mode="spherical", theta=10000, max_depth=10.0, group_split=None):
        super().__init__()
        if mode not in {"depth", "xyz", "spherical"}:
            raise ValueError(f"Unexpected geo_rope_fusion_mode: {mode}")

        self.head_dim = int(head_dim)
        self.mode = mode
        self.theta = theta
        self.max_depth = float(max_depth)
        if self.max_depth <= 0:
            raise ValueError(f"geo_rope_fusion_max_depth must be positive, got {max_depth}")

        weights, self.group_split = self._parse_group_split(group_split, mode)
        group_dims = self._split_even_dims(self.head_dim, weights=weights)
        self.group_dims = group_dims

        for idx, group_dim in enumerate(group_dims):
            inv_freq = self._build_inv_freq(group_dim)
            self.register_buffer(f"inv_freq_{idx}", inv_freq, persistent=False)

    @staticmethod
    def _parse_group_split(group_split, mode):
        if mode == "depth":
            if group_split is None or str(group_split).strip() == "1":
                return [1], "1"
            raise ValueError("depth mode expects group_split='1' or None")

        if group_split is None:
            group_split = "2,1,2"
        group_split = str(group_split).replace("|", ",").replace(";", ",")

        try:
            weights = [int(value.strip()) for value in group_split.split(",")]
        except ValueError as exc:
            raise ValueError(f"{mode} mode group_split must contain integers: {group_split}") from exc

        if len(weights) != 3:
            raise ValueError(f"{mode} mode expects exactly 3 split values")
        if any(weight <= 0 for weight in weights):
            raise ValueError("group split values must be positive")
        return weights, ",".join(str(weight) for weight in weights)

    @staticmethod
    def _split_even_dims(head_dim, weights):
        num_pairs = head_dim // 2
        total_weight = sum(weights)
        pair_counts = [(num_pairs * weight) // total_weight for weight in weights]
        pair_counts[-1] += num_pairs - sum(pair_counts)
        return [2 * count for count in pair_counts]

    def _build_inv_freq(self, group_dim):
        if group_dim <= 0:
            return torch.empty(0)
        return 1.0 / (self.theta ** (torch.arange(0, group_dim, 2).float() / group_dim))

    def _position_values(self, pos):
        pos = torch.nan_to_num(pos.float(), nan=0.0, posinf=self.max_depth, neginf=-self.max_depth)
        x = pos[..., 0]
        y = pos[..., 1]
        z = pos[..., 2]

        if self.mode == "depth":
            depth = z.clamp(min=0.0, max=self.max_depth)
            return [torch.log1p(depth) / math.log1p(self.max_depth)]

        if self.mode == "xyz":
            scale = max(self.max_depth, 1e-6)
            x = (x / scale).clamp(min=-1.0, max=1.0)
            y = (y / scale).clamp(min=-1.0, max=1.0)
            z = z.clamp(min=0.0, max=self.max_depth)
            z = torch.log1p(z) / math.log1p(self.max_depth)
            return [x, y, z]

        radius_xz = torch.sqrt(x * x + z * z + 1e-6)
        r = torch.sqrt(x * x + y * y + z * z + 1e-6)
        azimuth = torch.atan2(x, z)
        elevation = torch.atan2(y, radius_xz)
        log_r = torch.log1p(r.clamp(max=self.max_depth)) / math.log1p(self.max_depth)
        return [azimuth, elevation, log_r]

    @staticmethod
    def _rotate_group(x_group, position_value, inv_freq):
        if x_group.shape[-1] == 0:
            return x_group

        inv_freq = inv_freq.to(device=x_group.device, dtype=position_value.dtype)
        angle = position_value.unsqueeze(-1) * inv_freq
        cos = angle.cos().unsqueeze(1).to(dtype=x_group.dtype)
        sin = angle.sin().unsqueeze(1).to(dtype=x_group.dtype)

        x_even = x_group[..., 0::2]
        x_odd = x_group[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)

    def forward(self, x, pos):
        """
        Args:
            x: [B, H, N, head_dim]
            pos: [B, N, 3]
        Returns:
            x_rotated: [B, H, N, head_dim]
        """
        if x.dim() != 4:
            raise ValueError(f"GeoRoPEFusionRotary expects x as [B, H, N, D], got {tuple(x.shape)}")
        if pos.dim() != 3 or pos.shape[-1] != 3:
            raise ValueError(f"GeoRoPEFusionRotary expects pos as [B, N, 3], got {tuple(pos.shape)}")
        if x.shape[0] != pos.shape[0] or x.shape[2] != pos.shape[1]:
            raise ValueError(
                "GeoRoPEFusionRotary position shape must match x batch/token dims, "
                f"got x={tuple(x.shape)} and pos={tuple(pos.shape)}"
            )

        position_values = self._position_values(pos.to(device=x.device))
        rotated_groups = []
        start = 0
        for idx, group_dim in enumerate(self.group_dims):
            end = start + group_dim
            x_group = x[..., start:end]
            inv_freq = getattr(self, f"inv_freq_{idx}")
            rotated_groups.append(self._rotate_group(x_group, position_values[idx], inv_freq))
            start = end

        if start < x.shape[-1]:
            rotated_groups.append(x[..., start:])
        return torch.cat(rotated_groups, dim=-1)


class GeoRoPEFusionCrossAttention(nn.Module):
    GATE_TYPES = {"scalar", "forced", "per_head"}

    def __init__(
        self,
        d_clip,
        d_spatial_encoder,
        d_attn,
        num_heads,
        geo_rope_fusion_mode="spherical",
        max_depth=10.0,
        group_split=None,
        log_stats=False,
        log_attention_stats=False,
        gate_type="scalar",
        head_gate_init=0.99,
        dropout_rate=0.1,
    ):
        super().__init__()
        if d_attn % num_heads != 0:
            raise ValueError(f"d_attn ({d_attn}) must be divisible by num_heads ({num_heads})")
        if gate_type not in self.GATE_TYPES:
            raise ValueError(f"Unexpected GeoRoPE Fusion gate_type: {gate_type}")

        self.d_attn = d_attn
        self.num_heads = num_heads
        self.head_dim = d_attn // num_heads
        self.gate_type = gate_type

        self.clip_norm = nn.LayerNorm(d_clip)
        self.spatial_encoder_norm = nn.LayerNorm(d_spatial_encoder)

        self.clip_query_proj = nn.Linear(d_clip, d_attn)
        self.spatial_encoder_key_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.spatial_encoder_value_proj = nn.Linear(d_spatial_encoder, d_attn)

        # Manual equivalent of nn.MultiheadAttention's internal projections.
        self.attn_query_proj = nn.Linear(d_attn, d_attn)
        self.attn_key_proj = nn.Linear(d_attn, d_attn)
        self.attn_value_proj = nn.Linear(d_attn, d_attn)
        self.attn_out_proj = nn.Linear(d_attn, d_attn)
        self._reset_attention_parameters()

        self.geo_rope_fusion = GeoRoPEFusionRotary(
            head_dim=self.head_dim,
            mode=geo_rope_fusion_mode,
            max_depth=max_depth,
            group_split=group_split,
        )
        if gate_type == "scalar":
            self.geo_rope_fusion_gate_q = nn.Parameter(torch.zeros(()))
            self.geo_rope_fusion_gate_k = nn.Parameter(torch.zeros(()))
        elif gate_type == "per_head":
            init_gate = min(max(float(head_gate_init), 1e-6), 1.0 - 1e-6)
            init_logit = math.log(init_gate / (1.0 - init_gate))
            self.geo_rope_fusion_head_gate_logit = nn.Parameter(
                torch.full((num_heads,), init_logit)
            )
            self._last_head_gate_grad_norm = None
            self.geo_rope_fusion_head_gate_logit.register_hook(self._record_head_gate_grad_norm)

        self.out_proj = nn.Linear(d_attn, d_clip)
        self.out_norm = nn.LayerNorm(d_clip)
        self.dropout = nn.Dropout(dropout_rate)
        self.log_stats = log_stats
        self.log_attention_stats = log_attention_stats
        self.last_geo_rope_fusion_stats = {
            "gate_type": self.gate_type,
            "group_split": self.geo_rope_fusion.group_split,
            "group_dims": list(self.geo_rope_fusion.group_dims),
        }

    def _record_head_gate_grad_norm(self, grad):
        self._last_head_gate_grad_norm = float(grad.detach().float().norm().item())
        return grad

    def _reset_attention_parameters(self):
        in_proj_weight = torch.empty(3 * self.d_attn, self.d_attn)
        nn.init.xavier_uniform_(in_proj_weight)
        with torch.no_grad():
            self.attn_query_proj.weight.copy_(in_proj_weight[:self.d_attn])
            self.attn_key_proj.weight.copy_(in_proj_weight[self.d_attn:2 * self.d_attn])
            self.attn_value_proj.weight.copy_(in_proj_weight[2 * self.d_attn:])
            self.attn_query_proj.bias.zero_()
            self.attn_key_proj.bias.zero_()
            self.attn_value_proj.bias.zero_()

    def _reshape_to_heads(self, x):
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_attn)

    @staticmethod
    def _mean_std(x):
        x = x.detach().float()
        return {
            "mean": x.mean().item(),
            "std": x.std(unbiased=False).item(),
        }

    @staticmethod
    def _mean_abs(x):
        return x.detach().float().abs().mean().item()

    @staticmethod
    def _attention_entropy(attn):
        attn = attn.detach().float().clamp_min(1e-12)
        return float((-(attn * attn.log()).sum(dim=-1)).mean().item())

    def _record_stats(self, pos_clip, pos_spatial, q_delta, k_delta, q_effective_delta, k_effective_delta, gate=None):
        self.last_geo_rope_fusion_stats = {
            "gate_type": self.gate_type,
            "pos_clip": self._mean_std(pos_clip),
            "pos_spatial": self._mean_std(pos_spatial),
            "q_geo_rope_fusion_minus_q": self._mean_std(q_delta),
            "k_geo_rope_fusion_minus_k": self._mean_std(k_delta),
            "mean_abs_rope_delta_q": self._mean_abs(q_delta),
            "mean_abs_rope_delta_k": self._mean_abs(k_delta),
            "effective_rope_delta_q": self._mean_abs(q_effective_delta),
            "effective_rope_delta_k": self._mean_abs(k_effective_delta),
            "group_split": self.geo_rope_fusion.group_split,
            "group_dims": list(self.geo_rope_fusion.group_dims),
        }
        if self.gate_type == "scalar":
            self.last_geo_rope_fusion_stats.update({
                "geo_rope_fusion_gate_q": float(self.geo_rope_fusion_gate_q.detach().float().item()),
                "geo_rope_fusion_gate_k": float(self.geo_rope_fusion_gate_k.detach().float().item()),
            })
        elif self.gate_type == "per_head":
            gate = gate.detach().float().reshape(-1)
            self.last_geo_rope_fusion_stats.update({
                "gate_mean": float(gate.mean().item()),
                "gate_min": float(gate.min().item()),
                "gate_max": float(gate.max().item()),
                "gate_std": float(gate.std(unbiased=False).item()),
                "gate_per_head_values": [float(value) for value in gate.tolist()],
                "head_gate_shape": [1, self.num_heads, 1, 1],
                "gate_logit_grad_norm": self._last_head_gate_grad_norm,
            })

    def _record_attention_stats(self, attn, q_plain, k_plain):
        if not self.log_stats or not self.log_attention_stats:
            return
        with torch.no_grad():
            attn_plain = torch.matmul(q_plain, k_plain.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_plain = F.softmax(attn_plain.float(), dim=-1).to(dtype=attn.dtype)
            self.last_geo_rope_fusion_stats.update({
                "attention_delta_mean_abs": self._mean_abs(attn - attn_plain),
                "attention_entropy_rope": self._attention_entropy(attn),
                "attention_entropy_plain": self._attention_entropy(attn_plain),
            })

    def forward(self, clip_features, spatial_encoder_features, pos_clip, pos_spatial):
        """
        Args:
            clip_features: [B, N_clip, D_clip]
            spatial_encoder_features: [B, N_spatial, D_spatial_encoder]
            pos_clip: [B, N_clip, 3]
            pos_spatial: [B, N_spatial, 3]
        Returns:
            fused_features: [B, N_clip, D_clip]
            attn_weights: [B, N_clip, N_spatial]
        """
        clip_features_norm = self.clip_norm(clip_features)
        spatial_encoder_features_norm = self.spatial_encoder_norm(spatial_encoder_features)

        q = self.clip_query_proj(clip_features_norm)
        k = self.spatial_encoder_key_proj(spatial_encoder_features_norm)
        v = self.spatial_encoder_value_proj(spatial_encoder_features_norm)

        q = self.attn_query_proj(q)
        k = self.attn_key_proj(k)
        v = self.attn_value_proj(v)

        q = self._reshape_to_heads(q)
        k = self._reshape_to_heads(k)
        v = self._reshape_to_heads(v)
        q_plain = q
        k_plain = k

        q_geo_rope_fusion = self.geo_rope_fusion(q, pos_clip)
        k_geo_rope_fusion = self.geo_rope_fusion(k, pos_spatial)
        q_delta = q_geo_rope_fusion - q
        k_delta = k_geo_rope_fusion - k
        gate = None
        if self.gate_type == "scalar":
            q = q + self.geo_rope_fusion_gate_q * q_delta
            k = k + self.geo_rope_fusion_gate_k * k_delta
        elif self.gate_type == "forced":
            q = q_geo_rope_fusion
            k = k_geo_rope_fusion
        elif self.gate_type == "per_head":
            gate = torch.sigmoid(self.geo_rope_fusion_head_gate_logit).view(1, self.num_heads, 1, 1)
            q = q + gate * q_delta
            k = k + gate * k_delta
        if self.log_stats:
            self._record_stats(pos_clip, pos_spatial, q_delta, k_delta, q - q_plain, k - k_plain, gate=gate)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn.float(), dim=-1).to(dtype=q.dtype)
        self._record_attention_stats(attn, q_plain, k_plain)
        out = torch.matmul(attn, v)

        out = self._merge_heads(out)
        out = self.attn_out_proj(out)
        out = self.out_proj(out)
        out = self.out_norm(out)
        out = clip_features + out
        out = self.dropout(out)
        return out, attn.mean(dim=1)


class PostHocGeoRoPEPatchCrossAttention(nn.Module):
    """
    Eval-only variant for injecting Geometry-RoPE into a trained svf_patch_only
    checkpoint without changing the learned baseline parameter names.
    """
    def __init__(
        self,
        d_clip,
        d_spatial_encoder,
        d_attn,
        num_heads,
        geo_rope_fusion_mode="spherical",
        max_depth=10.0,
        group_split=None,
        eval_lambda=0.0,
        eval_lambda_q=None,
        eval_lambda_k=None,
        log_stats=False,
    ):
        super().__init__()
        if d_attn % num_heads != 0:
            raise ValueError(f"d_attn ({d_attn}) must be divisible by num_heads ({num_heads})")

        self.d_attn = d_attn
        self.num_heads = num_heads
        self.head_dim = d_attn // num_heads

        self.clip_norm = nn.LayerNorm(d_clip)
        self.spatial_encoder_norm = nn.LayerNorm(d_spatial_encoder)
        self.clip_query_proj = nn.Linear(d_clip, d_attn)
        self.spatial_encoder_key_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.spatial_encoder_value_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(d_attn)
        self.out_proj = nn.Linear(d_attn, d_clip)
        self.dropout = nn.Dropout(0.1)

        self.geo_rope_fusion = GeoRoPEFusionRotary(
            head_dim=self.head_dim,
            mode=geo_rope_fusion_mode,
            max_depth=max_depth,
            group_split=group_split,
        )
        self.eval_lambda_q = float(eval_lambda if eval_lambda_q is None else eval_lambda_q)
        self.eval_lambda_k = float(eval_lambda if eval_lambda_k is None else eval_lambda_k)
        self.log_stats = log_stats
        self.last_geo_rope_fusion_stats = {
            "eval_lambda_q": self.eval_lambda_q,
            "eval_lambda_k": self.eval_lambda_k,
            "group_split": self.geo_rope_fusion.group_split,
            "group_dims": list(self.geo_rope_fusion.group_dims),
        }

    def _reshape_to_heads(self, x):
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_attn)

    @staticmethod
    def _mean_std(x):
        x = x.detach().float()
        return {
            "mean": x.mean().item(),
            "std": x.std(unbiased=False).item(),
        }

    def _record_stats(self, pos_clip, pos_spatial, q_delta, k_delta):
        self.last_geo_rope_fusion_stats = {
            "pos_clip": self._mean_std(pos_clip),
            "pos_spatial": self._mean_std(pos_spatial),
            "q_geo_rope_fusion_minus_q": self._mean_std(q_delta),
            "k_geo_rope_fusion_minus_k": self._mean_std(k_delta),
            "eval_lambda_q": self.eval_lambda_q,
            "eval_lambda_k": self.eval_lambda_k,
            "group_split": self.geo_rope_fusion.group_split,
            "group_dims": list(self.geo_rope_fusion.group_dims),
        }

    def forward(self, clip_features, spatial_encoder_features, pos_clip, pos_spatial):
        clip_features_norm = self.clip_norm(clip_features)
        spatial_encoder_features_norm = self.spatial_encoder_norm(spatial_encoder_features)

        q = self.clip_query_proj(clip_features_norm)
        k = self.spatial_encoder_key_proj(spatial_encoder_features_norm)
        v = self.spatial_encoder_value_proj(spatial_encoder_features_norm)

        in_proj_weight = self.cross_attention.in_proj_weight
        in_proj_bias = self.cross_attention.in_proj_bias
        q = F.linear(q, in_proj_weight[:self.d_attn], in_proj_bias[:self.d_attn])
        k = F.linear(k, in_proj_weight[self.d_attn:2 * self.d_attn], in_proj_bias[self.d_attn:2 * self.d_attn])
        v = F.linear(v, in_proj_weight[2 * self.d_attn:], in_proj_bias[2 * self.d_attn:])

        q = self._reshape_to_heads(q)
        k = self._reshape_to_heads(k)
        v = self._reshape_to_heads(v)

        q_geo_rope_fusion = self.geo_rope_fusion(q, pos_clip)
        k_geo_rope_fusion = self.geo_rope_fusion(k, pos_spatial)
        q_delta = q_geo_rope_fusion - q
        k_delta = k_geo_rope_fusion - k
        if self.log_stats:
            self._record_stats(pos_clip, pos_spatial, q_delta, k_delta)

        q = q + self.eval_lambda_q * q_delta
        k = k + self.eval_lambda_k * k_delta

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn.float(), dim=-1).to(dtype=q.dtype)
        fused_features = torch.matmul(attn, v)
        fused_features = self._merge_heads(fused_features)
        fused_features = self.cross_attention.out_proj(fused_features)
        fused_features = self.out_proj(fused_features)
        fused_features = self.out_norm(fused_features)
        fused_features = fused_features + clip_features
        fused_features = self.dropout(fused_features)

        return fused_features, attn.mean(dim=1)


class PatchCrossAttentionCameraConcatFusion(nn.Module):
    def __init__(self, d_clip, d_spatial_encoder, d_attn, num_heads, dropout_rate=0.1, d_camera_encoder=None):
        super(PatchCrossAttentionCameraConcatFusion, self).__init__()

        # d_camera_encoder defaults to d_spatial_encoder for backward compatibility.
        # Set to 512 when using the new decoded schema (camera_decoder out_dim).
        _d_cam = d_camera_encoder or d_spatial_encoder

        self.clip_norm = nn.LayerNorm(d_clip)
        self.patch_norm = nn.LayerNorm(d_spatial_encoder)
        self.camera_norm = nn.LayerNorm(_d_cam)

        self.clip_query_proj = nn.Linear(d_clip, d_attn)
        self.patch_key_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.patch_value_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)

        self.out_proj = nn.Linear(d_attn, d_clip)
        self.out_norm = nn.LayerNorm(d_clip)
        self.camera_to_clip_proj = nn.Linear(_d_cam, d_clip)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, clip_features, patch_tokens, camera_tokens):
        clip_query = self.clip_query_proj(self.clip_norm(clip_features))
        patch_tokens_norm = self.patch_norm(patch_tokens)
        patch_key = self.patch_key_proj(patch_tokens_norm)
        patch_value = self.patch_value_proj(patch_tokens_norm)

        attn_output, attn_weights = self.cross_attention(
            query=clip_query,
            key=patch_key,
            value=patch_value,
        )

        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        fused_2d_tokens = self.out_norm(clip_features + attn_output)

        projected_camera_tokens = self.camera_to_clip_proj(self.camera_norm(camera_tokens))
        final_tokens = torch.cat((projected_camera_tokens, fused_2d_tokens), dim=1)
        return final_tokens, attn_weights


class GeometryBridgeFusion(nn.Module):
    def __init__(self, d_clip, d_spatial_encoder, d_attn, num_heads, dropout_rate=0.1, d_camera_encoder=None):
        super(GeometryBridgeFusion, self).__init__()

        # d_camera_encoder defaults to d_spatial_encoder for backward compatibility.
        # Set to 512 when using the new decoded schema (camera_decoder out_dim).
        _d_cam = d_camera_encoder or d_spatial_encoder

        # Stage 1: camera query attends to spatial patch tokens.
        self.camera_norm = nn.LayerNorm(_d_cam)
        self.patch_norm = nn.LayerNorm(d_spatial_encoder)
        self.camera_query_proj = nn.Linear(_d_cam, d_attn)
        self.patch_key_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.patch_value_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.geometry_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)

        # Stage 2: 2D tokens attend to geometry-aware 3D tokens from stage 1.
        self.clip_norm = nn.LayerNorm(d_clip)
        self.clip_query_proj = nn.Linear(d_clip, d_attn)
        self.geometry_key_proj = nn.Linear(d_attn, d_attn)
        self.geometry_value_proj = nn.Linear(d_attn, d_attn)
        self.bridge_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)

        self.out_proj = nn.Linear(d_attn, d_clip)
        self.out_norm = nn.LayerNorm(d_clip)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, clip_features, camera_tokens, patch_tokens):
        camera_tokens_norm = self.camera_norm(camera_tokens)
        patch_tokens_norm = self.patch_norm(patch_tokens)

        camera_query = self.camera_query_proj(camera_tokens_norm)
        patch_key = self.patch_key_proj(patch_tokens_norm)
        patch_value = self.patch_value_proj(patch_tokens_norm)

        geometry_tokens, camera_patch_attn = self.geometry_attention(
            query=camera_query,
            key=patch_key,
            value=patch_value,
        )

        clip_query = self.clip_query_proj(self.clip_norm(clip_features))
        geometry_key = self.geometry_key_proj(geometry_tokens)
        geometry_value = self.geometry_value_proj(geometry_tokens)

        fused_features, clip_geometry_attn = self.bridge_attention(
            query=clip_query,
            key=geometry_key,
            value=geometry_value,
        )

        fused_features = self.out_proj(fused_features)
        fused_features = self.dropout(fused_features)
        fused_features = self.out_norm(clip_features + fused_features)

        attn_weights = {
            "camera_to_patch": camera_patch_attn,
            "clip_to_geometry": clip_geometry_attn,
        }
        return fused_features, attn_weights

class SvfCatFeatFusion(nn.Module):
    """
    Comparison 1: concatenate camera_tokens and patch_tokens along the feature
    dimension, then let 2D features cross-attend the combined KV.

    camera_tokens: (F, P, d_camera)    e.g. (F, P, 512)
    patch_tokens:  (F, P, d_patch)     e.g. (F, P, 2048)
    combined:      (F, P, d_camera+d_patch)  e.g. (F, P, 2560)
    """
    def __init__(self, d_clip, d_camera_encoder, d_spatial_encoder, d_attn, num_heads, dropout_rate=0.1):
        super().__init__()
        d_combined = d_camera_encoder + d_spatial_encoder

        self.clip_norm    = nn.LayerNorm(d_clip)
        self.spatial_norm = nn.LayerNorm(d_combined)

        self.clip_query_proj    = nn.Linear(d_clip,     d_attn)
        self.spatial_key_proj   = nn.Linear(d_combined, d_attn)
        self.spatial_value_proj = nn.Linear(d_combined, d_attn)

        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)

        self.out_proj = nn.Linear(d_attn, d_clip)
        self.out_norm = nn.LayerNorm(d_clip)
        self.dropout  = nn.Dropout(dropout_rate)

    def forward(self, clip_features, camera_tokens, patch_tokens):
        # cat along feature dim: (F, P, 512+2048) = (F, P, 2560)
        spatial = torch.cat([camera_tokens, patch_tokens], dim=-1)

        Q = self.clip_query_proj(self.clip_norm(clip_features))
        spatial_norm = self.spatial_norm(spatial)
        K = self.spatial_key_proj(spatial_norm)
        V = self.spatial_value_proj(spatial_norm)

        attn_out, attn_weights = self.cross_attention(Q, K, V)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)
        out = self.out_norm(clip_features + attn_out)
        return out, attn_weights


class SvfPosePrependFusion(nn.Module):
    """
    Comparison 3: compress camera branch into a single pose token via Pi3's
    camera_head (4×4 → first 3 rows → 12 values), project to d_clip, and
    prepend it to the 2D-fused sequence.

    camera_pose:  (F, 12)              first 3 rows of the 4×4 pose matrix
    patch_tokens: (F, P, d_patch)      3D spatial features
    clip_features:(F, N_clip, d_clip)  2D vision features

    Output: (F, 1+N_clip, d_clip)   — 1 extra pose token per frame
    """
    def __init__(self, d_clip, d_spatial_encoder, d_attn, num_heads, dropout_rate=0.1, d_pose=12):
        super().__init__()
        # Pose token projection: 12 → d_clip
        self.pose_proj = nn.Linear(d_pose, d_clip)
        self.pose_norm = nn.LayerNorm(d_clip)

        # 2D cross-attend to patch
        self.clip_norm  = nn.LayerNorm(d_clip)
        self.patch_norm = nn.LayerNorm(d_spatial_encoder)

        self.clip_query_proj  = nn.Linear(d_clip,            d_attn)
        self.patch_key_proj   = nn.Linear(d_spatial_encoder, d_attn)
        self.patch_value_proj = nn.Linear(d_spatial_encoder, d_attn)

        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)

        self.out_proj = nn.Linear(d_attn, d_clip)
        self.out_norm = nn.LayerNorm(d_clip)
        self.dropout  = nn.Dropout(dropout_rate)

    def forward(self, clip_features, camera_pose, patch_tokens):
        # Step 1: pose token  (F, 1, d_clip)
        pose_token = self.pose_norm(self.pose_proj(camera_pose)).unsqueeze(1)

        # Step 2: 2D cross-attend patch
        Q = self.clip_query_proj(self.clip_norm(clip_features))
        K = self.patch_key_proj(self.patch_norm(patch_tokens))
        V = self.patch_value_proj(self.patch_norm(patch_tokens))

        attn_out, attn_weights = self.cross_attention(Q, K, V)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)
        fused_2d = self.out_norm(clip_features + attn_out)

        # Step 3: prepend pose token
        out = torch.cat([pose_token, fused_2d], dim=1)   # (F, 1+N_clip, d_clip)
        return out, attn_weights


class TransformerFusion(nn.Module):
    def __init__(self, d_clip, d_spatial_encoder, d_attn, num_heads, dropout_rate=0.1):
        super(TransformerFusion, self).__init__()
        
        # pre-norm
        self.clip_norm = nn.LayerNorm(d_clip)
        self.spatial_encoder_norm = nn.LayerNorm(d_spatial_encoder)
        
        # projection
        self.clip_query_proj = nn.Linear(d_clip, d_attn)
        self.spatial_encoder_key_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.spatial_encoder_value_proj = nn.Linear(d_spatial_encoder, d_attn)
        
        # cross attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)
        
        # post-norm
        self.out_norm = nn.LayerNorm(d_attn)
        
        # feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_attn, 4 * d_attn),
            nn.ReLU(),
            nn.Linear(4 * d_attn, d_clip)
        )
        
        # dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, clip_features, spatial_encoder_features):
        """
        Args:
            clip_features: [B, N, D_clip]
            spatial_encoder_features: [B, N, D_spatial_encoder]
        Returns:
            fused_features: [B, N, D_clip]
        """
        # pre-norm
        clip_features_norm = self.clip_norm(clip_features)  # [B, N, D_clip]
        spatial_encoder_features_norm = self.spatial_encoder_norm(spatial_encoder_features)  # [B, N, D_spatial_encoder]
        
        # projection to D_attn dimension
        clip_query_proj = self.clip_query_proj(clip_features_norm)  # [B, N, D_attn]
        spatial_encoder_key_proj = self.spatial_encoder_key_proj(spatial_encoder_features_norm)  # [B, N, D_attn]
        spatial_encoder_value_proj = self.spatial_encoder_value_proj(spatial_encoder_features_norm)  # [B, N, D_attn]
        
        # cross attention
        attention_output, _ = self.cross_attention(
            query=clip_query_proj,
            key=spatial_encoder_key_proj,
            value=spatial_encoder_value_proj
        )
        
        # dropout
        attention_output = self.dropout(attention_output)
        
        # add residual connection
        attention_output_add_residual = attention_output + clip_features
        
        # pre-norm
        attention_output_add_residual_norm = self.out_norm(attention_output_add_residual)
        
        # feed-forward network
        feed_forward_output = self.ffn(attention_output_add_residual_norm)   # [B, N_clip, D_clip]
        
        # dropout
        feed_forward_output = self.dropout(feed_forward_output)
        
        # add residual connection
        fused_features = feed_forward_output + attention_output_add_residual
        
        return fused_features

class llava_3d_fusion_block(nn.Module):
    def __init__(self, patch_size=14, latent_dim=1152):
        super(llava_3d_fusion_block, self).__init__()
        self.patch_size = patch_size
        self.points_enc = nn.Sequential(
            nn.Linear(3, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
            )
        
    def forward(self, clip_features, points):
        # points shape: [B, H, W, 3]
        B, H, W, _ = points.shape
        
        # 1. 获取patch中心点
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        
        # 重塑points为patches
        patches = points.view(B, patch_h, self.patch_size, patch_w, self.patch_size, 3)
        
        # 计算每个patch的中心点
        patch_center_points = patches.mean(dim=(2, 4))  # [B, patch_h, patch_w, 3]
        
        # 2. 编码中心点
        encoded_centers = self.points_enc(patch_center_points)  # [B, patch_h, patch_w, latent_dim]
        
        # 3. 将编码后的特征与clip_features融合
        # 确保维度匹配
        encoded_centers = encoded_centers.view(B, patch_h * patch_w, -1)
        
        # 直接相加融合
        fused_features = clip_features + encoded_centers
        
        return fused_features

class MLPFusion(nn.Module):
    def __init__(self, d_llm, d_spatial_encoder, fusion_block_type):
        super(MLPFusion, self).__init__()
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", fusion_block_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            
            def create_mlp():
                modules = [nn.Linear(d_spatial_encoder, d_llm)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(d_llm, d_llm))
                return nn.Sequential(*modules)

            self.spatial_features_mlp = create_mlp()
        
    def forward(self, clip_features, spatial_features):
        # project spatial features to llm dimension
        projected_spatial_features = self.spatial_features_mlp(spatial_features)

        # add residual connection
        fused_features = clip_features + projected_spatial_features

        return fused_features

class ConcatMLPFusion(nn.Module):
    def __init__(self, d_llm, d_spatial_encoder):
        super(ConcatMLPFusion, self).__init__()
        
        # MLP to project spatial features (similar to mlp2x_gelu)
        self.spatial_features_proj_mlp = nn.Sequential(
            nn.Linear(d_spatial_encoder, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm)
        )
        
        # MLP for fused features after concatenation
        # Input dim: 2 * d_llm, Output dim: d_llm
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_llm, 4 * d_llm), 
            nn.GELU(),
            nn.Linear(4 * d_llm, d_llm)
        )

    def forward(self, clip_features, spatial_features):
        # Project spatial features to llm dimension
        projected_spatial_features = self.spatial_features_proj_mlp(spatial_features) # [B, N, D_llm]

        # Concatenate features
        concatenated_features = torch.cat([clip_features, projected_spatial_features], dim=-1) # [B, N, 2 * D_llm]
        
        # Pass through fusion MLP
        fused_features = self.fusion_mlp(concatenated_features) # [B, N, D_llm]

        return fused_features

class ConcatSelfAttentionFusion(nn.Module):
    def __init__(self, d_llm, d_spatial_encoder, num_heads, dropout_rate=0.1):
        super(ConcatSelfAttentionFusion, self).__init__()
        
        # MLP to project spatial features
        self.spatial_features_proj_mlp = nn.Sequential(
            nn.Linear(d_spatial_encoder, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm)
        )
        
        # Pre-attention normalization
        self.pre_attn_norm = nn.LayerNorm(2 * d_llm)
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=2 * d_llm, 
            num_heads=num_heads, 
            dropout=dropout_rate, 
            batch_first=True
        )
        
        # Post-attention normalization
        self.post_attn_norm = nn.LayerNorm(2 * d_llm)

        # Final linear projection
        self.output_proj = nn.Linear(2 * d_llm, d_llm)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, clip_features, spatial_features):
        # Project spatial features to llm dimension
        projected_spatial_features = self.spatial_features_proj_mlp(spatial_features) # [B, N, D_llm]

        # Concatenate features
        concatenated_features = torch.cat([clip_features, projected_spatial_features], dim=-1) # [B, N, 2 * D_llm]
        
        # Pre-attention normalization
        normed_features = self.pre_attn_norm(concatenated_features)

        # Self-attention
        attn_output, _ = self.self_attention(
            query=normed_features, 
            key=normed_features, 
            value=normed_features
        ) # [B, N, 2 * D_llm]

        # Dropout and residual connection 1
        attn_output = self.dropout(attn_output)
        attn_output_residual = attn_output + concatenated_features # Residual connection before final projection

        # Post-attention normalization
        normed_attn_output = self.post_attn_norm(attn_output_residual)

        # Final projection
        fused_features = self.output_proj(normed_attn_output) # [B, N, D_llm]

        return fused_features

def build_multimodal_fusion_block(config, delay_load=False, **kwargs):
    fusion_block_type = getattr(config, "fusion_block", "cross_attention")
    geo_rope_fusion_mode_aliases = {
        "svf_depth_geo_rope_fusion": "depth",
        "svf_xyz_geo_rope_fusion": "xyz",
        "svf_spherical_geo_rope_fusion": "spherical",
        # Backward compat: old names before the GeoRoPE Fusion rename (commit 8561c4d)
        "svf_depth_rope": "depth",
        "svf_xyz_rope": "xyz",
        "svf_spherical_rope": "spherical",
    }
    d_clip = config.mm_hidden_size
    d_llm = config.hidden_size
    d_attn = d_clip
    d_spatial_encoder = getattr(config, "spatial_feature_dim", 768)
    # d_camera_encoder is 512 when using the new decoded schema (camera_decoder branch).
    # Defaults to None (falls back to d_spatial_encoder) for backward compatibility.
    d_camera_encoder = getattr(config, "spatial_camera_encoder_dim", None)
    if fusion_block_type == "cross_attention_with_mlp":
        return CrossAttentionFusionWithMLP(
            d_clip=d_clip,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            num_heads=18,
            mlp_ratio=4.0,
            proj_drop=0.1
        )
    elif fusion_block_type in ["cross_attention", "svf_baseline"]:
        return CrossAttentionFusion(
            d_clip=d_clip,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            num_heads=18
        )
    elif fusion_block_type == "svf_patch_cam_concat":
        return PatchCrossAttentionCameraConcatFusion(
            d_clip=d_clip,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            num_heads=18,
            dropout_rate=0.1,
            d_camera_encoder=d_camera_encoder,
        )
    elif fusion_block_type == "svf_geometry_bridge":
        return GeometryBridgeFusion(
            d_clip=d_clip,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            num_heads=18,
            dropout_rate=0.1,
            d_camera_encoder=d_camera_encoder,
        )
    elif fusion_block_type == "mlp_after_clip_proj":
        return MLPFusion(
            d_llm=d_llm,
            d_spatial_encoder=d_spatial_encoder,
            fusion_block_type="mlp2x_gelu"
        )
    elif fusion_block_type == "transformer":
        return TransformerFusion(
            d_clip=d_clip,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            num_heads=18,
            dropout_rate=0.1
        )
    elif fusion_block_type == "concat_mlp":
        return ConcatMLPFusion(
            d_llm=d_llm,
            d_spatial_encoder=d_spatial_encoder
        )
    elif fusion_block_type == "concat_self_attention":
        return ConcatSelfAttentionFusion(
            d_llm=d_llm,
            d_spatial_encoder=d_spatial_encoder,
            num_heads=36,
            dropout_rate=0.1
        )
    elif fusion_block_type == "llava_3d_fusion_block":
        return llava_3d_fusion_block(
            patch_size=16,
            latent_dim=1152
        )
    elif fusion_block_type == "video_3d_llm_fusion_block":
        return video_3d_llm_fusion_block(
            patch_size=14,
            latent_dim=d_llm
        )
    elif fusion_block_type.endswith("_layer_cross_attention"):
        num_layers = int(fusion_block_type.split("_")[0])
        return MultiLayerCrossAttentionFusion(
            num_layers=num_layers,
            d_query=d_llm,
            d_kv=d_spatial_encoder,
            num_heads=64
        )
    elif fusion_block_type == "svf_patch_only":
        # Baseline: 2D cross-attends patch_tokens only. Reuses CrossAttentionFusion.
        return CrossAttentionFusion(
            d_clip=d_clip,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            num_heads=18,
        )
    elif fusion_block_type == "svf_patch_only_geo_rope_eval":
        geo_rope_fusion_mode = (
            getattr(config, "geo_rope_fusion_mode", None)
            or getattr(config, "geometry_rope_mode", "spherical")
        )
        log_stats = getattr(config, "geo_rope_fusion_log_stats", False) or getattr(config, "geometry_rope_log_stats", False)
        if isinstance(log_stats, str):
            log_stats = log_stats.lower() in {"1", "true", "yes", "y", "on"}
        max_depth = getattr(config, "geo_rope_fusion_max_depth", None) or getattr(config, "geometry_rope_max_depth", 10.0)
        group_split = getattr(config, "geo_rope_fusion_group_split", None) or getattr(config, "geometry_rope_group_split", None)
        eval_lambda = getattr(config, "geo_rope_fusion_eval_lambda", None)
        if eval_lambda is None:
            eval_lambda = getattr(config, "geometry_rope_eval_lambda", 0.0)
        eval_lambda_q = getattr(config, "geo_rope_fusion_eval_lambda_q", None)
        eval_lambda_k = getattr(config, "geo_rope_fusion_eval_lambda_k", None)
        return PostHocGeoRoPEPatchCrossAttention(
            d_clip=d_clip,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            num_heads=18,
            geo_rope_fusion_mode=geo_rope_fusion_mode,
            max_depth=max_depth,
            group_split=group_split,
            eval_lambda=eval_lambda,
            eval_lambda_q=eval_lambda_q,
            eval_lambda_k=eval_lambda_k,
            log_stats=log_stats,
        )
    elif fusion_block_type in {
        "svf_geo_rope_fusion",
        "svf_geo_rope_fusion_forced",
        "svf_geo_rope_fusion_per_head_gate",
    } or fusion_block_type in geo_rope_fusion_mode_aliases:
        if fusion_block_type in geo_rope_fusion_mode_aliases:
            # Explicit alias (new or old name) → mode is fully determined by the name.
            geo_rope_fusion_mode = geo_rope_fusion_mode_aliases[fusion_block_type]
        else:
            # svf_geo_rope_fusion: fall back to config; check both new and old key names.
            geo_rope_fusion_mode = (
                getattr(config, "geo_rope_fusion_mode", None)
                or getattr(config, "geometry_rope_mode", "spherical")
            )
        log_stats = getattr(config, "geo_rope_fusion_log_stats", False) or getattr(config, "geometry_rope_log_stats", False)
        if isinstance(log_stats, str):
            log_stats = log_stats.lower() in {"1", "true", "yes", "y", "on"}
        log_attention_stats = (
            getattr(config, "geo_rope_fusion_log_attention_stats", False)
            or getattr(config, "geometry_rope_log_attention_stats", False)
        )
        if isinstance(log_attention_stats, str):
            log_attention_stats = log_attention_stats.lower() in {"1", "true", "yes", "y", "on"}
        # Accept both new and old config key names for max_depth and group_split.
        max_depth = getattr(config, "geo_rope_fusion_max_depth", None) or getattr(config, "geometry_rope_max_depth", 10.0)
        group_split = getattr(config, "geo_rope_fusion_group_split", None) or getattr(config, "geometry_rope_group_split", None)
        gate_type = getattr(config, "geo_rope_gate_type", None) or getattr(config, "geometry_rope_gate_type", None)
        if fusion_block_type == "svf_geo_rope_fusion_forced":
            gate_type = "forced"
        elif fusion_block_type == "svf_geo_rope_fusion_per_head_gate":
            gate_type = "per_head"
        elif gate_type is None:
            gate_type = "scalar"
        head_gate_init = (
            getattr(config, "geo_rope_head_gate_init", None)
            or getattr(config, "geometry_rope_head_gate_init", 0.99)
        )
        return GeoRoPEFusionCrossAttention(
            d_clip=d_clip,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            num_heads=18,
            geo_rope_fusion_mode=geo_rope_fusion_mode,
            max_depth=max_depth,
            group_split=group_split,
            log_stats=log_stats,
            log_attention_stats=log_attention_stats,
            gate_type=gate_type,
            head_gate_init=head_gate_init,
        )
    elif fusion_block_type == "svf_cat_feat":
        # Comparison 1: feature-dim concat of [camera, patch] as KV.
        _d_cam = d_camera_encoder or 512  # camera_decoder out_dim
        return SvfCatFeatFusion(
            d_clip=d_clip,
            d_camera_encoder=_d_cam,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            num_heads=18,
            dropout_rate=0.1,
        )
    elif fusion_block_type == "svf_pose_geometry_bridge":
        # Comparison 2: camera tokens (camera_decoder branch) query patch tokens
        # to form geometry-aware tokens, then 2D queries those geometry-aware tokens.
        _d_cam = d_camera_encoder or 512  # camera_decoder out_dim
        return GeometryBridgeFusion(
            d_clip=d_clip,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            num_heads=18,
            dropout_rate=0.1,
            d_camera_encoder=_d_cam,
        )
    elif fusion_block_type == "svf_pose_prepend":
        # Comparison 3: Pi3 camera_head → 12-value pose → prepended token.
        return SvfPosePrependFusion(
            d_clip=d_clip,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            num_heads=18,
            dropout_rate=0.1,
            d_pose=12,
        )
    raise ValueError(f"Unknown fusion block type: {fusion_block_type}")
