from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor





def _groups(channels: int) -> int:
    # Keep at least two channels per group. This is robust for very small smoke-test
    # volumes whose deepest spatial feature may be 1x1x1.
    upper = max(1, channels // 2)
    for value in (32, 16, 8, 4, 2, 1):
        if value <= upper and channels % value == 0:
            return value
    return 1


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, value: Tensor) -> Tensor:
        value = value.float().reshape(-1)
        half = self.dim // 2
        if half == 0:
            return value[:, None]
        exponent = -math.log(10000.0) * torch.arange(half, device=value.device, dtype=value.dtype) / max(half - 1, 1)
        frequencies = torch.exp(exponent)
        angles = value[:, None] * frequencies[None]
        result = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if result.shape[1] < self.dim:
            result = F.pad(result, (0, self.dim - result.shape[1]))
        return result


class TimeIntervalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.sin = SinusoidalEmbedding(dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 3, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor, r: Tensor) -> Tensor:
        return self.net(torch.cat([self.sin(t), self.sin(r), self.sin(t-r)], dim=1))


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int | None = None) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, 1)
        self.time = nn.Linear(time_dim, out_ch * 2) if time_dim is not None else None

    def forward(self, x: Tensor, emb: Tensor | None = None) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        if self.time is not None:
            if emb is None:
                raise ValueError("Time-conditioned block requires an embedding")
            scale, shift = self.time(F.silu(emb)).chunk(2, dim=1)
            h = h * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class Down3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Up3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)

    def forward(self, x: Tensor, target_shape: tuple[int, int, int]) -> Tensor:
        x = F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)
        return self.conv(x)


class UNet3D(nn.Module):
    """Four-level 3D U-Net used for the deterministic OCT-to-OCTA anchor."""
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_ch: int = 50, max_ch: int = 512) -> None:
        super().__init__()
        channels = [base_ch, min(base_ch*2,max_ch), min(base_ch*4,max_ch), min(base_ch*8,max_ch)]
        self.in_conv = nn.Conv3d(in_ch, channels[0], 3, padding=1)
        self.enc0 = ResBlock3D(channels[0], channels[0])
        self.down0 = Down3D(channels[0])
        self.enc1 = ResBlock3D(channels[0], channels[1])
        self.down1 = Down3D(channels[1])
        self.enc2 = ResBlock3D(channels[1], channels[2])
        self.down2 = Down3D(channels[2])
        self.enc3 = ResBlock3D(channels[2], channels[3])
        self.down3 = Down3D(channels[3])
        self.mid0 = ResBlock3D(channels[3], channels[3])
        self.mid1 = ResBlock3D(channels[3], channels[3])
        self.up3 = Up3D(channels[3], channels[3])
        self.dec3 = ResBlock3D(channels[3] + channels[3], channels[3])
        self.up2 = Up3D(channels[3], channels[2])
        self.dec2 = ResBlock3D(channels[2] + channels[2], channels[2])
        self.up1 = Up3D(channels[2], channels[1])
        self.dec1 = ResBlock3D(channels[1] + channels[1], channels[1])
        self.up0 = Up3D(channels[1], channels[0])
        self.dec0 = ResBlock3D(channels[0] + channels[0], channels[0])
        self.out_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
        self.out_conv = nn.Conv3d(channels[0], out_ch, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        s0 = self.enc0(self.in_conv(x))
        s1 = self.enc1(self.down0(s0))
        s2 = self.enc2(self.down1(s1))
        s3 = self.enc3(self.down2(s2))
        h = self.mid1(self.mid0(self.down3(s3)))
        h = self.dec3(torch.cat([self.up3(h, s3.shape[-3:]), s3], dim=1))
        h = self.dec2(torch.cat([self.up2(h, s2.shape[-3:]), s2], dim=1))
        h = self.dec1(torch.cat([self.up1(h, s1.shape[-3:]), s1], dim=1))
        h = self.dec0(torch.cat([self.up0(h, s0.shape[-3:]), s0], dim=1))
        return torch.tanh(self.out_conv(F.silu(self.out_norm(h))))


class ConditionalUNet3D(nn.Module):
    """Time-conditioned 3D U-Net for direct or residual pMF data prediction."""
    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 1,
        base_ch: int = 25,
        max_ch: int = 256,
        time_dim: int = 128,
        zero_output: bool = True,
    ) -> None:
        super().__init__()
        channels = [base_ch, min(base_ch*2,max_ch), min(base_ch*4,max_ch), min(base_ch*8,max_ch)]
        self.time_embed = TimeIntervalEmbedding(time_dim)
        self.in_conv = nn.Conv3d(in_ch, channels[0], 3, padding=1)
        self.enc0 = ResBlock3D(channels[0], channels[0], time_dim)
        self.down0 = Down3D(channels[0])
        self.enc1 = ResBlock3D(channels[0], channels[1], time_dim)
        self.down1 = Down3D(channels[1])
        self.enc2 = ResBlock3D(channels[1], channels[2], time_dim)
        self.down2 = Down3D(channels[2])
        self.enc3 = ResBlock3D(channels[2], channels[3], time_dim)
        self.down3 = Down3D(channels[3])
        self.mid0 = ResBlock3D(channels[3], channels[3], time_dim)
        self.mid1 = ResBlock3D(channels[3], channels[3], time_dim)
        self.up3 = Up3D(channels[3], channels[3])
        self.dec3 = ResBlock3D(channels[3] + channels[3], channels[3], time_dim)
        self.up2 = Up3D(channels[3], channels[2])
        self.dec2 = ResBlock3D(channels[2] + channels[2], channels[2], time_dim)
        self.up1 = Up3D(channels[2], channels[1])
        self.dec1 = ResBlock3D(channels[1] + channels[1], channels[1], time_dim)
        self.up0 = Up3D(channels[1], channels[0])
        self.dec0 = ResBlock3D(channels[0] + channels[0], channels[0], time_dim)
        self.out_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
        self.out_conv = nn.Conv3d(channels[0], out_ch, 3, padding=1)
        if zero_output:
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, z: Tensor, oct_cond: Tensor, t: Tensor, r: Tensor) -> Tensor:
        emb = self.time_embed(t, r)
        x = torch.cat([z, oct_cond], dim=1)
        s0 = self.enc0(self.in_conv(x), emb)
        s1 = self.enc1(self.down0(s0), emb)
        s2 = self.enc2(self.down1(s1), emb)
        s3 = self.enc3(self.down2(s2), emb)
        h = self.mid1(self.mid0(self.down3(s3), emb), emb)
        h = self.dec3(torch.cat([self.up3(h, s3.shape[-3:]), s3], dim=1), emb)
        h = self.dec2(torch.cat([self.up2(h, s2.shape[-3:]), s2], dim=1), emb)
        h = self.dec1(torch.cat([self.up1(h, s1.shape[-3:]), s1], dim=1), emb)
        h = self.dec0(torch.cat([self.up0(h, s0.shape[-3:]), s0], dim=1), emb)
        return self.out_conv(F.silu(self.out_norm(h)))


class DataPredictionModel(nn.Module):
    """Unified parameterization for direct, frozen-reference, joint, and endpoint controls."""
    VALID_VARIANTS = {"frozen_pmf", "joint_pmf"}

    def __init__(
        self,
        variant: str,
        anchor: nn.Module | None,
        predictor: nn.Module,
    ) -> None:
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown variant {variant}")
        if anchor is None:
            raise ValueError(f"Variant {variant} requires an anchor")
        self.variant = variant
        self.anchor = anchor
        self.predictor = predictor
        if variant == "frozen_pmf":
            assert self.anchor is not None
            self.anchor.requires_grad_(False)
            self.anchor.eval()

    @property
    def uses_anchor(self) -> bool:
        return self.anchor is not None

    @property
    def anchor_frozen(self) -> bool:
        return self.variant == "frozen_pmf"

    def train(self, mode: bool = True):
        super().train(mode)
        if self.anchor_frozen and self.anchor is not None:
            self.anchor.eval()
        return self

    def anchor_prediction(self, oct_cond: Tensor) -> Tensor:
        if self.anchor is None:
            return torch.zeros_like(oct_cond)
        if self.anchor_frozen:
            with torch.no_grad():
                return self.anchor(oct_cond)
        return self.anchor(oct_cond)

    def data_prediction(
        self,
        z: Tensor,
        oct_cond: Tensor,
        t: Tensor,
        r: Tensor,
        reference: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        correction = self.predictor(z, oct_cond, t, r)
        if reference is None:
            reference = self.anchor_prediction(oct_cond)
        return reference + correction, correction


    def velocity(
        self,
        z: Tensor,
        oct_cond: Tensor,
        t: Tensor,
        r: Tensor,
        t_clip: float,
        reference: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        prediction, correction = self.data_prediction(
            z, oct_cond, t, r, reference=reference
        )
        tc = t.clamp_min(float(t_clip)).reshape(t.shape[0], 1, 1, 1, 1)
        velocity = (z - prediction) / tc
        return prediction, correction, velocity





def build_model(config):
    task = config.get("task", "joint_pmf")
    if task == "anchor":
        return UNet3D(base_ch=config.get("anchor_base_ch", 50), max_ch=config.get("anchor_max_ch", 512))
    if task not in {"frozen_pmf", "joint_pmf"}:
        raise ValueError("Only reference-centered pMF is supported")
    # Keep the original predictor-first initialization order.
    predictor = ConditionalUNet3D(base_ch=config.get("predictor_base_ch", 25), max_ch=config.get("predictor_max_ch", 256), time_dim=config.get("time_dim", 128))
    anchor = UNet3D(base_ch=config.get("anchor_base_ch", 50), max_ch=config.get("anchor_max_ch", 512))
    return DataPredictionModel(task, anchor, predictor)
