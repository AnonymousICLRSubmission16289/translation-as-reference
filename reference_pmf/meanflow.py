from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor

from .models import DataPredictionModel


@dataclass
class PMFOutput:
    prediction: Tensor
    correction: Tensor
    velocity: Tensor
    derivative: Tensor
    local_velocity: Tensor
    target_velocity: Tensor
    t: Tensor
    r: Tensor


def expand_time(value: Tensor, ndim: int = 5) -> Tensor:
    return value.reshape(value.shape[0], *([1] * (ndim - 1)))


def logit_normal(shape: tuple[int, ...], mean: float, std: float, device, dtype) -> Tensor:
    return torch.sigmoid(torch.randn(shape, device=device, dtype=dtype) * std + mean)


def sample_intervals(batch: int, p_mean: float, p_std: float, t_clip: float, device, dtype) -> tuple[Tensor, Tensor]:
    t = logit_normal((batch,), p_mean, p_std, device, dtype).clamp_min(float(t_clip))
    r = torch.rand((batch,), device=device, dtype=dtype) * t
    return t, r


def adaptive_unit_loss(per_sample_error: Tensor, p: float = 1.0, eps: float = 0.01) -> tuple[Tensor, Tensor]:
    raw = per_sample_error.float()
    denom = (raw.detach() + float(eps)).pow(float(p))
    unit = raw / denom
    return unit.mean(), raw.mean()


def pmf_forward(
    model: DataPredictionModel,
    oct_cond: Tensor,
    target: Tensor,
    *,
    p_mean: float,
    p_std: float,
    noise_scale: float,
    t_clip: float,
    jvp_mode: str,
) -> PMFOutput:
    batch = target.shape[0]
    t, r = sample_intervals(batch, p_mean, p_std, t_clip, target.device, target.dtype)
    epsilon = torch.randn_like(target) * float(noise_scale)
    z_t = (1.0 - expand_time(t, target.ndim)) * target + expand_time(t, target.ndim) * epsilon
    target_velocity = epsilon - target

    # Compute the condition-only reference once. This is especially important
    # for joint_pmf: the 64M anchor graph is shared by the local-velocity and
    # interval-velocity branches instead of being instantiated repeatedly.
    reference = model.anchor_prediction(oct_cond) if model.uses_anchor else None

    _, _, local_velocity = model.velocity(
        z_t, oct_cond, t, t, t_clip, reference=reference
    )
    tangent = local_velocity.detach()

    # The trainable primal branch is evaluated normally. The derivative term is
    # intentionally semi-gradient in pmf_loss (derivative.detach()). Computing
    # the forward-mode JVP under no_grad therefore preserves the exact optimizer
    # gradient while avoiding a reverse-mode graph through the JVP transform.
    prediction, correction, velocity = model.velocity(
        z_t, oct_cond, t, r, t_clip, reference=reference
    )

    if jvp_mode == "none":
        derivative = torch.zeros_like(velocity)
    elif jvp_mode == "torch":
        from torch.func import jvp

        def velocity_only(z_arg: Tensor, t_arg: Tensor, r_arg: Tensor) -> Tensor:
            return model.velocity(
                z_arg, oct_cond, t_arg, r_arg, t_clip, reference=reference
            )[2]

        # torch.no_grad disables reverse-mode graph construction but does not
        # disable forward-mode AD, so the numerical JVP remains available.
        with torch.no_grad():
            _, derivative = jvp(
                velocity_only,
                (z_t, t, r),
                (tangent, torch.ones_like(t), torch.zeros_like(r)),
                strict=False,
            )
        derivative = derivative.detach()
    else:
        raise ValueError(f"Unsupported JVP mode: {jvp_mode}")

    return PMFOutput(
        prediction=prediction,
        correction=correction,
        velocity=velocity,
        derivative=derivative,
        local_velocity=local_velocity,
        target_velocity=target_velocity,
        t=t,
        r=r,
    )


def pmf_loss(
    output: PMFOutput,
    target: Tensor,
    *,
    flow_weight: float,
    aux_weight: float,
    lambda_l1: float,
    norm_p: float,
    norm_eps: float,
) -> tuple[Tensor, dict[str, float]]:
    interval = expand_time(output.t - output.r, output.velocity.ndim)
    finite_velocity = output.velocity + interval * output.derivative.detach()
    flow_per_sample = (finite_velocity - output.target_velocity.detach()).abs().flatten(1).mean(1)
    aux_per_sample = (output.local_velocity - output.target_velocity.detach()).abs().flatten(1).mean(1)
    loss_flow, raw_flow = adaptive_unit_loss(flow_per_sample, norm_p, norm_eps)
    loss_aux, raw_aux = adaptive_unit_loss(aux_per_sample, norm_p, norm_eps)
    l1 = (output.prediction - target).abs().mean()
    total = float(flow_weight) * loss_flow + float(aux_weight) * loss_aux + float(lambda_l1) * l1
    derivative_abs = output.derivative.detach().abs().mean()
    velocity_abs = output.velocity.detach().abs().mean()
    jvp_ratio = float((interval.detach() * output.derivative.detach()).abs().mean() / (velocity_abs + 1e-8))
    target_flat = output.target_velocity.detach().flatten(1)
    velocity_flat = output.velocity.detach().flatten(1)
    tangent_cos = torch.nn.functional.cosine_similarity(velocity_flat, target_flat, dim=1).mean()
    stats = {
        "loss": float(total.detach()),
        "l1": float(l1.detach()),
        "flow_unit": float(loss_flow.detach()),
        "aux_unit": float(loss_aux.detach()),
        "flow_raw": float(raw_flow.detach()),
        "aux_raw": float(raw_aux.detach()),
        "jvp_ratio": jvp_ratio,
        "du_abs": float(derivative_abs),
        "u_abs": float(velocity_abs),
        "tangent_cos": float(tangent_cos),
        "correction_abs": float(output.correction.detach().abs().mean()),
    }
    return total, stats


def one_step_prediction(
    model: DataPredictionModel,
    oct_cond: Tensor,
    *,
    noise: Tensor | None,
    t_clip: float,
) -> tuple[Tensor, Tensor]:
    batch = oct_cond.shape[0]
    if noise is None:
        noise = torch.randn_like(oct_cond)
    z = noise
    t = torch.ones((batch,), device=oct_cond.device, dtype=oct_cond.dtype)
    r = torch.zeros_like(t)
    prediction, correction = model.data_prediction(z, oct_cond, t, r)
    return prediction, correction
