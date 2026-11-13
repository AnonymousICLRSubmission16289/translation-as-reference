from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor


def zeropower_via_newtonschulz5(matrix: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the orthogonalized update used by Muon.

    Conv kernels are flattened to [out_channels, -1]. Computation is done in
    bfloat16 when supported and returned in the original dtype.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(matrix.shape)}")
    original_dtype = matrix.dtype
    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(dtype=original_dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=float(lr),
            base_lr=float(lr),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            ns_steps=int(ns_steps),
            weight_decay=float(weight_decay),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            weight_decay = float(group["weight_decay"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(grad)
                update = grad.add(buffer, alpha=momentum) if nesterov else buffer
                if update.ndim < 2:
                    raise RuntimeError("Muon received a parameter with ndim < 2")
                matrix = update.reshape(update.shape[0], -1)
                orthogonal = zeropower_via_newtonschulz5(matrix, ns_steps).reshape_as(update)
                scale = max(1.0, update.shape[0] / max(update.numel() / update.shape[0], 1.0)) ** 0.5
                parameter.add_(orthogonal, alpha=-lr * scale)
        return loss


def split_muon_parameters(named_parameters: Iterable[tuple[str, Tensor]]) -> tuple[list[Tensor], list[Tensor]]:
    matrix: list[Tensor] = []
    adam: list[Tensor] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and not any(token in name.lower() for token in ("embed", "norm", "bias")):
            matrix.append(parameter)
        else:
            adam.append(parameter)
    return matrix, adam


def build_hybrid_optimizers(
    named_parameters: Iterable[tuple[str, Tensor]],
    *,
    muon_lr: float,
    adamw_lr: float,
    momentum: float,
    ns_steps: int,
    beta2: float,
    weight_decay: float,
) -> list[torch.optim.Optimizer]:
    matrix, adam = split_muon_parameters(named_parameters)
    optimizers: list[torch.optim.Optimizer] = []
    if matrix:
        optimizers.append(Muon(matrix, lr=muon_lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay))
    if adam:
        optimizer = torch.optim.AdamW(adam, lr=adamw_lr, betas=(0.9, beta2), weight_decay=weight_decay)
        for group in optimizer.param_groups:
            group["base_lr"] = float(adamw_lr)
        optimizers.append(optimizer)
    return optimizers
