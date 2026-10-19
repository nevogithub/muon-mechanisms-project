import math

import torch
from torch.optim.optimizer import Optimizer, required

from .muon import muon_update


class MuonGradDiff(Optimizer):
    """
    Muon + AdamW with a simple consecutive-gradient difference correction.

    2D+ params:
    - keep Muon's exact update path
    - feed Muon with: g_t + gamma_t * (g_t - g_{t-1})

    Non-2D params:
    - unchanged AdamW branch
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=5,
        gamma=0.1,
        gamma_end=0.1,
        gamma_schedule="none",
        gamma_start_epoch=1,
        gamma_end_epoch=1,
        nesterov=True,
        adamw_lr=3e-4,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-8,
        adamw_weight_decay=0.0,
    ):
        if lr is required:
            raise ValueError("Learning rate is required")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            gamma=gamma,
            gamma_end=gamma_end,
            gamma_schedule=gamma_schedule,
            gamma_start_epoch=gamma_start_epoch,
            gamma_end_epoch=gamma_end_epoch,
            nesterov=nesterov,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_weight_decay=adamw_weight_decay,
        )
        super().__init__(params, defaults)
        self.current_epoch = 1

    def set_epoch(self, epoch: int):
        self.current_epoch = max(1, int(epoch))

    def _scheduled_gamma(self, group):
        gamma_start = float(group.get("gamma", 0.0))
        gamma_end = float(group.get("gamma_end", gamma_start))
        start_epoch = int(group.get("gamma_start_epoch", 1))
        end_epoch = int(group.get("gamma_end_epoch", 1))
        mode = str(group.get("gamma_schedule", "none")).lower()

        if mode == "none":
            gamma_t = gamma_start
        else:
            if self.current_epoch <= start_epoch:
                frac = 0.0
            elif self.current_epoch >= end_epoch:
                frac = 1.0
            else:
                denom = max(1, end_epoch - start_epoch)
                frac = float(self.current_epoch - start_epoch) / float(denom)

            if mode == "linear":
                pass
            elif mode == "cosine":
                frac = 0.5 - 0.5 * math.cos(frac * math.pi)
            else:
                raise ValueError(f"Unsupported gamma_schedule mode: {mode}")

            gamma_t = gamma_start + frac * (gamma_end - gamma_start)

        return float(gamma_t)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum = float(group["momentum"])
            ns_steps = int(group["ns_steps"])
            gamma_t = self._scheduled_gamma(group)
            nesterov = bool(group["nesterov"])

            adamw_lr = float(group["adamw_lr"])
            adamw_beta1, adamw_beta2 = group["adamw_betas"]
            adamw_eps = float(group["adamw_eps"])
            adamw_weight_decay = float(group["adamw_weight_decay"])

            if "step" not in group:
                group["step"] = 0
            group["step"] += 1
            step = int(group["step"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                state = self.state[p]

                if p.ndim >= 2:
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["last_grad"] = torch.zeros_like(p)

                    momentum_buffer = state["momentum_buffer"]
                    last_grad = state["last_grad"]
                    corrected_grad = grad + gamma_t * (grad - last_grad)

                    update = muon_update(
                        corrected_grad,
                        momentum_buffer,
                        beta=momentum,
                        ns_steps=ns_steps,
                        nesterov=nesterov,
                    )

                    p.data.mul_(1.0 - lr * weight_decay)
                    p.data.add_(update.to(p.dtype), alpha=-lr)
                    last_grad.copy_(grad)
                else:
                    if len(state) == 0:
                        state["adamw_exp_avg"] = torch.zeros_like(p)
                        state["adamw_exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg = state["adamw_exp_avg"]
                    exp_avg_sq = state["adamw_exp_avg_sq"]

                    exp_avg.mul_(adamw_beta1).add_(grad, alpha=1.0 - adamw_beta1)
                    exp_avg_sq.mul_(adamw_beta2).addcmul_(grad, grad, value=1.0 - adamw_beta2)

                    bias_correction1 = 1.0 - adamw_beta1 ** step
                    bias_correction2 = 1.0 - adamw_beta2 ** step
                    scale = bias_correction1 / (bias_correction2 ** 0.5)
                    update = exp_avg / (adamw_eps + exp_avg_sq.sqrt())

                    p.data.mul_(1.0 - adamw_lr * adamw_weight_decay)
                    p.data.add_(update, alpha=-adamw_lr / scale)

        return loss
