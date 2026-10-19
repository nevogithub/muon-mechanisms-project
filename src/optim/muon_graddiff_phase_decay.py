import math

import torch
from torch.optim.optimizer import Optimizer, required

from .muon import muon_update


class MuonGradDiffPhaseDecay(Optimizer):
    """
    Muon + AdamW with consecutive-gradient difference correction on the 2D path,
    while decaying Muon's own momentum and the grad-diff coefficient by epoch.

    2D+ params:
    - corrected_grad = grad + gamma_t * (grad - last_grad)
    - feed corrected_grad into Muon's exact update path
    - decay Muon's momentum beta over epochs

    Non-2D params:
    - unchanged AdamW branch
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.95,
        weight_decay=0.1,
        ns_schedule="constant",
        ns_steps_low=5,
        ns_steps_high=5,
        ns_warmup_epochs=1,
        gamma=0.1,
        gamma_end=0.0,
        gamma_schedule="linear",
        gamma_start_epoch=None,
        gamma_end_epoch=None,
        decay_schedule="linear",
        decay_start_epoch=15,
        decay_end_epoch=40,
        momentum_end=0.5,
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
            ns_schedule=ns_schedule,
            ns_steps_low=ns_steps_low,
            ns_steps_high=ns_steps_high,
            ns_warmup_epochs=ns_warmup_epochs,
            gamma=gamma,
            gamma_end=gamma_end,
            gamma_schedule=gamma_schedule,
            gamma_start_epoch=gamma_start_epoch,
            gamma_end_epoch=gamma_end_epoch,
            decay_schedule=decay_schedule,
            decay_start_epoch=decay_start_epoch,
            decay_end_epoch=decay_end_epoch,
            momentum_end=momentum_end,
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

    def _schedule_frac(self, group):
        start_epoch = int(group["decay_start_epoch"])
        end_epoch = int(group["decay_end_epoch"])
        mode = str(group.get("decay_schedule", "linear")).lower()

        if mode == "none":
            return 0.0

        if self.current_epoch <= start_epoch:
            frac = 0.0
        elif self.current_epoch >= end_epoch:
            frac = 1.0
        else:
            denom = max(1, end_epoch - start_epoch)
            frac = float(self.current_epoch - start_epoch) / float(denom)

        if mode == "linear":
            return frac
        if mode == "cosine":
            return 0.5 - 0.5 * math.cos(frac * math.pi)
        raise ValueError(f"Unsupported decay_schedule mode: {mode}")

    def _scheduled_momentum(self, group):
        start = float(group["momentum"])
        end = float(group["momentum_end"])
        frac = self._schedule_frac(group)
        beta_t = start + frac * (end - start)
        return max(0.0, min(0.999, float(beta_t)))

    def _scheduled_gamma(self, group):
        start = float(group["gamma"])
        end = float(group.get("gamma_end", start))
        mode = str(group.get("gamma_schedule", group.get("decay_schedule", "linear"))).lower()
        start_epoch = group.get("gamma_start_epoch", None)
        end_epoch = group.get("gamma_end_epoch", None)

        if start_epoch is None:
            start_epoch = int(group["decay_start_epoch"])
        else:
            start_epoch = int(start_epoch)

        if end_epoch is None:
            end_epoch = int(group["decay_end_epoch"])
        else:
            end_epoch = int(end_epoch)

        if mode == "none":
            return float(start)

        if self.current_epoch < start_epoch:
            return 0.0
        if self.current_epoch > end_epoch:
            return float(end)

        if self.current_epoch == end_epoch:
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

        gamma_t = start + frac * (end - start)
        return float(gamma_t)

    def _scheduled_ns_steps(self, group):
        low = int(group["ns_steps_low"])
        high = int(group["ns_steps_high"])
        warmup_epochs = int(group["ns_warmup_epochs"])
        mode = str(group.get("ns_schedule", "constant")).lower()

        if mode == "constant":
            return max(1, low)
        if mode == "linear":
            if warmup_epochs <= 1:
                return max(1, high)
            frac = min(1.0, float(self.current_epoch - 1) / float(warmup_epochs - 1))
            return max(1, int(round(low + frac * (high - low))))
        raise ValueError(f"Unsupported ns_schedule mode: {mode}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum = self._scheduled_momentum(group)
            gamma_t = self._scheduled_gamma(group)
            ns_steps = self._scheduled_ns_steps(group)
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
