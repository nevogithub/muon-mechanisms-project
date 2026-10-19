import math

import torch
from torch.optim.optimizer import Optimizer, required

from .muon import muon_update


class MuonMomentumDecayAdamW(Optimizer):
    """Plain Muon + auxiliary AdamW with an epoch-scheduled Muon momentum.

    Matrix parameters use the unmodified gradient as Muon's input. There is no
    MARS gradient correction and no Mu2 direction estimator. Non-matrix
    parameters use the same auxiliary AdamW update as the other Muon variants.
    """

    def __init__(
        self,
        params,
        lr=required,
        weight_decay=0.0,
        momentum=0.95,
        momentum_end=0.95,
        decay_schedule="none",
        decay_start_epoch=1,
        decay_end_epoch=1,
        ns_steps=5,
        nesterov=True,
        adamw_lr=3.0e-4,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1.0e-8,
        adamw_weight_decay=0.0,
    ):
        if lr is required:
            raise ValueError("Learning rate is required")
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if not 0.0 <= float(momentum_end) < 1.0:
            raise ValueError("momentum_end must be in [0, 1)")
        if int(decay_end_epoch) < int(decay_start_epoch):
            raise ValueError("decay_end_epoch must be >= decay_start_epoch")
        if str(decay_schedule).lower() not in {"none", "linear", "cosine"}:
            raise ValueError("decay_schedule must be none, linear, or cosine")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            momentum_end=momentum_end,
            decay_schedule=decay_schedule,
            decay_start_epoch=decay_start_epoch,
            decay_end_epoch=decay_end_epoch,
            ns_steps=ns_steps,
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

    def scheduled_momentum(self, group=None) -> float:
        if group is None:
            group = self.param_groups[0]

        start = float(group["momentum"])
        end = float(group["momentum_end"])
        mode = str(group["decay_schedule"]).lower()
        start_epoch = int(group["decay_start_epoch"])
        end_epoch = int(group["decay_end_epoch"])

        if mode == "none" or self.current_epoch <= start_epoch:
            fraction = 0.0
        elif self.current_epoch >= end_epoch:
            fraction = 1.0
        else:
            duration = max(1, end_epoch - start_epoch)
            fraction = float(self.current_epoch - start_epoch) / float(duration)

        if mode == "cosine":
            fraction = 0.5 - 0.5 * math.cos(math.pi * fraction)
        beta = start + fraction * (end - start)
        return max(0.0, min(0.999, float(beta)))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta = self.scheduled_momentum(group)
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])
            nesterov = bool(group["nesterov"])

            adamw_lr = float(group["adamw_lr"])
            adamw_beta1, adamw_beta2 = group["adamw_betas"]
            adamw_eps = float(group["adamw_eps"])
            adamw_weight_decay = float(group["adamw_weight_decay"])
            group["step"] = int(group.get("step", 0)) + 1
            adam_step = int(group["step"])

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad.detach()
                state = self.state[parameter]

                if parameter.ndim >= 2:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    update = muon_update(
                        grad,
                        state["momentum_buffer"],
                        beta=beta,
                        ns_steps=ns_steps,
                        nesterov=nesterov,
                    )
                    parameter.mul_(1.0 - lr * weight_decay)
                    parameter.add_(update.to(parameter.dtype), alpha=-lr)
                    continue

                if "adamw_exp_avg" not in state:
                    state["adamw_exp_avg"] = torch.zeros_like(parameter)
                    state["adamw_exp_avg_sq"] = torch.zeros_like(parameter)
                exp_avg = state["adamw_exp_avg"]
                exp_avg_sq = state["adamw_exp_avg_sq"]
                exp_avg.lerp_(grad, 1.0 - adamw_beta1)
                exp_avg_sq.lerp_(grad.square(), 1.0 - adamw_beta2)

                bias_correction1 = 1.0 - adamw_beta1**adam_step
                bias_correction2 = 1.0 - adamw_beta2**adam_step
                corrected_avg = exp_avg / bias_correction1
                corrected_avg_sq = exp_avg_sq / bias_correction2
                update = corrected_avg / (corrected_avg_sq.sqrt() + adamw_eps)
                parameter.mul_(1.0 - adamw_lr * adamw_weight_decay)
                parameter.add_(update, alpha=-adamw_lr)

        return loss
