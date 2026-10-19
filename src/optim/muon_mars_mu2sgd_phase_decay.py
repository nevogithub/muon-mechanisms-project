import math

import torch
from torch.optim.optimizer import Optimizer, required

from .muon import muon_update


class MuonMarsMu2SGDPhaseDecay(Optimizer):
    """
    Muon + AdamW with MARS-style gradient difference, Mu2SGD-style direction
    smoothing, and scheduled Muon momentum decay on the 2D path.

    2D+ params:
    - mars_grad = grad + gamma_t * (grad - last_grad)
    - mu2_estimator = mars_grad + alpha_t * (mu2_estimator - last_mars_grad)
    - feed mu2_estimator into Muon's update path with beta_t

    Non-2D params:
    - AdamW branch, matching the other Muon+AdamW variants in this repo.
    """

    def __init__(
        self,
        params,
        lr=required,
        mu2_betas=(0.95, 0.1),
        weight_decay=0.1,
        ns_schedule="constant",
        ns_steps_low=5,
        ns_steps_high=5,
        ns_warmup_epochs=1,
        gamma=0.5,
        gamma_end=0.1,
        gamma_schedule="cosine",
        gamma_start_epoch=None,
        gamma_end_epoch=None,
        mu2_alpha_init=0.2,
        mu2_alpha_end=0.05,
        mu2_decay_schedule="cosine",
        mu2_decay_start_epoch=6,
        mu2_decay_end_epoch=18,
        mu2_beta_end=0.5,
        nesterov=True,
        adamw_lr=3e-4,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-8,
        adamw_weight_decay=0.0,
        non2d_optimizer="adamw",
        adamu2w_betas=(0.95, 0.1, 0.99),
        adamu2w_gamma=0.025,
    ):
        if lr is required:
            raise ValueError("Learning rate is required")

        defaults = dict(
            lr=lr,
            mu2_betas=mu2_betas,
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
            mu2_alpha_init=mu2_alpha_init,
            mu2_alpha_end=mu2_alpha_end,
            mu2_decay_schedule=mu2_decay_schedule,
            mu2_decay_start_epoch=mu2_decay_start_epoch,
            mu2_decay_end_epoch=mu2_decay_end_epoch,
            mu2_beta_end=mu2_beta_end,
            nesterov=nesterov,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_weight_decay=adamw_weight_decay,
            non2d_optimizer=non2d_optimizer,
            adamu2w_betas=adamu2w_betas,
            adamu2w_gamma=adamu2w_gamma,
        )
        super().__init__(params, defaults)
        self.current_epoch = 1

    def set_epoch(self, epoch: int):
        self.current_epoch = max(1, int(epoch))

    def _schedule_frac(self, start_epoch, end_epoch, mode):
        mode = str(mode).lower()
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
        raise ValueError(f"Unsupported schedule mode: {mode}")

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

    def _scheduled_muon_beta(self, group):
        beta_start = float(group["mu2_betas"][0])
        beta_end = float(group["mu2_beta_end"])
        mode = str(group.get("mu2_decay_schedule", "cosine")).lower()
        frac = self._schedule_frac(
            int(group["mu2_decay_start_epoch"]),
            int(group["mu2_decay_end_epoch"]),
            mode,
        )
        beta_t = beta_start + frac * (beta_end - beta_start)
        return max(0.0, min(0.999, float(beta_t)))

    def _scheduled_alpha(self, group):
        alpha_start = float(group.get("mu2_alpha_init", group["mu2_betas"][1]))
        alpha_end = float(group.get("mu2_alpha_end", alpha_start))
        mode = str(group.get("mu2_decay_schedule", "cosine")).lower()
        frac = self._schedule_frac(
            int(group["mu2_decay_start_epoch"]),
            int(group["mu2_decay_end_epoch"]),
            mode,
        )
        alpha_t = alpha_start + frac * (alpha_end - alpha_start)
        return max(0.0, float(alpha_t))

    def _scheduled_gamma(self, group):
        gamma_start = float(group.get("gamma", 0.0))
        gamma_end = float(group.get("gamma_end", gamma_start))
        mode = str(group.get("gamma_schedule", group.get("mu2_decay_schedule", "cosine"))).lower()

        start_epoch = group.get("gamma_start_epoch", None)
        end_epoch = group.get("gamma_end_epoch", None)
        if start_epoch is None:
            start_epoch = int(group["mu2_decay_start_epoch"])
        else:
            start_epoch = int(start_epoch)
        if end_epoch is None:
            end_epoch = int(group["mu2_decay_end_epoch"])
        else:
            end_epoch = int(end_epoch)

        if mode == "none":
            return gamma_start

        frac = self._schedule_frac(start_epoch, end_epoch, mode)
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
            muon_beta = self._scheduled_muon_beta(group)
            alpha_t = self._scheduled_alpha(group)
            gamma_t = self._scheduled_gamma(group)
            ns_steps = self._scheduled_ns_steps(group)
            nesterov = bool(group["nesterov"])

            adamw_lr = float(group["adamw_lr"])
            adamw_beta1, adamw_beta2 = group["adamw_betas"]
            adamw_eps = float(group["adamw_eps"])
            adamw_weight_decay = float(group["adamw_weight_decay"])
            non2d_optimizer = str(group.get("non2d_optimizer", "adamw")).lower()
            adamu2w_beta1, adamu2w_beta2, adamu2w_beta3 = group.get(
                "adamu2w_betas", (0.95, 0.1, 0.99)
            )
            adamu2w_gamma = float(group.get("adamu2w_gamma", 0.025))

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
                        state["mu2_estimator"] = torch.zeros_like(p)
                        state["last_grad"] = torch.zeros_like(p)
                        state["last_mars_grad"] = torch.zeros_like(p)

                    momentum_buffer = state["momentum_buffer"]
                    mu2_estimator = state["mu2_estimator"]
                    last_grad = state["last_grad"]
                    last_mars_grad = state["last_mars_grad"]

                    grad_diff = grad - last_grad
                    mars_grad = grad + gamma_t * grad_diff
                    mu2_estimator.copy_(mars_grad + alpha_t * (mu2_estimator - last_mars_grad))

                    update = muon_update(
                        mu2_estimator,
                        momentum_buffer,
                        beta=muon_beta,
                        ns_steps=ns_steps,
                        nesterov=nesterov,
                    )

                    p.data.mul_(1.0 - lr * weight_decay)
                    p.data.add_(update.to(p.dtype), alpha=-lr)
                    last_grad.copy_(grad)
                    last_mars_grad.copy_(mars_grad)
                else:
                    if non2d_optimizer == "adamu2w":
                        if len(state) == 0:
                            state["adamu2w_exp_avg"] = torch.zeros_like(p)
                            state["adamu2w_second_exp_avg"] = torch.zeros_like(p)
                            state["adamu2w_exp_avg_sq"] = torch.zeros_like(p)
                            state["adamu2w_last_grad"] = torch.zeros_like(p)

                        exp_avg = state["adamu2w_exp_avg"]
                        second_exp_avg = state["adamu2w_second_exp_avg"]
                        exp_avg_sq = state["adamu2w_exp_avg_sq"]
                        last_grad = state["adamu2w_last_grad"]

                        grad_diff = grad - last_grad
                        corrected_grad = grad + grad_diff * (
                            adamu2w_gamma * (adamu2w_beta1 / (1.0 - adamu2w_beta1))
                        )

                        exp_avg.mul_(adamu2w_beta1).add_(corrected_grad, alpha=1.0 - adamu2w_beta1)
                        second_exp_avg.mul_(adamu2w_beta2).add_(exp_avg, alpha=1.0 - adamu2w_beta2)
                        exp_avg_sq.mul_(adamu2w_beta3).addcmul_(
                            corrected_grad, corrected_grad, value=1.0 - adamu2w_beta3
                        )

                        bias_correction1 = 1.0 - adamu2w_beta1 ** step
                        bias_correction2 = 1.0 - adamu2w_beta3 ** step
                        denom = (
                            exp_avg_sq.sqrt()
                            .mul(1.0 / (bias_correction2 ** 0.5))
                            .add_(adamw_eps)
                            .mul_(bias_correction1)
                        )
                        update = second_exp_avg / denom

                        p.data.mul_(1.0 - adamw_lr * adamw_weight_decay)
                        p.data.add_(update, alpha=-adamw_lr)
                        last_grad.copy_(grad)
                        continue

                    if non2d_optimizer != "adamw":
                        raise ValueError(f"Unsupported non2d_optimizer: {non2d_optimizer}")

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
