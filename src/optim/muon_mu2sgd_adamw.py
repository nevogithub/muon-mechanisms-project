import math

import torch
from torch.optim.optimizer import Optimizer, required

from .muon import muon_update


class MuonMu2SGDAdamW(Optimizer):
    """
    Muon + AdamW with a Mu2-inspired long-history estimator inserted only into
    Muon's input-direction path.

    2D+ params:
    - maintain a recursive estimator d_t = g_t + alpha_t * (d_{t-1} - g_{t-1})
    - feed d_t into Muon's exact update machinery

    Non-2D params:
    - unchanged AdamW branch
    """

    def __init__(
        self,
        params,
        lr=required,
        mu2_betas=(0.95, 0.1),
        weight_decay=0.1,
        ns_steps=5,
        mu2_alpha_init=0.1,
        mu2_alpha_end=0.0,
        mu2_alpha_schedule="none",
        mu2_alpha_start_epoch=1,
        mu2_alpha_end_epoch=1,
        matched_adamw_rms=0.2,
        nesterov=True,
        adamw_lr=3e-4,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-8,
        adamw_weight_decay=0.0,
        preserve_mu2_estimator=False,
        diagnostics=False,
    ):
        if lr is required:
            raise ValueError("Learning rate is required")

        defaults = dict(
            lr=lr,
            mu2_betas=mu2_betas,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            mu2_alpha_init=mu2_alpha_init,
            mu2_alpha_end=mu2_alpha_end,
            mu2_alpha_schedule=mu2_alpha_schedule,
            mu2_alpha_start_epoch=mu2_alpha_start_epoch,
            mu2_alpha_end_epoch=mu2_alpha_end_epoch,
            matched_adamw_rms=matched_adamw_rms,
            nesterov=nesterov,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_weight_decay=adamw_weight_decay,
            preserve_mu2_estimator=preserve_mu2_estimator,
            diagnostics=diagnostics,
        )
        super().__init__(params, defaults)
        self.current_epoch = 1
        self.diagnostic_history = []
        self._last_diagnostic_epoch = None

    def set_epoch(self, epoch: int):
        self.current_epoch = max(1, int(epoch))

    def _scheduled_alpha(self, group):
        alpha_start = float(group.get("mu2_alpha_init", 0.0))
        alpha_end = float(group.get("mu2_alpha_end", 0.0))
        start_epoch = int(group.get("mu2_alpha_start_epoch", 1))
        end_epoch = int(group.get("mu2_alpha_end_epoch", 1))
        mode = str(group.get("mu2_alpha_schedule", "none")).lower()

        if mode == "none":
            alpha_t = alpha_start
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
                frac = 0.5 - 0.5 * torch.cos(torch.tensor(frac * 3.141592653589793)).item()
            else:
                raise ValueError(f"Unsupported mu2_alpha_schedule mode: {mode}")

            alpha_t = alpha_start + frac * (alpha_end - alpha_start)

        return max(0.0, float(alpha_t))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        collect_diagnostics = (
            self._last_diagnostic_epoch != self.current_epoch
            and any(bool(group.get("diagnostics", False)) for group in self.param_groups)
        )
        diagnostics = {
            "grad_sq": None,
            "estimator_sq": None,
            "correction_sq": None,
            "grad_estimator_dot": None,
            "pre_raw_sq": None,
            "pre_estimator_sq": None,
            "pre_dot": None,
            "post_raw_sq": None,
            "post_estimator_sq": None,
            "post_dot": None,
            "matrix_parameters": 0,
        }

        def add_scalar(key, value):
            value = value.float()
            diagnostics[key] = value if diagnostics[key] is None else diagnostics[key] + value

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            muon_beta = float(group["mu2_betas"][0])
            ns_steps = int(group["ns_steps"])
            nesterov = bool(group["nesterov"])

            adamw_lr = float(group["adamw_lr"])
            adamw_beta1, adamw_beta2 = group["adamw_betas"]
            adamw_eps = float(group["adamw_eps"])
            adamw_weight_decay = float(group["adamw_weight_decay"])

            if "step" not in group:
                group["step"] = 0
            group["step"] += 1
            step = int(group["step"])
            alpha_t = self._scheduled_alpha(group)
            preserve_estimator = bool(group.get("preserve_mu2_estimator", False))

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

                    momentum_buffer = state["momentum_buffer"]
                    mu2_estimator = state["mu2_estimator"]
                    last_grad = state["last_grad"]

                    mu2_estimator.copy_(grad + alpha_t * (mu2_estimator - last_grad))

                    if collect_diagnostics:
                        old_momentum = momentum_buffer.clone()
                        raw_momentum = old_momentum.lerp(grad, 1.0 - muon_beta)
                        estimator_momentum = old_momentum.lerp(mu2_estimator, 1.0 - muon_beta)
                        if nesterov:
                            raw_prepolar = grad.lerp(raw_momentum, muon_beta)
                            estimator_prepolar = mu2_estimator.lerp(estimator_momentum, muon_beta)
                        else:
                            raw_prepolar = raw_momentum
                            estimator_prepolar = estimator_momentum

                        correction = mu2_estimator - grad
                        add_scalar("grad_sq", torch.sum(grad.float().square()))
                        add_scalar("estimator_sq", torch.sum(mu2_estimator.float().square()))
                        add_scalar("correction_sq", torch.sum(correction.float().square()))
                        add_scalar("grad_estimator_dot", torch.sum(grad.float() * mu2_estimator.float()))
                        add_scalar("pre_raw_sq", torch.sum(raw_prepolar.float().square()))
                        add_scalar("pre_estimator_sq", torch.sum(estimator_prepolar.float().square()))
                        add_scalar("pre_dot", torch.sum(raw_prepolar.float() * estimator_prepolar.float()))

                    # The historical, author-approved Mu2-Muon integration stores
                    # Muon's Nesterov-combined pre-polar signal for the next Mu2
                    # recurrence. preserve_mu2_estimator=True instead retains the
                    # literal pre-Muon d_t and is used as a diagnostic no-op
                    # control: with a preceding-minibatch last_grad, that literal
                    # recurrence collapses algebraically to d_t == grad_t.
                    muon_input = mu2_estimator.clone() if preserve_estimator else mu2_estimator
                    update = muon_update(
                        muon_input,
                        momentum_buffer,
                        beta=muon_beta,
                        ns_steps=ns_steps,
                        nesterov=nesterov,
                    )

                    if collect_diagnostics:
                        raw_counterfactual = muon_update(
                            grad.clone(),
                            old_momentum,
                            beta=muon_beta,
                            ns_steps=ns_steps,
                            nesterov=nesterov,
                        )
                        add_scalar("post_raw_sq", torch.sum(raw_counterfactual.float().square()))
                        add_scalar("post_estimator_sq", torch.sum(update.float().square()))
                        add_scalar("post_dot", torch.sum(raw_counterfactual.float() * update.float()))
                        diagnostics["matrix_parameters"] += 1

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

        if collect_diagnostics and diagnostics["matrix_parameters"]:
            eps = 1e-30

            def scalar(key):
                return float(diagnostics[key].item())

            def cosine(dot_key, left_key, right_key):
                denom = math.sqrt(max(eps, scalar(left_key) * scalar(right_key)))
                return max(-1.0, min(1.0, scalar(dot_key) / denom))

            grad_estimator_cos = cosine("grad_estimator_dot", "grad_sq", "estimator_sq")
            prepolar_cos = cosine("pre_dot", "pre_raw_sq", "pre_estimator_sq")
            postpolar_cos = cosine("post_dot", "post_raw_sq", "post_estimator_sq")
            self.diagnostic_history.append(
                {
                    "Epoch": int(self.current_epoch),
                    "LR": float(self.param_groups[0]["lr"]),
                    "Muon Beta": float(self.param_groups[0]["mu2_betas"][0]),
                    "Mu2 Alpha": float(self._scheduled_alpha(self.param_groups[0])),
                    "Preserve Mu2 Estimator": bool(
                        self.param_groups[0].get("preserve_mu2_estimator", False)
                    ),
                    "Correction Relative Norm": math.sqrt(
                        scalar("correction_sq") / max(eps, scalar("grad_sq"))
                    ),
                    "Gradient Estimator Cosine": grad_estimator_cos,
                    "Gradient Estimator Angle Degrees": math.degrees(math.acos(grad_estimator_cos)),
                    "Prepolar Counterfactual Cosine": prepolar_cos,
                    "Prepolar Counterfactual Angle Degrees": math.degrees(math.acos(prepolar_cos)),
                    "Postpolar Counterfactual Cosine": postpolar_cos,
                    "Postpolar Counterfactual Angle Degrees": math.degrees(math.acos(postpolar_cos)),
                    "Matrix Parameter Tensors": int(diagnostics["matrix_parameters"]),
                }
            )
            self._last_diagnostic_epoch = self.current_epoch

        return loss
