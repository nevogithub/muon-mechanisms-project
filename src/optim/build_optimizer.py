"""Optimizer factory for the Muon mechanisms evaluated in this project."""

from .muon import SingleDeviceMuonWithAuxAdam
from .muon_graddiff import MuonGradDiff
from .muon_graddiff_phase_decay import MuonGradDiffPhaseDecay
from .muon_mars_mu2sgd_phase_decay import MuonMarsMu2SGDPhaseDecay
from .muon_momentum_decay_adamw import MuonMomentumDecayAdamW
from .muon_mu2sgd_adamw import MuonMu2SGDAdamW


SUPPORTED_OPTIMIZERS = {
    "muon_adamw",
    "muon_graddiff",
    "muon_graddiff_phase_decay",
    "muon_mars_mu2sgd_phase_decay",
    "muon_momentum_decay_adamw",
    "muon_mu2sgd_adamw",
}


def _cli_or(value, default):
    return default if value is None else value


def _adamw_settings(specific_cfg, args):
    beta1, beta2 = specific_cfg.get("adamw_betas", [0.9, 0.999])
    if args.non2d_betas is not None:
        beta1, beta2 = args.non2d_betas
    return {
        "adamw_lr": float(_cli_or(args.non2d_lr, specific_cfg.get("adamw_lr", 1e-3))),
        "adamw_betas": (float(beta1), float(beta2)),
        "adamw_eps": float(specific_cfg.get("adamw_eps", 1e-8)),
        "adamw_weight_decay": float(specific_cfg.get("adamw_weight_decay", 0.0)),
    }


def build_optimizer(model, cfg, args):
    """Build one of the six Muon-family optimizers used in the final study."""

    opt_cfg = cfg["optimizer"]
    name = args.optimizer or opt_cfg["name"]
    if name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer {name!r}. This public snapshot contains only "
            f"the final project mechanisms: {sorted(SUPPORTED_OPTIMIZERS)}"
        )

    lr = float(_cli_or(args.lr, opt_cfg["lr"]))
    weight_decay = float(_cli_or(args.weight_decay, opt_cfg.get("weight_decay", 0.0)))
    specific_cfg = opt_cfg.get(name, {}).copy()
    adamw = _adamw_settings(specific_cfg, args)

    if name == "muon_adamw":
        momentum = float(_cli_or(args.momentum, specific_cfg.get("momentum", 0.95)))
        ns_steps = int(specific_cfg.get("ns_steps", 5))
        nesterov = bool(specific_cfg.get("nesterov", True))
        exclude_output_head = bool(specific_cfg.get("exclude_output_head", False))

        named_params = [
            (param_name, param)
            for param_name, param in model.named_parameters()
            if param.requires_grad
        ]

        def is_output_head(param_name):
            return param_name == "head.fc.weight" or param_name in {
                "head.weight",
                "fc.weight",
                "classifier.weight",
            }

        muon_params = [
            param
            for param_name, param in named_params
            if param.ndim >= 2
            and not (exclude_output_head and is_output_head(param_name))
        ]
        auxiliary_params = [
            param
            for param_name, param in named_params
            if param.ndim < 2
            or (exclude_output_head and is_output_head(param_name))
        ]
        groups = [
            {
                "params": muon_params,
                "use_muon": True,
                "lr": lr,
                "momentum": momentum,
                "ns_steps": ns_steps,
                "nesterov": nesterov,
                "weight_decay": weight_decay,
            },
            {
                "params": auxiliary_params,
                "use_muon": False,
                "lr": adamw["adamw_lr"],
                "betas": adamw["adamw_betas"],
                "eps": adamw["adamw_eps"],
                "weight_decay": adamw["adamw_weight_decay"],
            },
        ]
        return SingleDeviceMuonWithAuxAdam(groups)

    if name == "muon_graddiff":
        momentum = float(_cli_or(args.momentum, specific_cfg.get("momentum", 0.95)))
        gamma = float(_cli_or(args.mvr_gamma, specific_cfg.get("gamma", 0.1)))
        return MuonGradDiff(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=int(specific_cfg.get("ns_steps", 5)),
            gamma=gamma,
            gamma_end=float(specific_cfg.get("gamma_end", gamma)),
            gamma_schedule=str(specific_cfg.get("gamma_schedule", "none")),
            gamma_start_epoch=int(specific_cfg.get("gamma_start_epoch", 1)),
            gamma_end_epoch=int(specific_cfg.get("gamma_end_epoch", 1)),
            nesterov=bool(specific_cfg.get("nesterov", True)),
            **adamw,
        )

    if name == "muon_mu2sgd_adamw":
        beta1_default, alpha_default = specific_cfg.get("mu2_betas", [0.95, 0.1])
        beta1 = float(_cli_or(args.momentum, beta1_default))
        alpha = float(_cli_or(args.second_momentum, alpha_default))
        return MuonMu2SGDAdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            mu2_betas=(beta1, alpha),
            ns_steps=int(specific_cfg.get("ns_steps", 5)),
            mu2_alpha_init=float(specific_cfg.get("mu2_alpha_init", alpha)),
            mu2_alpha_end=float(specific_cfg.get("mu2_alpha_end", 0.0)),
            mu2_alpha_schedule=str(specific_cfg.get("mu2_alpha_schedule", "none")),
            mu2_alpha_start_epoch=int(specific_cfg.get("mu2_alpha_start_epoch", 1)),
            mu2_alpha_end_epoch=int(specific_cfg.get("mu2_alpha_end_epoch", 1)),
            matched_adamw_rms=float(specific_cfg.get("matched_adamw_rms", 0.2)),
            nesterov=bool(specific_cfg.get("nesterov", True)),
            preserve_mu2_estimator=bool(specific_cfg.get("preserve_mu2_estimator", False)),
            diagnostics=bool(specific_cfg.get("diagnostics", False)),
            **adamw,
        )

    if name == "muon_mars_mu2sgd_phase_decay":
        beta1_default, alpha_default = specific_cfg.get("mu2_betas", [0.95, 0.1])
        beta1 = float(_cli_or(args.momentum, beta1_default))
        alpha = float(_cli_or(args.second_momentum, alpha_default))
        gamma = float(_cli_or(args.mvr_gamma, specific_cfg.get("gamma", 0.5)))
        return MuonMarsMu2SGDPhaseDecay(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            mu2_betas=(beta1, alpha),
            ns_schedule=str(specific_cfg.get("ns_schedule", "constant")),
            ns_steps_low=int(specific_cfg.get("ns_steps_low", 5)),
            ns_steps_high=int(specific_cfg.get("ns_steps_high", 5)),
            ns_warmup_epochs=int(specific_cfg.get("ns_warmup_epochs", 1)),
            gamma=gamma,
            gamma_end=float(specific_cfg.get("gamma_end", 0.1)),
            gamma_schedule=str(specific_cfg.get("gamma_schedule", "cosine")),
            gamma_start_epoch=specific_cfg.get("gamma_start_epoch"),
            gamma_end_epoch=specific_cfg.get("gamma_end_epoch"),
            mu2_alpha_init=float(specific_cfg.get("mu2_alpha_init", alpha)),
            mu2_alpha_end=float(specific_cfg.get("mu2_alpha_end", 0.05)),
            mu2_decay_schedule=str(specific_cfg.get("mu2_decay_schedule", "cosine")),
            mu2_decay_start_epoch=int(specific_cfg.get("mu2_decay_start_epoch", 6)),
            mu2_decay_end_epoch=int(specific_cfg.get("mu2_decay_end_epoch", 18)),
            mu2_beta_end=float(specific_cfg.get("mu2_beta_end", 0.5)),
            nesterov=bool(specific_cfg.get("nesterov", True)),
            non2d_optimizer="adamw",
            **adamw,
        )

    if name == "muon_graddiff_phase_decay":
        momentum = float(_cli_or(args.momentum, specific_cfg.get("momentum", 0.95)))
        gamma = float(_cli_or(args.mvr_gamma, specific_cfg.get("gamma", 0.1)))
        decay_schedule = str(specific_cfg.get("decay_schedule", "linear"))
        return MuonGradDiffPhaseDecay(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_schedule=str(specific_cfg.get("ns_schedule", "constant")),
            ns_steps_low=int(specific_cfg.get("ns_steps_low", 5)),
            ns_steps_high=int(specific_cfg.get("ns_steps_high", 5)),
            ns_warmup_epochs=int(specific_cfg.get("ns_warmup_epochs", 1)),
            gamma=gamma,
            gamma_end=float(specific_cfg.get("gamma_end", 0.0)),
            gamma_schedule=str(specific_cfg.get("gamma_schedule", decay_schedule)),
            gamma_start_epoch=specific_cfg.get("gamma_start_epoch"),
            gamma_end_epoch=specific_cfg.get("gamma_end_epoch"),
            decay_schedule=decay_schedule,
            decay_start_epoch=int(specific_cfg.get("decay_start_epoch", 15)),
            decay_end_epoch=int(specific_cfg.get("decay_end_epoch", 40)),
            momentum_end=float(specific_cfg.get("momentum_end", 0.5)),
            nesterov=bool(specific_cfg.get("nesterov", True)),
            **adamw,
        )

    if name == "muon_momentum_decay_adamw":
        momentum = float(_cli_or(args.momentum, specific_cfg.get("momentum", 0.95)))
        return MuonMomentumDecayAdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            momentum_end=float(specific_cfg.get("momentum_end", momentum)),
            decay_schedule=str(specific_cfg.get("decay_schedule", "none")),
            decay_start_epoch=int(specific_cfg.get("decay_start_epoch", 1)),
            decay_end_epoch=int(specific_cfg.get("decay_end_epoch", 1)),
            ns_steps=int(specific_cfg.get("ns_steps", 5)),
            nesterov=bool(specific_cfg.get("nesterov", True)),
            **adamw,
        )

    raise AssertionError(f"Unhandled optimizer {name}")
