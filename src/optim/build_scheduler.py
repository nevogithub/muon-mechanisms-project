"""Learning-rate scheduler used by the final constant and cosine-LR experiments."""

import torch


def build_scheduler(optimizer, cfg, args):
    sched_cfg = cfg.get("scheduler", {})
    name = args.scheduler or sched_cfg.get("name")
    if not name:
        return None
    if name != "cosine":
        raise ValueError(
            f"Unsupported scheduler {name!r}. The final study used constant LR "
            "or cosine annealing only."
        )

    epochs = int(cfg.get("training", {}).get("epochs", 50))
    eta_min = sched_cfg.get("eta_min")
    if eta_min is None and "eta_min_ratio" in sched_cfg:
        matrix_lr = float(cfg.get("optimizer", {}).get("lr", 0.0))
        eta_min = float(sched_cfg["eta_min_ratio"]) * matrix_lr
    if eta_min is None:
        eta_min = 0.0

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(sched_cfg.get("T_max", epochs)),
        eta_min=float(eta_min),
    )
