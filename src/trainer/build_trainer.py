from .clf_trainer import CLFTrainer

TRAINER_REGISTRY = {
    "clf": CLFTrainer,
}


def build_trainer(model, optimizer, train_loader, test_loader, device, cfg, args, scheduler, train_eval_loader=None):
    """
    Build trainer according to cfg["training"]["trainer"].
    """
    trainer_name = cfg["training"].get("trainer", "clf")

    if trainer_name not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown trainer: {trainer_name}")

    return TRAINER_REGISTRY[trainer_name](
        model,
        optimizer,
        train_loader,
        test_loader,
        device,
        cfg,
        args,
        scheduler,
        train_eval_loader,
    )
