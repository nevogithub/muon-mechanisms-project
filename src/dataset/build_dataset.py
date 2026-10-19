from torch.utils.data import DataLoader
from .cifar100 import CIFAR100
from .imagenet100 import ImageNet100


DATASET_REGISTRY = {
    "cifar100": CIFAR100,
    "imagenet100": ImageNet100,
}


# ==========================================================
# Build Dataset
# ==========================================================
def build_dataset(cfg):
    """
    Build dataset according to cfg["dataset"].
    Returns:
        train_dataset, test_dataset
    """

    dataset_cfg = cfg["dataset"]
    dataset_name = dataset_cfg["name"]

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_class = DATASET_REGISTRY[dataset_name]

    dataset_kwargs = {}
    if "root" in dataset_cfg:
        dataset_kwargs["root"] = dataset_cfg["root"]
    for key in [
        "recipe",
        "randaugment_num_ops",
        "randaugment_magnitude",
        "color_jitter",
        "auto_augment",
        "interpolation",
        "train_crop_scale",
        "train_crop_ratio",
        "horizontal_flip_p",
        "random_erasing_p",
        "random_erasing_mode",
        "random_erasing_count",
        "train_subdir",
        "val_subdir",
        "image_size",
        "crop_pct",
        "class_manifest_sha256",
    ]:
        if key in dataset_cfg:
            dataset_kwargs[key] = dataset_cfg[key]

    dataset = dataset_class(**dataset_kwargs)

    train_eval = getattr(dataset, "trainset_eval", dataset.trainset)
    return dataset.trainset, dataset.testset, train_eval


# ==========================================================
# Build Dataloader
# ==========================================================
def build_dataloader(train_dataset, test_dataset, cfg, train_eval_dataset=None):
    """
    Build train, test, and (optionally) train-eval dataloaders.
    train_eval_dataset: clean train set without augmentation, for measuring train accuracy.
    """

    dataset_cfg = cfg["dataset"]

    batch_size = dataset_cfg["batch_size"]
    num_workers = dataset_cfg["num_workers"]

    import torch
    use_pin_memory = bool(dataset_cfg.get("pin_memory", torch.cuda.is_available()))
    train_persistent = bool(
        dataset_cfg.get("persistent_workers", num_workers > 0)
    ) and num_workers > 0
    eval_persistent = bool(
        dataset_cfg.get("eval_persistent_workers", False)
    ) and num_workers > 0
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = dataset_cfg.get("prefetch_factor", 4)
    train_loader_kwargs = {
        **loader_kwargs,
        "persistent_workers": train_persistent,
    }
    eval_loader_kwargs = {
        **loader_kwargs,
        "persistent_workers": eval_persistent,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=bool(dataset_cfg.get("train_shuffle", True)),
        **train_loader_kwargs,
        drop_last=bool(dataset_cfg.get("train_drop_last", True)),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=bool(dataset_cfg.get("eval_shuffle", False)),
        **eval_loader_kwargs,
        drop_last=bool(dataset_cfg.get("eval_drop_last", False)),
    )

    # Clean train eval loader (no augmentation) — falls back to train_loader if not provided
    eval_dataset = train_eval_dataset if train_eval_dataset is not None else train_dataset
    train_eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=bool(dataset_cfg.get("eval_shuffle", False)),
        **eval_loader_kwargs,
        drop_last=bool(dataset_cfg.get("eval_drop_last", False)),
    )

    return train_loader, test_loader, train_eval_loader
