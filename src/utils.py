"""Configuration loading and reproducibility helpers."""

import random
import re

import numpy as np
import torch
import yaml


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?$")


def _normalize_scalar(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if _INT_RE.fullmatch(text):
        return int(text)
    if _FLOAT_RE.fullmatch(text):
        return float(text)
    return value


def _normalize(value):
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return _normalize_scalar(value)


def _set_if_present(target, key, value):
    if value is not None:
        target[key] = value


def override_config_with_args(cfg, args):
    """Apply only the command-line overrides supported by the public runner."""

    seed = getattr(args, "seed", None)
    if seed is not None:
        cfg["seeds"] = seed

    dataset = cfg.setdefault("dataset", {})
    _set_if_present(dataset, "name", getattr(args, "dataset", None))
    _set_if_present(dataset, "root", getattr(args, "data_root", None))
    _set_if_present(dataset, "batch_size", getattr(args, "batch_size", None))
    _set_if_present(dataset, "num_workers", getattr(args, "num_workers", None))

    model = cfg.setdefault("model", {})
    _set_if_present(model, "name", getattr(args, "model", None))
    _set_if_present(model, "num_classes", getattr(args, "num_classes", None))

    training = cfg.setdefault("training", {})
    _set_if_present(training, "epochs", getattr(args, "epochs", None))
    _set_if_present(training, "eval_interval", getattr(args, "eval_interval", None))
    _set_if_present(training, "grad_clip", getattr(args, "grad_clip", None))
    _set_if_present(training, "trainer", getattr(args, "trainer", None))

    optimizer = cfg.setdefault("optimizer", {})
    _set_if_present(optimizer, "name", getattr(args, "optimizer", None))
    _set_if_present(optimizer, "lr", getattr(args, "lr", None))
    _set_if_present(optimizer, "weight_decay", getattr(args, "weight_decay", None))

    scheduler = getattr(args, "scheduler", None)
    if scheduler is not None:
        cfg.setdefault("scheduler", {})["name"] = scheduler

    wandb = cfg.setdefault("wandb", {})
    if getattr(args, "wandb", False):
        wandb["enabled"] = True
    _set_if_present(wandb, "project", getattr(args, "wandb_project", None))
    _set_if_present(wandb, "entity", getattr(args, "wandb_entity", None))
    _set_if_present(wandb, "run_name", getattr(args, "wandb_run_name", None))
    return cfg


def load_config(arg):
    args = arg if hasattr(arg, "config") else None
    path = args.config if args is not None else arg
    with open(path, "r", encoding="utf-8") as stream:
        cfg = _normalize(yaml.safe_load(stream))
    return override_config_with_args(cfg, args) if args is not None else cfg


def set_seed(seed):
    if seed is None:
        raise ValueError("A seed is required for every experiment")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
