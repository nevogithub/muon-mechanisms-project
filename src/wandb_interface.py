from __future__ import annotations

from typing import Any, Dict, Optional, Iterable

import os
import pandas as pd

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


_METRIC_COLUMNS = [
    "Train Loss",
    "Train Accuracy",
    "Test Loss",
    "Test Accuracy",
    "VR Loss",
]


def _is_wandb_available() -> bool:
    return _wandb is not None


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): _to_serializable(v) for k, v in config.items()}


def _format_lr_value(lr: Optional[float]) -> Optional[str]:
    if lr is None:
        return None
    try:
        return f"{float(lr):g}"
    except (TypeError, ValueError):
        return str(lr)


def _get_scheduler_name(cfg: Dict[str, Any], params: Dict[str, Any]) -> str:
    scheduler = params.get("scheduler")
    if scheduler:
        return str(scheduler)
    return str(cfg.get("scheduler", {}).get("name") or "constant")


def _get_lr_value(cfg: Dict[str, Any], params: Dict[str, Any]) -> Optional[float]:
    lr = params.get("lr")
    if lr is not None:
        return lr
    return cfg.get("optimizer", {}).get("lr")


def build_run_name(cfg: Dict[str, Any], params: Dict[str, Any]) -> str:
    """
    Build a run name that includes optimizer, dataset, model, and LR schedule type.
    Example: adamw--mnist--simple_conv--lr-0.0003-constant
    """
    optimizer = params.get("optimizer") or cfg.get("optimizer", {}).get("name")
    dataset = params.get("dataset") or cfg.get("dataset", {}).get("name")
    model = params.get("model") or cfg.get("model", {}).get("name")

    lr_value = _format_lr_value(_get_lr_value(cfg, params))
    schedule_name = _get_scheduler_name(cfg, params)

    lr_tag = f"lr-{lr_value}-{schedule_name}" if lr_value is not None else f"lr-{schedule_name}"

    parts = [optimizer, dataset, model, lr_tag]
    return "--".join([str(p) for p in parts if p])


def build_experiment_run_name(experiment_name: str, exp_cfg: Dict[str, Any]) -> str:
    opt_name = exp_cfg.get("optimizer", {}).get("name", "unknown")
    dataset = exp_cfg.get("dataset", {}).get("name", "unknown")
    model = exp_cfg.get("model", {}).get("name", "unknown")
    lr = exp_cfg.get("optimizer", {}).get("lr", 0)
    sched = exp_cfg.get("scheduler", {}).get("name") or "constant"

    lr_value = _format_lr_value(lr) or str(lr)
    return f"{experiment_name}--{opt_name}--{dataset}--{model}--lr-{lr_value}-{sched}"


def _wandb_common_kwargs(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    extra = extra or {}
    out: Dict[str, Any] = {}

    for key in ["group", "job_type", "notes", "mode"]:
        value = extra.get(key)
        if value is not None:
            out[key] = value

    tags = extra.get("tags")
    if tags is not None:
        if isinstance(tags, str):
            out["tags"] = [tags]
        elif isinstance(tags, Iterable):
            out["tags"] = [str(tag) for tag in tags]

    return out


def init_training_run(cfg: Dict[str, Any], params: Dict[str, Any]):
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    if not _is_wandb_available():
        print("WARNING: WandB is enabled in config, but the wandb package is not installed.")
        return None

    run_name = wandb_cfg.get("run_name") or build_run_name(cfg, params)

    init_kwargs = {
        "project": wandb_cfg.get("project", "mlopt-benchmark"),
        "entity": wandb_cfg.get("entity"),
        "name": run_name,
        "config": sanitize_config(params),
        "reinit": True,
    }
    init_kwargs.update(_wandb_common_kwargs(wandb_cfg))
    return _wandb.init(**init_kwargs)


def init_summary_run(
    wandb_project: str,
    wandb_entity: Optional[str],
    run_name: str,
    config: Dict[str, Any],
    tags: Optional[Iterable[str]] = None,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    notes: Optional[str] = None,
    mode: Optional[str] = None,
):
    if not _is_wandb_available():
        print("WARNING: Skipping WandB summary logging because wandb is not installed.")
        return None

    init_kwargs = {
        "project": wandb_project,
        "entity": wandb_entity,
        "name": run_name,
        "config": sanitize_config(config),
        "reinit": True,
    }

    common_kwargs = _wandb_common_kwargs({
        "tags": tags,
        "group": group,
        "job_type": job_type,
        "notes": notes,
        "mode": mode,
    })
    init_kwargs.update(common_kwargs)

    return _wandb.init(**init_kwargs)


def log_metrics(run: Any, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
    if run is None:
        return
    payload = sanitize_config(metrics)
    if step is None:
        run.log(payload, commit=commit)
    else:
        run.log(payload, step=step, commit=commit)


def set_run_summary(run: Any, metrics: Dict[str, Any]) -> None:
    if run is None:
        return
    for key, value in sanitize_config(metrics).items():
        run.summary[key] = value


def finish_run(run: Any) -> None:
    if run is None:
        return
    run.finish()


def _get_raw_data(results_folder: str) -> pd.DataFrame:
    # Collect results.csv across run folders without importing torch-heavy utils.
    if not os.path.isdir(results_folder):
        return pd.DataFrame()

    dfs = []
    for item in os.listdir(results_folder):
        item_path = os.path.join(results_folder, item)
        if os.path.isdir(item_path):
            results_path = os.path.join(item_path, "results.csv")
            if os.path.exists(results_path):
                dfs.append(pd.read_csv(results_path))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _get_averaged_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    non_empty_columns = [col for col in raw_df.columns if not raw_df[col].isna().all()]
    metric_columns = [col for col in _METRIC_COLUMNS if col in raw_df.columns]

    if not metric_columns:
        return pd.DataFrame()

    group_columns = [
        col
        for col in non_empty_columns
        if col not in metric_columns + ["seed"]
    ]

    grouped = raw_df.groupby(group_columns)
    result_mean = grouped[metric_columns].mean().reset_index()
    result_std = grouped[metric_columns].std().reset_index()
    return pd.merge(result_mean, result_std, on=group_columns, suffixes=("_mean", "_std"))


def aggregate_averaged_results(results_dir: str) -> pd.DataFrame:
    """
    Load per-run CSVs and return aggregated mean/std results over seeds.
    """
    raw_df = _get_raw_data(results_dir)
    return _get_averaged_results(raw_df)


def print_averaged_results(results_dir: str, sort_by: str = "optimizer") -> pd.DataFrame:
    """
    Print and return averaged results over runs/optimizers.
    """
    averaged = aggregate_averaged_results(results_dir)
    if averaged.empty:
        print(f"No results found in {results_dir}")
        return averaged

    if sort_by in averaged.columns:
        averaged = averaged.sort_values(sort_by)

    print(averaged.to_string(index=False))
    return averaged

