"""
Experiment runner using hash-table based configuration.

This script:
1. Loads experiments from config/experiments.yaml
2. Each experiment is self-contained with its own optimizer, model, dataset, training params
3. Runs experiments for each seed
4. Aggregates results (mean/std) across seeds
5. Reports aggregated metrics to W&B

Usage:
    # Run all active experiments
    python run_experiment_suite.py --config config/experiments.yaml --wandb

    # Run specific experiments
    python run_experiment_suite.py --experiments adamw_baseline mu2muon_aux_adam --wandb

    # Override seeds
    python run_experiment_suite.py --experiments mu2muon_aux_adam --seeds "42,43,44,45,46" --wandb
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "0"

import argparse
import copy
import subprocess
import sys
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import tempfile
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any, Optional
from src.utils import load_config
from src.wandb_interface import (
    build_experiment_run_name,
    init_summary_run,
    log_metrics,
    set_run_summary,
    finish_run,
)

def parse_seeds(seeds_value) -> List[int]:
    if isinstance(seeds_value, int):
        return [seeds_value]
    if isinstance(seeds_value, list):
        return [int(s) for s in seeds_value]
    if isinstance(seeds_value, str):
        text = seeds_value.strip()
        if not text:
            return [42]

        seeds: List[int] = []
        chunks = [c for c in re.split(r"[,\s]+", text) if c]

        for chunk in chunks:
            if re.fullmatch(r"-?\d+", chunk):
                seeds.append(int(chunk))
                continue

            range_match = re.fullmatch(r"(-?\d+)-(-?\d+)(?::(\d+))?", chunk)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                step = int(range_match.group(3) or "1")
                if step <= 0:
                    raise ValueError(f"Invalid seed range step '{step}' in '{chunk}'.")
                direction = 1 if end >= start else -1
                actual_step = step * direction
                for seed in range(start, end + direction, actual_step):
                    seeds.append(seed)
                continue

            raise ValueError(
                f"Invalid seeds value '{chunk}'. Use ints like '1,2,3' "
                "or ranges like '1-5' / '10-2:2'."
            )

        # Preserve order while de-duplicating.
        return list(dict.fromkeys(seeds))
    return [42]


def run_single_experiment(
    experiment_name: str,
    exp_cfg: Dict[str, Any],
    seed: int,
    suite_run_id: str,
    device: str,
    seed_wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    results_dir: str = "results",
) -> None:
    opt_cfg       = exp_cfg.get("optimizer", {})
    model_cfg     = exp_cfg.get("model", {})
    dataset_cfg   = exp_cfg.get("dataset", {})
    training_cfg  = exp_cfg.get("training", {})
    scheduler_cfg = exp_cfg.get("scheduler", {})
    opt_name = opt_cfg.get("name", "adamw")

    top_level_keys = {"name", "lr", "weight_decay"}
    nested_opt_params = {k: v for k, v in opt_cfg.items() if k not in top_level_keys}

    tmp_cfg = {
        "seeds": seed,
        "device": device,
        "wandb": {
            "enabled": seed_wandb_enabled,
            "project": wandb_project,
            "entity": wandb_entity,
            "run_name": (
                f"{build_experiment_run_name(experiment_name, exp_cfg)}"
                f"--suite-{suite_run_id}--seed-{seed}"
            ),
            "group": f"{experiment_name}--suite-{suite_run_id}",
            "job_type": "seed-training",
            "tags": ["suite", "seed", experiment_name, opt_name, f"suite-{suite_run_id}"],
        },
        "dataset": dataset_cfg,
        "model": model_cfg,
        "training": training_cfg,
        "scheduler": scheduler_cfg,
        "optimizer": {
            "name": opt_name,
            "lr": opt_cfg.get("lr", 3e-4),
            "weight_decay": opt_cfg.get("weight_decay", 0),
            opt_name: nested_opt_params,
        },
        "results_dir": results_dir,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix=f"exp_{experiment_name}_"
    ) as f:
        yaml.dump(tmp_cfg, f, default_flow_style=False)
        tmp_cfg_path = f.name

    try:
        command = [
            sys.executable, "-m", "src.main",
            "--config", tmp_cfg_path,
            "--seed", str(seed),
            "--device", device,
        ]

        env = os.environ.copy()
        env["MKL_THREADING_LAYER"] = "GNU"
        env["MKL_SERVICE_FORCE_INTEL"] = "0"

        tag = f"[{experiment_name} | seed={seed}]"
        sep = "=" * 70
        print(f"\n{sep}\n{tag}  START\n{sep}\n", flush=True)

        # Stream output in real-time so nothing is lost if the process is killed
        process = subprocess.Popen(
            command, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
        process.wait()

        status = "OK" if process.returncode == 0 else f"FAILED (rc={process.returncode})"
        print(f"\n{sep}\n{tag}  {status}\n{sep}\n", flush=True)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

    finally:
        os.unlink(tmp_cfg_path)


def aggregate_results(results_dir: str = "results") -> pd.DataFrame:
    if not os.path.isdir(results_dir):
        return pd.DataFrame()
    dfs = []
    for root, _, files in os.walk(results_dir):
        for fname in sorted(files):
            if fname == "results.csv":
                dfs.append(pd.read_csv(os.path.join(root, fname)))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def compute_averaged_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    accuracy_cols = ["Train Accuracy", "Test Accuracy", "Test Top-5 Accuracy"]
    available_cols = [c for c in accuracy_cols if c in df.columns]
    if len(available_cols) < 2:
        return {}
    final_metrics = (
        df.groupby("seed")[available_cols].last()
        if "seed" in df.columns
        else df[available_cols].tail(1)
    )
    metrics = {
        "Train Accuracy_mean": final_metrics["Train Accuracy"].mean(),
        "Test Accuracy_mean": final_metrics["Test Accuracy"].mean(),
    }
    if "Test Top-5 Accuracy" in final_metrics:
        metrics["Test Top-5 Accuracy_mean"] = final_metrics["Test Top-5 Accuracy"].mean()
    return metrics


def compute_averaged_curve(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    step_col = None
    if "Epoch" in df.columns:
        step_col = "Epoch"
    elif "Iteration" in df.columns:
        step_col = "Iteration"
    if step_col is None:
        return pd.DataFrame()

    metric_cols = [
        col for col in [
            "Train Loss",
            "Train Accuracy",
            "Test Loss",
            "Test Accuracy",
            "Test Top-5 Accuracy",
            "VR Loss",
        ]
        if col in df.columns
    ]
    if not metric_cols:
        return pd.DataFrame()

    tmp = df.copy()
    tmp[step_col] = pd.to_numeric(tmp[step_col], errors="coerce")
    tmp = tmp.dropna(subset=[step_col])
    grouped = tmp.groupby(step_col)[metric_cols]
    mean_df = grouped.mean().reset_index()
    std_df = grouped.std().reset_index()
    return pd.merge(mean_df, std_df, on=step_col, suffixes=("_mean", "_std"))


def compute_seed_final_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per seed (last epoch/iteration), preserving all available columns.
    """
    if df.empty:
        return pd.DataFrame()

    if "seed" not in df.columns:
        return df.tail(1).copy()

    rank_col = None
    if "Epoch" in df.columns:
        rank_col = "Epoch"
    elif "Iteration" in df.columns:
        rank_col = "Iteration"

    if rank_col is not None:
        tmp = df.copy()
        tmp[rank_col] = pd.to_numeric(tmp[rank_col], errors="coerce")
        tmp = tmp.sort_values(["seed", rank_col], ascending=[True, True])
        return tmp.groupby("seed", as_index=False).tail(1).copy()

    return df.groupby("seed", as_index=False).tail(1).copy()


def log_summary_to_wandb(
    experiment_name: str,
    exp_cfg: Dict[str, Any],
    planned_seeds: List[int],
    successful_seeds: List[int],
    failed_seeds: List[int],
    suite_run_id: str,
    metrics: Dict[str, float],
    averaged_curve: Optional[pd.DataFrame],
    wandb_project: str,
    wandb_entity: Optional[str],
) -> None:
    run_name = f"{build_experiment_run_name(experiment_name, exp_cfg)}--suite-{suite_run_id}--summary"
    opt_cfg      = exp_cfg.get("optimizer", {})
    model_cfg    = exp_cfg.get("model", {})
    dataset_cfg  = exp_cfg.get("dataset", {})
    training_cfg = exp_cfg.get("training", {})
    scheduler_cfg = exp_cfg.get("scheduler", {})

    config = {
        "experiment_name": experiment_name,
        "suite_run_id": suite_run_id,
        "planned_seeds": planned_seeds,
        "successful_seeds": successful_seeds,
        "failed_seeds": failed_seeds,
        "num_seeds_planned": len(planned_seeds),
        "num_seeds_successful": len(successful_seeds),
        "num_seeds_failed": len(failed_seeds),
        **{f"optimizer_{k}": v for k, v in opt_cfg.items()},
        **{f"model_{k}": v for k, v in model_cfg.items()},
        **{f"dataset_{k}": v for k, v in dataset_cfg.items()},
        **{f"training_{k}": v for k, v in training_cfg.items()},
        **{f"scheduler_{k}": v for k, v in scheduler_cfg.items()},
    }

    summary_metrics = {
        **metrics,
        "num_seeds_planned": len(planned_seeds),
        "num_seeds_successful": len(successful_seeds),
        "num_seeds_failed": len(failed_seeds),
    }

    run = init_summary_run(
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        run_name=run_name,
        config=config,
        tags=[
            "summary",
            "multi-seed",
            experiment_name,
            opt_cfg.get("name", "unknown"),
            f"suite-{suite_run_id}",
        ],
        group=f"{experiment_name}--suite-{suite_run_id}",
        job_type="suite-summary",
    )

    step_key = None
    if averaged_curve is not None and not averaged_curve.empty:
        step_key = "Epoch" if "Epoch" in averaged_curve.columns else "Iteration" if "Iteration" in averaged_curve.columns else None
        if step_key is not None:
            for _, row in averaged_curve.iterrows():
                payload = row.dropna().to_dict()
                step = int(payload.pop(step_key))
                log_metrics(run, payload, step=step)

    log_metrics(run, summary_metrics)
    set_run_summary(run, summary_metrics)
    finish_run(run)


def run_single_experiment_config(
    exp_name: str,
    exp_cfg: Dict[str, Any],
    seeds: List[int],
    suite_run_id: str,
    device: str,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    keep_seed_results: bool,
    log_seed_wandb: bool,
) -> None:
    results_dir = os.path.join("results", f"exp_{exp_name}")
    seeds_dir = os.path.join(results_dir, "seeds", f"suite_{suite_run_id}")
    os.makedirs(seeds_dir, exist_ok=True)

    sep = "#" * 70
    print(f"\n{sep}")
    print(f"# {exp_name}")
    print(f"# opt={exp_cfg.get('optimizer',{}).get('name')}  "
          f"model={exp_cfg.get('model',{}).get('name')}  "
          f"dataset={exp_cfg.get('dataset',{}).get('name')}")
    print(f"{sep}\n")

    successful_seeds: List[int] = []
    failed_seeds: List[int] = []
    for seed in seeds:
        try:
            run_single_experiment(
                experiment_name=exp_name,
                exp_cfg=exp_cfg,
                seed=seed,
                suite_run_id=suite_run_id,
                device=device,
                seed_wandb_enabled=(use_wandb and log_seed_wandb),
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                results_dir=seeds_dir,
            )
            successful_seeds.append(seed)
        except subprocess.CalledProcessError:
            failed_seeds.append(seed)
            continue

    df = aggregate_results(seeds_dir)
    if df.empty:
        print(f"[{exp_name}] No results found, skipping.\n")
        if os.path.isdir(seeds_dir) and not keep_seed_results:
            shutil.rmtree(seeds_dir, ignore_errors=True)
        return

    metrics = compute_averaged_metrics(df)
    averaged_curve = compute_averaged_curve(df)
    avg_csv_path = os.path.join(results_dir, "results_averaged.csv")
    pd.DataFrame([metrics]).to_csv(avg_csv_path, index=False)
    if not averaged_curve.empty:
        avg_curve_csv_path = os.path.join(results_dir, "results_averaged_curve.csv")
        averaged_curve.to_csv(avg_curve_csv_path, index=False)
    print(f"[{exp_name}] Averaged results ג†’ {avg_csv_path}")

    print(f"\n[{exp_name}] Aggregated metrics over {len(successful_seeds)} successful seed(s):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    if failed_seeds:
        print(f"[{exp_name}] Failed seeds: {failed_seeds}")
    print(f"[{exp_name}] Seed run directory: {seeds_dir}")

    if use_wandb and metrics:
        print(f"[{exp_name}] Logging to W&B...")
        log_summary_to_wandb(
            experiment_name=exp_name,
            exp_cfg=exp_cfg,
            planned_seeds=seeds,
            successful_seeds=successful_seeds,
            failed_seeds=failed_seeds,
            suite_run_id=suite_run_id,
            metrics=metrics,
            averaged_curve=averaged_curve,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        )
        print(f"[{exp_name}] Done.\n")

    if os.path.isdir(seeds_dir) and not keep_seed_results:
        shutil.rmtree(seeds_dir, ignore_errors=True)
        print(f"[{exp_name}] Removed seed-level artifacts (kept averaged CSV only).")


def run_experiment_suite(
    cfg: Dict[str, Any],
    experiment_names: List[str],
    seeds: List[int],
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    epochs_override: Optional[int] = None,
    keep_seed_results: bool = False,
    log_seed_wandb: bool = False,
    parallel_experiments: int = 1,
) -> None:
    """Run experiments with optional experiment-level parallelism."""
    experiments = cfg.get("experiments", {})
    device = cfg.get("device", "cuda")
    # Epoch priority: CLI flag > top-level config key > per-experiment value > 50
    global_epochs = epochs_override if epochs_override is not None else cfg.get("epochs", None)

    valid = [n for n in experiment_names if n in experiments]
    skipped = [n for n in experiment_names if n not in experiments]
    for n in skipped:
        print(f"WARNING: experiment '{n}' not found in config, skipping.")

    workers = max(1, int(parallel_experiments))
    mode = "parallel" if workers > 1 else "sequential"
    print(f"\nRunning {len(valid)} experiments ({mode}) on {device}")
    print(f"Seeds per experiment: {seeds}\n")

    suite_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Suite run id: {suite_run_id}\n")
    print(f"Seed W&B logging: {log_seed_wandb}")
    print(f"Keep seed-level results on disk: {keep_seed_results}\n")

    if workers > 1:
        print(f"Experiment-level workers: {workers}\n")
        prepared: List[tuple[str, Dict[str, Any]]] = []
        for exp_name in valid:
            exp_cfg = copy.deepcopy(experiments[exp_name])
            exp_cfg.setdefault("training", {})
            if global_epochs is not None:
                exp_cfg["training"]["epochs"] = global_epochs
            elif "epochs" not in exp_cfg["training"]:
                exp_cfg["training"]["epochs"] = 50
            prepared.append((exp_name, exp_cfg))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    run_single_experiment_config,
                    exp_name=exp_name,
                    exp_cfg=exp_cfg,
                    seeds=seeds,
                    suite_run_id=suite_run_id,
                    device=device,
                    use_wandb=use_wandb,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                    keep_seed_results=keep_seed_results,
                    log_seed_wandb=log_seed_wandb,
                )
                for exp_name, exp_cfg in prepared
            ]
            for future in as_completed(futures):
                future.result()

        print("\n" + "=" * 70)
        print("All experiments completed!")
        print("=" * 70)
        return

    for exp_name in valid:
        exp_cfg     = experiments[exp_name]

        # Apply epoch count: CLI/top-level always wins over per-experiment hardcoded value
        exp_cfg = copy.deepcopy(exp_cfg)
        exp_cfg.setdefault("training", {})
        if global_epochs is not None:
            exp_cfg["training"]["epochs"] = global_epochs
        elif "epochs" not in exp_cfg["training"]:
            exp_cfg["training"]["epochs"] = 50  # absolute fallback

        results_dir = os.path.join("results", f"exp_{exp_name}")
        seeds_dir = os.path.join(results_dir, "seeds", f"suite_{suite_run_id}")
        os.makedirs(seeds_dir, exist_ok=True)

        sep = "#" * 70
        print(f"\n{sep}")
        print(f"# {exp_name}")
        print(f"# opt={exp_cfg.get('optimizer',{}).get('name')}  "
              f"model={exp_cfg.get('model',{}).get('name')}  "
              f"dataset={exp_cfg.get('dataset',{}).get('name')}")
        print(f"{sep}\n")

        successful_seeds: List[int] = []
        failed_seeds: List[int] = []
        for seed in seeds:
            try:
                run_single_experiment(
                    experiment_name=exp_name,
                    exp_cfg=exp_cfg,
                    seed=seed,
                    suite_run_id=suite_run_id,
                    device=device,
                    seed_wandb_enabled=(use_wandb and log_seed_wandb),
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                    results_dir=seeds_dir,
                )
                successful_seeds.append(seed)
            except subprocess.CalledProcessError:
                failed_seeds.append(seed)
                continue

        df = aggregate_results(seeds_dir)
        if df.empty:
            print(f"[{exp_name}] No results found, skipping.\n")
            if os.path.isdir(seeds_dir) and not keep_seed_results:
                shutil.rmtree(seeds_dir, ignore_errors=True)
            continue

        metrics = compute_averaged_metrics(df)
        averaged_curve = compute_averaged_curve(df)

        # Write single averaged CSV — this is the only output that matters
        avg_csv_path = os.path.join(results_dir, "results_averaged.csv")
        pd.DataFrame([metrics]).to_csv(avg_csv_path, index=False)
        if not averaged_curve.empty:
            avg_curve_csv_path = os.path.join(results_dir, "results_averaged_curve.csv")
            averaged_curve.to_csv(avg_curve_csv_path, index=False)
        print(f"[{exp_name}] Averaged results → {avg_csv_path}")

        print(f"\n[{exp_name}] Aggregated metrics over {len(successful_seeds)} successful seed(s):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        if failed_seeds:
            print(f"[{exp_name}] Failed seeds: {failed_seeds}")
        print(f"[{exp_name}] Seed run directory: {seeds_dir}")

        if use_wandb and metrics:
            print(f"[{exp_name}] Logging to W&B...")
            log_summary_to_wandb(
                experiment_name=exp_name,
                exp_cfg=exp_cfg,
                planned_seeds=seeds,
                successful_seeds=successful_seeds,
                failed_seeds=failed_seeds,
                suite_run_id=suite_run_id,
                metrics=metrics,
                averaged_curve=averaged_curve,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
            )
            print(f"[{exp_name}] Done.\n")

        if os.path.isdir(seeds_dir) and not keep_seed_results:
            shutil.rmtree(seeds_dir, ignore_errors=True)
            print(f"[{exp_name}] Removed seed-level artifacts (kept averaged CSV only).")

    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiments", type=str, nargs="+", default=None)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Seeds as CSV/space list or ranges, e.g. '1,2,3', '1 2 3', '1-5', '10-2:2'",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--log-seed-wandb",
        action="store_true",
        help="Also log every individual seed run to W&B (default: summary run only).",
    )
    parser.add_argument(
        "--keep-seed-results",
        action="store_true",
        help="Keep seed-level local artifacts under results/exp_*/seeds (default: remove after averaging).",
    )
    parser.add_argument(
        "--parallel-experiments",
        type=int,
        default=1,
        help="Number of experiments to run concurrently.",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count for all experiments (default: use config value)")
    parser.add_argument("--list", action="store_true", help="List experiments and exit")

    args = parser.parse_args()
    cfg  = load_config(args)

    if args.list:
        active = cfg.get("active_experiments", [])
        for name, ecfg in cfg.get("experiments", {}).items():
            flag = "[ACTIVE]" if name in active else ""
            print(f"  {name} ({ecfg.get('optimizer',{}).get('name','?')}) {flag}")
        return

    experiment_names = args.experiments or cfg.get("active_experiments", [])
    if not experiment_names:
        print("ERROR: no experiments specified.")
        return

    seeds = parse_seeds(args.seeds) if args.seeds else parse_seeds(cfg.get("seeds", 42))

    wandb_cfg     = cfg.get("wandb", {})
    use_wandb     = args.wandb or wandb_cfg.get("enabled", False)
    wandb_project = args.wandb_project or wandb_cfg.get("project", "mlopt-benchmark")
    wandb_entity  = args.wandb_entity  or wandb_cfg.get("entity")

    print(f"\n{'#'*70}")
    print(f"# Experiment Suite Runner")
    print(f"# config={args.config}  experiments={experiment_names}")
    print(f"# seeds={seeds}  wandb={use_wandb}")
    print(
        f"# log_seed_wandb={args.log_seed_wandb}  keep_seed_results={args.keep_seed_results}  "
        f"parallel_experiments={args.parallel_experiments}"
    )
    print(f"{'#'*70}\n")

    run_experiment_suite(
        cfg,
        experiment_names,
        seeds,
        use_wandb,
        wandb_project,
        wandb_entity,
        epochs_override=args.epochs,
        keep_seed_results=args.keep_seed_results,
        log_seed_wandb=args.log_seed_wandb,
        parallel_experiments=args.parallel_experiments,
    )


if __name__ == "__main__":
    main()

