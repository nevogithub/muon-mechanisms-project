#!/usr/bin/env python3
"""Collect only verified epoch-50 mechanism artifacts, never checkpoints."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import tarfile


def read_run(path, seed):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or int(float(rows[-1]["Epoch"])) != 50:
        raise ValueError(f"Not complete: {path}")
    epochs = [int(float(r["Epoch"])) for r in rows]
    if epochs != list(range(5, 51, 5)):
        raise ValueError(f"Unexpected evaluation epochs: {path}: {epochs}")
    for r in rows:
        if int(float(r["seed"])) != seed:
            raise ValueError(f"Seed mismatch: {path}")
        for col in ("Train Accuracy", "Test Loss", "Test Accuracy") + tuple(c for c in ("Train Loss", "Test Top-5 Accuracy") if c in r):
            if not math.isfinite(float(r[col])):
                raise ValueError(f"Nonfinite {col}: {path}")
    return float(rows[-1]["Test Accuracy"]) * 100


def latest_complete(root, name, seed):
    valid = []
    for p in (root / "results" / f"exp_{name}" / "seeds").glob("suite_*/run*/results.csv"):
        try:
            acc = read_run(p, seed)
        except (ValueError, KeyError, FileNotFoundError):
            continue
        valid.append((p, acc))
    if not valid:
        raise ValueError(f"No eligible run: {name}, seed {seed}")
    p, acc = sorted(valid)[-1]
    return p, acc, len(valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root, logs = args.root, args.logs
    cfg = json.loads((root / "config/_generated_mechanism_followup_grid.json").read_text())
    diag = json.loads((root / "config/_generated_mu2_diagnostic_ablation_workers4.json").read_text())
    manifest, extra_logs = [], []
    mapping = {
        "lr_momentum_factorial": (89059, "mech-mom-factorial-r2"),
        "mars_gamma_sweep": (89060, "mech-mars-gamma-r2"),
        "mars_momentum_factorial": (89061, "mech-mars-mom-factorial"),
        "stressed_lr_mechanism_confirmation": (89064, "mech-stress-confirm"),
    }
    for group, (job, label) in mapping.items():
        for i, entry in enumerate(cfg["experiment_groups"][group]):
            actual_job, actual_label = job, label
            if job == 89064 and i in (4, 5):
                actual_job = {4: 90137, 5: 90052}[i]
                actual_label = "mech-stress-recover"
                extra_logs += [logs / f"{label}-{job}_{i}.out", logs / f"{label}-{job}_{i}.err"]
            log = logs / f"{actual_label}-{actual_job}_{i}.out"
            logtext = log.read_text(errors="replace")
            if "FAILED (rc=" in logtext or "Seed run directory:" not in logtext:
                raise ValueError(f"Unsuccessful job: {log}")
            suite = re.search(r"Suite run id: (\d{8}_\d{6})", logtext).group(1)
            name, seed = entry["experiment"], entry["seed"]
            path = root / "results" / f"exp_{name}" / "seeds" / f"suite_{suite}" / "run1/results.csv"
            acc = read_run(path, seed)
            manifest.append(dict(block=group, experiment=name, seed=seed, job=f"{actual_job}_{i}", path=path.relative_to(root).as_posix(), accuracy_pct=acc, eligible_candidates=1))
            extra_logs += [log, log.with_suffix(".err")]
    for name in diag["active_experiments"]:
        p, acc, count = latest_complete(root, name, 42)
        manifest.append(dict(block="mu2_diagnostic", experiment=name, seed=42, job="archived_seed42", path=p.relative_to(root).as_posix(), accuracy_pct=acc, eligible_candidates=count))
    baseline_names = ["tv5m_in100_clr0025_mom_const_b098", "tv5m_in100_clr0025_mom_cos_b098_b050_e36_e48"]
    for name in baseline_names:
        for seed in (42, 43, 44):
            p, acc, count = latest_complete(root, name, seed)
            manifest.append(dict(block="plain_constant_lr", experiment=name, seed=seed, job="archived", path=p.relative_to(root).as_posix(), accuracy_pct=acc, eligible_candidates=count))
    for family in ("muon", "graddiff"):
        for lr in ("0p005", "0p01", "0p02", "0p04", "0p08"):
            name = f"tinyvit5m_imagenet100_lrrobust_{family}_lr{lr}"
            for seed in ((42, 43, 44) if lr in ("0p005", "0p01") else (42,)):
                p, acc, count = latest_complete(root, name, seed)
                manifest.append(dict(block="archived_lr_robustness", experiment=name, seed=seed, job="archived", path=p.relative_to(root).as_posix(), accuracy_pct=acc, eligible_candidates=count))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as tar:
        for entry in manifest:
            parent = (root / entry["path"]).parent
            for name in ("results.csv", "config.json", "metrics_log.txt", "optimizer_diagnostics.csv"):
                p = parent / name
                if p.exists():
                    tar.add(p, arcname=p.relative_to(root).as_posix())
        for p in extra_logs:
            if p.exists():
                tar.add(p, arcname=f"slurm_logs/{p.name}")
        for p in (root / "config/_generated_mechanism_followup_grid.json", root / "config/_generated_mu2_diagnostic_ablation_workers4.json"):
            tar.add(p, arcname=p.relative_to(root).as_posix())
        data = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    print(f"Verified and bundled {len(manifest)} seed-level records to {args.output}")


if __name__ == "__main__":
    main()
