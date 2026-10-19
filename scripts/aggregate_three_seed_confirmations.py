#!/usr/bin/env python3
"""Aggregate the targeted post-selection seed-42--44 confirmation experiments.

The downloaded artifact tree may contain canceled or failed retries.  A run is
eligible only when its final CSV row is epoch 50 and its embedded seed is one of
the requested confirmation seeds.  If more than one completed artifact exists
for a seed, the lexicographically latest suite path is retained and recorded.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np
import pandas as pd


TINY_EXPERIMENTS = [
    ("lr", "tinyvit5m_imagenet100_lrrobust_muon_lr0p005", "Muon+AdamW", 0.005),
    ("lr", "tinyvit5m_imagenet100_lrrobust_graddiff_lr0p005", "MARS-Muon+AdamW", 0.005),
    ("lr", "tinyvit5m_imagenet100_lrrobust_mars_mu2_lr0p005", "MARS+Mu2-Muon+AdamW", 0.005),
    ("lr", "tinyvit5m_imagenet100_lrrobust_muon_lr0p01", "Muon+AdamW", 0.01),
    ("lr", "tinyvit5m_imagenet100_lrrobust_graddiff_lr0p01", "MARS-Muon+AdamW", 0.01),
    ("lr", "tinyvit5m_imagenet100_lrrobust_mars_mu2_lr0p01", "MARS+Mu2-Muon+AdamW", 0.01),
    ("momentum", "tv5m_in100_clr0025_mom_const_b095", "Constant beta=0.95", 0.0025),
    ("momentum", "tv5m_in100_clr0025_mom_const_b098", "Constant beta=0.98", 0.0025),
    (
        "momentum",
        "tv5m_in100_clr0025_mom_cos_b098_b050_e36_e48",
        "Late decay beta=0.98->0.50",
        0.0025,
    ),
]

# Preserve the archived seed-42 snapshot for every condition except the one
# exact rerun commissioned to verify the anomalous Muon LR=0.01 collapse.
TINY_SEED42_ARTIFACT_OVERRIDES = {
    "tinyvit5m_imagenet100_lrrobust_muon_lr0p01",
}

RESNET_FAMILIES = {
    "muon": ("Muon+AdamW", "muon"),
    "graddiff": ("MARS-Muon+AdamW", "graddiff"),
    "mu2": ("Mu2-Muon+AdamW", "mu2"),
    "mars_mu2": ("MARS+Mu2-Muon+AdamW", "mars_mu2"),
    "final_cosine": ("Joint schedule", "final"),
}
RESNET_LRS = [0.005, 0.01, 0.02, 0.04, 0.08]
RESNET_EXTRA = [
    "resnet18_component_muon_lr0p08",
    "resnet18_component_mars_g0p75_lr0p08",
    "resnet18_component_mu2_a0p35_lr0p08",
    "resnet18_algmom_lowconst_plain_b0p5_lr0p04",
    "resnet18_algmom_highconst_plain_b0p98_lr0p04",
    "resnet18_algmom_cos_plain_b0p98_to_b0p5_lr0p04",
    "resnet18_algmom_lowconst_plain_b0p5_lr0p08",
    "resnet18_algmom_highconst_plain_b0p98_lr0p08",
    "resnet18_algmom_cos_plain_b0p98_to_b0p5_lr0p08",
    "resnet18_algmom_highconst_mars_mu2_b0p98_lr0p08",
    "resnet18_constantlr_muon_lr0p04",
    "resnet18_constantlr_final_cosine_lr0p01",
]


def lr_token(lr: float) -> str:
    return str(lr).replace(".", "p")


def final_csv_rows(experiment_dir: Path, seeds: set[int]) -> dict[int, dict]:
    candidates: dict[int, list[tuple[str, dict]]] = {seed: [] for seed in seeds}
    if not experiment_dir.exists():
        return {}
    for path in experiment_dir.glob("**/results.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or "Epoch" not in frame or "seed" not in frame:
            continue
        row = frame.iloc[-1]
        try:
            epoch = int(float(row["Epoch"]))
            seed = int(float(row["seed"]))
        except (TypeError, ValueError):
            continue
        if epoch < 50 or seed not in seeds:
            continue
        record = row.to_dict()
        record["artifact_path"] = path.as_posix()
        candidates[seed].append((path.as_posix(), record))
    return {
        seed: sorted(rows, key=lambda item: item[0])[-1][1]
        for seed, rows in candidates.items()
        if rows
    }


def summary_rows(seed_frame: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for experiment, group in seed_frame.groupby("experiment", sort=False):
        values = group["accuracy"].astype(float)
        first = group.iloc[0]
        groups.append(
            {
                "benchmark": first["benchmark"],
                "group": first["group"],
                "experiment": experiment,
                "label": first["label"],
                "matrix_lr": first["matrix_lr"],
                "n": len(values),
                "accuracy_mean": values.mean(),
                "accuracy_std": values.std(ddof=1) if len(values) > 1 else math.nan,
                "accuracy_min": values.min(),
                "accuracy_max": values.max(),
            }
        )
    return pd.DataFrame(groups)


def load_tiny_seed42(snapshot: Path) -> dict[str, float]:
    frame = pd.read_csv(snapshot)
    frame = frame[(frame["seed"] == 42) & (frame["status"] == "complete")]
    return {
        row["experiment"]: float(row["test_accuracy"])
        for _, row in frame.iterrows()
    }


def load_resnet_seed42(report_csv: Path, constant_csv: Path) -> dict[str, float]:
    report = pd.read_csv(report_csv)
    values: dict[str, float] = {}
    for family, (_, old_optimizer) in RESNET_FAMILIES.items():
        for lr in RESNET_LRS:
            match = report[
                (report["suite"] == "lr_stability")
                & (report["model"] == "resnet18")
                & (report["optimizer"] == old_optimizer)
                & np.isclose(report["lr"].astype(float), lr)
            ]
            if len(match) != 1:
                raise ValueError(f"Seed-42 primary lookup failed for {family=} {lr=}")
            name = f"resnet18_cifar100_{family}_lr{lr_token(lr)}"
            values[name] = float(match.iloc[0]["test_acc"])

    for name in RESNET_EXTRA:
        exact = report[report["experiment"] == f"exp_{name}"]
        if len(exact) == 1:
            values[name] = float(exact.iloc[0]["test_acc"])

    constant = pd.read_csv(constant_csv)
    for name in RESNET_EXTRA:
        exact = constant[constant["run"] == f"exp_{name}"]
        if len(exact) == 1:
            values[name] = float(exact.iloc[0]["test_acc"])

    missing = sorted(set(RESNET_EXTRA) - set(values))
    if missing:
        raise ValueError(f"Missing seed-42 ResNet values: {missing}")
    return values


def write_tiny_tex(summary: pd.DataFrame, output: Path) -> None:
    order = [name for _, name, _, _ in TINY_EXPERIMENTS]
    indexed = summary.set_index("experiment").loc[order]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Targeted post-selection TinyViT-5M/ImageNet-100 confirmations. Values are mean $\pm$ sample standard deviation of final-epoch validation top-1 accuracy across seeds 42--44.}",
        r"\label{tab:tinyvit-three-seed}",
        r"\begin{tabular}{L{0.55\linewidth}rr}",
        r"\toprule",
        r"Condition & LR & Top-1 (\%) \\",
        r"\midrule",
    ]
    for _, row in indexed.iterrows():
        label = str(row["label"]).replace("beta", r"$\beta$").replace("->", r"$\rightarrow$")
        lines.append(
            f"{label} & {row['matrix_lr']:.4g} & "
            f"{100*row['accuracy_mean']:.2f} $\\pm$ {100*row['accuracy_std']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def write_resnet_tex(summary: pd.DataFrame, output: Path) -> None:
    primary = summary[(summary["group"] == "lr_primary") & (summary["matrix_lr"] <= 0.02)]
    rows = []
    for family, (label, _) in RESNET_FAMILIES.items():
        fam = primary[primary["experiment"].str.contains(f"cifar100_{family}_lr")]
        best = fam.loc[fam["accuracy_mean"].idxmax()]
        rows.append(
            (
                label,
                best["matrix_lr"],
                best["accuracy_mean"],
                best["accuracy_std"],
                fam["accuracy_mean"].mean(),
                fam["accuracy_mean"].min(),
            )
        )
    rows.sort(key=lambda item: item[2], reverse=True)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{ResNet18/CIFAR-100 LR confirmation across seeds 42--44. Following the advisor's reporting scope, the main table stops at LR $0.02$; the complete five-LR grid remains in the artifact CSV. Best-cell uncertainty is the sample standard deviation across seeds; mean and worst aggregate the three displayed seed-averaged LR cells.}",
        r"\label{tab:prelim-lr}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Optimizer & Best LR & Best $\pm$ SD & Mean & Worst \\",
        r"\midrule",
    ]
    for label, lr, mean, std, across, worst in rows:
        lines.append(
            f"{label} & {lr:.3g} & {100*mean:.2f} $\\pm$ {100*std:.2f} & "
            f"{100*across:.2f} & {100*worst:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def plot_tiny(seed_frame: pd.DataFrame, output_base: Path) -> None:
    """Show the paired seed structure instead of hiding it behind mean/error bars."""
    panels = [
        (
            "LR 0.005",
            [
                "tinyvit5m_imagenet100_lrrobust_muon_lr0p005",
                "tinyvit5m_imagenet100_lrrobust_graddiff_lr0p005",
                "tinyvit5m_imagenet100_lrrobust_mars_mu2_lr0p005",
            ],
            ["Muon", "MARS", "MARS+Mu2"],
        ),
        (
            "LR 0.01",
            [
                "tinyvit5m_imagenet100_lrrobust_muon_lr0p01",
                "tinyvit5m_imagenet100_lrrobust_graddiff_lr0p01",
                "tinyvit5m_imagenet100_lrrobust_mars_mu2_lr0p01",
            ],
            ["Muon", "MARS", "MARS+Mu2"],
        ),
        (
            "Constant LR 0.0025",
            [
                "tv5m_in100_clr0025_mom_const_b095",
                "tv5m_in100_clr0025_mom_const_b098",
                "tv5m_in100_clr0025_mom_cos_b098_b050_e36_e48",
            ],
            [r"Const. $\beta=.95$", r"Const. $\beta=.98$", "Late decay"],
        ),
    ]
    seed_colors = {42: "#3b6fb6", 43: "#d17a22", 44: "#4b9b58"}
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    for ax, (title, experiments, labels) in zip(axes, panels):
        pivot = (
            seed_frame[seed_frame["experiment"].isin(experiments)]
            .pivot(index="seed", columns="experiment", values="accuracy")
            .loc[[42, 43, 44], experiments]
        )
        x = np.arange(len(experiments))
        for seed, row in pivot.iterrows():
            values = 100 * row.to_numpy(dtype=float)
            ax.plot(
                x,
                values,
                marker="o",
                linewidth=1.7,
                markersize=5,
                color=seed_colors[int(seed)],
                label=f"Seed {int(seed)}",
            )
        flat = 100 * pivot.to_numpy(dtype=float).ravel()
        pad = max(0.8, 0.07 * (flat.max() - flat.min()))
        ax.set_ylim(max(0, flat.min() - pad), min(100, flat.max() + pad))
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Final validation top-1 accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("TinyViT-5M / ImageNet-100: paired confirmations", y=1.01)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_constant_lr_learning_curves(
    repo: Path, seed_frame: pd.DataFrame, output_base: Path
) -> None:
    """Restore the advisor-approved two-panel constant-LR trajectory presentation."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))

    # ResNet18: retain the original seed-42, best-final-cell-per-family presentation.
    resnet_root = repo / "analysis/constantlr_pre_shutdown_snapshot/results"
    family_labels = {
        "muon": "Muon",
        "graddiff": "MARS",
        "mu2": "Mu2",
        "mars_mu2": "MARS+Mu2",
        "final_cosine": "Joint schedule",
    }
    family_colors = {
        "muon": "#3b6fb6",
        "graddiff": "#d17a22",
        "mu2": "#4b9b58",
        "mars_mu2": "#8f5fbf",
        "final_cosine": "#c23b55",
    }
    for family, label in family_labels.items():
        candidates = []
        for path in resnet_root.glob(
            f"exp_resnet18_constantlr_{family}_lr*/results_averaged_curve.csv"
        ):
            curve = pd.read_csv(path)
            if curve.empty:
                continue
            final = float(curve.iloc[-1]["Test Accuracy_mean"])
            candidates.append((final, path, curve))
        if not candidates:
            raise ValueError(f"No ResNet constant-LR curves found for {family}")
        _, path, curve = max(candidates, key=lambda item: item[0])
        lr = path.parent.name.rsplit("lr", 1)[-1].replace("p", ".")
        axes[0].plot(
            curve["Epoch"],
            100 * curve["Test Accuracy_mean"],
            marker="o",
            label=f"{label}, LR {lr}",
            color=family_colors[family],
        )

    axes[0].set_title("ResNet18 / CIFAR-100 (seed 42)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Evaluation accuracy (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7.5)

    # TinyViT: selected schedule and its two constant-momentum controls over all three seeds.
    tiny_curves = pd.read_csv(
        repo
        / "analysis/tinyvit_imagenet100_summary/snapshot/tinyvit_imagenet100_curves.csv"
    )
    momentum_specs = [
        ("tv5m_in100_clr0025_mom_const_b095", r"Constant $\beta=.95$", "#3b6fb6"),
        ("tv5m_in100_clr0025_mom_const_b098", r"Constant $\beta=.98$", "#d17a22"),
        (
            "tv5m_in100_clr0025_mom_cos_b098_b050_e36_e48",
            r"Late decay $.98\to.50$",
            "#4b9b58",
        ),
    ]
    for experiment, label, color in momentum_specs:
        curves = []
        seed42 = tiny_curves[
            (tiny_curves["experiment"] == experiment) & (tiny_curves["seed"] == 42)
        ][["epoch", "test_accuracy"]].rename(
            columns={"epoch": "Epoch", "test_accuracy": "accuracy"}
        )
        seed42["seed"] = 42
        curves.append(seed42)
        rows = seed_frame[seed_frame["experiment"] == experiment]
        for row in rows.itertuples(index=False):
            if int(row.seed) == 42:
                continue
            path = Path(str(row.artifact_path))
            if not path.exists():
                raise ValueError(f"Missing TinyViT curve artifact: {path}")
            curve = pd.read_csv(path)[["Epoch", "Test Accuracy"]].rename(
                columns={"Test Accuracy": "accuracy"}
            )
            curve["seed"] = int(row.seed)
            curves.append(curve)
        combined = pd.concat(curves, ignore_index=True)
        grouped = combined.groupby("Epoch")["accuracy"]
        mean = 100 * grouped.mean()
        low = 100 * grouped.min()
        high = 100 * grouped.max()
        axes[1].plot(mean.index, mean.values, marker="o", color=color, label=label)
        axes[1].fill_between(mean.index, low.values, high.values, color=color, alpha=0.14)

    axes[1].axvspan(36, 48, color="#777777", alpha=0.08, label="Decay window")
    axes[1].set_title("TinyViT-5M / ImageNet-100 (seeds 42-44)")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_resnet(summary: pd.DataFrame, output_base: Path) -> None:
    displayed_lrs = [lr for lr in RESNET_LRS if lr <= 0.02]
    primary = summary[
        (summary["group"] == "lr_primary") & (summary["matrix_lr"].isin(displayed_lrs))
    ].copy()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for family, (label, _) in RESNET_FAMILIES.items():
        frame = primary[primary["experiment"].str.contains(f"cifar100_{family}_lr")].sort_values("matrix_lr")
        ax.errorbar(
            frame["matrix_lr"],
            100 * frame["accuracy_mean"],
            yerr=100 * frame["accuracy_std"],
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=label,
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(displayed_lrs))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in displayed_lrs]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.axhline(
        74.5,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="74.5% descriptive reference",
        zorder=0,
    )
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title("ResNet18 / CIFAR-100 LR grid (seeds 42-44)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    artifacts = Path(args.artifact_root).resolve()
    results_root = artifacts / "results"
    output_dir = repo / "analysis" / "three_seed_confirmation"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = repo / "report" / "figures"
    tables = repo / "report" / "tables"

    tiny42 = load_tiny_seed42(
        repo / "analysis/tinyvit_imagenet100_summary/snapshot/tinyvit_imagenet100_final.csv"
    )
    tiny_records = []
    for group, name, label, lr in TINY_EXPERIMENTS:
        if name not in tiny42:
            raise ValueError(f"Missing TinyViT seed-42 snapshot for {name}")
        seed42_accuracy = tiny42[name]
        seed42_artifact = (
            "analysis/tinyvit_imagenet100_summary/snapshot/"
            "tinyvit_imagenet100_final.csv"
        )
        if name in TINY_SEED42_ARTIFACT_OVERRIDES:
            verified = final_csv_rows(results_root / f"exp_{name}", {42})
            if set(verified) != {42}:
                raise ValueError(f"Missing verified TinyViT seed-42 artifact for {name}")
            seed42_accuracy = float(verified[42]["Test Accuracy"])
            seed42_artifact = verified[42]["artifact_path"]
        tiny_records.append(
            {
                "benchmark": "tinyvit5m_imagenet100",
                "group": group,
                "experiment": name,
                "label": label,
                "matrix_lr": lr,
                "seed": 42,
                "accuracy": seed42_accuracy,
                "artifact_path": seed42_artifact,
            }
        )
        found = final_csv_rows(results_root / f"exp_{name}", {43, 44})
        if set(found) != {43, 44}:
            raise ValueError(f"Incomplete TinyViT confirmation for {name}: {sorted(found)}")
        for seed, row in sorted(found.items()):
            tiny_records.append(
                {
                    "benchmark": "tinyvit5m_imagenet100",
                    "group": group,
                    "experiment": name,
                    "label": label,
                    "matrix_lr": lr,
                    "seed": seed,
                    "accuracy": float(row["Test Accuracy"]),
                    "artifact_path": row["artifact_path"],
                }
            )

    resnet42 = load_resnet_seed42(
        repo / "analysis/cifar100_report/cifar100_all_results.csv",
        repo / "analysis/constantlr_control/constantlr_all_results.csv",
    )
    resnet_meta = []
    for family, (label, _) in RESNET_FAMILIES.items():
        for lr in RESNET_LRS:
            name = f"resnet18_cifar100_{family}_lr{lr_token(lr)}"
            resnet_meta.append(("lr_primary", name, label, lr))
    for name in RESNET_EXTRA:
        resnet_meta.append(("matched_extra", name, name.replace("_", " "), math.nan))

    resnet_records = []
    for group, name, label, lr in resnet_meta:
        resnet_records.append(
            {
                "benchmark": "resnet18_cifar100",
                "group": group,
                "experiment": name,
                "label": label,
                "matrix_lr": lr,
                "seed": 42,
                "accuracy": resnet42[name],
                "artifact_path": "seed-42 local analysis snapshot",
            }
        )
        found = final_csv_rows(results_root / f"exp_{name}", {43, 44})
        if set(found) != {43, 44}:
            raise ValueError(f"Incomplete ResNet confirmation for {name}: {sorted(found)}")
        for seed, row in sorted(found.items()):
            resnet_records.append(
                {
                    "benchmark": "resnet18_cifar100",
                    "group": group,
                    "experiment": name,
                    "label": label,
                    "matrix_lr": lr,
                    "seed": seed,
                    "accuracy": float(row["Test Accuracy"]),
                    "artifact_path": row["artifact_path"],
                }
            )

    tiny_seed = pd.DataFrame(tiny_records)
    resnet_seed = pd.DataFrame(resnet_records)
    tiny_summary = summary_rows(tiny_seed)
    resnet_summary = summary_rows(resnet_seed)
    tiny_seed.to_csv(output_dir / "tinyvit_seed_results.csv", index=False)
    tiny_summary.to_csv(output_dir / "tinyvit_summary.csv", index=False)
    resnet_seed.to_csv(output_dir / "resnet_seed_results.csv", index=False)
    resnet_summary.to_csv(output_dir / "resnet_summary.csv", index=False)

    paired_specs = [
        (
            "TinyViT LR 0.01: MARS - Muon",
            "tinyvit5m_imagenet100_lrrobust_graddiff_lr0p01",
            "tinyvit5m_imagenet100_lrrobust_muon_lr0p01",
        ),
        (
            "TinyViT LR 0.01: MARS+Mu2 - Muon",
            "tinyvit5m_imagenet100_lrrobust_mars_mu2_lr0p01",
            "tinyvit5m_imagenet100_lrrobust_muon_lr0p01",
        ),
        (
            "TinyViT LR 0.0025: late decay - beta0.98",
            "tv5m_in100_clr0025_mom_cos_b098_b050_e36_e48",
            "tv5m_in100_clr0025_mom_const_b098",
        ),
        (
            "TinyViT LR 0.0025: beta0.95 - beta0.98",
            "tv5m_in100_clr0025_mom_const_b095",
            "tv5m_in100_clr0025_mom_const_b098",
        ),
    ]
    paired_rows = []
    pivot = tiny_seed.pivot(index="seed", columns="experiment", values="accuracy")
    for label, treatment, control in paired_specs:
        deltas = 100 * (pivot[treatment] - pivot[control])
        paired_rows.append(
            {
                "contrast": label,
                "n": len(deltas),
                "delta_pp_mean": deltas.mean(),
                "delta_pp_std": deltas.std(ddof=1),
                "seed42_delta_pp": deltas.loc[42],
                "seed43_delta_pp": deltas.loc[43],
                "seed44_delta_pp": deltas.loc[44],
            }
        )
    pd.DataFrame(paired_rows).to_csv(output_dir / "key_contrasts.csv", index=False)

    write_tiny_tex(tiny_summary, tables / "three_seed_confirmations.tex")
    write_resnet_tex(resnet_summary, tables / "preliminary_lr_stability.tex")
    plot_tiny(tiny_seed, figures / "tinyvit_three_seed_confirmations")
    plot_resnet(resnet_summary, figures / "resnet_three_seed_lr_grid")
    plot_constant_lr_learning_curves(
        repo, tiny_seed, figures / "constantlr_learning_curves_resnet_tinyvit"
    )
    print(f"TinyViT seed rows: {len(tiny_seed)}; conditions: {len(tiny_summary)}")
    print(f"ResNet seed rows: {len(resnet_seed)}; conditions: {len(resnet_summary)}")
    print(output_dir / "key_contrasts.csv")


if __name__ == "__main__":
    main()
