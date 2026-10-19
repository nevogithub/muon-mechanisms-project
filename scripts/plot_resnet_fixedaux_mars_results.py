#!/usr/bin/env python3
"""Create slide-ready plots for the clean ResNet MARS confirmation grid."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAME_RE = re.compile(
    r"exp_r18_c100_fixedaux_coslr_mars_g(?P<gamma>[0-9p]+)_lr(?P<lr>[0-9p]+)_seed(?P<seed>\d+)"
)
COLORS = {0.0: "#4C566A", 0.25: "#5E81AC", 0.5: "#D08770", 0.75: "#A3BE8C"}


def decode(token: str) -> float:
    return float(token.replace("p", "."))


def load(root: Path) -> pd.DataFrame:
    frames = []
    for path in root.glob("results/exp_r18_c100_fixedaux_*/seeds/*/run1/results.csv"):
        match = NAME_RE.search(path.as_posix())
        if not match:
            continue
        frame = pd.read_csv(path)
        frame["lr"] = decode(match.group("lr"))
        frame["gamma"] = decode(match.group("gamma"))
        frame["seed"] = int(match.group("seed"))
        frame["source"] = str(path)
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No clean result files found under {root}")
    data = pd.concat(frames, ignore_index=True)
    expected = data.groupby(["lr", "gamma", "seed"])["source"].nunique()
    if len(expected) != 42 or not (expected == 1).all():
        raise RuntimeError(f"Expected 42 unique runs, found {len(expected)}")
    return data


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def lr_robustness(final: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    lr_values = sorted(final.lr.unique())
    for gamma, label, marker in ((0.0, "Muon control, γ=0", "o"), (0.5, "MARS correction, γ=0.5", "s")):
        sub = final[final.gamma == gamma]
        stats = sub.groupby("lr")["accuracy"].agg(["mean", "std"]).reindex(lr_values)
        axes[0].errorbar(
            lr_values,
            100 * stats["mean"],
            yerr=100 * stats["std"],
            color=COLORS[gamma],
            marker=marker,
            linewidth=2.2,
            markersize=7,
            capsize=4,
            label=label,
        )
    plain_best = 100 * final[final.gamma == 0].groupby("lr")["accuracy"].mean().max()
    axes[0].axhline(plain_best, color="#777777", linestyle="--", linewidth=1.4)
    axes[0].text(0.0052, plain_best + 0.08, f"Best Muon mean: {plain_best:.2f}%", color="#666666", fontsize=9)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(lr_values, [f"{v:g}" for v in lr_values])
    axes[0].set_xlabel("Initial matrix learning rate")
    axes[0].set_ylabel("Epoch-50 test accuracy (%)")
    axes[0].set_title("Accuracy across the learning-rate grid")
    axes[0].grid(axis="y", color="#D8DEE9", linewidth=0.8)
    axes[0].legend(frameon=False, loc="lower right")

    deltas = []
    delta_sd = []
    for lr in lr_values:
        a = final[(final.lr == lr) & (final.gamma == 0)].set_index("seed")["accuracy"]
        b = final[(final.lr == lr) & (final.gamma == 0.5)].set_index("seed")["accuracy"]
        values = 100 * (b - a)
        deltas.append(values.mean())
        delta_sd.append(values.std(ddof=1))
    axes[1].bar(range(len(lr_values)), deltas, color=["#BF616A" if d < 0 else "#A3BE8C" for d in deltas], width=0.68)
    axes[1].errorbar(range(len(lr_values)), deltas, yerr=delta_sd, fmt="none", ecolor="#2E3440", capsize=4, linewidth=1.3)
    axes[1].axhline(0, color="#2E3440", linestyle="--", linewidth=1.3)
    axes[1].set_xticks(range(len(lr_values)), [f"{v:g}" for v in lr_values])
    axes[1].set_xlabel("Initial matrix learning rate")
    axes[1].set_ylabel("Paired accuracy change (points)")
    axes[1].set_title("MARS helps only at the higher rates")
    axes[1].grid(axis="y", color="#D8DEE9", linewidth=0.8)

    fig.suptitle("Clean ResNet-18 MARS learning-rate ablation", fontsize=18, fontweight="bold", y=1.03)
    fig.text(
        0.5,
        -0.02,
        "CIFAR-100, 50 epochs, seeds 42–44, β=0.95, cosine matrix LR to 5%, auxiliary AdamW LR fixed at 0.001",
        ha="center",
        fontsize=10,
        color="#4C566A",
    )
    fig.tight_layout()
    save(fig, out, "01_resnet_clean_mars_lr_robustness")


def gamma_sweep(final: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for lr, marker in ((0.04, "o"), (0.08, "s")):
        sub = final[final.lr == lr]
        stats = sub.groupby("gamma")["accuracy"].agg(["mean", "std"]).sort_index()
        xs = stats.index.to_numpy(float)
        ax.errorbar(
            xs,
            100 * stats["mean"],
            yerr=100 * stats["std"],
            marker=marker,
            linewidth=2.2,
            markersize=7,
            capsize=4,
            label=f"Matrix LR {lr:g}",
        )
        for gamma in xs:
            ys = 100 * sub[sub.gamma == gamma]["accuracy"].to_numpy()
            offsets = np.linspace(-0.012, 0.012, len(ys))
            ax.scatter(np.full(len(ys), gamma) + offsets, ys, s=24, alpha=0.65, color="#2E3440", zorder=3)
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.set_xlabel("MARS coefficient γ")
    ax.set_ylabel("Epoch-50 test accuracy (%)")
    ax.set_title("The preferred correction strength increases under LR stress", fontsize=16, fontweight="bold")
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.8)
    ax.legend(frameon=False)
    fig.text(
        0.5,
        -0.015,
        "ResNet-18 on CIFAR-100, 50 epochs, seeds 42–44, β=0.95, fixed auxiliary AdamW LR 0.001",
        ha="center",
        fontsize=10,
        color="#4C566A",
    )
    fig.tight_layout()
    save(fig, out, "02_resnet_clean_mars_gamma_sweep")


def learning_curves(data: pd.DataFrame, out: Path) -> None:
    comparisons = [
        (0.04, 0.0, 0.25, "LR 0.04: γ=0 versus γ=0.25"),
        (0.08, 0.0, 0.75, "LR 0.08: γ=0 versus γ=0.75"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), sharey=True)
    for ax, (lr, base_gamma, best_gamma, title) in zip(axes, comparisons):
        for gamma, label in ((base_gamma, "Muon control"), (best_gamma, f"MARS, γ={best_gamma:g}")):
            sub = data[(data.lr == lr) & (data.gamma == gamma)]
            curve = sub.groupby("Epoch")["Test Accuracy"].agg(["mean", "std"]).reset_index()
            color = COLORS[gamma]
            ax.plot(curve.Epoch, 100 * curve["mean"], color=color, linewidth=2.4, label=label)
            ax.fill_between(
                curve.Epoch.to_numpy(),
                100 * (curve["mean"] - curve["std"]).to_numpy(),
                100 * (curve["mean"] + curve["std"]).to_numpy(),
                color=color,
                alpha=0.17,
                linewidth=0,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color="#D8DEE9", linewidth=0.8)
        ax.legend(frameon=False, loc="lower right")
    axes[0].set_ylabel("Test accuracy (%)")
    fig.suptitle("MARS improves the late trajectory at stressed learning rates", fontsize=18, fontweight="bold", y=1.03)
    fig.text(
        0.5,
        -0.02,
        "Means and sample-SD bands over seeds 42–44, evaluated every five epochs",
        ha="center",
        fontsize=10,
        color="#4C566A",
    )
    fig.tight_layout()
    save(fig, out, "03_resnet_clean_mars_learning_curves")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    data = load(args.input)
    final = data[data.Epoch == 50].copy()
    final["accuracy"] = final["Test Accuracy"]
    summary = (
        final.groupby(["lr", "gamma"])["accuracy"]
        .agg(n="count", mean="mean", sample_sd="std")
        .reset_index()
    )
    final[["lr", "gamma", "seed", "accuracy", "source"]].sort_values(["lr", "gamma", "seed"]).to_csv(
        args.output / "resnet_clean_mars_seed_results.csv", index=False
    )
    summary.to_csv(args.output / "resnet_clean_mars_summary.csv", index=False)
    lr_robustness(final, args.output)
    gamma_sweep(final, args.output)
    learning_curves(data, args.output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
