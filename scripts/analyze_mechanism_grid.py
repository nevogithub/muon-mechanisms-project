"""Analyze verified mechanism artifacts without touching the paper.

Usage: bundled-python scripts/analyze_mechanism_grid.py --bundle tmp/mechanism_grid_20260917
All accuracy values are percentage points. Variability is sample SD, not SEM.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FACTOR_BLOCKS = {"plain_constant_lr", "lr_momentum_factorial", "mars_momentum_factorial"}


def canonical(name):
    return re.sub(r"(?:_confirm)?_seed\d+$", "", name)


def params(cfg):
    opt = cfg["optimizer"]
    p = dict(opt)
    p.update(opt.get(opt["name"], {}))
    p.pop(opt["name"], None)
    return p


def table_text(df):
    # Avoid a non-bundled tabulate dependency.
    rows = [[str(x) for x in df.columns]] + [[str(x) for x in row] for row in df.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * len(rows[0])) + " |"] + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def summarize(df, keys, metric="accuracy_pct"):
    return df.groupby(keys, dropna=False)[metric].agg(n="count", mean="mean", sample_sd="std", minimum="min", maximum="max").reset_index()


def save(fig, out, name):
    fig.savefig(out / "plots" / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "plots" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=ROOT / "analysis/mechanism_grid_20260917")
    args = ap.parse_args()
    bundle, out = args.bundle.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert len(manifest) == 154
    followups = [r for r in manifest if r["block"] in {"lr_momentum_factorial", "mars_gamma_sweep", "mars_momentum_factorial", "stressed_lr_mechanism_confirmation"}]
    assert len(followups) == 37
    records, curves, diagnostic_rows, exact = [], [], [], []
    thresholds = []
    for r in manifest:
        path = bundle / r["path"]
        raw = pd.read_csv(path)
        assert list(raw.Epoch.astype(int)) == list(range(5, 51, 5)), path
        assert set(raw.seed.astype(int)) == {r["seed"]}, path
        numeric = [c for c in ["Train Accuracy", "Test Accuracy", "Test Loss", "Train Loss", "Test Top-5 Accuracy"] if c in raw]
        assert np.isfinite(raw[numeric].to_numpy(dtype=float)).all(), path
        acc = raw["Test Accuracy"].to_numpy(dtype=float) * 100
        assert abs(acc[-1] - r["accuracy_pct"]) < 1e-8
        cfg = json.loads(path.with_name("config.json").read_text())["config"]
        p = params(cfg)
        assert cfg['training']['epochs'] == 50
        assert cfg['dataset']['batch_size'] == 128
        assert cfg['training']['grad_clip'] == 1.0
        assert cfg['optimizer']['weight_decay'] == .0005
        assert p.get('nesterov', True)
        assert p.get('ns_steps', p.get('ns_steps_low', 5)) == 5
        name = canonical(r["experiment"])
        scheduler = cfg.get("scheduler", {})
        beta = p.get("momentum", p.get("mu2_betas", [np.nan])[0])
        alpha = p.get("mu2_alpha_init", 0.0)
        rec = dict(r, condition=name, dataset=cfg["dataset"]["name"], model=cfg["model"]["name"],
                   optimizer=p["name"], lr=p["lr"], lr_policy="cosine" if scheduler.get("name") == "cosine" else "constant",
                   beta=beta, beta_end=p.get("momentum_end", beta), momentum_policy="late" if p.get("decay_schedule", "none") != "none" else "constant",
                   gamma=p.get("gamma", 0.0), alpha=alpha, alpha_end=p.get("mu2_alpha_end", alpha),
                   alpha_schedule=p.get("mu2_alpha_schedule", "none"), preserve_estimator=p.get("preserve_mu2_estimator", False),
                   alpha_start_epoch=p.get('mu2_alpha_start_epoch', 1), alpha_end_epoch=p.get('mu2_alpha_end_epoch', 1),
                   beta_start_epoch=p.get('decay_start_epoch', 1), beta_end_epoch=p.get('decay_end_epoch', 1),
                   workers=cfg["dataset"].get("num_workers"), auxiliary_lr_initial=p.get("adamw_lr"),
                   auxiliary_lr_policy="scheduled_to_common_absolute_floor" if p["name"] == "muon_adamw" and scheduler.get("name") == "cosine" else "fixed_internal",
                   accuracy_auc_e5_e50_pct=np.trapezoid(acc, raw.Epoch.to_numpy()) / 45)
        for epoch in (1, 36, 48, 50):
            floor = scheduler.get('eta_min', scheduler.get('eta_min_ratio', 0) * p['lr'])
            rec[f'auxiliary_lr_epoch{epoch}'] = floor + (p.get('adamw_lr', .001)-floor) * (1 + np.cos(np.pi * (epoch-1)/50)) / 2 if rec['auxiliary_lr_policy'] == 'scheduled_to_common_absolute_floor' else p.get('adamw_lr', .001)
        records.append(rec)
        for epoch, value in zip(raw.Epoch.astype(int), acc):
            curves.append(dict(condition=name, block=r["block"], seed=r["seed"], epoch=epoch, accuracy_pct=value))
        for threshold in (70, 80, 83, 84, 85):
            hits = raw.loc[acc >= threshold, "Epoch"]
            thresholds.append(dict(condition=name, seed=r["seed"], threshold_pct=threshold,
                                   first_observed_epoch=int(hits.iloc[0]) if len(hits) else np.nan,
                                   reached_by_epoch50=bool(len(hits))))
        exact.append(dict(condition=name, seed=r["seed"], source_path=r["path"],
                          optimizer_json=json.dumps(cfg["optimizer"], sort_keys=True), scheduler_json=json.dumps(scheduler, sort_keys=True),
                          training_json=json.dumps(cfg["training"], sort_keys=True), dataset_json=json.dumps(cfg["dataset"], sort_keys=True),
                          model_json=json.dumps(cfg["model"], sort_keys=True)))
        diagpath = path.with_name("optimizer_diagnostics.csv")
        if diagpath.exists():
            diag = pd.read_csv(diagpath)
            assert len(diag) == 50 and list(diag.Epoch.astype(int)) == list(range(1, 51)), diagpath
            for col in ("Correction Relative Norm", "Prepolar Counterfactual Angle Degrees", "Postpolar Counterfactual Angle Degrees"):
                assert np.isfinite(diag[col].to_numpy()).all(), diagpath
            for dr in diag.to_dict("records"):
                diagnostic_rows.append(dict(condition=name, seed=r["seed"], **dr))
    df, learning = pd.DataFrame(records), pd.DataFrame(curves)
    assert not df.duplicated(["condition", "seed"]).any()
    df.to_csv(out / "seed_results.csv", index=False)
    summary = summarize(df, ["condition", "dataset", "lr", "lr_policy", "beta", "gamma", "alpha", "alpha_schedule", "preserve_estimator"])
    summary.to_csv(out / "condition_summary.csv", index=False)
    summarize(df, ["condition"], "accuracy_auc_e5_e50_pct").to_csv(out / "accuracy_auc_summary.csv", index=False)
    pd.DataFrame(exact).to_csv(out / "exact_run_configurations.csv", index=False)
    df[["condition", "seed", "dataset", "optimizer", "lr", "lr_policy", "auxiliary_lr_initial", "auxiliary_lr_policy", "beta", "beta_end", "momentum_policy", "beta_start_epoch", "beta_end_epoch", "gamma", "alpha", "alpha_end", "alpha_schedule", "alpha_start_epoch", "alpha_end_epoch", "preserve_estimator", "workers"]].to_csv(out / "hyperparameter_coverage.csv", index=False)
    df[['condition', 'seed', 'auxiliary_lr_policy'] + [f'auxiliary_lr_epoch{e}' for e in (1,36,48,50)]].to_csv(out / 'auxiliary_lr_trajectory_audit.csv', index=False)
    learning.to_csv(out / "learning_curves_seed.csv", index=False)
    curve_summary = learning.groupby(["condition", "epoch"]).accuracy_pct.agg(n="count", mean="mean", sample_sd="std").reset_index()
    curve_summary.to_csv(out / "learning_curves_summary.csv", index=False)
    times = pd.DataFrame(thresholds)
    times.to_csv(out / "time_to_threshold_seed.csv", index=False)
    times.groupby(["condition", "threshold_pct"]).agg(n=("seed", "count"), reached=("reached_by_epoch50", "sum"), mean_epoch_among_reached=("first_observed_epoch", "mean"), sample_sd_epoch_among_reached=("first_observed_epoch", "std")).reset_index().to_csv(out / "time_to_threshold_summary.csv", index=False)

    # Confirm that the collected historical controls match the preserved primary table.
    preserved = pd.read_csv(ROOT / "analysis/three_seed_confirmation/tinyvit_seed_results.csv")
    old = df[df.block.isin({"plain_constant_lr", "archived_lr_robustness"})]
    audit = old.merge(preserved[["experiment", "seed", "accuracy"]], left_on=["condition", "seed"], right_on=["experiment", "seed"], how="left", validate="one_to_one")
    audit["difference_pp"] = audit.accuracy_pct - audit.accuracy * 100
    assert audit.loc[audit.accuracy.notna(), "difference_pp"].abs().max() < 1e-4
    audit[["condition", "seed", "accuracy_pct", "accuracy", "difference_pp"]].to_csv(out / "preserved_control_audit.csv", index=False)

    # Full 2 x 2 x 2 factorial: all eight cells have the same fixed auxiliary LR.
    fact = df[df.block.isin(FACTOR_BLOCKS)].copy()
    assert len(fact) == 24
    fact["mars"] = (fact.gamma > 0).astype(int)
    assert fact.groupby(["lr_policy", "mars", "momentum_policy"]).size().eq(3).all()
    factorial_summary = summarize(fact, ["lr_policy", "mars", "momentum_policy"])
    factorial_summary.to_csv(out / "momentum_mars_factorial_summary.csv", index=False)
    pivot = fact.pivot(index="seed", columns=["lr_policy", "mars", "momentum_policy"], values="accuracy_pct")
    contrast_rows = []

    def contrast(label, values, note="matched seed labels; not bitwise paired trajectories"):
        for seed, value in values.items():
            contrast_rows.append(dict(contrast=label, seed=int(seed), difference_pp=float(value), interpretation=note))

    for policy in ("constant", "cosine"):
        for mars in (0, 1):
            contrast(f"late_minus_constant_beta__{policy}_lr__mars{mars}", pivot[policy, mars, "late"] - pivot[policy, mars, "constant"])
        for mom in ("constant", "late"):
            contrast(f"mars_minus_plain__{policy}_lr__{mom}_beta", pivot[policy, 1, mom] - pivot[policy, 0, mom])
        contrast(f"mars_x_momentum_DID__{policy}_lr", (pivot[policy, 1, "late"] - pivot[policy, 1, "constant"]) - (pivot[policy, 0, "late"] - pivot[policy, 0, "constant"]))
    for mars in (0, 1):
        contrast(f"momentum_gain_constant_minus_cosine_lr__mars{mars}", (pivot["constant", mars, "late"] - pivot["constant", mars, "constant"]) - (pivot["cosine", mars, "late"] - pivot["cosine", mars, "constant"]))
    contrast("three_way_LR_x_MARS_x_momentum_DID", ((pivot["constant", 1, "late"] - pivot["constant", 1, "constant"]) - (pivot["constant", 0, "late"] - pivot["constant", 0, "constant"])) - ((pivot["cosine", 1, "late"] - pivot["cosine", 1, "constant"]) - (pivot["cosine", 0, "late"] - pivot["cosine", 0, "constant"])))

    fixed = df[df.condition.str.match(r"mu2diag_tv5m_lr0p01_b.*_a.*_fixed$") & df.beta.isin([.5, .9, .95]) & df.alpha.isin([0, .2])].copy()
    assert len(fixed) == 18
    fixed_summary = summarize(fixed, ["beta", "alpha"])
    fixed_summary.to_csv(out / "stressed_beta_alpha_summary.csv", index=False)
    fp = fixed.pivot(index="seed", columns=["beta", "alpha"], values="accuracy_pct")
    for beta in (.5, .9, .95):
        contrast(f"mu2_alpha02_minus_0__beta{beta}", fp[beta, .2] - fp[beta, 0])
    for beta in (.5, .9):
        contrast(f"mu2_x_beta_DID__beta{beta}_minus095", (fp[beta, .2] - fp[beta, 0]) - (fp[.95, .2] - fp[.95, 0]))
    tuned = df[df.condition == "tv5m_in100_coslr01_marsg035_b09"].set_index("seed").accuracy_pct
    contrast("mars_addition_after_beta09_tuning", tuned - fp[.9, 0], "same auxiliary LR treatment; different optimizer classes and loader realizations")
    robust = df[df.block == "archived_lr_robustness"].copy()
    robust["family"] = np.where(robust.gamma > 0, "MARS-inspired", "Muon")
    robust_summary = summarize(robust, ["family", "lr"])
    robust_summary.to_csv(out / "archived_lr_robustness_summary.csv", index=False)
    for lr in (.005, .01):
        rp = robust[robust.lr == lr].pivot(index="seed", columns="family", values="accuracy_pct")
        contrast(f"historical_mars_minus_muon__lr{lr}", rp["MARS-inspired"] - rp["Muon"], "auxiliary LR policy differs: not an isolated MARS effect")
    mars095 = robust[(robust.lr == .01) & (robust.family == 'MARS-inspired')].set_index('seed').accuracy_pct
    contrast('cross_block_mars_minus_alpha0_beta095_fixed_aux', mars095 - fp[.95,0], 'fixed auxiliary LR in both; historical MARS versus diagnostic alpha-zero controls, cross-block')
    cr = pd.DataFrame(contrast_rows)
    cr.to_csv(out / "matched_seed_contrasts.csv", index=False)
    contrast_summary = summarize(cr, ["contrast", "interpretation"], "difference_pp")
    contrast_summary.to_csv(out / "contrast_summary.csv", index=False)

    diag = pd.DataFrame(diagnostic_rows)
    diag.to_csv(out / "optimizer_diagnostics_seed_epoch.csv", index=False)
    diagnostics_summary = diag[diag.Epoch >= 2].groupby(["condition", "seed"])[["Correction Relative Norm", "Gradient Estimator Angle Degrees", "Prepolar Counterfactual Angle Degrees", "Postpolar Counterfactual Angle Degrees"]].mean().reset_index()
    diagnostics_summary.to_csv(out / "diagnostic_means_first_batch_e2_e50.csv", index=False)
    surfaces = df[(df.block == "mu2_diagnostic") & df.condition.str.endswith("_fixed")]
    assert len(surfaces) == 78
    surfaces.to_csv(out / "mu2_fixed_surface_seed42.csv", index=False)
    other_diag = df[(df.block == "mu2_diagnostic") & ~df.condition.str.endswith("_fixed")]
    other_diag.to_csv(out / "mu2_schedule_literal_controls_seed42.csv", index=False)
    gamma_curve = pd.concat([df[df.block == "mars_gamma_sweep"], robust[(robust.lr == .01) & (robust.seed == 42)]], ignore_index=True).sort_values("gamma")
    gamma_curve[["gamma", "accuracy_pct", "condition", "auxiliary_lr_policy"]].to_csv(out / "mars_gamma_curve_seed42.csv", index=False)
    range_rows = []
    for family, group in robust_summary.groupby("family"):
        for criterion in (70, 80, 83):
            supported = group[(group.n == 3) & (group.minimum >= criterion)]
            range_rows.append(dict(family=family, criterion_pct_all3seeds=criterion, tested_lrs_all3seeds_pass=json.dumps(supported.lr.tolist()), max_tested_passing_lr=supported.lr.max() if len(supported) else np.nan, caveat="two three-seed LRs only; historical auxiliary schedule differs"))
    pd.DataFrame(range_rows).to_csv(out / "lr_robustness_tested_range.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {0: "#31688e", 1: "#b04a20"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True, layout="constrained")
    for ax, policy in zip(axes, ("constant", "cosine")):
        for mars in (0, 1):
            g = factorial_summary[(factorial_summary.lr_policy == policy) & (factorial_summary.mars == mars)].set_index("momentum_policy").loc[["constant", "late"]]
            ax.errorbar([0, 1], g["mean"], yerr=g.sample_sd, marker="o", capsize=4, color=colors[mars], label="Muon" if mars == 0 else "MARS-inspired, gamma=0.35")
        ax.set(xticks=[0, 1], xticklabels=["Constant beta=0.98", "Late beta 0.98 to 0.50"], title=f"{policy.capitalize()} LR, initial 0.0025", ylabel="Epoch-50 top-1 accuracy (%)", ylim=(81.5, 86.3))
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("Momentum decay benefits constant LR, not cosine LR (mean and sample SD, 3 seeds)")
    save(fig, out, "momentum_mars_factorial")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, layout="constrained")
    for ax, policy in zip(axes, ("constant", "cosine")):
        for (mars, mom), g in fact[fact.lr_policy == policy].groupby(["mars", "momentum_policy"]):
            ids = set(g.condition)
            lc = learning[learning.condition.isin(ids)].groupby("epoch").accuracy_pct.agg(["mean", "std"])
            ax.plot(lc.index, lc["mean"], color=colors[mars], linestyle="--" if mom == "late" else "-", label=f"{'MARS' if mars else 'Muon'}, {'late decay' if mom == 'late' else 'constant beta'}")
            ax.fill_between(lc.index, lc["mean"]-lc["std"], lc["mean"]+lc["std"], color=colors[mars], alpha=.10)
        ax.axvspan(36, 48, color="grey", alpha=.08)
        ax.set(title=f"{policy.capitalize()} LR 0.0025", xlabel="Epoch", ylabel="Top-1 accuracy (%)", ylim=(0, 90))
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Learning curves (3-seed mean; shaded sample SD; beta decay window 36–48)")
    save(fig, out, "factorial_learning_curves")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    for ax, (dataset, lr, title) in zip(axes, [("cifar100", .08, "ResNet18 / CIFAR-100, LR 0.08"), ("imagenet100", .005, "TinyViT-5M / ImageNet-100, LR 0.005"), ("imagenet100", .01, "TinyViT-5M / ImageNet-100, LR 0.01")]):
        mat = surfaces[(surfaces.dataset == dataset) & (surfaces.lr == lr)].pivot(index="beta", columns="alpha", values="accuracy_pct").sort_index()
        im = ax.imshow(mat, vmin=0, vmax=90, cmap="viridis", aspect="auto")
        ax.set(xticks=range(len(mat.columns)), xticklabels=[f"{x:g}" for x in mat.columns], yticks=range(len(mat.index)), yticklabels=[f"{x:g}" for x in mat.index], xlabel="Mu2-inspired alpha", ylabel="Muon beta", title=title)
        for i in range(len(mat.index)):
            for j in range(len(mat.columns)):
                ax.text(j, i, f"{mat.iloc[i,j]:.1f}", ha="center", va="center", fontsize=8, color="white" if mat.iloc[i,j]<50 else "black")
    fig.colorbar(im, ax=axes, label="Epoch-50 top-1 accuracy (%)", shrink=.8)
    fig.suptitle("Full beta × alpha ablation surfaces (seed 42; exploratory)")
    save(fig, out, "mu2_beta_alpha_surfaces")

    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    for alpha, color in [(0, "#31688e"), (.2, "#b04a20")]:
        g = fixed_summary[fixed_summary.alpha == alpha].sort_values("beta")
        ax.errorbar(np.arange(3) + (-.07 if alpha == 0 else .07), g["mean"], yerr=g.sample_sd, marker="o", capsize=4, color=color, label=f"alpha={alpha:g}")
    ax.set(xticks=[0, 1, 2], xticklabels=["0.50", "0.90", "0.95"], xlabel="Muon beta", ylabel="Epoch-50 top-1 accuracy (%)", ylim=(0, 95), title="TinyViT LR 0.01: high beta destabilizes; alpha=0.20 does not rescue it")
    ax.legend(title="3-seed mean and sample SD")
    ax.grid(axis="y", alpha=.2)
    save(fig, out, "stress_beta_alpha_confirmation")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    for family, color in [("Muon", "#31688e"), ("MARS-inspired", "#b04a20")]:
        g = robust_summary[(robust_summary.family == family) & (robust_summary.n == 3)].sort_values("lr")
        axes[0].errorbar(g.lr, g["mean"], yerr=g.sample_sd, marker="o", capsize=4, color=color, label=family)
    axes[0].set(xlabel="Initial LR", ylabel="Epoch-50 top-1 accuracy (%)", title="Historical LR grid (auxiliary-schedule caveat)", xticks=[.005, .01], ylim=(50, 90))
    axes[0].axhline(75, linestyle="--", color="grey", linewidth=1, label="75% reference")
    axes[0].legend(fontsize=8)
    positive = gamma_curve[gamma_curve.gamma > 0]
    axes[1].plot(positive.gamma, positive.accuracy_pct, "o-", color="#b04a20", label='Fixed auxiliary LR')
    zero = gamma_curve[gamma_curve.gamma == 0]
    axes[1].scatter(zero.gamma, zero.accuracy_pct, marker='x', s=55, color='#31688e', label='Archived gamma=0, scheduled auxiliary')
    axes[1].set(xlabel="MARS-inspired gamma", ylabel="Epoch-50 top-1 accuracy (%)", title="LR 0.01, beta=0.95: gamma sweep (seed 42)")
    axes[1].legend(fontsize=7, loc='lower right')
    save(fig, out, "mars_lr_and_gamma")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    for beta, color in [(.5, "#31688e"), (.9, "#268a5b"), (.95, "#b04a20")]:
        g = surfaces[(surfaces.dataset == "imagenet100") & (surfaces.lr == .01) & (surfaces.beta == beta)]
        g = g.merge(diagnostics_summary[diagnostics_summary.seed == 42], on=["condition", "seed"]).sort_values("alpha")
        axes[0].plot(g.alpha, g["Correction Relative Norm"], "o-", color=color, label=f"beta={beta:g}")
        axes[1].plot(g.alpha, g["Postpolar Counterfactual Angle Degrees"], "o-", color=color, label=f"beta={beta:g}")
    axes[0].set(xlabel="Alpha", ylabel="Correction norm / gradient norm", title="Correction survives before Muon")
    axes[1].set(xlabel="Alpha", ylabel="Post-orthogonalization angle (degrees)", title="Orthogonalization does not erase the correction")
    axes[0].legend(fontsize=8)
    fig.suptitle("First-minibatch diagnostics, mean over epochs 2–50 (seed 42; not a variance estimate)")
    save(fig, out, "mu2_direction_diagnostics")

    def c(label):
        return contrast_summary.set_index("contrast").loc[label]
    def cs(label):
        x = c(label)
        return f"{x['mean']:+.2f} ± {x.sample_sd:.2f} pp"
    rendered_fact = factorial_summary.copy()
    rendered_fact["accuracy_mean_sampleSD_pct"] = rendered_fact.apply(lambda r: f"{r['mean']:.2f} ± {r.sample_sd:.2f}", axis=1)
    rendered_stress = fixed_summary.copy()
    rendered_stress["accuracy_mean_sampleSD_pct"] = rendered_stress.apply(lambda r: f"{r['mean']:.2f} ± {r.sample_sd:.2f}", axis=1)
    t_fact = times[times.condition.isin(fact.condition)]
    t84 = t_fact[t_fact.threshold_pct == 84].merge(fact[["condition", "seed", "lr_policy", "mars", "momentum_policy"]], on=["condition", "seed"], validate="one_to_one")
    t84_summary = t84.groupby(["lr_policy", "mars", "momentum_policy"]).agg(reached=("reached_by_epoch50", "sum"), mean_epoch_among_reached=("first_observed_epoch", "mean")).reset_index()
    t84_summary.to_csv(out / "factorial_threshold84_summary.csv", index=False)
    literal = other_diag[other_diag.preserve_estimator].copy()
    schedules = other_diag[~other_diag.preserve_estimator].copy()
    controls_view = other_diag[['dataset','lr','condition','alpha','alpha_end','alpha_start_epoch','alpha_end_epoch','preserve_estimator','accuracy_pct']].copy()
    controls_view['condition'] = controls_view.condition.str.replace(r'^mu2diag_.*_b0p95_', '', regex=True)
    controls_view['accuracy_pct'] = controls_view.accuracy_pct.round(2)
    # Independent scalar-memory analysis, before the nonlinear Muon map.
    pole_rows = []
    for beta in (.5,.9,.95,.98):
        for alpha in (0,.05,.1,.2,.35,.5):
            trace = beta + alpha*(1-beta**2)
            determinant = alpha*beta*(1-beta)
            pole = (trace + np.sqrt(trace**2-4*determinant))/2
            assert beta-1e-12 <= pole < 1
            pole_rows.append(dict(beta=beta, alpha=alpha, dominant_linear_memory_pole=pole))
    pd.DataFrame(pole_rows).to_csv(out / 'mu2_prepolar_memory_poles.csv', index=False)
    notes = f"""# Muon mechanism experiments — ordered results

As of 17 September 2026. All 37 follow-up executions finished at epoch 50. This analysis includes 154 verified seed-level records: 37 follow-ups, 93 earlier Mu2 diagnostic cells, six constant-LR controls and 18 historical LR-grid records. It does not replace or rebuild the paper.

## 1. Setup and what changes in each block

LR denotes the initial matrix-parameter LR. All tensors with ndim >= 2, including convolution and classifier weights, use Muon. Other tensors use auxiliary AdamW. Every run is from scratch for 50 epochs, batch size 128, evaluation every five epochs, global gradient clipping 1.0, matrix weight decay 0.0005, Nesterov enabled and five Newton–Schulz iterations. Auxiliary settings are LR 0.001, betas (0.9, 0.999), epsilon 1e-8, weight decay zero. These are not AdamW-only baselines.

TinyViT is the official tiny_vit_5m_224 architecture with a 100-class head (5,103,864 task parameters). ImageNet-100 is the ambityga Kaggle dataset at 224 px. The recipe uses Mixup 0.8, CutMix 1.0, mixing probability 1.0, switch probability 0.5, label smoothing 0.1, rand-m9-mstd0.5-inc1, color jitter 0.4, random erasing probability 0.25, random resized crop scale (0.08, 1), ratio (0.75, 4/3), bicubic interpolation, horizontal flip probability 0.5 and evaluation crop fraction 0.875. Normalization mean (0.485, 0.456, 0.406), SD (0.229, 0.224, 0.225). Drop path is zero. Precision is BF16 if supported else FP32, TF32 and cuDNN benchmarking enabled, deterministic algorithms disabled. Data-loader worker counts varied for technical recovery and are recorded per run.

ResNet18/CIFAR-100 diagnostics use LR 0.08 and the recorded base data recipe. Full model, dataset, training and optimizer settings for every execution are in `exact_run_configurations.csv`; this file, rather than this shortened prose, defines all per-run values.

| Block | Varied settings | Replication |
| --- | --- | --- |
| Momentum–MARS factorial | LR policy constant/cosine at 0.0025; gamma 0/0.35; beta constant 0.98 or cosine 0.98 to 0.50, epochs 36–48 | Eight cells, seeds 42–44 |
| MARS coefficient | gamma 0.10, 0.20, 0.50, 0.75 at LR 0.01, beta 0.95; archived gamma 0 and 0.35 anchors | Seed 42 exploratory |
| Mu2 full fixed surface | alpha 0, 0.05, 0.10, 0.20, 0.35, 0.50; ResNet beta 0, 0.50, 0.90, 0.95, 0.98; TinyViT beta 0.50, 0.90, 0.95, 0.98 at LR 0.005 and 0.01 | 78 cells, seed 42 |
| Mu2 recurrence/schedule controls | Two literal-state controls and early decay, late decay, early rise for each model/LR block | 15 cells, seed 42 |
| Stressed-LR confirmation | TinyViT LR 0.01; beta 0.50/0.90/0.95 × alpha 0/0.20 | Six cells, seeds 42–44 |
| MARS after beta tuning | gamma 0.35, beta 0.90, LR 0.01 versus alpha-zero diagnostic controls | Seeds 42–44 |

Cosine LR uses eta_min = 0.05 × initial matrix LR and is stepped after each epoch. At epoch e, the matrix LR is eta_min + (eta0 − eta_min)(1 + cos(pi(e−1)/50))/2; constant LR remains eta0. Beta decay uses r=(e−36)/12 clipped to [0,1], beta(e)=0.98+(0.50−0.98)(1−cos(pi r))/2. It holds 0.98 through epoch 36 and reaches 0.50 at epoch 48. This is the actual cosine window, not a global exponential decay. There is no LR warmup in these blocks. Any exponential theoretical model is an idealized comparison, not the schedule used here.

### Setup audit that limits the historical LR-grid claim

`muon_adamw` exposes separate matrix and auxiliary parameter groups. Its cosine scheduler changes both group LRs toward a common **absolute** minimum eta_min = 0.05 × initial matrix LR. `muon_graddiff`, the momentum-decay variant and Mu2 diagnostic variants use an internal `adamw_lr` that the scheduler does not change. Thus auxiliary hyperparameter values match initially, but the auxiliary trajectories in the historical plain-Muon versus MARS grid do not. The old LR-grid difference cannot be attributed solely to MARS. The new factorial and beta–alpha comparisons keep the auxiliary LR fixed across their cells. No implementation was changed during this analysis and no new training was launched.

The MARS and momentum variants also implement auxiliary Adam epsilon placement differently (uncorrected moments versus bias-corrected moments). With epsilon 1e-8 this is a small numerical distinction, but the cross-class comparisons are not bitwise-equivalent toggles. Historical constant-LR controls have different worker counts from the new factorial. Seed-label pairing is useful descriptively, not an identical-minibatch coupling.

## 2. Momentum decay and MARS together

All values below are epoch-50 top-1 accuracy, mean ± **sample SD** over three seeds.

{table_text(rendered_fact[['lr_policy', 'mars', 'momentum_policy', 'accuracy_mean_sampleSD_pct']])}

Late-minus-constant beta gains:

- Plain Muon, constant LR: {cs('late_minus_constant_beta__constant_lr__mars0')}.
- Plain Muon, cosine LR: {cs('late_minus_constant_beta__cosine_lr__mars0')}.
- MARS, constant LR: {cs('late_minus_constant_beta__constant_lr__mars1')}.
- MARS, cosine LR: {cs('late_minus_constant_beta__cosine_lr__mars1')}.

The difference-in-differences (MARS momentum-decay gain minus plain-Muon momentum-decay gain) is {cs('mars_x_momentum_DID__constant_lr')} under constant LR and {cs('mars_x_momentum_DID__cosine_lr')} under cosine LR. The three-way LR-policy × MARS × momentum interaction is {cs('three_way_LR_x_MARS_x_momentum_DID')}.

MARS after late decay adds {cs('mars_minus_plain__constant_lr__late_beta')} under constant LR, but {cs('mars_minus_plain__cosine_lr__late_beta')} under cosine LR. MARS with constant beta reduces accuracy under constant LR ({cs('mars_minus_plain__constant_lr__constant_beta')}). Therefore the combination is schedule-dependent; there is no universal MARS-plus-decay synergy.

The constant-LR late-decay policy gives the refreshed direction a non-vanishing step budget. Cosine LR reduces that budget during the same late window. This is a mechanism consistent with the interaction, not a causal proof: LR also changes training trajectory and accumulated weight decay. Muon approximately flattens singular values, not merely a single gradient norm. Rotation of the update can survive orthogonalization even when scalar magnitude information is suppressed.

## 3. Mu2-inspired estimator: what the hyperparameter ablation establishes

{table_text(rendered_stress[['beta', 'alpha', 'accuracy_mean_sampleSD_pct']])}

At LR 0.01, alpha=0.20 versus alpha=0 gives {cs('mu2_alpha02_minus_0__beta0.5')} at beta 0.50, {cs('mu2_alpha02_minus_0__beta0.9')} at beta 0.90 and {cs('mu2_alpha02_minus_0__beta0.95')} at beta 0.95. Low-beta controls already achieve roughly 84% and alpha 0.20 provides no consistent improvement. At beta 0.95 it substantially worsens a regime that is already unstable. This supports a negative result **for this integration and tested settings**, not for Mu2SGD generally.

MARS at beta 0.90 achieves {tuned.mean():.2f} ± {tuned.std(ddof=1):.2f}%. It adds {cs('mars_addition_after_beta09_tuning')} over the alpha-zero beta-0.90 control. The seed differences are {', '.join(f'{v:+.2f}' for v in (tuned-fp[.9,0]).sort_index())} pp. This is a modest descriptive gain with three seeds, much smaller than the historical high-beta LR-grid gap. Momentum tuning itself explains a large part of the apparent stability benefit.

The complete seed-42 surfaces are shown in `plots/mu2_beta_alpha_surfaces.png`, and all 78 values are retained. ResNet has some small favorable cells rather than a uniform negative response. TinyViT LR 0.005 is sensitive to alpha mainly at very high beta; at LR 0.01 beta dominates. The 15 schedule and literal-state controls are reported separately, not omitted or merged with fixed-alpha experiments.

The schedule tests use cosine alpha 0.20 to 0.05 over epochs 6–18 (early decay), 0.20 to 0.05 over epochs 36–48 (late decay), and 0.05 to 0.20 over epochs 6–18 (early rise). Beta stays 0.95. These schedule tests alone do not rescue the LR-0.01 TinyViT setting:

{table_text(controls_view)}

### What was actually implemented and diagnosed

The author-approved inspired integration computes d_t = g_t + alpha_t(h_(t−1) − g_(t−1)), where h_(t−1) is the **Nesterov-combined pre-orthogonalization signal** stored in the estimator tensor by Muon's in-place update. It is not an unmodified literal d_(t−1) recurrence. Alpha controls this historical correction and beta controls Muon's EMA. The second `mu2_betas` entry is stored in configuration; the update reads its first entry for beta and the explicit alpha schedule for alpha. `matched_adamw_rms` is stored but not read by this update. There is no fresh-gradient evaluation on the previous iterate, so this is not an exact implementation of full Mu2SGD variance reduction.

For `preserve_mu2_estimator=True`, the estimator is cloned before Muon. With zero initial state and consecutive-minibatch last gradients, the literal recurrence gives d_t=g_t by induction, for any alpha schedule. These literal-state controls diagnose an algebraic no-op, not another working variance-reduction method.

The recorded first minibatch at each epoch shows nonzero correction norms and post-orthogonalization angles for nonzero alpha. Thus Muon's orthogonalization does **not** simply cancel the estimator's directional effect. Persistent noisy or stale history compounding Muon's momentum is a plausible explanation for high-beta degradation, not an identified cause. The logged counterfactual uses the same current momentum buffer for raw and corrected input; it is a local step comparison, not a separate baseline trajectory. Norms/angles are aggregated over all matrix tensors and sampled once per epoch. They do not estimate gradient variance, establish heavy regularization as the cause, or distinguish architecture from dataset difficulty.

For a scalar entry of the pre-orthogonalization signal, with constant beta and alpha, the implementation gives m_t=beta m_(t−1)+(1−beta)d_t and h_t=(1−beta²)d_t+beta² m_(t−1). Substituting d_t=g_t+alpha(h_(t−1)−g_(t−1)) gives a two-state linear memory matrix A=[[beta, alpha(1−beta)], [beta², alpha(1−beta²)]], with input vector [(1−beta), (1−beta²)] times (g_t−alpha g_(t−1)). Its trace is beta+alpha(1−beta²) and determinant alpha beta(1−beta). The dominant memory pole is (trace+sqrt(trace²−4 determinant))/2. Positive alpha increases that pole above beta for 0<beta<1, lengthening memory. For beta 0.95, alpha 0.20, the pole is approximately 0.9596 versus 0.95 at alpha zero. For alpha<1 the frozen linear memory remains stable: this calculation does **not** predict an exploding estimator or prove the cause of nonconvergence. Gradient feedback, orthogonalization, augmentation and the LR still matter. It does show why adding a recursive correction is not automatically variance reduction and can increase history persistence.

## 4. MARS sensitivity and LR robustness

At beta 0.95, LR 0.01, seed-42 gamma 0.10/0.20/0.50/0.75 gives 69.24/72.72/81.58/83.06%. Positive-gamma comparisons keep auxiliary LR fixed. These show a coefficient response in this slice, not an optimal gamma or a general monotonic law. Gamma 0.35 from the historical run and gamma 0 plain-Muon are marked as archived anchors.

The historical three-seed MARS-minus-Muon gap at LR 0.01 is {cs('historical_mars_minus_muon__lr0.01')}, versus {cs('historical_mars_minus_muon__lr0.005')} at LR 0.005. These are descriptive optimizer-package comparisons because of the auxiliary-schedule mismatch. Both packages fail to learn well at historical single-seed LRs 0.02/0.04/0.08, so the evidence does not support unlimited LR robustness. No discarded high-LR jobs were rerun. `lr_robustness_tested_range.csv` reports only tested passing points, never an inferred continuous stability interval.

MARS beta 0.95, gamma 0.35 at LR 0.01 also exceeds the diagnostic alpha-zero beta-0.95 controls by {cs('cross_block_mars_minus_alpha0_beta095_fixed_aux')}. Both sides of this comparison have fixed auxiliary LR, so the evidence for high-beta stabilization is not limited to the mismatched older plain-Muon package. However, these are different experimental blocks with variable loading/GPU realizations, and the unstable alpha-zero controls have very large variability. Report that comparison separately rather than replacing either baseline or declaring the exact effect size identified.

MARS-inspired correction is u_t=(1+gamma)g_t−gamma g_(t−1). With constant beta the EMA's linear transfer function is (1−beta)[(1+gamma)−gamma z^(-1)]/(1−beta z^(-1)), where z^(-1) denotes a one-step delay. The added finite-difference term modifies responsiveness without adding a recursive state pole beyond beta, unlike the inspired Mu2 recurrence above. Its DC gain is one and its high-frequency gain can increase. This offers a concrete distinction between the two integrations, but proves neither noise reduction nor their optimization outcomes. Gradients here come from successive augmented minibatches, not paired fresh evaluations at two iterates. Both names therefore refer to inspired integrations, not tests of the full source algorithms' guarantees.

## 5. Convergence-speed measurement and presentation order

`time_to_threshold_seed.csv` records the first **observed evaluation epoch** reaching 70/80/83/84/85%. A missing epoch means not reached by epoch 50, not zero. The summary includes the reached count and reports mean epoch only among reached runs. Evaluation is every five epochs, so threshold times are interval-observed and not interpolated. `accuracy_auc_summary.csv` reports trapezoidal mean accuracy over epochs 5–50. Final-accuracy gains alone do not prove faster convergence, especially when beta decay only starts at epoch 36. Do not compare wall-clock speed across GPU/worker configurations.

For the 84% threshold (reached counts are out of three seeds):

{table_text(t84_summary.round(2))}

Plain-Muon cosine-LR late decay reaches the observed 84% threshold at mean epoch 41.67 versus 45.00 for constant beta, despite a lower final accuracy. Thus this experiment supports faster attainment of this particular threshold under cosine LR too. The much stronger constant-LR advantage is in final accuracy and reaching high thresholds reliably, not an assertion that decay never accelerates a cosine-LR run. When one policy fails to reach a threshold, do not compare conditional mean times as if every run reached it.

Present the results in this order: (1) fixed experimental contract and varied factors, (2) momentum/LR interaction, (3) MARS coefficient and controlled tuned-beta increment, (4) full Mu2 hyperparameter surface and replicated negative slice, (5) combinations and mechanism diagnostics. Keep ResNet's broader earlier ablations separate from TinyViT transfer evidence. The earlier three-seed ResNet grid remains in `analysis/three_seed_confirmation/resnet_summary.csv`; it was not recomputed from unavailable raw files in this bundle.

## 6. Provenance and limitations

Every selected output has all evaluations through epoch 50 and finite available losses/accuracies. The 37 follow-ups are selected by successful SLURM job log and suite ID, not best accuracy. All collected historical controls that have a preserved-table counterpart match it within 0.0001 pp. The old collapsed seed-42 plain-Muon run is preserved in earlier provenance; the verified 67.30% repeat is the historical primary-table value. Different reruns of alpha-zero diagnostic Muon are not silently substituted into that LR-grid mean.

Jobs 89064_4 and 89064_5 failed from shared-memory unlink errors and were replaced by 90137_4 and 90052_5. Partial 84856/84857 runs and other canceled/superseded attempts are excluded. Finite nonconvergent completed runs remain in the results. Only three seeds support the confirmation claims; the large hyperparameter sweep is seed-42 exploration. Shared-memory recovery changed loading parallelism, not the requested optimizer settings, and may change stochastic realizations. No new ablations or paper edits were performed.
"""
    (out / "FINDINGS.md").write_text(notes, encoding="utf-8")
    prior = out / 'preserved_primary_tables'
    prior.mkdir(exist_ok=True)
    for name in ('resnet_seed_results.csv', 'resnet_summary.csv', 'tinyvit_seed_results.csv', 'tinyvit_summary.csv', 'key_contrasts.csv'):
        shutil.copy2(ROOT / 'analysis/three_seed_confirmation' / name, prior / name)
    (prior / 'README.md').write_text('These are unchanged earlier primary tables, not recomputed from the new bundle. Accuracy columns are fractions (0 to 1), while delta_pp columns are percentage points. The new FINDINGS.md documents an auxiliary-LR trajectory mismatch in historical plain-Muon versus MARS comparisons. Treat those historical comparisons as optimizer-package evidence, not an isolated correction effect.\n', encoding='utf-8')
    shutil.copy2(bundle / "manifest.json", out / "source_manifest.json")
    audit_info = dict(records=len(df), followups=len(followups), diagnostic_seed42=int((df.block == "mu2_diagnostic").sum()), factorial_records=len(fact), stress_records=len(fixed), evaluation_epochs=list(range(5, 51, 5)), standard_deviation="sample, ddof=1", controls_matching_preserved_table=int(audit.accuracy.notna().sum()), source_bundle=str(bundle), outputs_paper_unchanged=True)
    (out / "verification.json").write_text(json.dumps(audit_info, indent=2))
    print(json.dumps(audit_info, indent=2))
    print(contrast_summary[["contrast", "n", "mean", "sample_sd"]].to_string(index=False))


if __name__ == "__main__":
    main()
