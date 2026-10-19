"""Project-wide, source-audited scientific figures. Does not modify the paper.

Run with the bundled Python and PYTHONPATH=tmp/pydeps. Outputs are static figures,
tidy CSVs, a local gallery, a graph book, and a portable ZIP.
"""
from pathlib import Path
import json
import re
import html
import shutil
import zipfile
import textwrap
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/project_benchmark_graphs_20260917'
MECH = ROOT / 'analysis/mechanism_grid_20260917'
BUNDLE = ROOT / 'tmp/mechanism_grid_20260917'
BOOK = ROOT / 'output/pdf/muon_project_graphbook_20260917.pdf'
ZIP = ROOT / 'output/analysis/muon_project_graphs_20260917.zip'
COLORS = {'Muon': '#0072B2', 'MARS': '#D55E00', 'Mu2': '#009E73',
          'MARS + Mu2': '#CC79A7', 'Joint schedule': '#6B548E'}
FAMILIES = list(COLORS)
FAMILY_IDS = {'muon': 'Muon', 'plain': 'Muon', 'graddiff': 'MARS', 'mars': 'MARS',
              'mu2': 'Mu2', 'mars_mu2': 'MARS + Mu2', 'final': 'Joint schedule',
              'final_cosine': 'Joint schedule'}
AUX_NOTE = 'Historical package comparison: auxiliary LR trajectories differ between the plain baseline and correction variants.'
FIXED_NOTE = 'Fixed auxiliary LR. Matched seed labels, not bitwise-identical stochastic trajectories.'
FIGURES = []
DATA = {}
CURVES = {}
CONFIGS = {}
REFRESH = None


def flatten(cfg):
    opt = cfg['optimizer']
    p = dict(opt)
    p.update(opt.get(opt['name'], {}))
    sch = cfg.get('scheduler', {})
    beta = p.get('momentum', p.get('mu2_betas', [.95])[0])
    bsch = p.get('decay_schedule', p.get('mu2_decay_schedule', 'none'))
    beta_end = p.get('momentum_end', p.get('mu2_beta_end', beta))
    gamma = p.get('gamma', 0.)
    alpha = p.get('mu2_alpha_init', 0.)
    family = 'Joint schedule' if p.get('gamma_schedule', 'none') != 'none' else (
        'MARS + Mu2' if gamma and alpha else 'MARS' if gamma else 'Mu2' if alpha else 'Muon')
    return dict(lr=float(p['lr']), lr_policy='cosine' if sch.get('name') == 'cosine' else 'constant',
                beta=beta, beta_end=beta_end, beta_schedule=bsch,
                beta_start_epoch=p.get('decay_start_epoch', p.get('mu2_decay_start_epoch', 1)),
                beta_end_epoch=p.get('decay_end_epoch', p.get('mu2_decay_end_epoch', 1)),
                gamma=gamma, gamma_end=p.get('gamma_end', gamma),
                gamma_schedule=p.get('gamma_schedule', 'none'), alpha=alpha,
                alpha_end=p.get('mu2_alpha_end', alpha),
                alpha_schedule=p.get('mu2_alpha_schedule', bsch),
                preserve_estimator=p.get('preserve_mu2_estimator', False), family=family,
                optimizer=p['name'], auxiliary_lr_initial=p.get('adamw_lr', .001),
                auxiliary_lr_policy='scheduled_to_common_absolute_floor'
                if p['name'] == 'muon_adamw' and sch.get('name') == 'cosine' else 'fixed',
                workers=cfg['dataset'].get('num_workers'), epochs=cfg['training']['epochs'],
                batch_size=cfg['dataset']['batch_size'], matrix_weight_decay=p.get('weight_decay'),
                grad_clip=cfg['training'].get('grad_clip'), nesterov=p.get('nesterov', True),
                ns_steps=p.get('ns_steps', p.get('ns_steps_low', 5)))


def put(condition, seed, benchmark, block, accuracy, curve, cfg, source, config_kind):
    assert benchmark in ('ResNet18 / CIFAR-100', 'TinyViT-5M / ImageNet-100')
    assert cfg['dataset']['name'] in ('cifar100', 'imagenet100')
    key = (condition, int(seed))
    f = flatten(cfg)
    assert f['epochs'] == 50 and f['batch_size'] == 128
    d = pd.read_csv(curve) if isinstance(curve, Path) else curve.copy()
    acc_col = 'Test Accuracy' if 'Test Accuracy' in d else 'Test Accuracy_mean'
    assert list(d.Epoch.astype(int)) == list(range(5, 51, 5)), (condition, source)
    assert np.isfinite(d[acc_col].to_numpy(dtype=float)).all()
    assert abs(float(d[acc_col].iloc[-1]) * 100 - accuracy) < .001, (condition, source)
    if key in DATA:
        assert abs(DATA[key]['accuracy_pct'] - accuracy) < .001 or condition == 'tinyvit5m_imagenet100_lrrobust_muon_lr0p01'
    DATA[key] = dict(condition=condition, seed=int(seed), benchmark=benchmark, block=block,
                     accuracy_pct=float(accuracy), source=source, configuration_source=config_kind, **f)
    fields = {'accuracy_pct': acc_col,
              'test_loss': 'Test Loss' if 'Test Loss' in d else 'Test Loss_mean',
              'train_accuracy_pct': 'Train Accuracy' if 'Train Accuracy' in d else 'Train Accuracy_mean',
              'train_loss': 'Train Loss' if 'Train Loss' in d else 'Train Loss_mean',
              'top5_accuracy_pct': 'Test Top-5 Accuracy' if 'Test Top-5 Accuracy' in d else 'Test Top-5 Accuracy_mean'}
    rows = []
    for _, r in d.iterrows():
        row = dict(condition=condition, seed=int(seed), epoch=int(r.Epoch))
        for dest, src in fields.items():
            row[dest] = float(r[src]) * (100 if dest.endswith('_pct') else 1) if src in d else np.nan
        rows.append(row)
    CURVES[key] = rows
    CONFIGS[key] = dict(condition=condition, seed=int(seed), configuration_source=config_kind,
                        source=source, config=cfg)


def collect():
    presets = {}
    for filename in ['_generated_report_resnet18_clean.json', '_generated_report_component_resnet18.json',
                     '_generated_report_algorithm_momentum_resnet18.json', '_generated_report_constant_lr_control.json',
                     '_generated_tinyvit_imagenet100_grid.json', '_generated_tinyvit5m_imagenet100_momentum_only.json',
                     '_generated_tinyvit5m_imagenet100_momentum_constantlr.json']:
        j = json.loads((ROOT / 'config' / filename).read_text())
        for k, v in j['experiments'].items():
            presets[k] = (v, f'preset: config/{filename}')
    old = pd.read_csv(ROOT / 'analysis/cifar100_report/cifar100_all_results.csv')
    for _, r in old[old.model.eq('resnet18')].iterrows():
        cond = r.experiment.removeprefix('exp_')
        if r.suite == 'lr_stability':
            cond = cond.replace('lrstab_resnet18_', 'resnet18_cifar100_').replace('_final_lr', '_final_cosine_lr')
        cfg, origin = presets[cond]
        path = ROOT / r.curve_path.replace('\\', '/')
        put(cond, 42, 'ResNet18 / CIFAR-100', 'resnet_' + r.suite, r.test_acc * 100, path, cfg, str(path.relative_to(ROOT)), origin)
    oldconst = pd.read_csv(ROOT / 'analysis/constantlr_control/constantlr_all_results.csv')
    for _, r in oldconst[oldconst.model.eq('resnet18')].iterrows():
        cond = r.run.removeprefix('exp_')
        path = ROOT / 'analysis/constantlr_pre_shutdown_snapshot/results' / r.run / 'results_averaged_curve.csv'
        cfg, origin = presets[cond]
        put(cond, 42, 'ResNet18 / CIFAR-100', 'resnet_constant_lr', r.test_acc * 100, path, cfg, str(path.relative_to(ROOT)), origin)
    tf = pd.read_csv(ROOT / 'analysis/tinyvit_imagenet100_summary/snapshot/tinyvit_imagenet100_final.csv')
    tc = pd.read_csv(ROOT / 'analysis/tinyvit_imagenet100_summary/snapshot/tinyvit_imagenet100_curves.csv')
    for _, r in tf.iterrows():
        cond = r.experiment
        if cond == 'tinyvit5m_imagenet100_lrrobust_muon_lr0p01':
            continue  # Explicitly commissioned verified rerun replaces this one old snapshot.
        if cond not in presets:
            assert r.block == 'protocol_equivalent_baseline'
            cfg, origin = presets['tinyvit5m_imagenet100_lrrobust_muon_lr0p005']
        else:
            cfg, origin = presets[cond]
        c = tc[tc.experiment.eq(cond)].rename(columns={'epoch': 'Epoch', 'test_accuracy': 'Test Accuracy',
                  'test_loss': 'Test Loss', 'train_accuracy': 'Train Accuracy', 'train_loss': 'Train Loss', 'top5_accuracy': 'Test Top-5 Accuracy'})
        put(cond, 42, 'TinyViT-5M / ImageNet-100', 'tiny_' + r.block, r.test_accuracy * 100, c, cfg,
            f'analysis/tinyvit_imagenet100_summary/snapshot/tinyvit_imagenet100_curves.csv#{cond}', origin)
    for label, benchmark in [('resnet', 'ResNet18 / CIFAR-100'), ('tinyvit', 'TinyViT-5M / ImageNet-100')]:
        table = pd.read_csv(ROOT / f'analysis/three_seed_confirmation/{label}_seed_results.csv')
        for _, r in table.iterrows():
            path = Path(r.artifact_path)
            if not path.exists() or not path.with_name('config.json').exists():
                assert r.seed == 42 and (r.experiment, 42) in DATA
                assert abs(DATA[r.experiment, 42]['accuracy_pct'] - r.accuracy * 100) < .001
                continue
            cfg = json.loads(path.with_name('config.json').read_text())['config']
            block = DATA.get((r.experiment, 42), {}).get('block', 'tiny_lr_robustness')
            put(r.experiment, r.seed, benchmark, block, r.accuracy * 100, path, cfg,
                str(path.relative_to(ROOT)), 'recorded per-run config.json')
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    for r in manifest:
        path = BUNDLE / r['path']
        cfg = json.loads(path.with_name('config.json').read_text())['config']
        from analyze_mechanism_grid import canonical
        cond = canonical(r['experiment'])
        existing = DATA.get((cond, r['seed']), {})
        block = existing.get('block', r['block'])
        put(cond, r['seed'], 'ResNet18 / CIFAR-100' if cfg['dataset']['name'] == 'cifar100' else 'TinyViT-5M / ImageNet-100',
            block, r['accuracy_pct'], path, cfg, f'tmp/mechanism_grid_20260917/{r["path"]}', 'recorded per-run config.json')
    df = pd.DataFrame(DATA.values())
    curves = pd.DataFrame([r for rows in CURVES.values() for r in rows])
    assert not df.duplicated(['condition', 'seed']).any()
    assert not curves.duplicated(['condition', 'seed', 'epoch']).any()
    df['accuracy_auc_e5_e50_pct'] = [np.trapezoid([r['accuracy_pct'] for r in CURVES[c, s]], [r['epoch'] for r in CURVES[c, s]]) / 45 for c, s in zip(df.condition, df.seed)]
    df.to_csv(OUT / 'data/seed_results.csv', index=False)
    curves.to_csv(OUT / 'data/learning_curves_seed.csv', index=False)
    (OUT / 'data/run_configurations.json').write_text(json.dumps(list(CONFIGS.values()), indent=2))
    summary = df.groupby('condition', sort=False).accuracy_pct.agg(n='count', mean='mean', sample_sd='std', minimum='min', maximum='max').reset_index()
    metadata = df.drop_duplicates('condition').drop(columns=['seed', 'accuracy_pct', 'source', 'accuracy_auc_e5_e50_pct'])
    summary = metadata.merge(summary, on='condition', validate='one_to_one')
    summary.to_csv(OUT / 'data/condition_summary.csv', index=False)
    tables = OUT / 'data/block_tables'
    tables.mkdir(exist_ok=True)
    for block, g in summary.groupby('block'):
        g.to_csv(tables / f'{block}.csv', index=False)
    robustness = []
    for benchmark, reference, block in [('ResNet18 / CIFAR-100', 74.5, 'resnet_lr_stability'),
                                         ('TinyViT-5M / ImageNet-100', 74.5, 'tiny_lr_robustness')]:
        for family, g in summary[summary.block.eq(block) & summary.lr.le(.02 if 'TinyViT' in benchmark else 1)].groupby('family'):
            robustness.append(dict(benchmark=benchmark, family=family, reference_pct=reference,
                                   tested_lrs=len(g), tested_mean_endpoints_at_or_above_reference=int(g['mean'].ge(reference).sum()),
                                   tested_lrs_json=json.dumps(sorted(g.lr.tolist())),
                                   interpretation='descriptive discrete tested points, not a continuous stability interval'))
    pd.DataFrame(robustness).to_csv(OUT / 'data/discrete_lr_robustness_references.csv', index=False)
    oldcollapse = tf[tf.experiment.eq('tinyvit5m_imagenet100_lrrobust_muon_lr0p01')]
    oldcollapse.to_csv(OUT / 'data/excluded_superseded_seed42_snapshot.csv', index=False)
    coverage = summary.groupby(['benchmark', 'block']).agg(conditions=('condition','count'), replicated_conditions=('n', lambda s: int((s == 3).sum())), retained_seed_records=('n','sum')).reset_index()
    coverage.to_csv(OUT / 'data/coverage.csv', index=False)
    for filename in ['matched_seed_contrasts.csv', 'contrast_summary.csv', 'time_to_threshold_seed.csv',
                     'time_to_threshold_summary.csv', 'optimizer_diagnostics_seed_epoch.csv',
                     'diagnostic_means_first_batch_e2_e50.csv', 'mu2_prepolar_memory_poles.csv']:
        shutil.copy2(MECH / filename, OUT / 'data' / filename)
    return df, curves, summary, coverage


def axes(n=1, width=None, height=5.2):
    return plt.subplots(1, n, figsize=(width or (10 if n == 1 else 6 * n), height), squeeze=False)[0:2]


def finish(fig, id, title, caption, takeaway, tier='appendix', settings='', caveat=''):
    fig.suptitle(title, fontsize=16, fontweight='normal', y=1.01)
    for ax in fig.axes:
        if ax.get_label() == '<colorbar>':
            continue
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=1.4)
    target = OUT / tier / id
    if REFRESH is None or id in REFRESH:
        for ext in ['png', 'svg', 'pdf']:
            fig.savefig(target.with_suffix('.' + ext), dpi=220, bbox_inches='tight', facecolor='white')
    else:
        assert all(target.with_suffix('.'+ext).exists() for ext in ['png','svg','pdf'])
    plt.close(fig)
    FIGURES.append(dict(id=id, title=title, caption=caption, takeaway=takeaway, tier=tier,
                        settings=settings, caveat=caveat, path=f'{tier}/{id}.png'))


def lr_axis(ax, values, ylabel='Epoch-50 test accuracy (%)'):
    ax.set_xscale('log', base=2)
    ax.set_xticks(values, [f'{x:g}' for x in values])
    ax.set_xlabel('LR')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.2)


def sweep(ax, frame, x='lr', marker='o', name=None, color=None):
    frame = frame.sort_values(x)
    ax.errorbar(frame[x], frame['mean'], yerr=frame.sample_sd.fillna(0), marker=marker,
                color=color, label=name, capsize=3, linewidth=2)


def endpoint(ax, frame, labels, colors=None, ylabel='Epoch-50 test accuracy (%)'):
    for i, (condition, label) in enumerate(labels):
        g = frame[frame.condition.eq(condition)].sort_values('seed')
        assert len(g)
        c = colors[i] if colors else COLORS.get(g.family.iloc[0], '#0072B2')
        ax.scatter(i + np.linspace(-.09, .09, len(g)), g.accuracy_pct, s=32, color=c, alpha=.55, zorder=3)
        ax.errorbar(i, g.accuracy_pct.mean(), yerr=g.accuracy_pct.std(ddof=1) if len(g) > 1 else 0,
                    marker='_', ms=20, lw=2, capsize=5, color=c)
        ax.text(i, g.accuracy_pct.max() + .10, f'n={len(g)}', ha='center', fontsize=9)
    ax.set_xticks(range(len(labels)), [x[1] for x in labels])
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.2)
    ax.margins(x=.15, y=.22)


def learning(ax, curves, conditions, colors=None, metric='accuracy_pct', shade=None, ylabel=None):
    for i, (condition, label) in enumerate(conditions):
        d = curves[curves.condition.eq(condition)]
        g = d.groupby('epoch')[metric].agg(['mean', 'std', 'count'])
        assert len(g) == 10
        c = colors[i] if colors else list(COLORS.values())[i % 5]
        ax.plot(g.index, g['mean'], color=c, marker='o', ms=3, label=f'{label} (n={int(g["count"].min())})')
        if (g['count'] > 1).all():
            ax.fill_between(g.index, g['mean'] - g['std'], g['mean'] + g['std'], color=c, alpha=.12)
    if shade:
        ax.axvspan(*shade, color='#777777', alpha=.10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel or ('Test accuracy (%)' if metric == 'accuracy_pct' else 'Test cross-entropy loss'))
    ax.set_xticks([5, 15, 25, 35, 45, 50])
    ax.grid(axis='y', alpha=.2)
    ax.legend(fontsize=9, loc='best', frameon=False)


def heat(ax, table, title, delta=False, vmin=None, vmax=None, annotate=True):
    values = table.to_numpy(dtype=float)
    if delta:
        lim = max(abs(np.nanmin(values)), abs(np.nanmax(values)), 1) if vmax is None else vmax
        im = ax.imshow(values, cmap='RdBu', norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect='auto')
    else:
        im = ax.imshow(values, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(title, pad=14)
    ax.set_xticks(range(len(table.columns)), [str(c) for c in table.columns])
    ax.set_yticks(range(len(table.index)), [str(x) for x in table.index])
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            if annotate and np.isfinite(value):
                rgba = im.cmap(im.norm(value))
                lum = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                ax.text(j, i, f'{value:+.2f}' if delta else f'{value:.2f}', ha='center', va='center', color='black' if lum > .55 else 'white', fontsize=10)
    return im


def figures(df, curves, summary, coverage):
    md = pd.read_csv(MECH / 'seed_results.csv')
    fact = md[md.block.isin(['plain_constant_lr', 'lr_momentum_factorial', 'mars_momentum_factorial'])].copy()
    fact['mars'] = fact.gamma.gt(0)
    fs = fact.groupby(['lr_policy', 'mars', 'momentum_policy']).accuracy_pct.agg(n='count', mean='mean', sample_sd='std').reset_index()
    def fc(policy, mars, momentum):
        return fact[(fact.lr_policy == policy) & (fact.mars == mars) & (fact.momentum_policy == momentum)].condition.iloc[0]
    fig, a = axes(2)
    e = np.arange(1, 51)
    a[0,0].plot(e, np.ones(50), color='#0072B2', label='Constant LR')
    a[0,0].plot(e, .05 + .95 * (1 + np.cos(np.pi * (e - 1) / 50)) / 2, color='#D55E00', label='Cosine LR')
    a[0,0].set(xlabel='Epoch', ylabel='LR / initial LR', title='Outer LR policy')
    a[0,0].legend(frameon=False)
    for start, end, label, color in [(6, 18, 'Early: epochs 6-18', '#009E73'), (36, 48, 'Late: epochs 36-48', '#CC79A7')]:
        x = np.clip((e-start)/(end-start), 0, 1)
        beta = .5 + (.98-.5)*(1+np.cos(np.pi*x))/2
        a[0,1].plot(e, beta, label=label, color=color)
    a[0,1].axhline(.98, ls='--', color='#444', label='Constant beta 0.98')
    a[0,1].set(xlabel='Epoch', ylabel='Momentum beta', title='Momentum transition window')
    a[0,1].legend(frameon=False, fontsize=10)
    finish(fig, '01_actual_schedules', 'Learning rate and momentum are separate controls', 'Schedules derived from the executable epoch conventions. Cosine LR approaches 5% of its initial value; the epoch-50 update precedes the last scheduler step. Momentum is held before and after its transition window.', 'Use these schedules to define the experiment axes before showing results.', 'main', '50 epochs. Cosine LR phase (epoch-1)/50. Momentum transition uses (epoch-start)/(end-start).')

    rmain = summary[summary.block.eq('resnet_lr_stability')]
    fig, a = axes()
    for fam in FAMILIES:
        sweep(a[0,0], rmain[rmain.family.eq(fam)], name=fam, color=COLORS[fam])
    lr_axis(a[0,0], sorted(rmain.lr.unique()))
    a[0,0].axhline(74.5, ls='--', color='#555', lw=1.5, label='74.5% descriptive reference')
    a[0,0].legend(ncol=3, fontsize=10, frameon=False)
    finish(fig, '02_resnet_lr_grid', 'ResNet18: the complete three-seed LR grid', 'All five tested LRs and five optimizer packages. Points are means; error bars are sample SD across seeds 42-44. The advisor-requested dashed reference is descriptive, not a statistical significance boundary.', 'Package ranking depends on LR; no family wins at every LR.', 'main', 'ResNet18 / CIFAR-100. Cosine LR. beta=0.95 except joint schedule. MARS gamma=0.5; Mu2 alpha=0.2.', AUX_NOTE)

    tmain = summary[(summary.block == 'tiny_lr_robustness') & summary.family.isin(['Muon','MARS','MARS + Mu2']) & summary.lr.le(.01)]
    fig, a = axes()
    for fam in ['Muon','MARS','MARS + Mu2']:
        sweep(a[0,0], tmain[tmain.family.eq(fam)], name=fam, color=COLORS[fam])
    lr_axis(a[0,0], [.005,.01], ylabel='Epoch-50 validation accuracy (%)')
    a[0,0].axhline(74.5, ls='--', color='#555', lw=1.5, label='74.5% descriptive reference')
    a[0,0].legend(frameon=False, fontsize=10)
    finish(fig, '03_tinyvit_lr_confirmation', 'TinyViT-5M: LR sensitivity across three seeds', 'Three-seed confirmations at the two selected LRs. The commissioned seed-42 rerun (67.30%) replaces the superseded 10.56% snapshot only for Muon LR 0.01. The original remains in provenance. The advisor-requested 74.5% reference is descriptive, not a published standard accuracy.', 'At LR 0.01, MARS retains more accuracy than the historical baseline package; adding Mu2 is not an improvement over MARS.', 'main', 'Kaggle ImageNet-100. beta=0.95; MARS gamma=0.35; combined alpha=0.2. Means +/- sample SD.', AUX_NOTE)

    fig, a = axes(2)
    for i, policy in enumerate(['constant','cosine']):
        endpoint(a[0,i], df, [(fc(policy,False,'constant'),'Constant beta\n0.98'), (fc(policy,False,'late'),'Late decay\n0.98 to 0.50')], ['#0072B2','#CC79A7'], ylabel='Epoch-50 validation accuracy (%)')
        a[0,i].set_title(f'{policy.capitalize()} LR')
        a[0,i].set_ylim(83,86)
    finish(fig, '04_matched_momentum_endpoints', 'Late momentum decay helps the constant-LR endpoint', 'Same initial LR 0.0025 and fixed auxiliary LR in both LR policies. Seed dots plus mean +/- sample SD. Constant LR: +1.58 +/- 0.33 pp paired-label difference. Cosine LR: -0.31 +/- 0.31 pp.', 'The endpoint effect changes with the outer LR policy.', 'main', 'TinyViT-5M / ImageNet-100. No MARS or Mu2. Late beta transition epochs 36-48. n=3.', FIXED_NOTE)

    for mars, id in [(False,'05_plain_matched_learning_curves'), (True,'06_mars_matched_learning_curves')]:
        fig, a = axes(2)
        for i, policy in enumerate(['constant','cosine']):
            learning(a[0,i], curves, [(fc(policy,mars,'constant'),'Constant beta 0.98'), (fc(policy,mars,'late'),'Late beta decay')], ['#0072B2','#CC79A7'], shade=(36,48), ylabel='Validation accuracy (%)')
            a[0,i].set_title(f'{policy.capitalize()} LR')
            a[0,i].set_ylim(50,87)
        finish(fig, id, ('MARS: ' if mars else '') + 'Momentum timing changes the learning trajectory', 'Means with +/- sample-SD bands at the ten observed evaluations. Shaded epochs 36-48 are the momentum transition window; curves connect observations, not interpolated measurements.', 'Distinguish an earlier threshold crossing from the epoch-50 endpoint.', 'main', 'TinyViT-5M / ImageNet-100. Initial LR 0.0025. ' + ('MARS gamma=0.35. ' if mars else 'No direction correction. ') + 'n=3 per cell.', FIXED_NOTE)

    contrast = pd.read_csv(MECH / 'matched_seed_contrasts.csv')
    fig, a = axes(2)
    for i, policy in enumerate(['constant','cosine']):
        for j, mars in enumerate([0,1]):
            g = contrast[contrast.contrast.eq(f'late_minus_constant_beta__{policy}_lr__mars{mars}')].sort_values('seed')
            a[0,i].scatter(j+np.linspace(-.06,.06,3), g.difference_pp, color=['#0072B2','#D55E00'][j], alpha=.65)
            a[0,i].errorbar(j,g.difference_pp.mean(),yerr=g.difference_pp.std(),fmt='_',ms=18,capsize=4,color=['#0072B2','#D55E00'][j])
        a[0,i].axhline(0,color='#555',ls='--')
        a[0,i].set(xticks=[0,1],xticklabels=['Plain Muon','MARS'],ylabel='Late minus constant beta (pp)',title=f'{policy.capitalize()} LR',ylim=(-1,3))
    finish(fig, '07_mars_momentum_interaction', 'MARS and momentum decay have a conditional interaction', 'Matched-label differences across seeds 42-44. Constant LR gains: plain +1.58 pp, MARS +2.34 pp. Cosine LR gains: plain -0.31 pp, MARS -0.54 pp. The MARS increment after late decay is only +0.18 +/- 0.29 pp under constant LR.', 'The combination is useful in this constant-LR setting, but there is no universal additive win.', 'main', 'Full 2 x 2 x 2 factorial: LR policy, MARS gamma 0 or 0.35, constant/late beta. Initial LR 0.0025. n=3.', FIXED_NOTE)

    times = pd.read_csv(MECH / 'time_to_threshold_seed.csv')
    fig, a = axes(2)
    for i, policy in enumerate(['constant','cosine']):
        labels = []
        for j,(mars,mom) in enumerate([(False,'constant'),(False,'late'),(True,'constant'),(True,'late')]):
            g = times[(times.condition == fc(policy,mars,mom)) & (times.threshold_pct == 84)].sort_values('seed')
            y = g.first_observed_epoch.fillna(54)
            for k,(_, r) in enumerate(g.iterrows()):
                a[0,i].scatter(j+(k-1)*.08, y.iloc[k], marker='o' if pd.notna(r.first_observed_epoch) else 'x', color='#D55E00' if mars else '#0072B2', s=50)
            labels.append(f'{"MARS" if mars else "Plain"}\n{mom}\n{g.first_observed_epoch.notna().sum()}/3 reached')
        a[0,i].set(xticks=range(4),xticklabels=labels,yticks=[30,35,40,45,50,54],yticklabels=['30','35','40','45','50','Not reached'],ylabel='First observed epoch at 84%',title=f'{policy.capitalize()} LR',ylim=(28,56))
        a[0,i].grid(axis='y',alpha=.2)
    finish(fig, '08_threshold_crossing', 'Earlier accuracy targets and better final accuracy are different', 'Every seed is shown. Crosses at the labelled top row did not reach 84% by epoch 50; they are not assigned an artificial crossing time. The selected 84% target is descriptive and crossings are resolved only every five epochs.', 'Late decay can cross 84% sooner under cosine LR even though its final accuracy is lower.', 'main', 'TinyViT-5M matched factorial, LR 0.0025; n=3. Epoch efficiency, not wall-clock speed.', FIXED_NOTE)

    surface = pd.read_csv(MECH / 'mu2_fixed_surface_seed42.csv')
    groups = [('cifar100',.08,'ResNet18, LR 0.08'), ('imagenet100',.005,'TinyViT, LR 0.005'), ('imagenet100',.01,'TinyViT, LR 0.01')]
    fig, a = axes(3,height=5)
    for i,(dataset,lr,title) in enumerate(groups):
        g = surface[(surface.dataset == dataset) & surface.lr.eq(lr)]
        table = g.pivot(index='beta',columns='alpha',values='accuracy_pct').sort_index()
        delta = table.subtract(table[0.],axis=0)
        im = heat(a[0,i],delta,title,delta=True,vmax=85)
        a[0,i].set(xlabel='Mu2 alpha',ylabel='Momentum beta')
        fig.colorbar(im,ax=a[0,i],fraction=.04,pad=.04,label='Difference from alpha=0 (pp)')
    finish(fig, '09_mu2_matched_alpha_deltas', 'Mu2 coefficient sweeps relative to the same-momentum control', 'All 78 fixed beta-alpha cells, seed 42. Each entry subtracts alpha=0 in the same row, avoiding comparison across momentum settings. Common scale retains severe failures. Independent nondeterministic runs differ even in the beta-zero no-op row.', 'Mu2 is not erased by Muon, but provides no consistent gain; high-memory stressed-LR cells deteriorate.', 'main', 'Fixed auxiliary LR. alpha in {0,0.05,0.1,0.2,0.35,0.5}; ResNet beta {0,0.5,0.9,0.95,0.98}; Tiny beta {0.5,0.9,0.95,0.98}. n=1 exploratory.', 'No universal negative result for full Mu2SGD. This is the approved Mu2-inspired integration, not a new gradient evaluated at the previous iterate.')

    stress = md[md.block.isin(['stressed_lr_mechanism_confirmation','mu2_diagnostic']) & md.dataset.eq('imagenet100') & md.lr.eq(.01) & md.beta.isin([.5,.9,.95]) & md.alpha.isin([0,.2]) & md.gamma.eq(0) & md.alpha_schedule.eq('none') & ~md.preserve_estimator]
    fig, a = axes()
    for i,alpha in enumerate([0,.2]):
        sub = stress[stress.alpha.eq(alpha)]
        st = sub.groupby('beta').accuracy_pct.agg(mean='mean',sample_sd='std').reset_index()
        sweep(a[0,0],st,x='beta',name=f'alpha={alpha:g}',color=['#0072B2','#009E73'][i])
        for _,r in sub.iterrows():
            a[0,0].scatter(r.beta+(r.seed-43)*.003, r.accuracy_pct, color=['#0072B2','#009E73'][i],alpha=.35,s=22)
    a[0,0].set(xticks=[.5,.9,.95],xlabel='Momentum beta',ylabel='Epoch-50 validation accuracy (%)',ylim=(0,90))
    a[0,0].legend(frameon=False)
    finish(fig, '10_mu2_stress_confirmation', 'Stressed LR: momentum tuning matters more than our history correction', 'Seed dots and mean +/- sample SD, three seeds per cell. At beta 0.5 and 0.9, alpha 0.2 changes accuracy by -0.03 pp on average. At beta 0.95 it changes 36.32 +/- 21.40% to 5.39 +/- 1.74%. Finite, poorly convergent runs are retained.', 'Our tested history correction does not rescue the stressed high-memory regime.', 'main', 'TinyViT-5M / ImageNet-100. Cosine LR 0.01. Fixed auxiliary LR. n=3.', FIXED_NOTE)

    tuned = md[md.block.eq('stressed_lr_mechanism_confirmation') & md.gamma.gt(0)].condition.iloc[0]
    plain = stress[stress.beta.eq(.9) & stress.alpha.eq(0)].condition.iloc[0]
    fig, a = axes(2)
    endpoint(a[0,0],df,[(plain,'Tuned plain\nbeta 0.90'),(tuned,'Tuned MARS\nbeta 0.90')],['#0072B2','#D55E00'], ylabel='Epoch-50 validation accuracy (%)')
    a[0,0].set_ylim(83.5,85)
    g1 = df[df.condition.eq(plain)].set_index('seed').accuracy_pct
    g2 = df[df.condition.eq(tuned)].set_index('seed').accuracy_pct
    difference = (g2-g1).sort_index()
    a[0,1].scatter(difference.index.astype(str),difference,color='#D55E00',s=65)
    a[0,1].axhline(0,color='#555',ls='--')
    a[0,1].set(xlabel='Seed',ylabel='MARS minus tuned plain (pp)',ylim=(-.1,.9))
    finish(fig, '11_tuned_mars_increment', 'MARS adds a modest gain after tuning momentum', 'At the same tuned beta 0.90, MARS reaches 84.46 +/- 0.18%, versus 84.12 +/- 0.28% for the alpha-zero control. Matched-label increments are +0.12, +0.14, +0.76 pp, mean +0.34 +/- 0.36 pp.', 'Report the modest residual gain, separately from the much larger gain due to reducing beta.', 'main', 'TinyViT-5M / ImageNet-100. Cosine LR 0.01. MARS gamma=0.35. n=3.', 'Fixed auxiliary LR in both classes, but loader/hardware regimes and floating-point implementations are not bitwise identical.')

    gamma = pd.read_csv(MECH / 'mars_gamma_curve_seed42.csv')
    fig, a = axes()
    positive = gamma[gamma.gamma.gt(0)]
    a[0,0].plot(positive.gamma,positive.accuracy_pct,'o-',color=COLORS['MARS'],label='Positive gamma: fixed auxiliary LR')
    zero = gamma[gamma.gamma.eq(0)]
    a[0,0].scatter(zero.gamma,zero.accuracy_pct,marker='x',s=90,color='#555',label='Historical zero: scheduled auxiliary LR')
    a[0,0].set(xlabel='MARS gamma',ylabel='Epoch-50 validation accuracy (%)',xticks=gamma.gamma,ylim=(60,87))
    a[0,0].legend(frameon=False,fontsize=10)
    finish(fig, 'A01_tinyvit_mars_gamma', 'TinyViT: the complete MARS gamma probe', 'Seed-42 positive-gamma probes at 0.10, 0.20, 0.35, 0.50, 0.75. The zero anchor uses a different auxiliary LR trajectory and is deliberately not connected to the positive-gamma line.', 'Larger tested gamma improves this exploratory stressed-LR sweep; the optimum and a continuous robustness interval are not established.', settings='TinyViT-5M, cosine LR 0.01, beta 0.95. n=1.',caveat=AUX_NOTE)

    comp = summary[summary.block.eq('resnet_component')]
    for fam,id,x in [('MARS','A02_resnet_mars_gamma','gamma'),('Mu2','A03_resnet_mu2_alpha','alpha')]:
        fig,a=axes(2)
        for i,lr in enumerate([.04,.08]):
            g=comp[comp.lr.eq(lr) & comp.family.eq(fam)]
            sweep(a[0,i],g,x=x,name=fam,color=COLORS[fam])
            base=comp[comp.lr.eq(lr)&comp.family.eq('Muon')].iloc[0]
            a[0,i].axhline(base['mean'],color=COLORS['Muon'],ls='--',label=f'Plain baseline (n={int(base.n)})')
            for _,r in g.iterrows():
                a[0,i].annotate(f'n={int(r.n)}',(r[x],r['mean']),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
            a[0,i].set(xlabel=f'{fam} {x}',ylabel='Epoch-50 test accuracy (%)',title=f'LR {lr:g}',ylim=(71,76))
            a[0,i].legend(frameon=False,fontsize=9)
        finish(fig,id,f'ResNet18: all historical {fam} coefficient probes','All tested coefficient values at each LR, with available replication labelled per point. Error bars are sample SD only where n=3. These are selected post-sweep confirmations, not independent held-out tuning evaluation.', 'Keep the historical positive cells visible alongside the newer diagnostic controls.',settings='ResNet18 / CIFAR-100. Cosine LR. beta 0.95.',caveat=AUX_NOTE)

    alg = summary[summary.block.eq('resnet_algmom')].copy()
    alg['policy'] = np.where(alg.beta_schedule.eq('none'),alg.beta.map(lambda b:f'Constant {b:g}'),'Early 0.98 to 0.50')
    fig,a=axes(2)
    for i,lr in enumerate([.04,.08]):
        g=alg[alg.lr.eq(lr)].copy()
        g['cell']=g['mean'].map(lambda x:f'{x:.2f}')+' (n='+g.n.astype(str)+')'
        table=g.pivot(index='policy',columns='family',values='mean').reindex(columns=FAMILIES[:4])
        im=heat(a[0,i],table,f'LR {lr:g}',vmin=70,vmax=76,annotate=False)
        for y,idx in enumerate(table.index):
            for x,col in enumerate(table.columns):
                r=g[(g.policy==idx)&(g.family==col)].iloc[0]
                a[0,i].text(x,y,f'{r["mean"]:.2f}\nn={int(r.n)}',ha='center',va='center',fontsize=10,color='white' if r['mean']<73 else 'black')
        a[0,i].tick_params(axis='x',labelsize=9)
        fig.colorbar(im,ax=a[0,i],fraction=.04,pad=.04,label='Test accuracy (%)')
    finish(fig,'A04_resnet_full_factorial','ResNet18: all 24 direction-by-momentum cells','Every historical factorial cell is present. Selected cells have three seeds, the rest one seed; the number is printed in each cell. A common accuracy scale is used across LRs.','Combinations and momentum policies do not produce a universal ranking.',settings='Cosine LR 0.04 or 0.08; gamma 0 or 0.5; alpha 0 or 0.2; constant beta 0.5/0.98 or early 0.98 to 0.50 epochs 6-18.',caveat=FIXED_NOTE)

    fig,a=axes(2)
    for i,lr in enumerate([.04,.08]):
        g=alg[alg.lr.eq(lr)&alg.family.eq('Muon')].sort_values('beta',ascending=True)
        g=pd.concat([g[g.beta_schedule.eq('none')],g[~g.beta_schedule.eq('none')]])
        endpoint(a[0,i],df,[(r.condition,r.policy.replace('Constant ','Constant\nbeta ').replace('Early ','Early decay\n')) for _,r in g.iterrows()],['#0072B2','#777','#CC79A7'])
        a[0,i].set(title=f'LR {lr:g}',ylim=(71.5,75.5))
    finish(fig,'A05_resnet_momentum_confirmation','ResNet18: low, high, and decayed momentum confirmations','Three seeds per displayed cell. At LR 0.08, constant beta 0.50 reaches 74.26 +/- 0.43%, constant beta 0.98 reaches 73.36 +/- 0.26%, early decay reaches 72.54 +/- 0.26%.','Early momentum decay is not a universal endpoint improvement.',settings='Plain direction within the same phase-decay implementation. Cosine LR; epochs 6-18 transition.',caveat=FIXED_NOTE)

    rc=summary[summary.block.eq('resnet_constant_lr')]
    fig,a=axes()
    for fam in FAMILIES:
        g=rc[rc.family.eq(fam)]
        sweep(a[0,0],g,name=fam,color=COLORS[fam])
        for _,r in g.iterrows():
            if r.n==3: a[0,0].annotate('n=3',(r.lr,r['mean']),xytext=(3,5),textcoords='offset points',fontsize=8)
    lr_axis(a[0,0],sorted(rc.lr.unique()))
    a[0,0].legend(ncol=3,frameon=False,fontsize=10)
    finish(fig,'A06_resnet_constant_lr_grid','ResNet18: every constant-LR control','All 20 cells. Unannotated points are seed 42 only; the two selected confirmations are labelled n=3 and have sample-SD bars. Joint schedule changes direction coefficients and momentum, not outer LR.','Constant-LR comparisons remain a separate block rather than being mixed with cosine-LR optima.',settings='LR {0.005,0.01,0.02,0.04}. Joint gamma 0.5 to 0.1, alpha 0.2 to 0.05, beta 0.95 to 0.5 epochs 6-18.',caveat='All auxiliary LRs are constant here; only selected cells were replicated.')
    fig,a=axes(2)
    for i,lr in enumerate([.01,.04]):
        g=rc[rc.lr.eq(lr)]
        learning(a[0,i],curves,[(g[g.family.eq(fam)].condition.iloc[0],fam) for fam in FAMILIES])
        a[0,i].set(title=f'Constant LR {lr:g}',ylim=(55,75))
    finish(fig,'A07_resnet_constant_lr_curves','ResNet18: constant-LR trajectories at two disclosed LRs','Display LRs 0.01 and 0.04 because they include the selected joint-schedule and plain confirmations. Other trajectories remain in the tidy learning-curve data. Sample-SD bands appear only for replicated cells.','Different winning LRs are not an isolated momentum-decay effect.',settings='Five families; 50 epochs. All 20 condition curves supplied in data/learning_curves_seed.csv.')

    tm=summary[summary.block.isin(['tiny_momentum_cosine_lr','tiny_momentum_constant_lr'])].copy()
    tm['policy_label']=[f'Const {r.beta:g}' if r.beta_schedule=='none' else f'{r.beta_schedule[:3]} {r.beta:g} to {r.beta_end:g}, {int(r.beta_start_epoch)}-{int(r.beta_end_epoch)}' for _,r in tm.iterrows()]
    for policy,lr,id in [('cosine',.005,'A08_tinyvit_cosine_13_policies'),('constant',.0025,'A09_tinyvit_constant0025_13_policies'),('constant',.005,'A10_tinyvit_constant005_13_policies')]:
        g=tm[tm.lr_policy.eq(policy)&tm.lr.eq(lr)].sort_values(['beta_schedule','beta_end','beta_start_epoch','beta'])
        assert len(g)==13
        fig,a=axes(width=11,height=7)
        ax=a[0,0]
        ax.errorbar(g['mean'],range(13),xerr=g.sample_sd.fillna(0),fmt='o',color='#0072B2',capsize=4)
        ax.set(yticks=range(13),yticklabels=[f'{r.policy_label} (n={int(r.n)})' for _,r in g.iterrows()],xlabel='Epoch-50 validation accuracy (%)',title=f'{policy.capitalize()} LR {lr:g}')
        ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
        finish(fig,id,'TinyViT: the complete momentum timing and endpoint sweep','All 13 policies in this LR block, including constant beta 0.50/0.95/0.98, four windows for cosine decay to 0.50 or 0.80, and two late linear transitions to 0.80. Seed counts are shown per policy.','This sweep establishes timing sensitivity, not a universal best schedule.',settings=f'{policy.capitalize()} LR {lr:g}. Plain momentum-decay implementation. Exploratory seed 42 except labelled confirmations.',caveat='Do not compare different initial LRs to identify the effect of LR policy; use the matched factorial in the main figures.')

    early=tm[tm.lr_policy.eq('cosine')&tm.lr.eq(.005)]
    fig,a=axes(2)
    labels=[]
    for beta in [.5,.98]:
        labels.append((early[early.beta.eq(beta)&early.beta_schedule.eq('none')].condition.iloc[0],f'Constant beta {beta:g}'))
    labels.append((early[early.beta_end.eq(.5)&early.beta_start_epoch.eq(6)&early.beta_schedule.eq('cosine')].condition.iloc[0],'Early beta decay 6-18'))
    learning(a[0,0],curves,labels,['#777','#0072B2','#CC79A7'],shade=(6,18),ylabel='Validation accuracy (%)')
    learning(a[0,1],curves,labels,['#777','#0072B2','#CC79A7'],shade=(6,18),ylabel='Validation accuracy (%)')
    a[0,1].set_xlim(5,20); a[0,1].set_ylim(40,77); a[0,1].set_title('Early-epoch detail')
    finish(fig,'A11_tinyvit_early_transient','TinyViT: an initial jump does not guarantee a better endpoint','Seed-42 cosine-LR timing comparison. Relative to constant beta 0.98, the early transition gains 2.06 pp at epoch 10, loses 1.48 pp at epoch 15, and loses 8.86 pp at epoch 50.','Show the whole trajectory alongside the early zoom.',settings='LR 0.005. beta 0.98 to 0.50 epochs 6-18. n=1.')

    fulltiny=summary[summary.block.eq('tiny_lr_robustness')&summary.lr.le(.02)]
    fig,a=axes()
    for fam in FAMILIES:
        g=fulltiny[fulltiny.family.eq(fam)]
        g=g.sort_values('lr')
        a[0,0].plot(g.lr,g['mean'],color=COLORS[fam],label=fam)
        for n, marker in [(1,'x'),(3,'o')]:
            sub=g[g.n.eq(n)]
            a[0,0].errorbar(sub.lr,sub['mean'],yerr=sub.sample_sd.fillna(0),fmt=marker,color=COLORS[fam],capsize=3)
    lr_axis(a[0,0],[.005,.01,.02], ylabel='Epoch-50 validation accuracy (%)'); a[0,0].set_ylim(0,90)
    a[0,0].legend(ncol=3,frameon=False,fontsize=9)
    a[0,0].text(.03,.14,'Circles: n=3; crosses: n=1\nError bars: sample SD',transform=a[0,0].transAxes,fontsize=10)
    finish(fig,'A12_tinyvit_all_families_lr','TinyViT: all five historical packages up to LR 0.02','The advisor-facing plot stops at 0.02. Archived 0.04/0.08 cells remain in the data coverage only, not extrapolated into a robustness interval. Circles show n=3 and crosses show n=1; error bars are sample SD.','Retain near-chance outcomes rather than reporting only successful settings.',settings='beta 0.95 except the early joint schedule; gamma 0.35; alpha 0.2.',caveat=AUX_NOTE)

    fig,a=axes(3,height=5)
    for i,(dataset,lr,title) in enumerate(groups):
        g=surface[surface.dataset.eq(dataset)&surface.lr.eq(lr)]
        table=g.pivot(index='beta',columns='alpha',values='accuracy_pct').sort_index()
        im=heat(a[0,i],table,title,vmin=0,vmax=90)
        accuracy_label = 'Validation accuracy (%)' if dataset == 'imagenet100' else 'Test accuracy (%)'
        a[0,i].set(xlabel='Mu2 alpha',ylabel='Momentum beta'); fig.colorbar(im,ax=a[0,i],fraction=.04,pad=.04,label=accuracy_label)
    finish(fig,'A13_mu2_absolute_surfaces','Mu2: absolute performance across all 78 fixed-coefficient cells','Identical absolute color scale across the three blocks, with numeric annotations. Each block uses one architecture-dataset pair and a stated LR; cross-block absolute accuracy is not a leaderboard.','Low beta resolves the TinyViT stressed-LR failures without needing a secondary correction.',settings='All fixed beta-alpha exploratory cells. Seed 42. Approved effective recurrence.')

    controls=pd.read_csv(MECH/'mu2_schedule_literal_controls_seed42.csv')
    for literal,id in [(False,'A14_mu2_alpha_schedule_endpoints'),(True,'A15_mu2_literal_state_controls')]:
        fig,a=axes(3,height=5.5)
        for i,(dataset,lr,title) in enumerate(groups):
            g=controls[controls.dataset.eq(dataset)&controls.lr.eq(lr)&controls.preserve_estimator.eq(literal)]
            baseline=surface[surface.dataset.eq(dataset)&surface.lr.eq(lr)&surface.beta.eq(.95)&surface.alpha.eq(0)].iloc[0]
            labels=[(baseline.condition,'alpha=0')]
            if literal:
                labels += [(r.condition,f'Literal\nalpha {r.alpha:g}') for _,r in g.iterrows()]
            else:
                labels += [(r.condition, 'Early fall' if r.alpha>r.alpha_end and r.alpha_start_epoch<10 else 'Late fall' if r.alpha>r.alpha_end else 'Early rise') for _,r in g.iterrows()]
            ylabel = 'Epoch-50 validation accuracy (%)' if dataset == 'imagenet100' else 'Epoch-50 test accuracy (%)'
            endpoint(a[0,i],df[df.seed.eq(42)],labels,ylabel=ylabel)
            a[0,i].set_title(title); a[0,i].tick_params(axis='x',labelsize=9)
        finish(fig,id,'Mu2: '+('literal-state diagnostic controls' if literal else 'all tested alpha schedules'), 'All five diagnostic controls per block are split into alpha-schedule and literal-state panels. Every displayed control is seed 42 only. The literal-state recurrence is an algebraic no-op when correctly initialized; different nondeterministic runs need not have identical curves.', 'The stressed beta 0.95 setting is not reliably rescued by these schedules.' if not literal else 'The literal control is an implementation diagnostic, not a new competitive optimizer.',settings='beta 0.95. Early alpha 0.20 to 0.05 or 0.05 to 0.20 epochs 6-18; late 0.20 to 0.05 epochs 36-48. Literal alpha 0.2/0.5.',caveat='This recurrence uses consecutive minibatch gradients and stored prepolar state; it is not the complete original Mu2SGD method.')

    diag=pd.read_csv(MECH/'diagnostic_means_first_batch_e2_e50.csv')
    joined=surface.merge(diag,on=['condition','seed'],validate='one_to_one')
    fig,a=axes(2)
    for i,metric in enumerate(['Prepolar Counterfactual Angle Degrees','Postpolar Counterfactual Angle Degrees']):
        for dataset,lr,title in groups:
            g=joined[joined.dataset.eq(dataset)&joined.lr.eq(lr)&joined.beta.eq(.95)].sort_values('alpha')
            a[0,i].plot(g.alpha,g[metric],marker='o',label=title)
        a[0,i].set(xlabel='Mu2 alpha',ylabel='Counterfactual angle (degrees)',title='Before orthogonalization' if i==0 else 'After orthogonalization')
        a[0,i].legend(frameon=False,fontsize=9)
    finish(fig,'A16_mu2_direction_diagnostics','Our history correction changes the direction passed to Muon','Diagnostic means from the first minibatch of each epoch, epochs 2-50. Prepolar and postpolar counterfactual angles are nonzero at positive alpha. These sparse aggregates are not a direct gradient-variance estimate.','Muon does not simply erase our history correction; lack of benefit is not explained by a zero directional effect.',settings='beta 0.95; alpha sweep; seed 42. Diagnostics sampled once per epoch.',caveat='Counterfactual directions are local diagnostics, not causal evidence for variance reduction or a proof of the failure mechanism.')

    fig,a=axes(3,height=5)
    for i,beta in enumerate([.5,.9,.95]):
        labels=[(stress[stress.beta.eq(beta)&stress.alpha.eq(alpha)].condition.iloc[0],f'alpha {alpha:g}') for alpha in [0,.2]]
        learning(a[0,i],curves,labels,['#0072B2','#009E73'],ylabel='Validation accuracy (%)'); a[0,i].set(title=f'beta {beta:g}',ylim=(0,90))
    finish(fig,'A17_mu2_stress_learning_curves','Mu2: three-seed stress trajectories at each momentum setting','All six confirmed beta-alpha cells, mean +/- sample-SD bands. Different panes share an accuracy scale and the same LR.','The high-beta regime fails throughout training; the lower-beta controls converge without Mu2.',settings='TinyViT-5M / ImageNet-100. Cosine LR 0.01. n=3.',caveat=FIXED_NOTE)

    fig,a=axes(2)
    schlabels=[(r.condition,'Early fall' if r.alpha>r.alpha_end and r.alpha_start_epoch<10 else 'Late fall' if r.alpha>r.alpha_end else 'Early rise') for _,r in controls[controls.dataset.eq('imagenet100')&controls.lr.eq(.01)&~controls.preserve_estimator].iterrows()]
    fixed=surface[surface.dataset.eq('imagenet100')&surface.lr.eq(.01)&surface.beta.eq(.95)&surface.alpha.eq(.2)].condition.iloc[0]
    learning(a[0,0],curves[curves.seed.eq(42)],[(fixed,'Fixed alpha 0.2')]+schlabels,ylabel='Validation accuracy (%)')
    for label,start,end,init,final in [('Early fall',6,18,.2,.05),('Late fall',36,48,.2,.05),('Early rise',6,18,.05,.2)]:
        t=np.clip((e-start)/(end-start),0,1)
        a[0,1].plot(e,final+(init-final)*(1+np.cos(np.pi*t))/2,label=label)
    a[0,1].set(xlabel='Epoch',ylabel='Mu2 alpha',title='Actual tested schedules'); a[0,1].legend(frameon=False)
    finish(fig,'A18_mu2_schedule_learning_curves','Mu2: schedules do not rescue this stressed setting','Seed-42 trajectories for the fixed-alpha and three scheduled-alpha controls, alongside their coefficient functions. No exponential schedule is claimed to have been used.','Changing alpha timing is insufficient in the tested beta 0.95 / LR 0.01 regime.',settings='TinyViT-5M / ImageNet-100. Cosine outer LR 0.01. n=1.')

    poles=pd.read_csv(MECH/'mu2_prepolar_memory_poles.csv')
    fig,a=axes()
    for beta,g in poles.groupby('beta'):
        a[0,0].plot(g.alpha,g.dominant_linear_memory_pole,marker='o',label=f'beta {beta:g}')
    a[0,0].set(xlabel='Mu2 alpha',ylabel='Dominant frozen-linear memory pole',ylim=(0,1.01)); a[0,0].axhline(1,color='#555',ls='--'); a[0,0].legend(ncol=3,frameon=False)
    finish(fig,'A19_mu2_linear_memory_analysis','Mu2: a limited linear-memory interpretation','Analytic poles of A=[[beta, alpha(1-beta)], [beta^2, alpha(1-beta^2)]], freezing the coefficient and external gradient sequence. This describes the stored prepolar recurrence, not the nonlinear training dynamics.','Positive alpha extends effective memory in this linear model, but the poles do not prove instability of the full algorithm.',settings='Derived quantity, not an empirical experiment.',caveat='Not a convergence theorem for Muon, neural networks, or stochastic gradients. No causal attribution to regularization follows.')

    fig,a=axes(2)
    for i,lrs in enumerate([[.005,.01],[.04,.08]]):
        for lr in lrs:
            floor=.05*lr
            aux=floor+(.001-floor)*(1+np.cos(np.pi*(e-1)/50))/2
            a[0,i].plot(e,aux,label=f'Historical plain, LR {lr:g}')
        a[0,i].axhline(.001,color='#555',ls='--',label='Correction variants: fixed 0.001')
        a[0,i].set(xlabel='Epoch',ylabel='Auxiliary LR',title='TinyViT historical selected LRs' if i==0 else 'ResNet18 historical high LRs')
        a[0,i].legend(frameon=False,fontsize=9)
    finish(fig,'A20_auxiliary_lr_audit','Why some historical comparisons are package comparisons','Derived from the current scheduler implementation and recorded historical configurations. The baseline exposes both parameter groups to CosineAnnealingLR with one absolute eta_min. Internal auxiliary AdamW paths remain at 0.001.','Same initial auxiliary LR does not imply the same auxiliary LR trajectory.',settings='Common absolute eta_min=0.05 * initial Muon LR. Scheduler steps after each epoch.',caveat='New matched factorial and Mu2 diagnostic comparisons hold the auxiliary LR fixed. These older trajectories limit isolated mechanism attribution.')

    oldc=pd.read_csv(ROOT/'analysis/tinyvit_imagenet100_summary/snapshot/tinyvit_imagenet100_curves.csv')
    oldc=oldc[oldc.experiment.eq('tinyvit5m_imagenet100_lrrobust_muon_lr0p01')]
    fig,a=axes(2)
    for i,metric in enumerate(['accuracy_pct','test_loss']):
        y=oldc.test_accuracy*100 if metric=='accuracy_pct' else oldc.test_loss
        a[0,i].plot(oldc.epoch,y,color='#777',ls='--',label='Superseded seed-42 attempt')
        learning(a[0,i],curves[curves.seed.eq(42)], [('tinyvit5m_imagenet100_lrrobust_muon_lr0p01','Commissioned seed-42 rerun')],['#0072B2'],metric=metric,ylabel='Validation accuracy (%)' if metric == 'accuracy_pct' else 'Validation cross-entropy loss')
    finish(fig,'A21_seed42_repeat_provenance','Seed-42 repeat: preserve the failed-convergence attempt as provenance','Original finite endpoint 10.56%, commissioned rerun 67.30%. Only the rerun enters the current three-seed summary; the original is supplied separately, not treated as another independent seed.','A same-seed repeat need not reproduce a nondeterministic trajectory.',settings='TinyViT-5M, Muon LR 0.01. Archived source tree was not immutable.',caveat='This pair cannot distinguish nondeterministic numerical effects from all historical source/environment differences, or establish a specific causal failure mechanism.')

    fig,a=axes(2)
    for i,policy in enumerate(['constant','cosine']):
        for j,(mars,mom) in enumerate([(False,'constant'),(False,'late'),(True,'constant'),(True,'late')]):
            g=df[df.condition.eq(fc(policy,mars,mom))]
            values=g.accuracy_auc_e5_e50_pct
            a[0,i].errorbar(j,values.mean(),yerr=values.std(),fmt='o',color='#D55E00' if mars else '#0072B2',capsize=4)
            a[0,i].scatter(j+np.linspace(-.06,.06,3),values,color='#D55E00' if mars else '#0072B2',alpha=.4)
        a[0,i].set(xticks=range(4),xticklabels=['Plain\nconstant','Plain\nlate','MARS\nconstant','MARS\nlate'],ylabel='Normalized accuracy AUC (%)',title=f'{policy.capitalize()} LR')
        a[0,i].grid(axis='y',alpha=.2)
    finish(fig,'A22_factorial_accuracy_auc','Trajectory summary: accuracy AUC is a secondary endpoint','Trapezoidal area over observed epochs 5-50 divided by 45, computed per seed before mean and sample SD. No unobserved epoch-0 accuracy is invented.','AUC, target crossings, and epoch-50 accuracy answer different questions.',settings='TinyViT-5M matched factorial. LR 0.0025. n=3. AUC expressed on an average-accuracy scale.',caveat='AUC is descriptive and excludes the first five epochs. It is not a wall-clock efficiency measurement.')

    fig,a=plt.subplots(2,3,figsize=(17,10))
    for ax,lr in zip(a.flat,sorted(rmain.lr.unique())):
        g=rmain[rmain.lr.eq(lr)]
        learning(ax,curves,[(g[g.family.eq(f)].condition.iloc[0],f) for f in FAMILIES]); ax.set(title=f'Cosine LR {lr:g}',ylim=(50,77))
    a.flat[-1].axis('off')
    finish(fig,'A23_resnet_full_grid_learning_curves','ResNet18: trajectories for all 25 primary cells','All primary families at all five LRs, mean +/- sample-SD bands from seeds 42-44. The shared scale prevents per-panel magnification of small differences.','Use endpoint plots for the concise comparison and this complete trajectory grid for backup questions.',settings='ResNet18 / CIFAR-100. n=3 per condition.',caveat=AUX_NOTE)

    fig,a=axes(width=13,height=7)
    labels=[('R18: ' if r.benchmark.startswith('ResNet') else 'Tiny: ')+r.block.removeprefix('resnet_').removeprefix('tiny_').replace('_',' ') for _,r in coverage.iterrows()]
    x=np.arange(len(coverage))
    one=coverage.conditions-coverage.replicated_conditions
    a[0,0].barh(x,one,color='#BBBBBB',label='Seed-42 exploration')
    a[0,0].barh(x,coverage.replicated_conditions,left=one,color='#0072B2',label='Three-seed conditions')
    for i,r in coverage.iterrows(): a[0,0].text(r.conditions+.4,i,str(int(r.conditions)),va='center',fontsize=10)
    a[0,0].set(yticks=x,yticklabels=labels,xlabel='Named conditions retained in each experiment block'); a[0,0].invert_yaxis(); a[0,0].legend(frameon=False)
    finish(fig,'A24_experiment_coverage','Project coverage: exploration and confirmation remain distinct','Counts describe named conditions retained in the provided result registry, not unique scientific configurations after deduplicating across independently named blocks. Archived high-LR cells are counted only as supplied data. Superseded technical failures and the old repeat are not confirmation seeds.','All substantive approved benchmark blocks are represented, without pretending every exploratory cell was replicated.',settings=f'{len(df)} retained seed records, {len(summary)} named conditions, only ResNet18/CIFAR-100 and real TinyViT-5M/ImageNet-100.',caveat='Approved workloads only. AdamW is auxiliary, not a standalone competing baseline. No new training jobs launched for this graph package.')
    assert len(FIGURES)==35, len(FIGURES)


def write_gallery(df, summary, coverage):
    index=pd.DataFrame(FIGURES)
    index.to_csv(OUT/'FIGURE_INDEX.csv',index=False)
    main=[f for f in FIGURES if f['tier']=='main']
    selected=['01_actual_schedules','02_resnet_lr_grid','03_tinyvit_lr_confirmation',
              '04_matched_momentum_endpoints','05_plain_matched_learning_curves',
              '07_mars_momentum_interaction','08_threshold_crossing',
              '09_mu2_matched_alpha_deltas','10_mu2_stress_confirmation','11_tuned_mars_increment']
    content='''# Muon project graph package - 17 September 2026

## What to show

For a 30-40 minute presentation, use the ten figures listed below after the background and setup. Figure 06 provides a useful extra trajectory slide if time permits. Keep the full sweeps as backup. This is a mechanism study, not a claim of universal optimizer superiority.

1. Actual LR and momentum schedules: define the independent controls.
2. ResNet18 complete LR grid: controlled primary package comparisons across all five LRs.
3. TinyViT LR confirmations: architecture-dataset transfer and stressed-LR sensitivity, with the auxiliary-policy caveat.
4. Matched momentum endpoints: isolate LR policy at initial LR 0.0025.
5. Matched plain learning curves: show when the momentum effect appears.
6. MARS-by-momentum contrasts: explain the conditional interaction, not automatic synergy.
7. Target crossings: distinguish speed in epochs from endpoint performance.
8. Full Mu2 alpha-zero differences: show the broad parameter search, not one unlucky coefficient.
9. Replicated Mu2 stress cells: low momentum already fixes most of the problem.
10. Tuned MARS increment: the remaining improvement is modest.

## Benchmarking convention

LR denotes the initial Muon-side matrix-parameter LR. Auxiliary AdamW starts at 0.001. Its trajectory is fixed in the matched mechanism blocks but differs in some older package comparisons; affected figures state this explicitly.

The primary metric is epoch-50 test accuracy, not best-checkpoint accuracy. Three-seed figures use seeds 42-44, means and sample SD (ddof=1), not confidence intervals. One-seed exploration is labelled n=1. Corrections are MARS-inspired and Mu2SGD-inspired implementations, not claims about the complete original algorithms. Same seed labels do not guarantee a common stochastic trajectory. Technical failures are excluded; finite nonconvergent outcomes are retained. The commissioned seed-42 repeat replaces only one superseded TinyViT Muon LR-0.01 observation, with both attempts preserved separately.

Targets and AUC are secondary descriptive measures. Targets are not a published standard accuracy and were not a held-out selection criterion. Target crossings are measured at five-epoch resolution; non-crossing seeds stay visibly censored. No wall-clock speedup is claimed because hardware and loader regimes differ. Test-set-driven exploratory tuning limits claims about an independently validated best method.

The actual schedules are cosine/linear transitions, not the exponential idealization in the separate theory note. Muon's approximate polar update reshapes singular values; it does not generally erase every directional difference. The approved Mu2 recurrence stores the Nesterov prepolar state, which the literal-state controls distinguish. Sparse direction diagnostics do not directly establish gradient variance reduction or a cause of collapse.

## Why these plots

The full LR grids, coefficient surfaces, and alpha-zero controls expose tuning sensitivity rather than comparing only selected winners. This follows the concerns in [Optimizer Benchmarking Needs to Account for Hyperparameter Tuning](https://arxiv.org/abs/1910.11758). Workload-specific views and retention of neutral results follow the motivation of [Descending through a Crowded Valley](https://proceedings.mlr.press/v139/schmidt21a.html). Threshold crossings are inspired by [AlgoPerf's time-to-result framing](https://arxiv.org/abs/2306.07179), but this project measures epochs, not its controlled-hardware wall-clock benchmark, and does not claim to implement AlgoPerf.

## Fixed scientific setup

Both workloads: from scratch, 50 epochs, batch 128, global norm clipping 1.0, matrix weight decay 0.0005, Nesterov enabled, five Newton-Schulz iterations. Tensors with ndim >= 2 use Muon, including convolution and classifier weights. Auxiliary AdamW starts at LR 0.001, betas (0.9,0.999), epsilon 1e-8, weight decay 0. Named condition settings and complete recorded/preset configuration sources are in data/condition_summary.csv and data/run_configurations.json. Preset provenance is explicitly distinguished from recorded per-run configs where older seed-42 raw configs are unavailable.

ResNet18 uses the CIFAR stem (3x3 stride 1, no initial max pool), CIFAR-100, crop/flip training augmentation. TinyViT uses the real TinyViT-5M topology, 100-class head (5,103,864 parameters; canonical 1000-class head approximately 5.4M), 224-pixel Kaggle ambityga ImageNet-100. ImageNet-100 training: Mixup 0.8, CutMix 1.0, probability 1, switch probability 0.5, label smoothing 0.1, RandAugment rand-m9-mstd0.5-inc1, random erasing 0.25, color jitter 0.4, crop scale (0.08,1), aspect (0.75,4/3), horizontal flip 0.5, bicubic interpolation, validation crop fraction 0.875, drop path 0. TF32/cuDNN benchmarking/nondeterministic kernels enabled; BF16 if supported otherwise FP32. Loader settings vary and are retained in each config, not silently declared identical.

Historical MARS coefficients: ResNet primary 0.5, probes 0.25/0.5/0.75; Tiny primary 0.35, extra probes 0.1/0.2/0.5/0.75. Historical Mu2 primary alpha 0.2; ResNet probes 0.1/0.2/0.35. The later beta-alpha diagnostics sweep alpha 0/0.05/0.1/0.2/0.35/0.5 and beta 0/0.5/0.9/0.95/0.98 for ResNet, without beta 0 for Tiny. Scheduled-alpha controls use early fall/rise epochs 6-18 and late fall epochs 36-48. Exact joint and momentum windows are retained in the per-condition metadata.

## Contents and coverage

PNG figures are suitable for slides, SVG and single-page PDF figures retain vector graphics. main/ contains 11 central figures, appendix/ contains 24 full-sweep and diagnostic figures. index.html is a local browsable gallery. The graph book is a convenience review copy; use the individual vector files for publication. Source data and original approved configs are retained; the current paper and its figures are not modified.

Counts below refer to named blocks, not unique configurations across repeated blocks. Archived TinyViT LRs 0.04/0.08 remain in CSV coverage only following the advisor's requested main-plot restriction. Only the two approved workloads are included.

'''
    content += f'Retained registry: {len(df)} seed records in {len(summary)} named conditions.\n\n'
    content += coverage.to_csv(index=False)+'\n## Figure inventory\n\n'
    for f in FIGURES:
        content+=f'### {f["id"]}: {f["title"]}\n\n{f["caption"]}\n\n{f["settings"]}\n\nTakeaway: {f["takeaway"]}\n\n'
        if f['caveat']: content+='Limit: '+f['caveat']+'\n\n'
    (OUT/'README.md').write_text(content,encoding='utf-8')
    (OUT/'PRESENTATION_SELECTION.md').write_text('# Recommended results sequence\n\n'+ '\n'.join(f'- [{id}](main/{id}.png)' for id in selected)+'\n\nUse A01-A24 as backup rather than giving every sweep a main-talk slide. Include A16 for the Mu2 directional explanation if time permits.\n',encoding='utf-8')
    cards=[]
    for f in FIGURES:
        cards.append(f'<article id="{f["id"]}"><h2>{html.escape(f["id"]+": "+f["title"])}</h2><img loading="lazy" src="{f["path"]}" alt="{html.escape(f["title"])}"><p>{html.escape(f["caption"])}</p><p><b>Settings:</b> {html.escape(f["settings"])}</p><p><b>Takeaway:</b> {html.escape(f["takeaway"])}</p><p class="limit">{html.escape(f["caveat"])}</p><p><a href="{f["path"]}">PNG</a> | <a href="{f["path"].replace(".png",".svg")}">SVG</a> | <a href="{f["path"].replace(".png",".pdf")}">Vector PDF</a></p></article>')
    nav=' '.join(f'<a href="#{f["id"]}">{f["id"]}</a>' for f in FIGURES)
    (OUT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Muon project graphs</title><style>body{font:17px Arial,sans-serif;color:#202735;max-width:1150px;margin:40px auto;padding:0 24px;background:#f7f8fa}h1{font-size:30px}h2{font-size:23px}article{background:white;margin:24px 0;padding:24px;border:1px solid #dde1e7}img{width:100%;height:auto}nav{line-height:2.2}nav a{margin-right:12px}a{color:#006398}.limit{color:#704b19}p{line-height:1.5}</style><h1>Muon project graphs</h1><p>17 September 2026. 11 main figures and 24 appendix figures. Points/bands are mean +/- sample SD where n=3; one-seed sweeps are labelled. Read the per-figure setup caveats before making mechanism claims.</p><p><a href="README.md">Full setup and interpretation</a> | <a href="PRESENTATION_SELECTION.md">Presentation selection</a> | <a href="FIGURE_INDEX.csv">Figure index</a></p><nav>'+nav+'</nav>'+''.join(cards)+'</html>',encoding='utf-8')


def graphbook():
    BOOK.parent.mkdir(parents=True,exist_ok=True)
    font='Helvetica'
    arial=Path('C:/Windows/Fonts/arial.ttf')
    if arial.exists():
        pdfmetrics.registerFont(TTFont('GraphArial',str(arial))); font='GraphArial'
    c=canvas.Canvas(str(BOOK),pagesize=(960,640))
    c.setTitle('Muon project: complete graph and ablation book, 17 September 2026')
    for page,f in enumerate(FIGURES,1):
        c.setFillColorRGB(.12,.16,.21); c.setFont(font,19)
        heading=f['id'].split('_')[0]+'  '+f['title']
        assert pdfmetrics.stringWidth(heading,font,19)<=888,heading
        c.drawString(36,605,heading)
        c.drawImage(ImageReader(str(OUT/f['path'])),40,200,width=880,height=390,preserveAspectRatio=True,anchor='c')
        lines=[]
        for text in [f['caption'], 'Settings: '+f['settings'], 'Takeaway: '+f['takeaway'], ('Limit: '+f['caveat']) if f['caveat'] else '']:
            if text: lines.extend(textwrap.wrap(text,146)); lines.append('')
        assert len(lines)<=15,(f['id'],len(lines))
        c.setFont(font,10.5)
        y=188
        for line in lines:
            assert pdfmetrics.stringWidth(line,font,10.5)<=880,(f['id'],line)
            c.drawString(40,y,line); y-=11.5
        c.setFont(font,9)
        c.drawString(40,22,'Muon mechanism study - full setups and source data in the companion package')
        c.drawRightString(920,22,f'{page} / {len(FIGURES)} - 17 September 2026')
        c.showPage()
    c.save()
    reader=PdfReader(BOOK)
    assert len(reader.pages)==35
    for page,f in zip(reader.pages,FIGURES):
        text=page.extract_text()
        assert f['id'].split('_')[0] in text and 'Takeaway:' in text
    shutil.copy2(BOOK,OUT/BOOK.name)


def main():
    global REFRESH
    ap=argparse.ArgumentParser()
    ap.add_argument('--refresh-figures',help='Comma-separated changed figure IDs; reuse existing files for all others.')
    args=ap.parse_args()
    if args.refresh_figures:
        REFRESH=set(args.refresh_figures.split(','))
    for sub in ['main','appendix','data','source_configs']:
        (OUT/sub).mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
                         'axes.labelsize':11,'legend.fontsize':10,'xtick.labelsize':10,
                         'ytick.labelsize':10,'lines.linewidth':2,'pdf.fonttype':42,'svg.fonttype':'none'})
    df,curves,summary,coverage=collect()
    figures(df,curves,summary,coverage)
    write_gallery(df,summary,coverage)
    graphbook()
    for f in ROOT.glob('config/_generated_*.json'):
        if any(word in f.name for word in ['resnet18_clean','component_resnet18','algorithm_momentum_resnet18',
                'constant_lr_control','tinyvit_imagenet100_grid','tinyvit5m_imagenet100_momentum',
                'mu2_diagnostic','mechanism_followup']):
            j=json.loads(f.read_text())
            eligible={k:v for k,v in j['experiments'].items()
                      if v['dataset']['name'] in ('cifar100','imagenet100')
                      and v['model']['name'] in ('resnet18','timm_tiny_vit')}
            assert eligible, f
            j['experiments']=eligible
            j['active_experiments']=[k for k in j.get('active_experiments',eligible) if k in eligible]
            j['package_filter_note']='Original experiment entries preserved for approved ResNet18/CIFAR-100 and real TinyViT-5M/ImageNet-100 only; discarded proxy/dataset entries omitted.'
            (OUT/'source_configs'/f.name).write_text(json.dumps(j,indent=2))
    (OUT/'data/mechanism_tables').mkdir(exist_ok=True)
    for f in MECH.glob('*.csv'):
        shutil.copy2(f,OUT/'data/mechanism_tables'/f.name)
    shutil.copy2(Path(__file__),OUT/Path(__file__).name)
    validation=dict(figures=35,main_figures=11,appendix_figures=24,pdf_pages=35,
                    seed_records=len(df),named_conditions=len(summary),curve_rows=len(curves),
                    complete_epochs=True,finite_accuracy=True,sample_sd_ddof=1,
                    excluded_compact_proxy=True,only_approved_datasets=True,
                    paper_modified=False,training_jobs_submitted=False)
    (OUT/'verification.json').write_text(json.dumps(validation,indent=2))
    ZIP.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob('*')):
            if f.is_file(): z.write(f,f.relative_to(OUT))
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
    print(json.dumps(validation,indent=2))
    print(BOOK); print(ZIP)


if __name__=='__main__':
    main()
