# Canonical Experiment Design

This document is the single source of truth for describing the experimental design in
the paper, presentation, plots, and result summaries. Configuration files remain the
executable source of truth; this document supplies the human-readable taxonomy.

## 1. Experimental objective

We use a controlled, high-throughput ablation environment to study mechanisms attached
to Muon. The objective is not to search for one composite optimizer that wins every
benchmark. It is to determine, under matched training conditions, how the direction
presented to Muon's orthogonalizer and the temporal memory used to construct that
direction affect optimization.

The three studied mechanisms are:

1. a MARS-inspired consecutive-gradient correction;
2. a Mu2SGD-inspired secondary direction estimator;
3. scheduling Muon's momentum coefficient.

We first run broad single-seed surfaces and then repeat only claim-defining or diagnostic
comparisons on seeds 42--44. Weak, neutral, and non-convergent cells remain in the
reported coverage.

## 2. Notation: four independent controls

To prevent overloaded uses of "decay," every experiment is described using four
separate controls.

| Symbol | Meaning | What it controls |
|---|---|---|
| \(\eta_t\) | outer Muon learning rate | magnitude of the final matrix-parameter step |
| \(\beta_t\) | Muon momentum | temporal memory in the direction accumulator |
| \(\gamma_t\) | MARS-inspired coefficient | strength of the consecutive-gradient correction |
| \(\alpha_t\) | Mu2-inspired coefficient | strength of the secondary estimator correction |

Throughout the paper, **LR means the LR of the Muon matrix-parameter group**. The
auxiliary AdamW LR is always \(10^{-3}\) and is not swept. This convention should be
stated once in the introduction and once in the experimental setup; figure axes can then
simply use "LR."

### Outer LR policies

- **Cosine LR:** \(\eta_t\) decreases from the listed initial LR \(\eta_0\) to
  \(0.05\eta_0\) over 50 epochs.
- **Constant LR:** \(\eta_t=\eta_0\) for all 50 epochs.

An experiment described as "constant momentum" may still use cosine LR. An experiment
described as "constant LR" may still schedule momentum. These two axes must never be
conflated.

### Muon momentum policies

- **Constant momentum:** \(\beta_t=\beta_0\).
- **Cosine momentum transition:** \(\beta_t\) changes smoothly from
  \(\beta_0\) to \(\beta_1\) during a stated epoch window and then remains at
  \(\beta_1\).
- **Linear momentum transition:** the same, with linear interpolation.

Unless explicitly ablated, plain Muon and all direction-correction variants use
\(\beta=0.95\).

## 3. Fixed optimizer and training contract

Every comparison uses the same parameter split:

- tensors with `ndim >= 2`, including convolution kernels and classifier matrices, use
  the Muon-side update;
- lower-dimensional tensors use the unchanged auxiliary AdamW path;
- auxiliary AdamW: LR \(10^{-3}\), betas \((0.9,0.999)\), epsilon \(10^{-8}\),
  zero weight decay;
- Muon matrix weight decay: \(5\times10^{-4}\);
- Nesterov direction enabled;
- five Newton--Schulz steps;
- batch size 128;
- 50 training epochs;
- global gradient-norm clipping at 1.0;
- final-epoch accuracy is the primary endpoint.

Therefore, "Muon+AdamW" names the practical baseline parameter split. AdamW is not a
separate competing baseline in these experiments: it is deliberately held fixed so that
the study isolates Muon-side mechanisms.

## 4. Direction variants

For the matrix gradient \(g_t\), define the optional MARS-inspired correction

\[
\tilde g_t = g_t + \gamma_t(g_t-g_{t-1}).
\]

The tested correction is inspired by the variance-reduction mechanism in MARS. It is not
the complete MARS optimizer and is not identical to MARS-M: it omits MARS-M's
\(\beta/(1-\beta)\) scaling, corrected-gradient clipping, and Moonlight update scaling.

The optional Mu2-inspired secondary estimator is

\[
v_t = \tilde g_t + \alpha_t(v_{t-1}-\tilde g_{t-1}).
\]

This isolates a Mu2SGD-like history correction studied by the lab; it is not presented as
a reproduction of the complete published Mu2SGD optimizer. The direction is then passed
through Muon's momentum and Nesterov construction before Newton--Schulz
orthogonalization.

The five main direction families are:

| Family | \(\gamma_t\) | \(\alpha_t\) | \(\beta_t\) | Purpose |
|---|---:|---:|---:|---|
| Muon | 0 | 0 | 0.95 | matched baseline |
| MARS | fixed nonzero | 0 | 0.95 | isolate consecutive-gradient correction |
| Mu2 | 0 | fixed nonzero | 0.95 | isolate secondary estimator |
| MARS+Mu2 | fixed nonzero | fixed nonzero | 0.95 | test additivity |
| Joint schedule | scheduled | scheduled | scheduled | test the original combined recipe |

The joint schedule is a composite intervention. It must not be used to attribute an
effect to any one component.

## 5. Benchmark contracts

### ResNet18 / CIFAR-100

- torchvision ResNet18 with a 3x3 stride-1 CIFAR stem and no max pool;
- standard CIFAR-100 50,000/10,000 train/test split;
- random crop with padding 4, horizontal flip, and standard normalization;
- training from scratch for 50 epochs;
- main LR set: \(\{0.005,0.01,0.02,0.04,0.08\}\).

For the main direction grid, MARS uses \(\gamma=0.50\), Mu2 uses
\(\alpha=0.20\), and \(\beta=0.95\). The joint schedule changes
\(\gamma:0.50\rightarrow0.10\), \(\alpha:0.20\rightarrow0.05\), and
\(\beta:0.95\rightarrow0.50\) over epochs 6--18.

### TinyViT-5M / ImageNet-100

- published `tiny_vit_5m_224` topology;
- dimensions [64, 128, 160, 320], depths [2, 2, 6, 2], heads [2, 4, 5, 10],
  windows [7, 7, 14, 7];
- 100-way classifier and 5,103,864 trainable parameters;
- Kaggle `ambityga/imagenet100` train/validation split;
- 224x224 inputs, random-resized crop, horizontal flip, RandAugment,
  RandomErasing 0.25, Mixup 0.8, CutMix 1.0, and label smoothing 0.1;
- training from scratch for 50 epochs;
- main LR set: \(\{0.005,0.01,0.02,0.04,0.08\}\).

For the main direction grid, MARS uses \(\gamma=0.35\), Mu2 transfers
\(\alpha=0.20\) without TinyViT-specific retuning, and \(\beta=0.95\). The joint
schedule changes \(\gamma:0.35\rightarrow0.05\),
\(\alpha:0.20\rightarrow0.05\), and \(\beta:0.95\rightarrow0.50\) over epochs
6--18.

TinyViT uses BF16 when supported, TF32, cuDNN benchmarking, and stochastic augmentation.
The runs are not bitwise deterministic; a nominal seed fixes the requested random seed
but does not guarantee an identical trajectory across hardware and data-loader recovery
settings.

## 6. Experiment blocks

Each block answers one question. Results from different blocks must not be silently mixed
because their LR or momentum policies differ.

### Block A: main LR robustness

**Question:** How do the direction families behave as the outer LR is increased?

- five direction families x five LRs;
- cosine outer LR;
- constant \(\beta=0.95\), except for the explicitly labeled joint schedule;
- ResNet18: all 25 cells on seeds 42--44;
- TinyViT: all 25 cells on seed 42, with Muon, MARS, and MARS+Mu2 at LRs 0.005
  and 0.01 confirmed on seeds 42--44.

This block supports the MARS stressed-LR robustness result. It does not support a general
claim that the combined optimizer is best.

### Block B: coefficient probes

**Question:** Is a result tied to one coefficient?

ResNet18 at LRs 0.04 and 0.08, cosine outer LR, \(\beta=0.95\):

- MARS: \(\gamma\in\{0.25,0.50,0.75\}\);
- Mu2: \(\alpha\in\{0.10,0.20,0.35\}\);
- matched Muon control at each LR.

All cells are explored on seed 42. At LR 0.08, Muon, MARS \(\gamma=0.75\), and
Mu2 \(\alpha=0.35\) receive seeds 42--44.

### Block C: mechanism--momentum factorial

**Question:** Do direction corrections interact with the momentum level or schedule?

ResNet18, cosine outer LR:

- LR in \(\{0.04,0.08\}\);
- MARS toggle \(\gamma\in\{0,0.5\}\);
- Mu2 toggle \(\alpha\in\{0,0.2\}\);
- momentum policy in {constant 0.50, constant 0.98, cosine
  \(0.98\rightarrow0.50\) over epochs 6--18}.

This is a 2 x 2 x 3 x 2-LR factorial. It must be presented as an interaction study, not
as part of the main LR sweep.

### Block D: constant-LR direction control

**Question:** Does removing the outer LR schedule change the ordering of direction
families?

- five direction families;
- ResNet18 LRs \(\{0.005,0.01,0.02,0.04\}\);
- constant outer LR;
- the same component coefficients and internal joint schedule as Block A.

This block is a scheduler control. It is not the isolated momentum-decay study.

### Block E: isolated TinyViT momentum timing

**Question:** When should Muon's momentum be reduced, and how does the answer depend on
the outer LR policy?

Only plain Muon is used: \(\gamma=0\) and \(\alpha=0\).

The 13 momentum policies are:

- constants \(\beta\in\{0.50,0.95,0.98\}\);
- cosine \(0.98\rightarrow0.50\) during epochs 6--18, 16--28, 26--38, or 36--48;
- cosine \(0.98\rightarrow0.80\) during the same four windows;
- linear \(0.98\rightarrow0.80\) during epochs 26--38 or 36--48.

They are evaluated under:

1. cosine outer LR with initial LR 0.005: 13 cells;
2. constant LR 0.0025: 13 cells;
3. constant LR 0.005: 13 cells.

The claim-defining confirmation compares, at constant LR 0.0025 and seeds 42--44:

- constant \(\beta=0.95\);
- constant \(\beta=0.98\);
- cosine momentum \(0.98\rightarrow0.50\) over epochs 36--48.

This block supports a conditional claim: late momentum decay improves the endpoint in
the tested low constant-LR regime. It does not imply that lower momentum or momentum
decay is universally beneficial.

### Block F: Mu2 mechanism diagnostic

**Question:** Why is the Mu2-inspired term weak or destabilizing when attached to Muon?

ResNet18:

- LR 0.08 with cosine outer LR;
- \(\beta\in\{0,0.5,0.9,0.95,0.98\}\);
- \(\alpha\in\{0,0.05,0.10,0.20,0.35,0.50\}\);
- 30 fixed-coefficient cells;
- two literal-estimator controls at \(\beta=0.95\), \(\alpha\in\{0.2,0.5\}\);
- three schedules at \(\beta=0.95\): early \(0.20\rightarrow0.05\), late
  \(0.20\rightarrow0.05\), and early \(0.05\rightarrow0.20\).

TinyViT:

- LR in \(\{0.005,0.01\}\) with cosine outer LR;
- \(\beta\in\{0.5,0.9,0.95,0.98\}\);
- \(\alpha\in\{0,0.05,0.10,0.20,0.35,0.50\}\);
- 48 fixed-coefficient cells;
- at each LR, two literal-estimator controls and the same three alpha schedules.

The early schedule is \(\alpha:0.20\rightarrow0.05\) over epochs 6--18; the late
schedule uses epochs 36--48; the early rise is
\(\alpha:0.05\rightarrow0.20\) over epochs 6--18. Muon momentum is fixed within these
scheduled-alpha controls.

The diagnostic records the correction-to-gradient norm and the angular change before and
after polar orthogonalization. Its purpose is to identify redundancy and destructive
\(\alpha\)-by-\(\beta\) interactions, not to select another headline optimizer.

## 7. Evidence policy

- **Exploratory surface:** seed 42, used to locate regimes and interactions.
- **Targeted confirmation:** seeds 42--44 with the hyperparameters fixed before the
  additional seeds are examined.
- **Exact repeat:** repeats the same nominal condition to assess trajectory sensitivity;
  it is not a new hyperparameter cell.
- **Optimization non-convergence:** the run completes without an infrastructure or
  numerical exception but fails to learn useful accuracy.
- **Technical failure:** missing epoch-50 output because of data loading, scheduler,
  process, storage, or hardware failure; excluded and safely rerun.

Final tables must report seed-level values alongside mean and sample standard deviation
for claim-defining comparisons. Small differences from a one-seed exploratory surface
must not be presented as statistically established improvements.

## 8. Required presentation order

The experimental setup should be explained in this order:

1. fixed parameter split and training contract;
2. four independent controls \((\eta,\beta,\gamma,\alpha)\);
3. five direction families;
4. outer LR policies;
5. isolated momentum policies;
6. experiment blocks and seed policy.

Only after this setup should the results be presented. This prevents the main LR grid,
the joint schedule, the isolated momentum study, and the Mu2 diagnostic from appearing to
be one irregular optimizer sweep.
