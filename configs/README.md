# Frozen experiment grids

| File | Purpose |
|---|---|
| `resnet18_lr_grid.json` | Main ResNet-18/CIFAR-100 LR robustness grid |
| `resnet18_constant_lr_control.json` | ResNet-18 constant-LR scheduler control |
| `resnet18_mars_gamma_grid.json` | Fixed-auxiliary MARS coefficient and LR sweep |
| `tinyvit_lr_grid.json` | TinyViT-5M/ImageNet-100 direction-family LR grid |
| `tinyvit_momentum_cosine_lr.json` | Thirteen momentum policies under cosine LR |
| `tinyvit_momentum_constant_lr.json` | Momentum policies under constant LR |
| `mu2_diagnostics.json` | Mu²-inspired alpha-by-beta and schedule diagnostics |
| `mechanism_followups.json` | Focused TinyViT mechanism and interaction follow-ups |

All dataset roots are portable public defaults. Changing a root does not alter the frozen optimizer, scheduler, augmentation, or training hyperparameters.
