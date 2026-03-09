# CS229 Project Log

## Current Status (2026-03-08)

**Active direction**: Attention Mechanism Zoo — Representation Convergence in Vision Transformers
**Stage**: All analyses complete, paper/poster document written
**Code**: root directory (`src/`, `analyze_*.py`, `train_modal.py`)
**Results**: `results/` and `figures/`
**Paper doc**: `PAPER.md`

### Repo Reorganization (2026-03-08)
Moved all Phase 1 and abandoned Phase 2 code/outputs to `old/`:
- Root-level scripts (`train.py`, `train_modal.py`, `analyze_*.py`, etc.) → `old/`
- Root `src/` (Phase 1 models/trainer) → `old/src/`
- `gcp/`, `context.md`, `CS_229_Project_Proposal_Template.pdf` → `old/`
- `output/` (~746MB), `modal_output/` (~10GB) → `old/`
- `experiments/optimizer_arch/` (abandoned direction) → `old/`
- Deleted `modal_train.py` (was already removed from working tree)
- Updated `.gitignore` for new paths

**Current top-level structure**:
```
cs229_project/
├── src/                  # Models, trainer, data, CKA analysis
├── analyze_*.py          # Analysis scripts (zoo, mechanistic, pruning, depth, extra)
├── train_modal.py        # Modal cloud training entry point
├── launch_zoo.py         # Launches all 6 training runs
├── results/              # Per-model checkpoints, metrics, features
├── figures/              # All generated plots
├── archive/              # ImageNet-100 val images (val.X/)
├── old/                  # All Phase 1 + abandoned code/outputs
├── PAPER.md              # Paper draft
├── project_log.md        # This file
└── requirements.txt
```

---

## Project Timeline

### Phase 1: Original Project (completed, DEAD)

**Title**: "Eliminating High-Norm Artifacts in Vision Transformers via Gated Attention"

Built 5 ViT-Small variants on ImageNet-100 with DeiT-III recipe (LAMB, LayerScale):
1. VanillaViT — standard DeiT-III
2. RegisterViT — +4 register tokens (Darcet et al. ICLR 2024)
3. SDPAGatedViT — +sigmoid gate after SDPA (Qiu et al. NeurIPS 2025)
4. ValueGatedViT — +sigmoid gate after value projection
5. AsymmetricGatedViT — SDPAGated but CLS always ungated

Trained 11 runs across two Modal workspaces (vincexxwu/, v2/).

#### Key findings that killed the project:
1. **No artifacts at ViT-S scale.** L2 norms are clean unimodal Gaussians in ALL models. Artifacts only emerge at ViT-L/G scale (Darcet et al.).
2. **All architectures converge to ~83.8%** when properly trained (LS=1e-4). Gating provides zero benefit. CKA >0.95 across architectures.
3. **Gates are passive.** Activations ~0.5 (sigmoid of ~0). Gates neither help nor hurt.
4. **LAMB + LayerScale pathology discovered.** LS=1e-6 models fail (50-69% acc) due to LAMB trust ratio trapping gammas: `effective_lr ≈ base_lr × ||γ||/||update|| ≈ 4e-3 × 1e-3 ≈ 4e-6`. MLP gammas completely dead (1e-4 to 1e-10). This is an optimizer bug, not architecture research.

#### Files created:
- `src/models.py` — all 5 architectures
- `src/trainer.py` — LAMB training with L2/gate diagnostics
- `src/data.py` — ImageNet-100 with 3-Augment
- `train_modal.py` — Modal deployment (A100/H100)
- `analyze_l2_norms.py` — L2 norm plots across all checkpoints
- `analyze_layerscale.py` — gamma extraction and visualization
- `analyze_cka.py`, `analyze_cls_attn_vs_norm.py`, etc.

### Phase 2: Direction Search (completed)

Evaluated 20+ ideas. Full tracker in `notes/project_ideas_context.md`.

#### Ideas explored and rejected:
| # | Idea | Why Rejected |
|---|------|-------------|
| 1-5 | Artifact/gating-dependent ideas | Artifacts don't exist at this scale |
| 6 | Model merging | Different architectures, field saturated |
| 7 | Vision SAEs | Field exploded 2025, crowded |
| 8 | Apply-X-to-Y ideas (Magma, WSD, etc.) | No novelty |
| 9 | Hivemind LLM project | Domain pivot, scope too large |
| 10 | LR sensitivity of gated ViTs | Too simple |
| 11 | SAEs for cross-arch comparison | Weak "so what" — same accuracy |
| 12 | Training dynamics / critical periods | 50-60% boring risk, needs new runs |
| 13 | Data efficiency / label noise | Poorly motivated, 60-70% null risk |
| 14 | DINOv2 + gating | Wrong pathway (artifacts from MLP), scale problem |
| 15 | ViT confidence from internals | Each component done: CalAttn, X-Mahalanobis, AttenDence, InternalInspector, Entropy-Lens |
| 16 | Loss landscape flatness | Hide & Seek (ICML 2025) showed standard measures fail for transformers; expected result |
| 17 | KD from heterogeneous teachers | Wu et al. ICML 2022 covers it; gated-student angle likely null (gates passive) |
| 18 | Head taxonomy across objectives | Walmer et al. CVPR 2023 scooped it |
| 19 | Layer criticality | Methodology well-trodden (ShortGPT, Gromov et al.) |

#### Winner: #20 Optimizer-Architecture Silent Failures
- **Novelty**: HIGH — no prior characterization of LAMB+LS trust ratio trap
- **Risk of boring**: ~25% — SGD almost certainly also fails, giving multiple failure modes
- **Compute**: ~$333 total
- **Weakness**: Tagline ("we found optimizer bugs") is not as sexy as a positive result

### Phase 3: Attention Mechanism Zoo (active, 2026-03-05 — 2026-03-07)

**Pivot**: Abandoned optimizer-arch direction in favor of training 6 ViT-S models with different attention mechanisms and comparing learned representations via CKA (testing Platonic Representation Hypothesis).

**6 Attention Variants** (all ViT-S/16, 384-dim, 12 layers, 6 heads):
1. **Softmax** — standard F.scaled_dot_product_attention
2. **Sigmoid** — sigmoid(QK^T + bias), Apple ICLR 2025
3. **ReLU** — ReLU(QK^T) / N, Wortsman 2023
4. **Gated SDPA** — sigmoid gate after softmax attention, Qiu NeurIPS 2025
5. **Linear** — φ(Q)φ(K)^T V, Katharopoulos 2020
6. **DyT+Softmax** — Dynamic Tanh replacing LayerNorm, He/LeCun CVPR 2025

**Training**: ImageNet-100, AdamW lr=1e-3, cosine schedule, 200 epochs on Modal A100-40GB

**200-Epoch Results**:
| Variant | Accuracy | CKA vs Softmax (CLS) | CKA vs Softmax (Patch) |
|---|---|---|---|
| ReLU | 64.1% | 0.718 | ~0.8 |
| Softmax | 63.0% | — | — |
| Gated SDPA | 56.3% | 0.776 | ~0.85 |
| Sigmoid | 55.3% | 0.674 | ~0.7 |
| DyT | 37.3% | 0.589 | ~0.65 |
| Linear | diverged | — | — |

**Mechanistic Findings**:
1. **ReLU vs Softmax (accuracy-matched pair)**:
   - ReLU has 2.8x higher head specialization (entropy std 0.733 vs 0.266)
   - ReLU has 2.6x more sparse attention (64.6% vs 24.6% weights < 1e-3)
   - Patch CKA consistently higher than CLS CKA across all layers (gap ~0.24)
   - → Same accuracy, different internal strategy; "what" converges, "how" diverges
2. **Temporal dynamics**: Divergence crystallizes by epoch 25-50, early layers diverge first
3. **CLS vs Patch gap**: Universal pattern — spatial features more similar than aggregated CLS

**Bugs fixed during training**:
- `--detach` needed for Modal to avoid local client disconnections
- Cosine scheduler not extending on resume (old t_initial restored) — fixed by not restoring scheduler state
- Feature file alphabetical sort bug (epoch99 > epoch199) — fixed with numerical sort
- pos_embed size mismatch in attention hooks (model adds pos_embed before CLS concat)

**Key files**:
- `train_modal.py` — Modal training entry point
- `src/models.py` — all 6 attention variants
- `src/trainer.py` — training loop with feature extraction
- `analyze_zoo.py` — CKA analysis + plots
- `analyze_mechanistic.py` — per-head attention, layer-resolved CKA
- `analyze_pruning.py` — head pruning sensitivity + CS229 regression
- `analyze_depth.py` — attention distance, linear probes
- `analyze_extra.py` — progressive pruning curves, attention maps, PCA
- `figures/` — all generated figures

**Modal**: workspace `olympian1738`, volumes `gated-attn-checkpoints` + `imagenet100-data`

#### Head Pruning & CS229 Regression (2026-03-07)

**Analysis**: `analyze_pruning.py`

**Head Pruning Results** (5000 ImageNet-100 val images, accuracy drop metric):
| Variant | Baseline Acc | Mean Drop | Max Drop |
|---|---|---|---|
| Sigmoid | 56.74% | **0.54pp** | 1.94pp |
| ReLU | 66.18% | 0.42pp | **2.40pp** |
| Gated SDPA | 58.40% | 0.29pp | 1.92pp |
| Softmax | 65.04% | 0.22pp | 1.22pp |

Key findings:
- **ReLU has highest max single-head drop (2.40pp)** — its specialized heads carry uniquely critical information (layer 0 head 1 is a bottleneck). This aligns with 2.8x entropy specialization from mechanistic analysis.
- **Gated SDPA concentrates importance in layers 4-6**, with the rest near-zero. The gate creates a few critical bottleneck heads.
- **Softmax is most pruning-robust** — no single head drops accuracy by more than 1.22pp. Uniform heads = distributed redundancy.
- **Cosine distance metric gave misleading rankings** — ReLU appeared "least important" on cosine distance but has the largest accuracy impact. Always use task-relevant metrics.

**CS229 Regression** (predict head importance from entropy/sparsity/max_attn/layer_idx):
- Cross-model (leave-one-model-out): R² ≈ −0.1 → attention stats don't transfer across mechanisms
- Within-model (6-fold CV): Gated SDPA R²=0.61 (Ridge), others mixed (−0.5 to 0.28)
- This is itself a finding: the importance-entropy relationship is mechanism-specific

**Figures generated**:
- `head_importance_heatmap.png` — (layer, head) importance per variant
- `head_importance_distribution.png` — mean/std bars + per-layer profiles
- `head_importance_variance.png` — per-layer importance std
- `importance_prediction.png` — regression actual-vs-predicted + CV comparison

#### Deeper Analyses (2026-03-08)

**Analysis**: `analyze_depth.py`

**1. Attention Distance** — mean spatial distance between query and attended key, per head:
- ReLU has widest local-global range (3.40 patch units), with most-local head at 4.26
- Gated SDPA has narrowest range (2.42), gate homogenizes attention distances
- Confirms ReLU head specialization extends to spatial receptive field

**2. Linear Probing** — logistic regression on CLS features at each layer:
- ReLU back-loads discriminative info (peak gain at layer 10), softmax front-loads (peak at layer 5)
- By layer 6: softmax 36.3% vs ReLU 28.7%, but final layer: both ~58%
- Explains why ReLU's early heads are pruning bottlenecks: they feed the late-layer synthesis

**3. Cross-Model Head Transfer** — REMOVED. Control experiment (softmax_seed42) showed same-mechanism transfer also fails at ~1%, confirming it's just rotational non-alignment, not mechanism-specific. Not worth reporting.

**Figures**: `attn_distance_heatmap.png`, `attn_distance_distribution.png`, `linear_probe_accuracy.png`, `probe_gain_by_layer.png`

#### Progressive Pruning Curves (2026-03-08)

**Analysis**: `analyze_extra.py`

Progressively prune heads ordered by entropy (specialist-first vs generalist-first), evaluate accuracy at each step (6-head increments on 1K val subset).

**Key results**:
- **Gated SDPA is catastrophically fragile**: 65.9% → 17.5% after pruning 17% specialist heads (−48pp)
- **Softmax degrades most gracefully**: retains 61.4% at 33% pruned
- **ReLU specialist/generalist asymmetry**: 2.2x more damage from specialist-first pruning at 17%
- **Sigmoid inverted pattern**: generalist-first hurts more (unusual — high-entropy heads are the critical ones)

**Also ran** (not paper-worthy):
- Effective dimensionality (PCA): all models converge to ~140-160 components at final layer, ReLU most isotropic (PR=80.1)
- Attention map visualization: qualitative, good for poster only
- Specialization dynamics: only 1 checkpoint per model usable, insufficient for dynamics plot

**Figure**: `pruning_curves.png`

### Phase 3-old: Optimizer-Arch Experiment Setup (completed 2026-03-04, ABANDONED)

Created `experiments/optimizer_arch/` with all code:

```
experiments/optimizer_arch/
├── src/
│   ├── models.py      — VanillaViT with optional LayerScale (None = no LS)
│   ├── trainer.py      — Multi-optimizer + per-param diagnostics + effective LR
│   ├── data.py         — ImageNet-100 (copied from original)
│   └── analysis.py     — L2 norms (copied from original)
├── train_modal.py      — Modal entry point, per-optimizer LR defaults
├── launch_grid.py      — Launches 16 grid or 6 deep runs
├── analyze_results.py  — Generates 5 key figures
├── results/prior/      — Prior metrics CSVs for validation
└── README.md           — Compute estimates and run instructions
```

**Key changes from original trainer**:
- Added optimizers: AdamW, SGD+momentum, Lion (+ existing timm_lamb, torch_lamb)
- Added `_log_param_diagnostics()`: saves per-param norm, grad norm, Adam second moment stats, full gamma values as JSON every 5 epochs
- Added `_log_effective_lr()`: snapshots params before/after optimizer step to measure TRUE effective LR at key epochs (0, 1, 5, 10, 25, 50, 75, 99)
- `Block` now supports `layer_scale_init=None` to disable LayerScale entirely
- Per-optimizer LR defaults: AdamW=1e-3, LAMB=4e-3, SGD=0.05, Lion=3e-4
- Uses new Modal volume `optarch-checkpoints` (separate from original)

**Prior results copied** (for validation, not direct reuse):
- LAMB + LS=1e-6 metrics from v2/vanilla_baseline (47.2% at epoch 400)
- LAMB + LS=1e-4 metrics from vincexxwu/sdpa_gated_ls1e4 (83.6%)
- Various other architecture runs for reference

### Phase 4: Run Experiments (NOT STARTED)

**Phase 4a — Full grid (16 runs × 100 epochs)**:
- {AdamW, LAMB, SGD, Lion} × {none, 1e-6, 1e-4, 1e-2}
- ~53 GPU-hours, ~$133, ~2 days wall-clock (4 parallel)
- Command: `python launch_grid.py`

**Phase 4b — Deep runs (6 runs × 400 epochs)**:
- 4 no-LS baselines + adamw_ls1e-6 + timm_lamb_ls1e-4
- ~80 GPU-hours, ~$200, ~2-3 days wall-clock
- Command: `python launch_grid.py --deep`

### Phase 5: Analysis (NOT STARTED)

Key figures to generate (`python analyze_results.py`):
1. **Accuracy heatmap** — 4×4 grid, red=fail, green=success
2. **Training curves** — val acc over epochs, grouped by LS init
3. **Effective LR trajectories** — THE MONEY FIGURE: effective LR of LS gammas per optimizer
4. **Gamma evolution** — per-layer gamma values over training
5. **ELR ratio vs accuracy** — diagnostic metric scatter plot

Additional analysis needed:
- Mechanistic derivations (why each optimizer fails/succeeds)
- CS229 angle: logistic regression on epoch-10 features → predict convergence
- Connection to muP / rotational equilibrium theory

### Phase 6: Paper (NOT STARTED)

---

## Important Technical Details

### LAMB Trust Ratio Pathology (the core finding)
- LAMB: `update = lr × trust_ratio × adam_update`, where `trust_ratio = ||params||₂ / ||adam_update||₂`
- For LS gamma ≈ 1e-6 (384-dim): `||γ||₂ ≈ 1.96e-5`
- Adam update norm ≈ O(1e-2)
- → `trust_ratio ≈ 1e-3`, `effective_lr ≈ 4e-6`
- Weight decay (0.05) constantly pulls toward 0
- MLP gammas die completely; attention gammas partially escape (gradient from CLS classification path)
- `trainer.py` uses FLAT param list — no weight-decay exclusion for gammas

### Optimizer LR Conventions
- AdamW: lr=1e-3, wd=0.05 (standard ViT recipe)
- LAMB: lr=4e-3, wd=0.05 (DeiT-III recipe)
- SGD+mom: lr=0.05, wd=0.05 (needs ~50x higher LR than Adam)
- Lion: lr=3e-4, wd=0.3 (lower LR, higher WD per Chen et al.)

### Model Architecture
- ViT-Small: embed_dim=384, depth=12, num_heads=6, patch_size=16
- ~22M params (without gating)
- Pre-norm (LayerNorm before attention/MLP, not after)
- pos_embed shape: (1, 196, 384) — patches only, CLS concatenated after

### Data
- ImageNet-100: 100 classes, ~130K train, ~5K val
- Stored as tar on Modal volume `imagenet100-data`
- DeiT-III 3-Augment: grayscale/solarize/blur + color jitter + RRC + horizontal flip
- Mixup α=0.8, CutMix α=1.0, label smoothing 0.1, BCE loss

### Infrastructure
- Modal cloud GPUs (A100 ~$2.50/hr, H100 ~$3.95/hr)
- ~120s/epoch on A100 for ViT-S/ImageNet-100
- Auth: `modal setup` (last authed 2026-03-04 to new account)
- Data volume: `imagenet100-data` (shared with original project)
- Output volume: `optarch-checkpoints` (NEW, separate from original)
