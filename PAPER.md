# Same Task, Different Wiring: How Attention Mechanisms Shape Vision Transformer Representations

**CS229 Final Project — Vince Wu & Parth Sheth**

---

## 1. Abstract

We train six Vision Transformer (ViT-S/16) models that differ only in their attention mechanism — softmax, ReLU, sigmoid, gated SDPA, linear, and DyT+softmax — on ImageNet-100. Despite near-identical final accuracy for the top performers, the models learn structurally distinct representations. Using Centered Kernel Alignment (CKA), attention entropy analysis, and a novel cross-mechanism head pruning study, we show that (1) spatial patch features converge across mechanisms while aggregated CLS representations diverge, (2) ReLU attention produces 2.8x more specialized heads than softmax yet achieves equal accuracy, and (3) head importance is mechanism-specific — attention statistics that predict a head's contribution in one mechanism fail to transfer to another. These findings challenge the implicit assumption that attention mechanism choice is a minor architectural detail, and provide the first empirical evidence that functional head roles are shaped more by the attention function than by the task.

---

## 2. Introduction & Motivation

### The Platonic Representation Hypothesis
Recent work (Huh et al. 2024) proposes that neural networks trained on the same data converge to similar internal representations regardless of architecture — the "Platonic Representation Hypothesis." If true, architectural choices like attention mechanism are implementation details with no effect on learned representations.

### Our Question
We test a sharper version: **do models with identical architecture except for the attention function learn the same representations?** This controls for all confounds (depth, width, training recipe, data) and isolates the attention mechanism as the only variable.

### Why This Matters
- If representations converge: attention mechanism is purely a compute/efficiency choice
- If representations diverge: the attention function actively shapes what the model learns, with implications for model merging, distillation, and interpretability
- Prior work comparing attention mechanisms (Wortsman 2023, Ramapuram 2025, Qiu 2025) only reports final accuracy; no one has examined the internal representations

---

## 3. Experimental Setup

### 3.1 Model Architecture
All models are ViT-Small/16 with:
- `embed_dim=384`, `depth=12`, `num_heads=6`, `patch_size=16`
- ~22M parameters
- LayerScale with init=1e-4 (Touvron et al., DeiT-III)
- Pre-norm (LayerNorm before attention/MLP)
- `pos_embed` shape: (1, 196, 384) — patches only, CLS token concatenated after

### 3.2 Attention Variants

| Variant | Formulation | Reference |
|---|---|---|
| **Softmax** | `softmax(QK^T / sqrt(d)) V` via `F.scaled_dot_product_attention` | Vaswani et al. 2017 |
| **Sigmoid** | `sigmoid(QK^T / sqrt(d) + b) V` with `b = -log(N)` | Ramapuram et al. ICLR 2025 |
| **ReLU** | `relu(QK^T / sqrt(d)) / N · V` | Wortsman et al. 2023 |
| **Gated SDPA** | `[softmax(QK^T/sqrt(d)) V] * sigmoid(X W_gate)` | Qiu et al. NeurIPS 2025 |
| **Linear** | `phi(Q)(phi(K)^T V)` with `phi(x) = elu(x) + 1` | Katharopoulos et al. 2020 |
| **DyT + Softmax** | Standard softmax attention, but `DynamicTanh` replaces all LayerNorms | Zhu, He, LeCun CVPR 2025 |

**Key architectural differences:**
- Softmax/Gated: distribution-preserving (weights sum to 1 per query)
- Sigmoid: fixed scalar bias, non-normalized
- ReLU: sparse (negative scores become 0), normalized by N
- Linear: O(Nd^2) complexity, no explicit attention matrix
- DyT: same attention, different normalization (`tanh(alpha*x) * weight + bias`)

### 3.3 Training Recipe
- **Dataset**: ImageNet-100 (100 classes, ~130K train, ~5K val)
- **Optimizer**: AdamW, lr=1e-3, weight_decay=0.05
- **Schedule**: Cosine decay with 5-epoch linear warmup from 1e-6
- **Augmentation**: DeiT-III 3-Augment (grayscale/solarize/blur + color jitter + RRC + hflip)
- **Regularization**: Mixup α=0.8, CutMix α=1.0, label smoothing 0.1
- **Loss**: BCE (binary cross-entropy, following DeiT-III)
- **Training**: 200 epochs on Modal A100-40GB, ~120s/epoch

### 3.4 Analysis Tools
- **CKA** (Centered Kernel Alignment): Linear CKA on 10K val samples at 9 checkpoints (epochs 0, 10, 25, 50, 75, 99, 100, 150, 199)
- **Attention entropy**: Per-head entropy of (normalized) attention weights
- **Head pruning**: Zero out each head via forward hook, measure accuracy drop on full val set
- **CS229 regression**: sklearn regressors (Linear, Ridge, Random Forest) to predict head importance from attention statistics

### 3.5 Infrastructure
- Modal cloud GPUs (A100-40GB, ~$2.50/hr)
- Total compute: ~6 × 200 epochs × 120s ≈ 40 GPU-hours (~$100)
- Analysis: local Mac MPS (head pruning ~2.5hrs on full val set)

---

## 4. Results

### 4.1 Training Outcomes

| Variant | Val Acc@1 | Val Acc@5 | CKA vs Softmax (CLS) | CKA vs Softmax (Patch) |
|---|---|---|---|---|
| **ReLU** | **64.1%** | 87.0% | 0.718 | ~0.80 |
| **Softmax** | 63.0% | 85.8% | — | — |
| Gated SDPA | 56.3% | 82.0% | 0.776 | ~0.85 |
| Sigmoid | 55.3% | 80.9% | 0.674 | ~0.70 |
| DyT + Softmax | 37.3% | 66.5% | 0.589 | ~0.65 |
| Linear | diverged | — | — | — |

**Key observations:**
- ReLU slightly outperforms softmax (+1.1pp), consistent with Wortsman et al. 2023
- Linear diverges entirely under this training recipe
- DyT significantly underperforms (~26pp below softmax), suggesting ViT-S at 200 epochs is insufficient for DyT convergence
- Gated SDPA underperforms despite additional parameters (gate projection)

**Figure**: `accuracy_curves.png`

### 4.2 CKA Analysis: "What" Converges, "How" Diverges

#### 4.2.1 Pairwise CKA
At the final layer (epoch 199), pairwise CKA reveals a consistent pattern:
- **Patch token CKA is always higher than CLS token CKA** across all model pairs
- CLS CKA ranges from 0.589 (DyT) to 0.776 (Gated)
- Patch CKA is ~0.1-0.15 higher than CLS CKA for each pair

**Interpretation**: Spatial features ("what" is at each position) converge more than the aggregated CLS representation ("how" the model summarizes the image). The attention mechanism shapes aggregation strategy more than feature extraction.

**Figures**: `cka_pairwise_cls.png`, `cka_pairwise_patch.png`

#### 4.2.2 Layer-Resolved CKA Gap
For the accuracy-matched softmax-ReLU pair:
- CLS CKA decreases from ~0.9 (early layers) to ~0.72 (final layer)
- Patch CKA remains higher throughout, with a gap of ~0.24 at deeper layers
- The gap widens monotonically — early layers are similar, deep layers diverge

**Interpretation**: Representations start aligned (shared low-level features) and diverge as the attention mechanism increasingly shapes information flow. The CLS token, which depends entirely on attention-mediated aggregation, shows the largest divergence.

**Figure**: `cka_cls_vs_patch_by_layer.png`

#### 4.2.3 Temporal Dynamics
- CKA at epoch 0 (random init) is high (~0.9) for all pairs
- Divergence crystallizes by epoch 25-50
- Early layers diverge first, deeper layers follow
- Once divergence sets in, it is stable — no reconvergence at later epochs

**Figures**: `cka_layer_epoch_heatmap.png`, `cka_trajectory.png`

### 4.3 Mechanistic Analysis: Head Specialization

#### Attention Entropy
| Variant | Mean Entropy | Entropy Std (across heads) | Mean Sparsity |
|---|---|---|---|
| Softmax | 4.09 | 0.266 | 24.6% |
| ReLU | 3.80 | **0.733** | **64.6%** |
| Sigmoid | 3.97 | 0.414 | varies |
| Gated SDPA | 1.60 | 0.305 | varies |

**Key finding**: ReLU attention produces **2.8x higher head specialization** (entropy std 0.733 vs 0.266) and **2.6x more sparse** attention (64.6% vs 24.6% of weights < 1e-3). ReLU heads are either highly focused or broadly attending — fewer "average" heads.

**Figures**: `attn_per_head_entropy.png`, `attn_per_head_sparsity.png`, `attn_head_specialization.png`

### 4.4 Head Pruning: Specialization Creates Bottlenecks

This is our most novel analysis. We zero out individual attention heads (via forward hooks on the output projection) and measure accuracy drop on the full 5,000-image ImageNet-100 val set.

#### Method
For each model and each of 72 (layer, head) pairs:
1. Register a `forward_pre_hook` on `block.attn.proj` that zeros input columns `[H*64 : (H+1)*64]`
2. Evaluate full val set accuracy with this head disabled
3. Record `accuracy_drop = baseline_acc - pruned_acc` (in percentage points)
4. Remove hook and restore clean model

#### Results

| Variant | Baseline Acc | Mean Drop | Max Drop | Most Critical Head |
|---|---|---|---|---|
| **ReLU** | 66.18% | 0.42pp | **2.40pp** | Layer 0, Head 1 |
| Sigmoid | 56.74% | **0.54pp** | 1.94pp | Layer 2, Head 3 |
| Gated SDPA | 58.40% | 0.29pp | 1.92pp | Layer 4, Head 2 |
| Softmax | 65.04% | 0.22pp | 1.22pp | Layer 0, Head 0 |

#### Key Findings

1. **ReLU has the highest max single-head drop (2.40pp)** despite similar accuracy to softmax. Its specialized heads carry uniquely critical information. Layer 0, Head 1 is a bottleneck — removing it costs nearly 2.5 percentage points.

2. **Softmax is the most pruning-robust** — no single head drops accuracy by more than 1.22pp. Its uniform attention entropy creates distributed redundancy. Information is spread across heads, making each one individually dispensable.

3. **Gated SDPA concentrates importance in layers 4-6** while other layers are near-zero. The sigmoid gate creates a few critical bottleneck heads in the middle of the network, with early and late layers contributing minimally.

4. **Specialization ≠ redundancy**. ReLU's 2.8x higher entropy variance translates directly into functional importance variance. Specialized heads carry unique, non-redundant information.

5. **Per-layer importance profiles differ qualitatively**:
   - Softmax: roughly uniform across layers, slight peak at layer 0
   - ReLU: strong early-layer importance (layer 0), then distributed
   - Gated SDPA: concentrated mid-network (layers 4-6)
   - Sigmoid: distributed with peaks at layers 0, 2, 9

**Figures**: `head_importance_heatmap.png`, `head_importance_distribution.png`, `head_importance_variance.png`

### 4.5 Progressive Head Pruning: Specialization Creates Fragility

We extend single-head pruning to **progressive** pruning: iteratively removing heads ordered by entropy (specialist-first = lowest entropy removed first, generalist-first = highest entropy removed first) and evaluating accuracy at each step. This reveals how gracefully each model degrades as capacity is reduced.

#### Method
1. Compute per-head attention entropy on 64 val images to rank all 72 heads
2. For each model, progressively zero out heads in entropy-ranked order (specialist-first and generalist-first)
3. Evaluate accuracy on 1,000 val images at each step (increments of 6 heads = one layer's worth)

#### Results

| Variant | Specialist-first: acc after 8% pruned | after 17% | Generalist-first: acc after 8% pruned | after 17% |
|---|---|---|---|---|
| **Softmax** | 69.6% (−3.3pp) | 66.2% (−6.7pp) | 70.2% (−2.7pp) | 69.2% (−3.7pp) |
| **ReLU** | 65.3% (−7.6pp) | 57.7% (−15.2pp) | 69.6% (−3.3pp) | 65.9% (−7.0pp) |
| Sigmoid | 56.8% (−7.7pp) | 52.5% (−12.0pp) | 39.0% (−25.5pp) | 30.5% (−34.0pp) |
| **Gated SDPA** | **42.2% (−23.7pp)** | **17.5% (−48.4pp)** | 58.8% (−7.1pp) | 51.8% (−14.1pp) |

#### Key Findings

1. **Gated SDPA is catastrophically fragile to specialist pruning.** Removing just 12 specialist heads (17% of total) drops accuracy from 65.9% to 17.5% — a 48pp collapse. The sigmoid gate funnels nearly all information through a small subset of low-entropy heads. Once those are removed, the model is destroyed.

2. **Softmax degrades most gracefully.** Even at 33% pruned (specialist-first), softmax retains 61.4% accuracy. Its uniform entropy distribution means no small subset of heads is disproportionately critical — the hallmark of distributed redundancy.

3. **ReLU shows clear specialist/generalist asymmetry.** Removing specialists first costs 15.2pp at 17% vs 7.0pp generalist-first — a 2.2x difference. This confirms that ReLU's specialized heads carry unique, non-redundant information.

4. **Sigmoid has an inverted pattern.** Generalist-first pruning hurts *more* than specialist-first (39.0% vs 56.8% at 8% pruned). In sigmoid attention, the high-entropy "generalist" heads are actually the critical ones — likely because sigmoid's non-normalized weights make broad attention patterns the primary information carriers.

**Figure**: `pruning_curves.png`

### 4.6 CS229 Component: Predicting Head Importance from Attention Statistics

**Question**: Can you predict how important a head is (accuracy drop when pruned) from its attention pattern statistics (entropy, sparsity, max attention weight, layer index)?

**Dataset**: 4 models × 12 layers × 6 heads = 288 data points, each with 4 features.

#### Experiment A: Leave-One-Model-Out Cross-Validation

| Regressor | Mean R² | Best Model | Worst Model |
|---|---|---|---|
| Linear | -0.490 | ReLU (-0.057) | Sigmoid (-0.784) |
| Ridge | -0.567 | ReLU (-0.055) | Sigmoid (-1.031) |
| Random Forest | -1.945 | ReLU (-0.249) | Softmax (-3.310) |

**All negative R² values**: attention statistics from one mechanism cannot predict head importance in another. The entropy-importance relationship is mechanism-specific.

#### Experiment B: Within-Model 6-Fold Cross-Validation

| Model | Linear | Ridge | Random Forest |
|---|---|---|---|
| Gated SDPA | 0.410 | 0.327 | 0.378 |
| ReLU | 0.161 | 0.201 | 0.032 |
| Sigmoid | -0.190 | -0.179 | -0.413 |
| Softmax | -0.108 | -0.009 | -0.016 |
| **Mean** | 0.068 | 0.085 | -0.005 |

**Within-model prediction is mechanism-dependent**: Gated SDPA shows moderate predictability (R²≈0.4), while softmax and sigmoid show none. This itself is a finding — the relationship between how a head attends (entropy/sparsity) and how much it matters (accuracy impact) is determined by the attention function.

#### Feature Importance

| Feature | Linear Coeff (std) | RF Importance |
|---|---|---|
| Entropy | +0.395 | 0.450 |
| Sparsity | +0.242 | 0.169 |
| Max Attn | +0.126 | 0.323 |
| Layer (norm) | -0.012 | 0.058 |

Higher entropy heads tend to be more important (pooled), but this relationship breaks down across mechanisms.

**Figure**: `importance_prediction.png`

---

### 4.7 Attention Distance: Local vs. Global Head Roles

We compute the mean spatial distance between query and key positions, weighted by attention, for each head. Distance is measured in patch units on a 14x14 grid (range 0-18.4).

| Variant | Mean Dist | Dist Std | Min Head | Max Head | Range |
|---|---|---|---|---|---|
| Softmax | 6.42 | 0.78 | 4.39 | 7.65 | **3.26** |
| **ReLU** | 6.55 | 0.73 | 4.26 | 7.66 | **3.40** |
| Sigmoid | 6.58 | 0.60 | 4.97 | 7.43 | 2.46 |
| Gated SDPA | 6.79 | 0.46 | 5.23 | 7.65 | 2.42 |

**Key findings:**
- **ReLU and softmax have the widest range** of head distances (3.3-3.4 patch units), meaning they develop the clearest local-vs-global head specialization
- **Gated SDPA has the narrowest range** (2.42) — the gate homogenizes attention distances, consistent with its concentrated layer 4-6 importance profile
- ReLU's most local head (4.26) is more local than any softmax head (4.39), while their most global heads are identical (7.65-7.66). ReLU extends the specialization range at the local end.

**Figures**: `attn_distance_heatmap.png`, `attn_distance_distribution.png`

### 4.8 Linear Probing: Where Does Discriminative Information Emerge?

We train logistic regression (sklearn, C=1.0) on CLS features at each of 13 layers (12 blocks + final norm) to predict ImageNet-100 classes. Stratified 80/20 split (4000 train, 1000 test).

| Variant | Layer 0 | Layer 6 | Final | Peak Gain | Peak Layer |
|---|---|---|---|---|---|
| Softmax | 12.1% | 36.3% | 57.9% | +6.7pp | Layer 5 |
| **ReLU** | 12.3% | 28.7% | **58.1%** | +7.6pp | **Layer 10** |
| Sigmoid | 12.1% | 33.8% | 48.7% | +8.2pp | Layer 4 |
| Gated SDPA | 8.0% | 30.6% | 49.6% | +7.5pp | Layer 4 |

**Key findings:**
1. **ReLU back-loads discriminative information**: its peak accuracy gain is at layer 10, while softmax/sigmoid/gated peak at layers 4-5. ReLU's early layers focus on building specialized features, with classification-relevant information crystallizing late.
2. **Softmax front-loads**: by layer 6, softmax already has 36.3% probe accuracy vs. ReLU's 28.7%. Softmax builds classification-ready features earlier, consistent with its uniform head strategy.
3. **This explains the pruning asymmetry**: ReLU's layer 0 head 1 has the highest pruning importance (2.40pp drop) because early-layer specialization is critical for the late-layer information synthesis. Disrupting an early specialist cascades through the remaining layers.
4. **Final probe accuracy tracks model accuracy** (softmax 57.9%, ReLU 58.1%, sigmoid 48.7%, gated 49.6%), validating that CLS features capture task-relevant information proportional to end-to-end performance.

**Figures**: `linear_probe_accuracy.png`, `probe_gain_by_layer.png`

---

## 5. Discussion

### 5.1 Implications for the Platonic Representation Hypothesis
Our results partially support and partially challenge the hypothesis:
- **Support**: Patch-level CKA is >0.8 for accuracy-matched pairs, suggesting models extract similar spatial features
- **Challenge**: CLS-level representations diverge significantly (CKA = 0.67-0.78), and head functional roles (pruning importance, information timing) differ qualitatively across mechanisms
- **Resolution**: Models converge on *what spatial features* to extract but diverge on *how to aggregate and organize* them. The attention mechanism shapes the division of labor among heads and the layer-wise timing of discriminative information

### 5.2 Specialization vs. Robustness Tradeoff
ReLU and softmax achieve similar accuracy but with fundamentally different strategies:
- **ReLU**: specialized heads with critical bottlenecks (high max drop, high variance)
- **Softmax**: uniform heads with distributed redundancy (low max drop, low variance)

Progressive pruning makes this concrete: softmax retains 61% accuracy at 33% heads pruned, while Gated SDPA collapses to 17.5% at just 17% pruned. The gate mechanism, despite adding parameters, creates catastrophic fragility by concentrating information flow through a few specialist heads. This has practical implications for model compression and fault tolerance.

### 5.3 Mechanism-Specific Head Roles
The failure of cross-mechanism importance prediction (R² < 0) reveals fundamentally different internal organization. Knowing a head has high entropy tells you different things depending on the attention function — a high-entropy softmax head is relatively unimportant, while a high-entropy ReLU head may be critical. The linear probing results add another dimension: ReLU back-loads discriminative information (peak gain at layer 10), while softmax front-loads it (peak at layer 5), explaining why early-layer ReLU heads are uniquely critical bottlenecks.

### 5.5 Limitations
- **Scale**: ViT-S on ImageNet-100 is small-scale; patterns may differ at ViT-L/G on ImageNet-1K
- **Training recipe**: All models use the same AdamW recipe; some mechanisms may benefit from different hyperparameters (e.g., ReLU's higher accuracy could be recipe-dependent)
- **DyT and Linear failures**: DyT significantly underperforms and linear diverges, limiting some comparisons to 4 mechanisms
- **Head pruning is a proxy**: zeroing a head measures structural importance but not functional redundancy — other heads may compensate during retraining

---

## 6. Conclusion

We show that attention mechanism choice in Vision Transformers is not merely an efficiency-accuracy tradeoff — it fundamentally shapes internal representation structure, head specialization, and information flow timing. ReLU attention creates specialist heads that carry unique, critical information and back-loads discriminative features to deep layers, while softmax creates generalist heads with distributed redundancy and front-loads classification-relevant features. The relationship between attention statistics and head importance is mechanism-specific, making cross-mechanism transfer of interpretability tools unreliable. These findings suggest that attention mechanism choice deserves the same careful consideration as depth, width, and training recipe in Vision Transformer design.

---

## 7. References

- Darcet et al. "Vision Transformers Need Registers." ICLR 2024.
- Davari et al. "Reliability of CKA as a Similarity Measure in Deep Learning." 2022.
- Huh et al. "The Platonic Representation Hypothesis." ICML 2024.
- Katharopoulos et al. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML 2020.
- Michel et al. "Are Sixteen Heads Really Better than One?" NeurIPS 2019.
- Qiu et al. "Unlocking the Power of Gated Attention in Vision Transformers." NeurIPS 2025.
- Ramapuram et al. "Theory, Analysis, and Best Practices for Sigmoid Self-Attention." ICLR 2025.
- Touvron et al. "DeiT III: Revenge of the ViT." ECCV 2022.
- Vaswani et al. "Attention Is All You Need." NeurIPS 2017.
- Voita et al. "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned." ACL 2019.
- Wortsman et al. "Replacing softmax with ReLU in Vision Transformers." arXiv 2023.
- Zhu, Chen, He, LeCun. "Transformers without Normalization." CVPR 2025.

---

## 8. Appendix: Code & Reproducibility

### Repository Structure
```
cs229_project/
├── src/
│   ├── models.py         # All 6 ViT attention variants
│   ├── trainer.py        # Training loop with feature extraction
│   ├── data.py           # ImageNet-100 dataloaders + 3-Augment
│   └── analysis.py       # CKA computation utilities
├── train_modal.py        # Modal cloud training entry point
├── launch_zoo.py         # Launches all 6 training runs
├── analyze_zoo.py        # CKA analysis + accuracy curves
├── analyze_mechanistic.py # Per-head entropy, sparsity, CKA trajectories
├── analyze_pruning.py    # Head pruning + CS229 regression
├── analyze_depth.py      # Attn distance, linear probes
├── analyze_extra.py      # Progressive pruning curves
├── results/              # Per-model checkpoints, metrics, features
│   ├── softmax/          # metrics.csv, checkpoint_latest.pth, features/, ...
│   ├── relu/
│   ├── sigmoid/
│   ├── gated/
│   ├── linear/
│   └── softmax_dyt/
├── figures/              # All generated plots
├── PAPER.md              # This document
└── project_log.md        # Full project history
```

### Running Analysis
```bash
# CKA and accuracy analysis
python analyze_zoo.py --results-dir ./results

# Mechanistic analysis (entropy, sparsity, CKA trajectories)
python analyze_mechanistic.py --results-dir ./results --run-attention \
    --data-path /path/to/archive

# Head pruning + CS229 regression
python analyze_pruning.py --results-dir ./results \
    --data-path /path/to/archive --output-dir ./figures
```

### Figures Inventory

| Figure | Script | Description |
|---|---|---|
| `accuracy_curves.png` | `analyze_zoo.py` | Val accuracy over 200 epochs for all 6 variants |
| `cka_pairwise_cls.png` | `analyze_zoo.py` | Pairwise CKA heatmap (CLS tokens, final layer) |
| `cka_pairwise_patch.png` | `analyze_zoo.py` | Pairwise CKA heatmap (patch tokens, final layer) |
| `cka_layerwise.png` | `analyze_zoo.py` | Layer-wise CKA grids vs softmax baseline |
| `cka_trajectory.png` | `analyze_zoo.py` | CKA over training epochs (all pairs) |
| `cka_accuracy_matched.png` | `analyze_zoo.py` | CKA at matched accuracy levels |
| `attention_entropy.png` | `analyze_zoo.py` | Per-layer attention entropy for each variant |
| `l2_norm_distributions.png` | `analyze_zoo.py` | Patch token L2 norm histograms |
| `cka_cls_vs_patch_by_layer.png` | `analyze_mechanistic.py` | Layer-resolved CLS vs patch CKA gap |
| `cka_layer_epoch_heatmap.png` | `analyze_mechanistic.py` | Layer × epoch CKA evolution |
| `attn_per_head_entropy.png` | `analyze_mechanistic.py` | Per-head entropy heatmaps |
| `attn_per_head_sparsity.png` | `analyze_mechanistic.py` | Per-head sparsity heatmaps |
| `attn_head_specialization.png` | `analyze_mechanistic.py` | Entropy variance per layer |
| `head_importance_heatmap.png` | `analyze_pruning.py` | Accuracy drop per (layer, head) per variant |
| `head_importance_distribution.png` | `analyze_pruning.py` | Mean vs max drop bars + per-layer profiles |
| `head_importance_variance.png` | `analyze_pruning.py` | Per-layer importance std |
| `importance_prediction.png` | `analyze_pruning.py` | CS229 regression results |
| `pruning_curves.png` | `analyze_extra.py` | Progressive pruning: accuracy vs % heads removed |
| `attn_distance_heatmap.png` | `analyze_depth.py` | Per-head attention distance (local/global) |
| `attn_distance_distribution.png` | `analyze_depth.py` | Distance spread, histogram, receptive field |
| `linear_probe_accuracy.png` | `analyze_depth.py` | Probe accuracy curves by layer |
| `probe_gain_by_layer.png` | `analyze_depth.py` | Per-layer accuracy gain with peaks |

### Key Technical Details

**Head pruning hook**: Importance is measured by zeroing head contributions before the output projection. A `register_forward_pre_hook` on `block.attn.proj` sets `input[:, :, H*64:(H+1)*64] = 0`, disabling head H's contribution to the residual stream. This is done for all 72 (layer, head) pairs per model, evaluating accuracy on the full 5,000-image val set each time.

**CKA computation**: We use linear CKA (Kornblith et al. 2019) on features saved at 9 training checkpoints. CLS features are (N, 384) vectors; patch features are mean-pooled across spatial positions to (N, 384).

**Attention statistics**: Computed by hooking into the QKV linear layer's output, reconstructing attention weights according to each mechanism's formula, then computing per-head entropy, sparsity (fraction of weights < 1e-3), and max attention weight.
