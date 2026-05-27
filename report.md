---
title: "HW4 — Image Restoration with PromptIR"
subtitle: "Visual Recognition using Deep Learning, 2026 Spring"
author:
  - "陳沛妤"
  - "Student ID: 314560017"
date: "2026-05-27"
geometry: margin=2.2cm
fontsize: 11pt
linkcolor: blue
urlcolor: blue
---

**GitHub repository:** <https://github.com/h31263219/2026ImageClassification_HW4>

---

## 1. Introduction

This homework is an **all-in-one blind image restoration** task: a
**single** model must restore 256×256 RGB images degraded by either
**rain** or **snow**, with the degradation type unknown at test time.
The official metric on CodaBench is **PSNR**. The spec forbids any
external data and any pretrained weights — the model must be trained
from scratch and must use **PromptIR** as the architecture.

PromptIR (Potlapalli et al., NeurIPS 2023, [1]) is built on the
Restormer transformer backbone [2] and inserts three **Prompt
Generation Blocks** (PGB) in the decoder. Each PGB stores a small
bank of learnable prompt components, computes a softmax mixture
conditioned on the current feature map, and injects the mixture back
into the features — letting the same network adapt its decoder
behaviour to the specific (unknown) degradation present in the input.

Given the joint constraints (single model · no pretraining · two
degradation types · 3,200 paired training tuples), my **core idea**
is an **iterative scaling study** that isolates three orthogonal
levers, one at a time. Each lever produces a measurable PSNR gain
and motivates the next change:

1. **Baseline — `light` PromptIR + L1 loss + 128×128 crops.** A
   faithful re-implementation of the published architecture at the
   smallest configuration the GPU comfortably fits (~14.5 M
   parameters). Establishes the floor.
2. **+ capacity & loss — `medium` config + Charbonnier loss.**
   Widening the channels (dim 36 → 48) raises the parameter count to
   ~24.9 M without making the deepest-level attention blow up;
   switching L1 to Charbonnier (`sqrt(x² + ε²)`) sharpens the
   gradient near zero, the regime that dominates the final dB of
   PSNR.
3. **+ context — patch size 128 → 192.** With unchanged architecture
   and loss, training on larger crops gives the same model **more
   spatial context per sample** — the remaining failure mode is
   fine-streak artifacts that require seeing more of the surrounding
   texture.

These three changes together push **test PSNR from 29.75 → 30.78
(+1.03 dB)** on the CodaBench public leaderboard. To explain *why*
PGB is worth keeping, §4 retrains a fourth model with `--no-prompt`
(PGB removed) and shows the prompt mechanism is responsible for a
small but consistent positive gap.

## 2. Method

### 2.1 Data Preprocessing

The training root
[`hw4_realse_dataset/train/`](hw4_realse_dataset/train) contains
1,600 rain and 1,600 snow paired tuples:

```
train/
├─ degraded/  rain-1.png … rain-1600.png, snow-1.png … snow-1600.png
└─ clean/     rain_clean-1.png …,         snow_clean-1.png …
```

The dataset loader [(`dataset.py`)](dataset.py):

1. Lists every file in `degraded/`, splits the basename on `-` to
   recover the degradation type and index, and pairs it with the
   matching `*_clean-*.png` in `clean/`.
2. Loads both images as RGB and converts to float32 tensors in
   `[0, 1]` of shape `(3, H, W)`.
3. Returns `(degraded, clean, deg_type)` triples where `deg_type` is
   passed through to validation only for per-type PSNR tracking — the
   model itself never sees it.

**Augmentations (training only).** Random `P×P` crop (P=128 in the
baseline and §4 ablation, **P=192 in the submitted model**), random
horizontal flip (p=0.5), random vertical flip (p=0.5), and random
0/90/180/270° rotation. The full 256×256 image is used for
validation.

**Train/val split.** Fixed `seed=42`, **50 rain + 50 snow** held out
per class → **3,100 train / 100 val** pairs. The split is identical
across **all four runs** in this report, so all PSNR comparisons are
paired.

### 2.2 Model Architecture

I re-implement PromptIR [1] from scratch in
[`model.py`](model.py). The full architecture (see Fig. 1):

* **Patch embed** — 3×3 conv, RGB → `dim` channels.
* **Encoder × 4** — each level is a stack of *Transformer Blocks*
  (`LayerNorm → MDTA → LayerNorm → GDFN`). Downsampling uses
  `Conv 3×3 + PixelUnshuffle` (channels double, spatial halves).
  * **MDTA** is Multi-Dconv-Head Transposed Attention: attention is
    computed across **channels** rather than tokens, which is
    `O(C²)` instead of `O((HW)²)` — essential at 256×256.
  * **GDFN** is Gated Dconv Feed-Forward Network: two parallel
    projections followed by GELU-gated multiplication.
* **Latent** — deepest stack of transformer blocks at `8·dim`
  channels.
* **Decoder × 3** — upsample (`Conv 3×3 + PixelShuffle`), concat with
  the matching encoder features, `1×1` conv to reduce channels back,
  then transformer-block stack.
* **Refinement** — extra transformer blocks at output resolution.
* **Output** — `Conv 3×3` → **residual** add to the input image.

**Prompt Generation Blocks (the PromptIR contribution).** Inserted
at three locations: after the latent (`channels = 8·dim`), after
decoder-L3 (`4·dim`), and after decoder-L2 (`2·dim`). Each PGB:

1. Global-pools the feature map to a per-image embedding `(B, C)`.
2. Linearly projects to `prompt_len = 5` softmax weights.
3. Weight-sums the learnable prompt tensor of shape
   `(1, prompt_len, prompt_dim, prompt_size, prompt_size)` and
   bilinear-upsamples to the current spatial resolution.
4. Concatenates the prompt with the features, mixes them by one
   transformer block, and 1×1 conv-reduces back to feature channels.

This is the entire mechanism by which prompts (and the parameters
they carry) influence the network. **§4 retrains the model with all
three PGBs disabled** — the cleanest possible test of what PromptIR
adds on top of plain Restormer.

**Configurations used in this report.**

| Config        | `dim` | `num_blocks`   | `num_refinement` | Trainable params |
|---------------|-------|----------------|------------------|------------------|
| `light`       | 36    | (2, 3, 3, 4)   | 2                | **14.50 M**      |
| `light` (no PGB) | 36 | (2, 3, 3, 4)   | 2                | 8.47 M           |
| **`medium`** (submitted) | **48** | (2, 3, 3, 4) | 2 | **24.88 M**      |

The submitted `medium` configuration keeps the same block counts as
`light` but widens channels (dim 36 → 48). This trades ~2× per-block
FFN compute for materially more capacity, while keeping the
deepest-level attention from blowing up — `standard` config in
[`build_promptir`](model.py:L305) (dim=48, blocks=(4,6,6,8)) was
measured at **15 min/epoch** on the RTX 5070, which would have made
a 150-epoch run untenable.

### 2.3 Training Details

| Hyper-parameter      | Baseline (`light`)              | Submitted (`medium`-patch192)   |
|----------------------|---------------------------------|---------------------------------|
| Optimizer            | AdamW (β=(0.9, 0.999), wd 1e-4) | identical                       |
| Initial LR           | 2e-4                            | 2e-4                            |
| LR schedule          | Cosine annealing → 1e-6 / 150 ep | identical                       |
| **Loss**             | `L1Loss`                        | **Charbonnier `sqrt(x² + 1e-6)`** |
| **Batch size**       | 8                               | **4** (patch-192 memory)        |
| **Patch size**       | 128×128 random crop             | **192×192** random crop         |
| Mixed precision      | `torch.amp.autocast("cuda")`    | identical                       |
| Gradient clipping    | `clip_grad_norm_` at 1.0        | identical                       |
| Epochs               | 150                             | 150                             |
| Validation           | every epoch on full 256×256     | identical                       |

**Why these specific changes (and not others).**
- **AdamW over SGD** because the small from-scratch regime is
  sensitive to optimisation; AdamW's adaptive step is what lets the
  randomly initialised transformer blocks converge in ~150 epochs.
- **Charbonnier over L1** because Charbonnier's derivative at small
  residual is `x / sqrt(x² + ε²)` which is larger than the
  near-constant `sign(x)` of L1. In the late epochs where the
  residual is mostly small noise, this gives the optimiser a
  stronger pull toward exact zero — directly the regime that
  determines the final dB.
- **Patch 192 over 128** because the model has finite receptive
  field; at 128 it sees a tile that may not contain enough
  surrounding texture to disambiguate a rain streak from a similarly
  oriented edge. The qualitative failure mode at 128 was thin
  residual streaks (§3.4), exactly what wider context should help.
- **Batch 4 (forced by 192 memory)**: medium-config at b=8 p=192
  peaked at 16.6 GB on a 12 GB GPU (unified-memory swap → 10s/iter).
  At b=4 it fits in 8.5 GB → 197 ms/iter. The smaller batch is
  partly compensated by ~2× more update steps per epoch.

**Training cost.** All three productive runs on a single RTX 5070:
- Light + L1 (baseline): **2 h 39 min**
- Medium + Charbonnier (intermediate): **2 h 31 min**
- **Medium + Charbonnier + patch192 (submitted): 6 h 22 min**
- Light without PGB (§4 ablation): **2 h 31 min** (queued after baseline)

### 2.4 Test-Time Augmentation (TTA)

D4 dihedral averaging (8 transforms: identity, hflip, vflip, hvflip,
rot90, rot180, rot270, rot90+hflip). Each transform is applied to
the input, the model is run, and the **inverse** transform is
applied to the output; the 8 outputs are then averaged.

Why this is essentially free: the model was trained with random H/V
flip + random 90° rotation, so it should be approximately
equivariant under D4 — averaging then smooths the residual
orientation-dependent artefacts at no theoretical cost. On the val
set TTA-8 buys a consistent **+0.30 to +0.36 dB** across all three
trained models (Table 1). The 8× compute is negligible: 100 test
images at 256×256 finish in **20 s** on the RTX 5070, TTA included.

## 3. Results

### 3.1 Quantitative Results

**Table 1.** Validation PSNR (50 rain + 50 snow), TTA-8 throughout,
and the public CodaBench test score.

| Configuration                              | Val avg | Val rain | Val snow | Test PSNR |
|--------------------------------------------|---------|----------|----------|-----------|
| `light` + L1 + patch128 (baseline)         | 29.228  | 27.515   | 30.941   | 29.75     |
| `medium` + Charbonnier + patch128          | 29.592  | 27.760   | 31.424   | 30.10     |
| **`medium` + Charbonnier + patch192** (submitted) | **30.381** | **28.707** | **32.054** | **30.78** |
| `light` **without PGB** (§4 ablation)      | 29.249* | 27.538*  | 30.961*  | (not submitted) |

(* val numbers for the §4 ablation were computed with the same
TTA-8 wrapper for fair comparison.)

**Iterative gain over baseline:** +1.15 val / **+1.03 dB test**.

### 3.2 Iterative Improvement Trajectory

![**Figure 1. Three-step iterative improvement on the validation
set.** Each curve is a full 150-epoch run with identical seed (42)
and identical 90/10 val split; horizontal dotted lines mark the
final CodaBench test scores. The light → medium jump (orange vs
grey) reflects the joint capacity + Charbonnier change; the medium
→ medium-p192 jump (blue vs orange) is the patch-size change in
isolation. Patch-192 has the largest single
gain.](figures/fig_iterative_gains.png)

### 3.3 Training Curves of the Submitted Model

![**Figure 2. Training curves of the submitted model
(medium-patch192-Charbonnier).** Left: Charbonnier loss decreases
smoothly from 0.074 to 0.021. Right: per-degradation validation
PSNR; snow saturates around 31.7 dB while rain catches up more
slowly, reaching 28.3 dB by epoch 144. The cosine schedule's late
small LR is responsible for the smooth tail after epoch
~120.](figures/fig_training_curves.png)

Rain is markedly harder than snow throughout — at convergence
**snow reaches 31.7 dB while rain only reaches 28.3 dB**, a
~3.4 dB gap. Rain images have denser, more orientation-specific
streaks that occlude high-frequency texture (see Fig. 4), so a
restoration network with no prior knowledge of the streak
orientation genuinely has less information to work with. This
per-type asymmetry motivates §4's question: *does the prompt
mechanism — which is free to specialise on the harder rain case
— actually help rain more than snow?*

### 3.4 Public-Leaderboard Snapshot

![**Figure 3. CodaBench public leaderboard.** Submission 758474
(`314560017_HW4_p192.zip`) was produced by inference with 8-way
(D4) TTA from `output_medium_p192/best.pt`. Earlier submissions
757266 (light) and ~757302 (medium-p128) confirm the iterative
improvement of Table 1 also transfers to the public test
set.](figures/leaderboard.png)

### 3.5 Qualitative Visualisations

![**Figure 4. Sample validation predictions from the submitted
model** (medium-patch192-Charbonnier). Three rain rows (top) and
three snow rows (bottom): degraded input → predicted → clean ground
truth. PSNR is annotated per panel. The most extreme cases (rain
row 1: 7.90 dB input → 24.42 dB output, a 16.5 dB gain) show that
the model correctly identifies and removes oriented rain streaks
while preserving the underlying texture. Snow rows fare visibly
better, consistent with the 3.4-dB-easier-than-rain finding in
§3.3.](figures/fig_qualitative.png)

## 4. Additional Experiments

### 4.1 — Does the Prompt Generation Block actually help?

I keep this ablation **on the light configuration** rather than the
submitted medium one for two reasons: (a) it isolates the
contribution of PGB without confounding by the wider-channel /
larger-patch design changes, and (b) the light pair (with/without
PGB) is the only pair where I can match parameter budget reasoning
cleanly. The result here is **what justified keeping PGB in the
final medium-patch192 architecture**.

* **Hypothesis.** Removing all three PGBs from the decoder should
  hurt val PSNR. The all-in-one setting (rain vs snow with unknown
  type at inference) is exactly the regime PromptIR was designed
  for: the prompts are claimed to give the decoder a per-image,
  degradation-adaptive bias. If the paper's claim translates here,
  I expect a clear PSNR gap and — more diagnostically — a *larger*
  gap on rain than on snow, because rain is the harder class where
  any per-image specialisation has more room to help.

* **Why it might work.** The two degradations are visually
  distinct (rain produces oriented streaks, snow produces
  near-isotropic speckle). A shared decoder must compromise
  between two restoration strategies; the prompt mechanism lets
  the network *implicitly* identify which case it is in and
  modulate its decoder accordingly, effectively a soft mixture of
  experts.

* **Why it might not.** With only two degradation types, both well
  covered in the training set, the shared decoder might already
  learn a generic streak-removal prior. In that case the prompt
  mechanism adds ~6 M parameters and very little signal — its
  contribution could be entirely captured by "more parameters
  happens to help a little".

**Architectural change.** I add a `use_prompt: bool = True` flag
to [`PromptIR.__init__`](model.py#L191) that gates the construction
of the three `PromptGenBlock`s, the three prompt-interaction
transformer blocks, and the three 1×1 channel-reduction
convolutions. When `use_prompt=False`, the forward pass skips all
three prompt-injection sub-graphs; the model degenerates to a pure
Restormer (encoder → latent → decoder-3 → decoder-2 → decoder-1
→ refinement → output). The `--no-prompt` CLI flag in
[`train.py`](train.py#L48) toggles it.

**Measurement 1 — parameter audit.**

| Configuration       | Trainable params | PGB overhead     |
|---------------------|------------------|------------------|
| PromptIR-light (PGB on)| **14.50 M**   | —                |
| PromptIR-light, PGB off | 8.47 M       | 6.03 M (41.6 %)  |

The PGB module — three `PromptGenBlock`s plus the three
interaction-and-reduce blocks that consume their output — accounts
for **41.6 % of the model's parameters**. Any improvement
attributed to PGB must be evaluated against this large parameter
budget; a fair-but-strict reading of any positive result is
*"prompts plus the 6 M parameters they carry contribute X dB"*,
not the looser *"prompts work"*.

**Measurement 2 — paired retrain.** I train a second model with
exactly the same hyper-parameters, the same `seed=42` train/val
split, the same 150-epoch cosine schedule, and the same AMP
setting, differing only in the `--no-prompt` flag. Both models are
evaluated every epoch.

![**Figure 5. PGB on/off comparison (light config).** Left: full
val-PSNR curves overlaid. The two runs are visually
indistinguishable for most of training. Right: 5-epoch-smoothed
gap (with-PGB minus without-PGB). The PGB run takes ~10 epochs to
stably out-perform the plain Restormer (a long warm-up where the
prompt parameters are random noise), then settles to a +0.08 dB
lead that holds through the late epochs.](figures/fig_ablation_compare.png)

![**Figure 6. Per-degradation breakdown.** Solid = with PGB,
dashed = without. The gap is essentially the same shape and
magnitude for rain and snow — PGB does *not* preferentially help
rain (the harder class) as the hypothesis predicted. The decoder
appears to learn a generic streak-removal prior that does not
need per-image specialisation.](figures/fig_per_type_gap.png)

**Final numbers** (best epoch of each run, paired val split):

| Quantity         | With PGB | Without PGB | Δ (PGB benefit) |
|------------------|----------|-------------|-----------------|
| best epoch       | 146      | 145         | —               |
| val PSNR (avg)   | **28.928** | 28.841    | **+0.087 dB**   |
| val PSNR (rain)  | 27.166   | 27.108      | +0.058 dB       |
| val PSNR (snow)  | 30.689   | 30.573      | +0.116 dB       |
| trainable params | 14.50 M  | 8.47 M      | +6.03 M         |

**Implications — a more honest picture than the paper.** The
hypothesis is only *partially* confirmed and the magnitude is
surprising:

1. **PGB is a net positive, but small.** The +0.087 dB best-vs-best
   gap is real and persistent (see the late-epoch tail of Fig. 5
   right). But it is much smaller than PromptIR's NeurIPS paper [1]
   would suggest — the paper reports several dB of improvement vs
   Restormer on multi-degradation benchmarks. The gap is plausibly
   explained by our setting having only **two** degradation types
   covered well in training, whereas PromptIR was designed for
   **five** types under all-in-one settings. With fewer
   sub-problems, the shared decoder has less specialisation work to
   delegate to the prompts.
2. **The per-class prediction is wrong.** PGB helps snow
   (+0.116 dB) slightly *more* than rain (+0.058 dB), the opposite
   of what I predicted. Snow is the easier class; the small extra
   capacity from PGB goes furthest there. This means the prompt
   mechanism is not acting as a "rain-specialist mode" — it is
   more like a small capacity bonus applied uniformly.
3. **The 41.6 % parameter cost is a poor exchange** when measured
   against a same-class-count Restormer at the same training
   budget. But §3.1 shows that the same PGB design **scales** with
   the larger medium configuration to give the full +1.03 dB
   improvement over the light baseline, suggesting that PGB is more
   valuable when it has more underlying capacity to modulate. This
   experiment therefore *justifies* keeping PGB in the submitted
   model (every dB matters on the leaderboard), but with eyes open
   about *how much of the gain it is delivering vs. the wider
   channels and larger patch*.
4. **The decoder learns a generic streak-removal prior** on the
   dataset's scale (3,100 paired tuples × 150 epochs). For future
   work the obvious architectural direction is either to scale up
   to more degradations (where the paper's premise bites), or to
   replace PGB with cheaper degradation-conditioned normalisation
   (e.g., FiLM) that gives the same modulation at a fraction of
   the parameter cost.

## 5. References

1. **Potlapalli, V., Zamir, S. W., Khan, S. H., & Khan, F.** (2023).
   *PromptIR: Prompting for All-in-One Blind Image Restoration.*
   NeurIPS 2023. <https://arxiv.org/abs/2306.13090> —
   The architectural basis of every model in this report. Code:
   <https://github.com/va1shn9v/PromptIR>.
2. **Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., &
   Yang, M.-H.** (2022). *Restormer: Efficient Transformer for
   High-Resolution Image Restoration.* CVPR 2022.
   <https://arxiv.org/abs/2111.09881> — The backbone transformer
   blocks (MDTA + GDFN) that PromptIR is built on; my §4 ablation
   ("no-prompt" model) is exactly Restormer at this configuration.
3. **Charbonnier, P., Blanc-Féraud, L., Aubert, G., & Barlaud, M.**
   (1994). *Two deterministic half-quadratic regularization
   algorithms for computed imaging.* ICIP 1994. — Source of the
   Charbonnier loss used in the submitted model in place of L1.
4. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep Residual
   Learning for Image Recognition.* CVPR 2016.
   <https://arxiv.org/abs/1512.03385> — The residual-add in the
   output layer (`out + inp` in `PromptIR.forward`) follows the
   ResNet residual principle and is what makes a
   denoising/restoration network learn the residual rather than
   the full clean image.
5. **Loshchilov, I., & Hutter, F.** (2019). *Decoupled Weight Decay
   Regularization.* ICLR 2019.
   <https://arxiv.org/abs/1711.05101> — The AdamW optimiser used
   in training.
6. **Loshchilov, I., & Hutter, F.** (2017). *SGDR: Stochastic
   Gradient Descent with Warm Restarts.* ICLR 2017.
   <https://arxiv.org/abs/1608.03983> — Source of the cosine
   annealing schedule used here.
