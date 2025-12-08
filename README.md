# Progressive Token Drop for Efficient Vision Transformers

A training-time strategy that progressively reduces token attendance in Vision Transformers based on attention saliency, achieving faster training while maintaining inference performance on CIFAR-100.

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Method](#method)
- [Architecture](#architecture)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

Vision Transformers (ViTs) have demonstrated competitive performance on image recognition benchmarks but remain computationally expensive due to uniform processing of all image tokens. This work investigates **Progressive Token Drop (PTD)**, a training strategy that progressively reduces token attendance based on attention saliency.

Unlike pruning-based approaches that target inference-time acceleration, PTD reduces compute only during training through a token curriculum, leaving the model architecture and inference unchanged. All tokens are retained at test time, ensuring no inference overhead.

---

## Problem Statement

Vision Transformers operate on sequences of image patches and apply self-attention uniformly across all tokens, regardless of their semantic contribution. On low-resolution datasets such as CIFAR-100, this creates redundancy: many patches carry background or low-information content but are processed at the same computational cost as highly informative regions.

**Key challenges:**
- Redundant computation on non-informative patches during training
- Increased training time, especially on resource-constrained hardware
- Most existing token pruning methods alter model architecture or couple pruning with inference constraints

**Research question:** Can a curriculum-based token dropping strategy reduce training compute while maintaining or improving model generalization without modifying inference?

---

## Method

### Progressive Token Drop Schedule

PTD progressively reduces the token drop ratio over epochs using a polynomial curriculum:

$$d(t) = \begin{cases}
d_{\text{init}} & \text{if } t < t_s \\
d_{\text{final}} & \text{if } t \geq t_e \\
d_{\text{init}} + \left[\frac{t - t_s}{t_e - t_s}\right]^\gamma (d_{\text{final}} - d_{\text{init}}) & \text{otherwise}
\end{cases}$$

Where:
- $d_{\text{init}} = 0.0$ — Initial drop ratio (keep all tokens early)
- $d_{\text{final}} = 0.35$ — Final drop ratio (drop 35% of tokens in late epochs)
- $t_s = 40$ — Curriculum start epoch
- $t_e = 140$ — Curriculum end epoch
- $\gamma = 2.0$ — Controls curriculum shape (slower increase early, aggressive later)

**Intuition:** Early epochs expose the full token set for stable learning, while later epochs focus on a curated subset, encouraging the model to learn compact, discriminative features.

### Attention-Based Saliency

Token importance is computed as the mean attention each patch token receives from the class token, averaged over all attention heads:

$$s_i = \frac{1}{H} \sum_{h=1}^{H} A^{(h)}_{0,i}$$

Where:
- $A^{(h)} \in \mathbb{R}^{(N+1) \times (N+1)}$ — Attention weights from head $h$
- Index 0 denotes the class token
- Indices $1, \ldots, N$ correspond to patch tokens
- $H = 6$ — Number of attention heads

At epoch $t$, we keep the top-$k$ tokens where $k = \lfloor(1 - d(t)) \cdot N \rfloor$.

### Forward Pass with PTD

During training in the PTD-modified transformer block:

1. Compute self-attention outputs and weights
2. If $d(t) \approx 0$ or outside the curriculum window, use standard block behavior
3. Otherwise, compute saliency scores $\{s_i\}$, select top-$k$ patch tokens, and reassemble:
   $$X' = [x_{\text{cls}}; x_{i_1}; \ldots; x_{i_k}]$$
4. Apply residual updates and MLP with the reduced sequence

**Key property:** PTD operates only during training in a single transformer block. At evaluation, the model runs with the full token set, behaving identically to the baseline.

---

## Architecture

### Baseline Hybrid ViT

Both baseline and PTD models share the same backbone:

**Convolutional patch embedding:**
- 3-layer convolutional stem mapping input $x \in \mathbb{R}^{3 \times 32 \times 32}$ to embedding $z \in \mathbb{R}^{192 \times 16 \times 16}$
- Flattened into $N = 16 \times 16 = 256$ patch tokens

**Vision Transformer:**
- Learnable class token + positional embeddings
- 6 transformer blocks with:
  - Embedding dimension: $E = 192$
  - Attention heads: $H = 6$
  - MLP expansion ratio: 4.0
  - Dropout: $p = 0.1$
  - Stochastic depth: linearly increasing drop-path rate up to 0.1

**Standard transformer block:**
$$Y = X + \text{DropPath}(\text{MHA}(\text{LN}(X)))$$
$$Z = Y + \text{DropPath}(\text{MLP}(\text{LN}(Y)))$$

### PTD Modifications

The PTD variant modifies one transformer block to enable token dropping during training while maintaining an identical inference-time forward pass.

---

## Experimental Setup

### Dataset & Preprocessing

**CIFAR-100:**
- 50,000 training images, 10,000 test images
- Image size: 32×32, 3 channels
- Normalized with mean = (0.5071, 0.4867, 0.4408), std = (0.2675, 0.2565, 0.2761)

### Data Augmentation

**Baseline:**
- Random crop with 4-pixel padding
- Random horizontal flip
- RandAugment (2 operations, magnitude 10)
- Random erasing (probability 0.25)

**PTD variant:**
- Same random crop and flip
- RandAugment (magnitude 7 — slightly softer)
- Random erasing (probability 0.1 — more conservative)
- Mixup (α = 0.8) and CutMix (α = 1.0, probability 0.5)
- Exponential Moving Average (EMA) of weights with decay β = 0.9999

### Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (weight decay 0.05) |
| Learning rate | 4×10⁻⁴ with 5-epoch linear warmup |
| Schedule | Cosine decay over 200 epochs |
| Label smoothing | 0.05 (baseline), 0.03 (PTD) |
| Batch size | 128 (train), 256 (eval) |
| Total epochs | 200 |
| Gradient clipping | Global norm clipping at 1.0 |
| Hardware | Apple M2 Pro GPU (PyTorch MPS backend) |

**Regularization differences:**
- Baseline: standard dropout (p=0.1), drop-path up to 0.1
- PTD: no dropout (p=0.0), milder drop-path (0.05), Mixup/CutMix/EMA for regularization

---

## Results

### Quantitative Comparison

| Method | Train Acc | Val Acc | Training Time | Speedup |
|--------|-----------|---------|---------------|---------|
| Baseline (no PTD, no aug) | 92.4% | 71.59% | 13h 40m | 1.0× |
| PTD only (no aug, no EMA) | 97.9% | 68.83% | 10h 20m | 1.3× |
| **PTD + Mixup/CutMix + EMA** | **53.5%** | **74.12%** | **10h 20m** | **1.3×** |

### Training Dynamics

The training accuracy column reveals important behavior:

- **Baseline:** 92.4% train accuracy, 71.59% validation (20.81% gap)
- **PTD only:** 97.9% train accuracy, 68.83% validation (29.07% gap) — **severe overfitting**
- **PTD + Augmentation:** 53.5% train accuracy, 74.12% validation (20.73% gap) — **soft labels from Mixup/CutMix**

---

## Key Findings

### 1. PTD Alone Increases Overfitting

A counterintuitive result: **PTD without proper regularization harms generalization**, reducing validation accuracy from 71.59% to 68.83% (−2.76pp) despite reducing training time. The train-validation gap increases from 20.81% to 29.07%, indicating severe overfitting.

**Why does token reduction lead to worse overfitting?**

**Token compression and memorization:** When PTD removes low-saliency tokens, the remaining tokens must encode all necessary information. Rather than learning distributed, generalizable representations, the model memorizes training-specific patterns in retained high-saliency regions—analogous to extreme bottleneck layers in autoencoders.

**Attention feedback loops:** The saliency metric is based on class-to-patch attention weights. Retained tokens receive even more attention in subsequent epochs and dominate the learned representation. Dropped tokens never recover, preventing the model from exploring alternative feature combinations that might generalize better.

**Capacity-regularization mismatch:** Classical wisdom suggests reducing capacity acts as implicit regularization. However, PTD selectively prunes the token dimension while keeping all parameters fixed. This asymmetric reduction fails to provide regularization benefits and creates pathological training dynamics.

### 2. PTD + Strong Regularization Achieves Accuracy Gain

When combined with **Mixup, CutMix, and EMA**, PTD achieves 74.12% validation accuracy—a **2.53 percentage-point improvement** over baseline (71.59%)—while reducing training time by **25%** (13h 40m → 10h 20m).

The extreme training accuracy drop to 53.5% reflects soft labels from heavy data augmentation, which forces the model to learn robust features despite reduced token capacity.

### 3. Regularization is Essential

PTD's benefits come from the **synergistic combination** of token reduction with heavy augmentation and weight averaging. Token reduction alone is harmful; proper regularization transforms it into an effective training strategy.

### 4. Inference Remains Unchanged

Unlike pruning-based methods, PTD leaves inference unchanged: all tokens are retained at test time. This means:
- ✅ No inference latency overhead
- ✅ No deployment complications
- ✅ Training-time efficiency only

---

## Repository Structure

```
PTD-ViT/
├── README.md
├── LICENSE
├── .gitignore
│
├── Baseline_ViT/
│   ├── Baseline_ViT_CIFAR100.ipynb     # Baseline training notebook
│   ├── baseline_accuracy.png           # Training/validation accuracy curves
│   ├── baseline_loss.png               # Training/validation loss curves
│   ├── baseline_lr_schedule.png        # Learning rate schedule
│   ├── baseline_history.pkl            # Training history (pickled dict)
│   └── baseline_metrics.csv            # Final metrics summary
│
├── PTD_ViT/
│   ├── PTD_ViT_CIFAR100.ipynb          # PTD training notebook
│   ├── ptd_accuracy.png                # Training/validation accuracy curves
│   ├── ptd_loss.png                    # Training/validation loss curves
│   ├── ptd_lr_schedule.png             # Learning rate schedule
│   ├── ptd_history.pkl                 # Training history (pickled dict)
│   └── ptd_metrics.csv                 # Final metrics summary
│
└── .gitattributes
```

---

## Limitations

1. **Single dataset:** Results limited to CIFAR-100 (100 classes, 50K images). Generalization to ImageNet, medical imaging, or other domains untested.

2. **Single architecture:** Tested only on a compact hybrid ViT with convolutional stem. Results may differ for pure Vision Transformer, DeiT, Swin, or other architectures.

3. **Single hardware:** Training measured on Apple M2 Pro GPU via PyTorch MPS. Speedup may not transfer to NVIDIA GPUs (V100, A100), TPUs, or other accelerators.

4. **No multi-run statistics:** Due to computational constraints, mean ± std over multiple random seeds not reported. More rigorous statistical analysis needed.

5. **Saliency metric limitations:** Attention weights do not always correlate with semantic importance. Alternative measures (gradient-based saliency, information gain, learned importance) remain unexplored.

6. **Curriculum hyperparameters not systematically tuned:** Start epoch ($t_s$), end epoch ($t_e$), final drop ratio ($d_{\text{final}}$), and shape parameter ($\gamma$) are set manually without ablation studies.

7. **Uncontrolled comparison:** The baseline was not trained with Mixup/CutMix/EMA, making it difficult to isolate PTD's contribution from augmentation effects. Ideally, both should be tested independently and in combination.

8. **Single block modification:** PTD applied to only one transformer block (conservative design). Multi-block PTD could yield larger speedups but may impair stability—unexplored.

---

## Future Work

### High Priority

1. **Controlled augmentation ablation:** Apply Mixup/CutMix/EMA to the baseline (without PTD) to isolate PTD's contribution independent of augmentation effects.

2. **Augmentation sensitivity analysis:** Systematically vary Mixup/CutMix strength to identify the boundary where PTD helps versus hurts performance, and determine minimal regularization needed.

3. **Larger-scale validation:** Validate PTD with strong augmentation on ImageNet-1K and ImageNet-21K to assess generalizability beyond CIFAR-100.

4. **Multi-architecture testing:** Test on pure Vision Transformer, DeiT, Swin, and other ViT variants to understand architecture dependence.

### Medium Priority

1. **Alternative saliency metrics:** Investigate gradient-based saliency, Fisher information, or learned importance predictors. Compare against attention-based baseline.

2. **Curriculum shape exploration:** Ablate curriculum parameters ($t_s$, $t_e$, $d_{\text{final}}$, $\gamma$) to find optimal settings for different datasets and architectures.

3. **Multi-block PTD:** Study multi-block token dropping with appropriate stabilization techniques to achieve larger speedups.

4. **Token reinstatement strategies:** Explore mechanisms to recover dropped tokens or dynamically re-rank tokens across epochs.

### Research Directions

1. **Inference-time efficiency:** Combine PTD's training-time curriculum with inference-time pruning or token merging for end-to-end speedups.

2. **Theoretical analysis:** Why does curriculum-based token reduction improve generalization? Connection to curriculum learning theory and implicit regularization?

3. **Saliency feedback analysis:** Do attention-based saliency metrics create pathological feedback loops that contribute to overfitting?

4. **Cross-dataset transfer:** Fine-tune PTD-trained models on downstream tasks (CIFAR-10, medical imaging) to assess transfer learning benefits.

---

## References

[1] Rao, Y., et al. "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification." ICCV, 2021.

[2] Xu, Z., et al. "Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer." NeurIPS, 2021.

[3] Bolya, D., et al. "Token Merging: Your ViT But Faster." CVPR, 2023.

[4] Anonymous. "PatchDrop: Efficient Transformers via Compressed Token Representations." ICLR, 2024.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{chaudhary2025ptd,
  title={Progressive Token Drop for Efficient Vision Transformers},
  author={Chaudhary, Aumkesh},
  institution={Indian Institute of Technology, Patna},
  year={2025}
}
```

---

**Author:** Aumkesh Chaudhary, Indian Institute of Technology, Patna  
**Contact:** aumkeshchaudhary@gmail.com
