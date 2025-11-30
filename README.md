# Curriculum Token Drop: Training-Efficient Vision Transformers

## Table of Contents
- [Objective](#objective)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Architecture](#architecture)
- [Implementation Details](#implementation-details)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Analysis](#analysis)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Objective

This project investigates **Curriculum Token Drop (CTD)**, a training-time strategy that progressively reduces token attendance in Vision Transformers based on attention saliency. The goal is to achieve training-time acceleration while maintaining model architecture and inference performance, enabling efficient ViT training on resource-constrained hardware.

---

## Problem Statement

Vision Transformers (ViTs) have demonstrated strong performance on image recognition tasks but remain computationally expensive due to uniform processing of all image patches regardless of their semantic importance. During training, the model must process all patches equally, leading to:

- **Computational overhead:** Unnecessary computation on low-importance patches
- **Training latency:** Longer training times, especially on consumer hardware
- **Resource constraints:** Limited applicability for practitioners with restricted compute budgets

**Research Question:** Can a curriculum-based token dropping strategy reduce training compute while preserving model performance and generalization?

---

## Methodology

### Core Idea

CTD implements a **representation curriculum** where token retention decreases gradually during training according to a polynomial schedule based on attention saliency:

$$d(t) = d_{init} + (d_{final} - d_{init}) \times \left[\frac{t - t_{start}}{t_{end} - t_{start}}\right]^\gamma$$

Where:
- $d(t)$ = drop ratio at epoch $t$
- $d_{init}$ = initial drop ratio (0.0)
- $d_{final}$ = final drop ratio (0.35)
- $t_{start}$ = curriculum start epoch (20)
- $t_{end}$ = curriculum end epoch (160)
- $\gamma$ = curriculum aggressiveness parameter (1.5)

### Token Saliency Computation

Token importance is computed as mean attention received from the CLS token:

$$s_i = \frac{1}{H} \sum_{h=1}^{H} A^h_{0,i}$$

Where $H$ is the number of attention heads and $A^h_{0,i}$ is attention from CLS (position 0) to token $i$ in head $h$.

### Training Strategy

1. **Early epochs (0-20):** Full token exposure for stable global representation learning
2. **Curriculum phase (20-160):** Progressive token dropping following polynomial schedule
3. **Late epochs (160-200):** Final curriculum with 35% tokens dropped for efficiency

This "slow-early, aggressive-late" approach balances:
- Early stability (diverse data for robust features)
- Late efficiency (constrained token set enforces compact representations)

---

## Architecture

### Hybrid CNN-ViT Backbone

The model combines a convolutional stem with Vision Transformer blocks:

- **CNN Stem:** 3-layer convolutional stem (3 → 64 → 128 → 384 channels) with stride-2 pooling
- **ViT Blocks:** 8 transformer blocks with:
  - Embedding dimension: 384
  - Attention heads: 6
  - MLP ratio: 4.0
  - Stochastic depth: 0.1
  - Dropout: 0.1

**Why Hybrid?** CNN stem provides local feature extraction and data-efficient learning on small datasets like CIFAR-100.

### Modified Attention Block with Token Dropping

```python
def forward(self, x, epoch=None):
    # Standard attention + norm
    normed = self.norm1(x)
    attn_out, attn_weights = self.attn(normed, return_attn=True)
    
    # Compute saliency from CLS token attention
    saliency = attn_weights.mean(dim=1)[:, 0, 1:]  # (B, N_patches)
    
    # Determine drop ratio for current epoch
    drop_ratio = self._get_drop_ratio(epoch)
    keep = max(1, int(N_patches * (1.0 - drop_ratio)))
    
    # Select top-k salient tokens
    _, idx = torch.topk(saliency, k=keep, dim=1, largest=True)
    
    # Keep CLS token + selected patches
    kept_tokens = torch.gather(tokens, 1, idx)
    x_kept = torch.cat([cls_token, kept_tokens], dim=1)
    
    # Continue with reduced token set
    x = x_kept + self.drop_path(attn_out_kept)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
```

---

## Implementation Details

### Training Configuration

```python
config = {
    "image_size": 32,
    "patch_size": 4,
    "in_channels": 3,
    "num_classes": 100,
    "emb_dim": 384,
    "num_heads": 6,
    "depth": 8,
    "mlp_ratio": 4.0,
    "drop": 0.1,
    "drop_path": 0.1,
    "batch_size": 128,
    "epochs": 200,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "warmup_epochs": 5,
    "label_smoothing": 0.1,
    "seed": 42,
    # CTD-specific
    "use_token_drop": True,
    "initial_drop_ratio": 0.0,
    "final_drop_ratio": 0.35,
    "ctd_start_epoch": 20,
    "ctd_end_epoch": 160,
    "ctd_gamma": 1.5
}
```

### Optimizer & Scheduling

- **Optimizer:** AdamW (lr=3e-4, weight_decay=0.05)
- **LR Schedule:** Cosine annealing with 5-epoch linear warmup
- **Regularization:** Label smoothing (0.1), AutoAugment, Random Erasing, Stochastic depth (0.1)

### Hardware

- **Device:** Apple M2 Pro (16-core GPU via PyTorch MPS backend)
- **Framework:** PyTorch 2.0+
- **Precision:** FP32

---

## Repository Structure

```
CTD-ViT/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.py                          # Configuration parameters
├── models/
│   ├── __init__.py
│   ├── patch_embed.py                # Convolutional patch embedding
│   ├── attention.py                  # Multi-head attention with optional saliency
│   ├── mlp.py                        # MLP feedforward block
│   ├── transformer_block.py          # Transformer block with CTD
│   └── vit.py                        # Full ViT model
├── data/
│   ├── __init__.py
│   └── cifar100_loader.py            # CIFAR-100 data loading & augmentation
├── training/
│   ├── __init__.py
│   ├── train.py                      # Main training script
│   ├── utils.py                      # Training utilities (logging, checkpoints)
│   └── metrics.py                    # Evaluation metrics (accuracy, F1, etc.)
├── experiments/
│   ├── baseline.py                   # Train baseline model (no CTD)
│   ├── ctd_train.py                  # Train with CTD (γ=1.5)
│   └── ablations/
│       ├── gamma_sensitivity.py      # Ablate γ ∈ {0.5, 1.0, 1.5, 2.0, 3.0}
│       ├── saliency_metrics.py       # Compare saliency metrics
│       └── schedule_design.py        # Compare schedule shapes
├── notebooks/
│   ├── results_analysis.ipynb        # Visualize results & curves
│   ├── attention_visualization.ipynb # Visualize attention maps
│   └── saliency_analysis.ipynb       # Analyze token saliency patterns
├── checkpoints/
│   ├── baseline_final.pth            # Baseline model weights
│   └── ctd_gamma1.5_final.pth        # CTD model weights
└── results/
    ├── accuracy_curves.png           # Training/val accuracy over epochs
    ├── loss_curves.png               # Training/val loss over epochs
    └── results.csv                   # Quantitative results table
```

---

## Results

### Main Results: Accuracy & Efficiency

| Method | Validation Acc | Final Train Acc | Training Time | Train-Val Gap |
|--------|---|---|---|---|
| **Baseline (Hybrid ViT)** | 68.23% | 88.23% | 9h 10m | 20.0% |
| **CTD (γ=1.5)** | 67.53% ± 0.22% | 86.15% | 5h 43m | 18.6% |

**Key Finding:** CTD achieves **1.6× wall-clock training speedup** with a controlled **0.7% accuracy penalty** and **improved generalization** (18.6% vs 20.0% gap).

### Ablation Results: Curriculum Parameter Sensitivity

| γ | Val Acc | Train Time | Train-Val Gap | Status |
|---|---------|------------|---------------|--------|
| 0.5 | 68.10% | 5h 50m | 19.2% | Gentle |
| 1.0 | 67.82% ± 0.19% | 5h 45m | 18.8% | Moderate |
| **1.5** | **67.53% ± 0.22%** | **5h 43m** | **18.6%** | **Optimal** |
| 2.0 | 67.31% ± 0.20% | 5h 42m | 18.1% | Aggressive |
| 3.0 | 66.95% ± 0.25% | 5h 40m | 17.6% | Very aggressive |

**Observation:** γ=1.5 provides optimal trade-off. Method is robust across γ ∈ [1.0, 2.0] with <1% variation.

### Saliency Metric Comparison

| Metric | Val Acc | Train Time | Train-Val Gap |
|--------|---------|------------|---------------|
| **Attention-based** | **67.53% ± 0.22%** | **5h 43m** | **18.6%** |
| Gradient-based | 66.71% ± 0.28% | 6h 12m | 19.4% |
| Random dropping | 65.47% ± 0.31% | 5h 45m | 20.8% |

**Finding:** Attention-based saliency outperforms alternatives by 0.82% and is computationally cheaper (no backprop overhead).

---

## Analysis

### Why CTD Works

1. **Early Stability:** Full token exposure in epochs 0-20 allows stable global representation learning
2. **Progressive Specialization:** Gradual token removal encourages compact feature formation
3. **Regularization Effect:** Reduced token set acts as implicit regularizer, improving generalization
4. **Computational Efficiency:** Fewer tokens → fewer attention operations, linear speedup with token reduction

### Generalization Insight

The improved train-validation gap (20.0% → 18.6%) suggests curriculum-based token selection encourages robust feature learning:
- Model learns to focus on informative patches early
- Forced token selection prevents overfitting to entire patch set
- Results in more transferable representations

### When CTD Helps Most

✅ **Resource-constrained training** (limited GPU memory, edge devices)
✅ **Long training runs** (amortized overhead worth it)
✅ **Small image datasets** (CIFAR-100 scale or smaller)

❌ **Not ideal for** inference (no speedup—all tokens used)
❌ **Not ideal for** very large models (overhead > benefit)

---

## Limitations

1. **Single Dataset:** Validated on CIFAR-100 only (100 classes, 50K images). Generalization to ImageNet or domain-specific datasets untested.

2. **Single Architecture:** Tested on Hybrid CNN-ViT only. Results may vary for pure ViT, DeiT, Swin, or other architectures.

3. **Single Hardware:** Speedup measured on Apple M2 Pro MPS. May not generalize to GPUs (V100, A100), TPUs, or other hardware.

4. **No Statistical Significance Testing:** Only 3 random seeds per run. Larger sample needed for rigorous confidence intervals.

5. **Position Embedding Mismatch:** Position embeddings initialized for full 65 tokens (1 CLS + 64 patches) but reduced during training. More principled approaches (relative position bias, dynamic reindexing) unexplored.

6. **Saliency Metric Limitations:** Attention-based saliency may not capture all aspects of semantic importance. Gradient-based and information-theoretic metrics deserve deeper investigation.

---

## Future Work

### High Priority

1. **Validate on ImageNet-1K** — Test scalability to large, realistic datasets
2. **Test on Pure ViT & Other Architectures** — DeiT, Swin, Vision Transformer-B/L
3. **Compare Against Token Pruning Baselines** — Direct comparison with DynamicViT, Evo-ViT, ToMe
4. **Multiple Hardware Platforms** — GPU (V100, A100), TPU, edge devices

### Medium Priority

1. **Explore Alternative Saliency Metrics** — Fisher information, gradient signal-to-noise, information-theoretic measures
2. **Dynamic Position Embeddings** — Reindex position embeddings after token selection for principled approach
3. **Inference Optimization** — Extend speedup to inference time through knowledge distillation or pruning
4. **Theoretical Analysis** — Why does curriculum improve generalization? Connection to curriculum learning theory?

### Research Directions

1. **Learnable Saliency Weights** — Replace hard-coded attention with learned importance weights λ_i
2. **Multi-Scale Token Selection** — Different drop ratios per layer
3. **Combined Efficiency Techniques** — CTD + quantization + knowledge distillation
4. **Downstream Task Validation** — Fine-tuning on CIFAR-10, medical imaging, etc.

---

## References

[1] Dosovitskiy, A., et al. "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale." ICLR, 2021.

[2] Rao, Y., et al. "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification." ICCV, 2021.

[3] Xu, Z., et al. "Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer." NeurIPS, 2021.

[4] Bolya, D., et al. "Token Merging: Your ViT But Faster." CVPR, 2023.

[5] Lin, T.-Y., et al. "Focal Loss for Dense Object Detection." ICCV, 2017.

[6] He, K., et al. "Deep Residual Learning for Image Recognition." CVPR, 2016.

---

## How to Use

### Installation

```bash
git clone https://github.com/aumkeshchaudhary/CTD-ViT.git
cd CTD-ViT
pip install -r requirements.txt
```

### Training Baseline

```bash
python experiments/baseline.py
```

### Training with CTD

```bash
python experiments/ctd_train.py --gamma 1.5
```

### Ablation Studies

```bash
# Test different γ values
python experiments/ablations/gamma_sensitivity.py

# Compare saliency metrics
python experiments/ablations/saliency_metrics.py
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{chaudhary2024ctd,
  title={Curriculum Token Drop: Training-Efficient Vision Transformers},
  author={Chaudhary, Aumkesh},
  year={2024},
  howpublished={\url{https://github.com/aumkeshchaudhary/CTD-ViT}}
}
```

---

## Contact

For questions or suggestions, open an issue on GitHub or contact [your-email@example.com]

**Last Updated:** November 2024
