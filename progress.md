# Continuous Model Training Progress

## Summary
Training a LLaMA-based Transformer to predict mechanism joint coordinates from VAE latent vectors.

**Best Result So Far**: **4.62 mm** Val Euclidean Error (Phase 5)
**Current Best Candidate**: Linear 8-freq + Huber @ epoch 45 = **4.73 mm** (still improving)

---

## Experiment Log

| Phase | Fourier Type | Frequencies | Loss | d_model | Layers | Weight Decay | Train Loss | Val Loss | Best Val Euc Err | Notes |
|-------|--------------|-------------|------|---------|--------|--------------|------------|----------|------------------|-------|
| 1 | None (raw x,y) | - | MSE | 512 | 6 | 0.0001 | - | - | ~9+ mm | ❌ No learning |
| 1b | None + Norm | - | MSE | 512 | 6 | 0.0001 | ~0.99 | ~0.80 | 6.3 mm | ✅ Normalization fixed learning |
| 2 | None + Norm | - | MSE | 512 | 6 | 0.01 | - | - | 6.3 mm | ⚠️ Overfitting delayed |
| 3 | Deterministic 2^k | 32 | MSE | 256 | 4 | 0.01 | ~0.2 | ~0.72 | 5.8 mm | ⚠️ Smaller model, still overfits |
| 4 | Deterministic 2^k | 32 | MSE | 256 | 4 | 0.01 | ~0.22 | ~0.63 | 4.56 mm | ✅ Better, but plateaued |
| 5 | Deterministic 2^k | 32 | MSE | 512 | 6 | 0.001 | ~0.24 | ~0.617 | **4.62 mm** | ✅ **Previous Best** |
| 6a | Gaussian RFF | 256, σ=1.0 | MSE | 512 | 6 | 0.001 | ~0.04 | ~0.664 | 4.74 mm | ❌ Worse overfitting |
| 6b | Gaussian RFF | 256, σ=0.1 | Huber | 512 | 6 | 0.001 | ~0.03 | ~0.35 | 5.94 mm | ❌ Much worse |
| 7 | Linear 1,2,...,8 | 8 | Huber | 512 | 6 | 0.001 | - | ~0.27 | 4.73 mm | 🔄 Stalled/Marginal Gains |
| **8** | **Linear 1...32 (No Pi)** | **32** | **Huber** | 512 | 6 | 0.001 | ~0.257 | ~0.274 | **4.78 mm** | ✅ **Stable / No Overfitting** |

---

## Key Learnings

1. **Data Normalization is Critical** — Raw coordinates in [-10, 10] broke training completely.
2. **Exponential Frequencies (2^k) are Too High** — 2^31 ≈ 2 billion is way beyond useful for data scale ~10.
3. **Gaussian RFF Failed** — Random frequencies didn't help; made overfitting worse.
4. **Linear Frequencies Work** — 1, 2, 3, ..., 8 provides reasonable multi-scale encoding.
5. **Huber Loss** — Jury still out; may help with robustness to outliers.

---

## Current Configuration (Phase 7)

```python
num_epochs = 200
warmup_epochs = int(num_epochs * 0.1)  # 10% warmup
lr = 5e-4
weight_decay = 0.001

model_config = {
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 6,
    "dropout": 0.1,
    "num_freqs": 8,  # Linear: 1, 2, 3, ..., 8
}

loss = HuberLoss(delta=1.0) + BCE(stop_bit)
```
