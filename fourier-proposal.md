# Critical Review: Fourier Feature Insights from Tancik et al.

## Current Implementation
```python
# Deterministic frequencies: 2^0, 2^1, ..., 2^31
self.register_buffer("freq_bands", 2.0 ** torch.arange(32))
```
- **Type**: Positional encoding (deterministic)
- **Frequencies**: 32 (output = 128 dim)
- **Scale**: Fixed (no σ tuning)
- **Trainable**: No (buffer) ✓

---

## Insight 1: Gaussian RFF vs Deterministic
**Paper**: Gaussian random features N(0, σ²) outperform deterministic.

**Critical Analysis**: ⚠️ **Likely applies, BUT with caveats.**
- Deterministic 2^k frequencies are designed for *positional* encoding in sequences (each position gets unique representation).
- Our task is *coordinate regression* in 2D space — Gaussian RFF is explicitly designed for this.
- **Risk**: Gaussian RFF loses the "multi-scale" guarantee of 2^k. Low σ = only low frequencies; high σ = only high frequencies.

**Verdict**: ✅ **Apply** — Our task matches the paper's regression setting.

---

## Insight 2: σ is the Critical Hyperparameter
**Paper**: σ controls underfitting (low) vs overfitting (high).

**Critical Analysis**: ✅ **Highly relevant.**
- Our current implementation has NO tunable σ. The 2^k schedule implicitly covers σ ∈ [1, 2^31], but not uniformly.
- **Our data range**: [-10, 10]. The highest useful frequency is ~1/(data_resolution). If coordinates are ~0.1mm resolution, max useful freq ≈ 10 Hz.
- With 2^31 frequencies, we're encoding noise up to billions of Hz — almost certainly causing overfitting!

**Verdict**: ✅ **Apply** — This is likely our core issue. Need to sweep σ ∈ [0.1, 10] for our data scale.

---

## Insight 3: Keep Features Non-Trainable
**Paper**: Don't co-optimize Fourier parameters.

**Critical Analysis**: ✅ **Already implemented.**
- We use `register_buffer` which is non-trainable by default.

**Verdict**: ✓ No change needed.

---

## Insight 4: 256 Frequencies for Deeper Networks
**Paper**: Deeper networks need fewer features; suggests 256 frequencies as starting point.

**Critical Analysis**: ⚠️ **Partially applies.**
- We have 6 layers → medium depth.
- Currently using 32 frequencies (128-dim output).
- Paper used MLPs, not Transformers. Transformers already have powerful mixing via attention.
- 256 frequencies = 512-dim (matches d_model=512, convenient).

**Verdict**: 🔄 **Test** — Increase to 256 frequencies as part of the σ sweep.

---

# Implementation Plan

## Changes to `llama_latent_continuous.py`

### [MODIFY] `FourierFeatureEmbedding`
Replace deterministic 2^k with Gaussian RFF:
```python
class GaussianFourierFeatures(nn.Module):
    def __init__(self, input_dim=2, num_freqs=256, sigma=1.0):
        super().__init__()
        # Random frequencies from N(0, σ²)
        B = torch.randn(input_dim, num_freqs) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        # x: (..., 2)
        x_proj = 2 * math.pi * x @ self.B  # (..., num_freqs)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
```

### [MODIFY] `ContinuousInputEmbeddings`
- Use `GaussianFourierFeatures(input_dim=2, num_freqs=256, sigma=σ)`
- Output dim = 2 * 256 = 512 (matches d_model!)
- Update MLP: `Linear(512, 512)` → `Linear(512, 512)` (no change needed)

## Changes to `llama_latent_continuous_train.py`

### [MODIFY] Hyperparameter Grid
Add σ as a configurable parameter. Suggested sweep:
- σ ∈ [0.5, 1.0, 2.0, 5.0, 10.0]

## Verification
- Run 50-epoch training for each σ value.
- Compare Val MSE and Val Euc Err to find optimal σ.
