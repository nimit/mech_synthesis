# Path Synthesis: Latent-Conditioned Transformer for Mechanism Trajectory Prediction

## 1. Task Overview
This project addresses the problem of **mechanism path synthesis**: given a compressed representation of a bar linkage mechanism, predict the sequence of joint positions that the mechanism traces through.

**Input**: A 50-dimensional latent vector from a pre-trained VAE (representing the mechanism's geometry) + a mechanism type label (one of 17 classes).

**Output**: A sequence of up to 8 joint positions, each being a 2D coordinate $(x, y)$ plus a stop bit indicating whether the trajectory has ended.

**Goal**: Train a Transformer decoder to autoregressively predict these joint positions, achieving low Euclidean error (in mm) on unseen mechanism configurations within the same mechanism families.

---

## 2. Dataset
- **Source**: `dataset_17mechs/` — Pre-generated data from 17 types of bar linkage mechanisms.
- **Size**: ~439,000 training samples, ~110,000 validation samples.
- **Contents per sample**:
    - `vae_mu.npy`: The 50-dim latent vector ($\mu$ from VAE encoder).
    - `inputs.npy` / `labels.npy`: Continuous $(x, y)$ coordinates. Range: approximately $[-10, 10]$.
    - `encoded_labels.npy`: Integer mechanism type ID (0-16).
    - `attn_mask.npy`: Boolean mask for variable-length sequences.
- **Data Split**: 80% Train / 20% Validation (Random split).

---

## 3. The VAE
A pre-trained Variational Autoencoder encodes each mechanism's discrete token representation into a 50-dimensional latent space ($z \sim \mathcal{N}(\mu, \sigma^2)$).

The Transformer does **not** have access to raw mechanism tokens — it only sees the latent vector $\mu$. This tests whether the latent space captures enough geometric information to reconstruct the path.

---

## 4. Model Progression & Results

### Phase 0: Discrete Transformer (Baseline)
- **Model**: `LatentLLaMA_SingleToken` — Predicts discretized (binned) coordinates as token IDs (vocabulary size 200).
- **Architecture**: `d_model=512`, 8 heads, 6 layers, `dropout=0.1`.
- **Result**: ✅ **Works well.** Validation accuracy matches training. Classification over bins is forgiving.

---

### Phase 1: Continuous Transformer (Initial Attempt)
- **Model**: `LatentLLaMA_Continuous` — Predicts raw $(x, y)$ floats via regression.
- **Architecture**: Same as Discrete (`d_model=512`, 6 layers).
- **Problem**: ❌ **Flat MAE.** Training loss did not decrease meaningfully.

**Root Cause**: The raw data was unnormalized (range $[-10, 10]$, std ~5.6). Deep learning models struggle with such scales.

**Fix (Phase 1b)**:
1. Implemented per-channel **z-score normalization** (mean/std from training set).
2. Changed input projection from `Linear(2, 512)` to an **MLP** (`Linear -> ReLU -> Linear`) to provide non-linear feature extraction.
3. Added **LR warmup** (5 epochs linear ramp).

**Result**: ✅ Loss started decreasing. Val Euc Error: ~7.0 → ~6.3 over 10 epochs (test run) but full training run resulted in overfitting.

---

### Phase 2: Regularization (Weight Decay & Dropout)
- **Observation**: After Phase 1 fixes, model started *overfitting* around epoch 11. Train MSE << Val MSE.
- **Fix**:
    - Increased `weight_decay`: `0.0001` → `0.01`.
    - Passed `dropout` to LlamaConfig's `attention_dropout`.

**Result**: ⚠️ Marginal improvement. Overfitting delayed but not resolved.

---

### Phase 3: Aggressive Capacity Reduction
- **Hypothesis**: The model capacity (`d_model=512`, 6 layers) is too high for the regression task; it memorizes noise.
- **Fix**:
    - `d_model`: `512` → `256`
    - `num_layers`: `6` → `4`
    - `dropout`: `0.1` → `0.3`

**Result (50 epochs)**: ⚠️ **Underfitting slightly.** Train/Val gap smaller (~0.1), but absolute error still high. Val Euc Error ~6.3.

**Result (1000 epochs)**: ❌ **Still overfits.** Best Val saved at epoch 114 (Val Euc Err: 5.83). By epoch 999, Train MSE: 0.2, Val MSE: 0.72. Gap re-emerged.

---

### Phase 4: Fourier Features (Current)
- **Hypothesis**: The raw $(x, y)$ representation is "too low frequency" for the Transformer/MLP to learn smoothly. Standard networks have Spectral Bias (struggle with high-frequency details in low-dim domains).
- **Fix**: Implement **Fourier Feature Embeddings** (Tancik et al., NeurIPS 2020).
    - Each coordinate $v$ is mapped to: $[\sin(2^0 \pi v), \cos(2^0 \pi v), \dots, \sin(2^{L-1} \pi v), \cos(2^{L-1} \pi v)]$
    - This expands 2D input to 128D before the MLP.

**Status**: 🔄 **Training in progress (100 epochs).** Awaiting results.

---

## 5. Summary Table

| Phase | Key Change | Train MSE | Val MSE | Val Euc Err (mm) | Status |
|-------|-----------|-----------|---------|------------------|--------|
| 0 (Discrete) | Classification | N/A | N/A | Low | ✅ Baseline |
| 1 (Raw Continuous) | Regression | Flat | Flat | ~9+ | ❌ Failed |
| 1b (Normalized + MLP) | Normalization, MLP | 0.99→0.80 | 0.98→0.80 | 7.4→6.3 | ✅ Learning |
| 2 (Regularization) | Weight Decay, Dropout | 0.08 | 0.80 | ~6.3 | ⚠️ Overfitting |
| 3 (Shrink Brain) | Smaller model | 0.56 | 0.72 | 5.8 | ⚠️ Still Overfits |
| 4 (Fourier Features) | Coordinate Deconstruction | TBD | TBD | TBD | 🔄 In Progress |

---

## 6. Key Files
- **Model Definition**: `llama_latent_continuous.py`
- **Training Script**: `llama_latent_continuous_train.py`
- **Dataset Loader**: `dataset_continuous.py`
- **Discrete Baseline**: `llama_latent_model.py`, `llama_latent_train.py`
