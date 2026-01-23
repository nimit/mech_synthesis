# Fourier Features in Mechanism Path Synthesis

## 1. The Problem: Spectral Bias in Coordinate Regression
We are training a Transformer to output continuous 2D coordinates $(x, y)$ for mechanical linkages. Unlike standard NLP (discrete tokens), our input is low-dimensional (just 2 scalars).

**Theory**: Standard Neural Networks suffer from **Spectral Bias** (Rahaman et al., 2019). They naturally prioritize learning low-frequency components of a signal and struggle to capture high-frequency details (sharp turns, precise curvature) when the input dimension is low.

If we just feed raw $(x, y)$ into an MLP, the model tends to output "blurrier" or over-smoothed paths, failing to capture the strict geometric constraints of a bar linkage.

---

## 2. The Solution: Coordinate Deconstruction
To fix this, we map the input coordinates $v \in \mathbb{R}$ to a higher-dimensional feature space using sinusoidal functions, as recommended by *Tancik et al. (NeurIPS 2020)*.

$$ \gamma(v) = [\dots, \sin(f_k \pi v), \cos(f_k \pi v), \dots] $$

This allows the MLP to tune its weights to specific **frequencies** ($f_k$) of the input signal, effectively turning the "hard" regression problem into a richer interpolation task.

---

## 3. Evolution of Our Implementation

### Attempt A: Exponential Frequencies (The "Standard" Way)
Initially, we used the standard Positional Encoding schedule common in Transformers:
$$ f_k = 2^k \quad \text{for } k=0 \dots 31 $$
*   **Frequencies**: 1, 2, 4, 8, ..., $2^{31}$ (approx 2 billion).
*   **Result**: ⚠️ **Overfitting**.
*   **Why**: Our data $(x, y)$ is in the range $[-10, 10]$. A frequency of $2^{31}$ tries to capture variations at the scale of $10^{-9}$ (nanometer scale). This forced the model to memorize numerical noise rather than the mechanism's shape.

### Attempt B: Gaussian Random Fourier Features (RFF)
We tried sampling frequencies from a Gaussian distribution $\mathcal{N}(0, \sigma^2)$, matching Tancik et al.'s recommendation for regression.
*   **Result**: ❌ **Unstable**.
*   **Why**: Hard to tune $\sigma$. Low $\sigma$ underfitted; high $\sigma$ introduced high-frequency noise similar to Attempt A.

### Attempt C: Linear Harmonic Frequencies (The Winner)
We realized that mechanical linkages are physical systems governed by harmonic constraints (lengths, angles). Their paths are smooth curves decomposable into a few dominant harmonics.

We switched to **Linearly Spaced Frequencies**:
$$ f_k = k \quad \text{for } k=1 \dots N $$
*   **Frequencies**: 1, 2, 3, 4, ..., 64.
*   **Normalization**: Replaced the standard $\pi$ scaling with unity to stabilize gradients.

### Final Configuration
*   **Num Frequencies**: 64
*   **Spacing**: Linear ($1, 2, \dots, 64$)
*   **Scaling**: No $\pi$ factor (stabilizes gradients).
*   **Output Dim**: $2 \times 64 \times 2 = 256$ dimensions (embedded to 512).

---

## 4. Why This Works
By using linear frequencies $[1, 64]$, we provide the model with a "harmonic basis" that matches the physical bandwidth of the problem.
*   **Low Frequencies (1-10)**: Capture the general loop shape and size.
*   **Mid Frequencies (10-30)**: Capture the specific curvature changes.
*   **High Frequencies (30-64)**: Allow precision at the joints without encoding infinite-resolution noise.

This finding was empirically validated when the **Linear-64 model achieved our best-ever Euclidean error (~4.15 mm)**, significantly outperforming the exponential baseline (~4.62 mm) and minimizing generalization gap.
