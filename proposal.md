# Proposal: Address Overfitting in Continuous Model

## The Dilemma
The **Discrete Model** works well with `d_model=512`.
The **Continuous Model** overfits with the same architecture (Validation MSE rises after epoch 15).

Why the difference?
1.  **Task Nature**:
    *   **Discrete (Classification)**: The model predicts a probability distribution over 200 bins. It learns "regions" of probability.
    *   **Continuous (Regression)**: The model must output exact floating-point numbers. With a high-capacity model ($d_{model}=512$), it is easy for the network to aggressively memorize the specific coordinate noise in the training set to minimize MSE, rather than learning the smooth underlying geometric manifold.
2.  **Input Density**:
    *   **Discrete**: Inputs are tokens. Embedding layer ($200 \times 512$) is a dense lookup.
    *   **Continuous**: Input is just 2 scalars $(x, y)$. Projecting $2 \to 512$ creates a highly redundant, sparse representation that leaves massive freedom for the weights to fit noise.

## Option A: "Shrink the Brain" (Recommended)
**Reduce `d_model` to 256, `layers` to 4.**

**Reasoning**:
*   **Regression is "Simpler"**: predicting a continuous value often requires fewer parameters than predicting a complex multimodal distribution (classification).
*   **Force Generalization**: By creating a bottleneck, we force the model to capture the *dominant* geometric relationships (the mechanism constraints) rather than the specific noise of the training samples.
*   **Empirical Evidence**: High-dimensional regression almost always requires strong regularization. If we don't want to increase data size (fixed dataset), we must reduce model capacity.

## Option B: Coordinate Deconstruction / Fourier Features
**Keep `d_model=512`, but change input representation.**

**Reasoning**:
*   Instead of projecting raw $(x, y)$, map them to High-Dimensional Fourier Features ($\sin(2^k \pi x), \cos(2^k \pi x)$) before the MLP.
*   **Pros**: This is standard in fields/NeRFs. It helps the model learn high-frequency details.
*   **Cons**: It might actually **worsen** overfitting if the issue is memorization of high-frequency noise. NeRFs use this *to* memorize a scene. We want the opposite (generalization).

## Conclusion
I argue for **Option A**. The fact that the Discrete model works with 512 dims doesn't mean the Continuous model *needs* 512 dims. They are fundamentally different value mappings. The continuous model is overfitting because it has too much capacity to fit the "noise" in the exact float values. Constraining it will force it to learn the signal.

## Appendix A: Latent Token MLP
*See previous section.*

## Appendix B: Why MLP for Input Embeddings?
The user asked: *Why use an MLP for `ContinuousInputEmbeddings`? Won't it overfit?*

**Comparison**:
1.  **Discrete Model**: Uses `nn.Embedding(vocab_size, d_model)`.
    *   This learns a unique, independent vector for every token.
    *   The points can be placed *anywhere* in the `d_model`-dimensional space.
    *   It is maximally expressive.
2.  **Continuous Linear**: `nn.Linear(2, d_model)`.
    *   This maps all $(x, y)$ points onto a single 2D **flat plane** slicing through the `d_model`-dimensional space.
    *   It effectively restricts the Transformer to only see a 2D linear slice of the world.
3.  **Continuous MLP**: `Linear -> ReLU -> Linear`.
    *   This effectively "folds" or "warps" the 2D input plane into a non-linear manifold within the `d_model` space.
    *   This allows the model to represent complex relationships between coordinates, similar to how `nn.Embedding` allows arbitrary placement of tokens.
    *   **Overfitting**: The parameter count of this MLP ($256 \times 256 \approx 65k$) is tiny compared to the attention layers ($12 \times 256 \times 256 \approx 786k$ per layer). Validated by the fact that `loss` is now decreasing properly.


## Appendix C: Why Fourier Features Work
The user asked: *Make your case for Fourier features... cite any research.*

**The Theory: Spectral Bias**
Standard MLPs (and by extension, the Linear projections in Transformers) suffer from **Spectral Bias**: they are naturally biased towards learning low-frequency functions. They struggle to learn high-frequency variations in low-dimensional domains (like our 2D coordinates) unless the network is extremely deep or wide.

**The Solution: Coordinate Deconstruction**
By mapping input coordinates $v$ to a higher dimensional feature space Using sinusoidal functions:
$$ \gamma(v) = [\dots, \sin(2^k \pi v), \cos(2^k \pi v), \dots] $$
We enable the MLP to tune its weights to specific *frequencies* of the input signal.

**Key Research**
> **"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"**
> *Tancik et al., NeurIPS 2020*
>
> This paper proves that passing inputs through a Fourier feature mapping transforms the **Neural Tangent Kernel (NTK)** to be stationary and tunable. This allows us to control the frequency spectrum that the network can learn.
>
> **Relevance to Us**: Our linkage paths are smooth but complex geometric curves. The raw $(x,y)$ input is "too low frequency" for the model to capture the precise curvature and constraints. Fourier features give the model a multi-scale "grid" to latch onto, effectively turning the regression problem into a richer interpolation task.

