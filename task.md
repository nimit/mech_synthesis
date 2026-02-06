First goal: understand the "terrain" of the latent space.

start by modifying the code to forgo the use of KNN.
since we don't have mechanism data, we have to start by running all 17 mechanism types and then choosing the one with the best CD.

Then move on to the following steps towards the goal.

### Step 1: Sensitivity & Ruggedness Analysis

Perform a "line-search" in random directions from your query vector.

* **Goal:** Determine the **radius of stability**. How far can you move in  before the Chamfer Distance (CD) jumps from 0.1 to 5.0?
* **Metric:** Plot Distance from Seed vs. Chamfer Distance. If the graph looks like a jagged saw-blade, your latent space is "rugged," and you’ll need a more robust solver like CMA-ES.

### Step 2: The Search Phase (Candidate Generation)

Multi-Start Neighborhood Sampling (The "Smart Fan")

* **How it works:**
1. Generate 100 variations around  using a Gaussian distribution.
2. Sort by .
3. Take the top 5 "winners" and repeat the process around *each* of them with a smaller search radius (decaying noise).
4. Keep a "Hall of Fame" of unique mechanisms that hit the  mark.

---

### Alternative search method: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

CMA-ES is the "industry standard" for non-differentiable optimization in high-dimensional spaces.

* **How it works:** It starts with a "cloud" (search distribution) around your query. As it evaluates results, it doesn't just move the center of the cloud; it **stretches and rotates** the cloud to align with the "valleys" of low Chamfer Distance.