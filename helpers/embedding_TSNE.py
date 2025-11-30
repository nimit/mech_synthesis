import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# =========================================================
# Load model checkpoint
# =========================================================
ckpt_path = "./weights/transformer_weights_VIT_LLAMA/A100_LLAMA_d1024_h32_n6_bs512_lr0.0001_vit_llama.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(ckpt_path, map_location=device)
model_config = checkpoint["model_config"]

from vit_llama_model import SingleImageTransformerCLIP_LLaMA
model = SingleImageTransformerCLIP_LLaMA(**model_config).to(device)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()
print("✅ Model loaded successfully")

# =========================================================
# Extract mechanism embeddings
# =========================================================
with torch.no_grad():
    mech_embs = model.mech_embedding.weight.detach().cpu().numpy()  # (num_mechs, d_model)

num_mechs, d_model = mech_embs.shape
print(f"Mechanism embeddings shape: {mech_embs.shape}")

# =========================================================
# Define mechanism names (index -> name)
# =========================================================
mech_names = [
    "RRRR", "Steph1T1", "Steph1T2", "Steph1T3",
    "Steph3T1A1", "Steph3T1A2", "Steph3T2A1",
    "Watt1T1A1", "Watt1T1A2", "Watt1T2A1", "Watt1T2A2",
    "Watt1T3A1", "Watt1T3A2", "Watt2T1A1", "Watt2T1A2",
    "Watt2T2A1", "Watt2T2A2"
]
assert len(mech_names) == num_mechs, "Number of mechanism names must match embedding count!"

# =========================================================
# PCA reduction
# =========================================================
pca = PCA(n_components=2)
emb_pca = pca.fit_transform(mech_embs)
print(f"Explained variance ratio (PCA): {pca.explained_variance_ratio_}")

# =========================================================
# t-SNE reduction
# =========================================================
tsne = TSNE(n_components=2, perplexity=5, learning_rate=100, random_state=42)
emb_tsne = tsne.fit_transform(mech_embs)

# =========================================================
# Plot PCA vs t-SNE side-by-side
# =========================================================
plt.figure(figsize=(14, 6))
colors = plt.cm.tab20(np.linspace(0, 1, num_mechs))

# ---------------- PCA Plot ----------------
plt.subplot(1, 2, 1)
plt.scatter(emb_pca[:, 0], emb_pca[:, 1], s=200, c=colors, edgecolors="k", linewidths=1.2)
for i, name in enumerate(mech_names):
    plt.text(emb_pca[i, 0] + 0.4, emb_pca[i, 1] + 0.4, f"{i}: {name}", fontsize=9, weight="bold")
plt.title("PCA Projection of Mechanism Embeddings", fontsize=13, weight="bold")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(alpha=0.3)

# ---------------- t-SNE Plot ----------------
plt.subplot(1, 2, 2)
plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1], s=200, c=colors, edgecolors="k", linewidths=1.2)
for i, name in enumerate(mech_names):
    plt.text(emb_tsne[i, 0] + 0.4, emb_tsne[i, 1] + 0.4, f"{i}: {name}", fontsize=9, weight="bold")
plt.title("t-SNE Projection of Mechanism Embeddings", fontsize=13, weight="bold")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# =========================================================
# Optional: Cosine similarity matrix
# =========================================================
import torch.nn.functional as F
sim = F.cosine_similarity(
    torch.tensor(mech_embs)[:, None, :],
    torch.tensor(mech_embs)[None, :, :],
    dim=-1
).numpy()

print("\nCosine similarity between mechanism embeddings:")
print(np.round(sim, 3))
