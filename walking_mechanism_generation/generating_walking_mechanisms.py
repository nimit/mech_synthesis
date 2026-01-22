# =========================
# IMPORTS
# =========================

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import requests

from llama_latent_model import LatentLLaMA_SingleToken
from path_decomposition import computeSolSteps, linkMajor
from dataset_generation.curve_plot import get_pca_inclination, rotate_curve

# =========================
# DEVICE
# =========================

torch.set_float32_matmul_precision("medium")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# PATHS
# =========================

CHECKPOINT_PATH = "./weights/WEIGHTS_33/LATENT_LLAMA_d1536_h32_n6_bs512_lr0.0001_best.pth"

REF_DIR = "/home/anurizada/Documents/processed_dataset_33"
QUERY_LATENT_PATH = "vae_latents/vae_mu.npy"

OUTPUT_DIR = "results_curves_knn_normalized"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_MAPPING_PATH = "/home/anurizada/Documents/processed_dataset_33/label_mapping.json"
COUPLER_MAPPING_PATH = "/home/anurizada/Documents/transformer_gaussian/BSIdict.json"

NUM_NEIGHBORS = 100

# =========================
# LOAD REFERENCE DATA
# =========================

Z_ref = np.load(os.path.join(REF_DIR, "vae_mu.npy"))
GT_MECH = np.load(os.path.join(REF_DIR, "encoded_labels.npy"))

Z_ref_tensor = torch.from_numpy(Z_ref).float().to(device)

# =========================
# LOAD QUERY LATENTS
# =========================

Z_query = np.load(QUERY_LATENT_PATH)
N_query = Z_query.shape[0]

# =========================
# LOAD METADATA
# =========================

with open(LABEL_MAPPING_PATH) as f:
    index_to_label = json.load(f)["index_to_label"]

with open(COUPLER_MAPPING_PATH) as f:
    coupler_mapping = json.load(f)

# =========================
# LOAD MODEL
# =========================

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
cfg = checkpoint["model_config"]

model = LatentLLaMA_SingleToken(
    tgt_seq_len=cfg["tgt_seq_len"],
    d_model=cfg["d_model"],
    h=cfg["h"],
    N=cfg["N"],
    num_labels=cfg["num_labels"],
    vocab_size=cfg["vocab_size"],
    latent_dim=cfg["latent_dim"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("✅ Loaded Latent LLaMA")

# =========================
# CONSTANTS
# =========================

BIN_OFFSET = 3
NUM_BINS = 201

API_ENDPOINT = "http://localhost:4001/simulation"
API_ENDPOINT_8BAR = "http://localhost:4001/simulation-8bar"
HEADERS = {"Content-Type": "application/json"}

speedscale = 1
steps = 360
minsteps = int(steps * 20 / 360)

# =========================
# HELPERS
# =========================

def is_type8bar(name):
    return name.startswith("Type")

def safe_name(name, max_len=40):
    return "".join(c if c.isalnum() else "_" for c in name)[:max_len]

def coupler_index_for(name):
    try:
        return 10 if is_type8bar(name) else coupler_mapping[name]["c"].index(1)
    except Exception:
        return -1

def B2T(B):
    n = len(B[0])
    T = np.zeros((n, n))
    for i in range(n):
        if B[0][i]:
            T[i, i] = 1
    for row in B:
        for i in range(n):
            for j in range(i + 1, n):
                if row[i] and row[j]:
                    T[i, j] = T[j, i] = 1
    return T.astype(int).tolist()

class CoordinateBinner:
    def __init__(self):
        edges = np.linspace(-1, 1, NUM_BINS)
        self.centers = (edges[:-1] + edges[1:]) / 2

    def decode(self, tokens):
        idx = tokens.astype(int) - BIN_OFFSET
        if idx.size == 0 or np.any(idx < 0) or np.any(idx >= len(self.centers)):
            return None
        return self.centers[idx]

binner = CoordinateBinner()

# =========================
# REFERENCE-FREE NORMALIZATION
# =========================

def normalize_curve(curve):
    """
    Canonical per-curve normalization:
      - zero mean
      - PCA alignment
      - unit variance
    """
    x, y = curve[:, 0], curve[:, 1]

    # center
    x -= x.mean()
    y -= y.mean()

    # rotate to principal axis
    phi = -get_pca_inclination(x, y)
    x, y = rotate_curve(x, y, phi)

    # scale to unit energy
    scale = np.sqrt(np.var(x) + np.var(y)) + 1e-8
    x /= scale
    y /= scale

    return np.stack([x, y], axis=1)

# =========================
# SIMULATION
# =========================

def simulate_safe(points, mech_name):
    pts = np.round(points, 3)
    try:
        if is_type8bar(mech_name):
            B = coupler_mapping[mech_name]["B"]
            _, solSteps, _ = computeSolSteps(linkMajor(B))
            payload = {
                "T": B2T(B),
                "solSteps": solSteps,
                "params": pts.tolist(),
                "speedScale": speedscale,
                "steps": steps,
                "relativeTolerance": 0.1,
            }
            url = API_ENDPOINT_8BAR
        else:
            payload = {
                "params": pts.tolist(),
                "type": mech_name,
                "speedScale": speedscale,
                "steps": steps,
                "relativeTolerance": 0.1,
            }
            url = API_ENDPOINT

        r = requests.post(url, headers=HEADERS, data=json.dumps([payload]), timeout=30).json()
        if not r or "poses" not in r[0]:
            return None

        P = np.array(r[0]["poses"])
        if P.ndim < 3 or P.shape[0] < minsteps:
            return None
        return P
    except Exception:
        return None

# =========================
# DECODER
# =========================

def predict_safe(latent, mech_idx, max_len=25):
    tokens = torch.tensor([[0]], device=device)
    mech_labels = torch.tensor([mech_idx], device=device)

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(tokens, None, latent.unsqueeze(0), mech_labels)
            nxt = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            tokens = torch.cat([tokens, nxt], dim=1)
            if int(nxt.item()) == 1:
                break

    return tokens.squeeze().cpu().numpy()

# =========================
# BUILD KNN
# =========================

knn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS).fit(Z_ref)

# =========================
# MAIN LOOP
# =========================

for i in tqdm(range(N_query)):
    curve_dir = os.path.join(OUTPUT_DIR, f"curve_{i:04d}")
    os.makedirs(curve_dir, exist_ok=True)

    _, idxs = knn.kneighbors(Z_query[i].reshape(1, -1))

    for k, ref_idx in enumerate(idxs[0], start=1):
        latent = Z_ref_tensor[int(ref_idx)]
        mech_idx = int(GT_MECH[int(ref_idx)])
        mech_name = index_to_label[str(mech_idx)]
        mech_tag = safe_name(mech_name)

        tokens = predict_safe(latent, mech_idx)
        tokens = tokens[tokens >= BIN_OFFSET]
        if len(tokens) < 4:
            continue

        vals = binner.decode(tokens)
        if vals is None:
            continue

        pts = vals[: (len(vals) // 2) * 2].reshape(-1, 2)
        P = simulate_safe(pts, mech_name)
        if P is None:
            continue

        ci = coupler_index_for(mech_name)
        if ci < 0:
            continue

        curve = P[:, ci, :]
        curve_norm = normalize_curve(curve)

        plt.figure(figsize=(6, 6))
        plt.plot(curve_norm[:, 0], curve_norm[:, 1], "k", lw=2)
        plt.axis("equal")
        plt.axis("off")
        plt.savefig(
            os.path.join(curve_dir, f"decoded_{mech_tag}_{k:02d}.png"),
            dpi=200,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

print("\n✅ DONE — curves normalized and saved with mechanism names.")
