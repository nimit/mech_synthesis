"""
Short summary:
This script processes 100 random latent vectors, finds their 100 nearest neighbors,
predicts each neighbor's mechanism using its own GT mech type, simulates each
predicted mechanism, and:

1) Collects the predicted mechanism-type indices to build a histogram.
2) Computes DTW between ground-truth and predicted coupler curves (with 6
   symmetry variants) to report average DTW and % below a threshold.
"""

# =========================================================
# IMPORTS
# =========================================================
import os
import json
import requests
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tslearn.metrics import dtw_path

from llama_latent_model import LatentLLaMA_SingleToken
from dataset import BarLinkageDataset
from dataset_generation.curve_plot import get_pca_inclination, rotate_curve

torch.set_float32_matmul_precision("medium")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# =========================================================
# SIMULATION CONFIG
# =========================================================
API_ENDPOINT = "http://localhost:4000/simulation"
HEADERS = {"Content-Type": "application/json"}
speedscale = 1
steps = 360
minsteps = int(steps * 20 / 360)


# =========================================================
# MODEL + DATASET LOADING
# =========================================================
checkpoint_path = "./weights/CE_GAUS/LATENT_LLAMA_d768_h8_n6_bs512_lr0.0005_best.pth"
data_dir = "/home/anurizada/Documents/processed_dataset_17"
batch_size = 1

dataset = BarLinkageDataset(data_dir=data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

checkpoint = torch.load(checkpoint_path, map_location=device)
model_config = checkpoint["model_config"]

model = LatentLLaMA_SingleToken(
    tgt_seq_len=model_config["tgt_seq_len"],
    d_model=model_config["d_model"],
    h=model_config["h"],
    N=model_config["N"],
    num_labels=model_config["num_labels"],
    vocab_size=model_config["vocab_size"],
    latent_dim=model_config["latent_dim"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# =========================================================
# CONFIGURATION CONSTANTS
# =========================================================
NUM_QUERIES = 100     # how many random query latents to process
NUM_NEIGHBORS = 100   # first 100 KNN neighbors
NUM_MECH_TYPES = 17

SOS_TOKEN, EOS_TOKEN, PAD_TOKEN = 0, 1, 2
NUM_SPECIAL_TOKENS = 3
BIN_OFFSET = NUM_SPECIAL_TOKENS
NUM_BINS = 201
LATENT_DIM = model_config["latent_dim"]

tgt_seq_len = model_config["tgt_seq_len"]
FIXED_TEMPERATURE = 0.0
TOP_K = None


# =========================================================
# LOAD LATENTS / LABELS
# =========================================================
latent_path = "/home/anurizada/Documents/processed_dataset_17/vae_mu.npy"
labels_cont_path = "/home/anurizada/Documents/processed_dataset_17/labels_continuous.npy"
encoded_labels_path = "/home/anurizada/Documents/processed_dataset_17/encoded_labels.npy"

LATENTS_ALL = np.load(latent_path)
LABELS_CONT_ALL = np.load(labels_cont_path)
ENCODED_LABELS_ALL = np.load(encoded_labels_path)
N_SAMPLES = LATENTS_ALL.shape[0]


# =========================================================
# LABEL MAPPING
# =========================================================
label_mapping_path = "/home/anurizada/Documents/processed_dataset_17/label_mapping.json"
coupler_mapping_path = "/home/anurizada/Documents/transformer/BSIdict.json"

with open(label_mapping_path, "r") as f:
    label_mapping = json.load(f)
index_to_label = label_mapping["index_to_label"]

with open(coupler_mapping_path, "r") as f:
    coupler_mapping = json.load(f)


# =========================================================
# GLOBAL ACCUMULATORS
# =========================================================
PREDICTED_MECH_TYPES = []      # for histogram
GLOBAL_DTW_VALUES = []         # all DTW values
GLOBAL_DTW_BELOW_2 = 0         # count of DTW < 2.0


# =========================================================
# BASIC HELPERS
# =========================================================
def coupler_index_for(mech_type: str) -> int:
    """
    Return the coupler index (joint index) for a given mechanism type.
    If not found, return -1.
    """
    if mech_type in coupler_mapping and "c" in coupler_mapping[mech_type]:
        cvec = coupler_mapping[mech_type]["c"]
        if isinstance(cvec, list) and 1 in cvec:
            return cvec.index(1)
    return -1


def safe_name(name: str, max_len: int = 30) -> str:
    """
    Create a filesystem-safe short name.
    """
    cleaned = "".join(c if c.isalnum() else "_" for c in name)
    cleaned = cleaned[:max_len]
    return cleaned or "unk"


# =========================================================
# Normalization (for DTW)
# =========================================================
def normalize_curve(x: np.ndarray, y: np.ndarray):
    """
    Center, scale, and align 2D curve via PCA.
    Returns (x_norm, y_norm).
    """
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.var(x) + np.var(y)) + 1e-12
    x /= denom
    y /= denom
    phi = -get_pca_inclination(x, y)
    return rotate_curve(x, y, phi)


# =========================================================
# CoordinateBinner
# =========================================================
class CoordinateBinner:
    """
    Maps discrete bin indices to continuous coordinate values on [-kappa, kappa].
    """
    def __init__(self, kappa: float = 1.0, num_bins: int = 200):
        self.kappa = kappa
        self.num_bins = num_bins
        self.bin_edges = np.linspace(-kappa, kappa, num_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    def bin_to_value_torch(self, idx: torch.Tensor) -> torch.Tensor:
        idx = torch.clamp(idx, 0, self.num_bins - 1)
        centers = torch.tensor(self.bin_centers, device=idx.device, dtype=torch.float32)
        return centers[idx]


binner = CoordinateBinner(kappa=1.0, num_bins=NUM_BINS - 1)


# =========================================================
# Causal Mask for Autoregressive Decoding
# =========================================================
def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Build a standard causal (lower-triangular) attention mask.
    """
    m = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return m.unsqueeze(0).unsqueeze(0)


# =========================================================
# Autoregressive Latent Prediction (Token Generation)
# =========================================================
def predict_autoregressive_latent(
    model: torch.nn.Module,
    latent: torch.Tensor,
    mech_idx: int,
    max_seq_len: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k=None,
    eos_token: int = EOS_TOKEN,
    sos_token: int = SOS_TOKEN,
) -> np.ndarray:
    """
    Given a latent vector and a mechanism index, autoregressively generate
    a token sequence using the Transformer decoder, starting from SOS.
    """
    model.eval()

    if latent.dim() == 1:
        latent = latent.unsqueeze(0)
    latent = latent.to(device)

    mech_labels = torch.tensor([mech_idx], device=device, dtype=torch.long)
    decoder_input = torch.tensor([[sos_token]], device=device, dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_seq_len):
            mask = build_causal_mask(decoder_input.size(1), device)
            logits = model(decoder_input, mask, latent, mech_labels)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)

            if top_k is not None:
                k = min(top_k, probs.size(-1))
                topk_probs, topk_idx = torch.topk(probs, k=k)
                next_token = topk_idx.gather(-1, torch.multinomial(topk_probs, 1))
            elif temperature == 0.0:
                # Greedy decoding
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            token = int(next_token.item())
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            if token == eos_token:
                break

    return decoder_input.squeeze(0).cpu().numpy()


# =========================================================
# Clean joint label rows (GT)
# =========================================================
def clean_and_reshape_label(row: np.ndarray) -> np.ndarray:
    """
    Remove invalid entries (2.0, -1.0), ensure even length, and reshape to (N, 2).
    """
    mask = (row != 2.0) & (row != -1.0)
    row = row[mask]
    if row.size % 2:
        row = row[:-1]
    if row.size == 0:
        return np.zeros((0, 2))
    return row.reshape(-1, 2)


# =========================================================
# Simulation Wrapper
# =========================================================
def simulate_curve(params: np.ndarray, mech_type: str) -> np.ndarray | None:
    """
    Call the external simulator with given parameters and mechanism type.
    Returns poses array or None if the call fails.
    """
    ex = {
        "params": params.tolist(),
        "type": mech_type,
        "speedScale": speedscale,
        "steps": steps,
        "relativeTolerance": 0.1,
    }

    try:
        r = requests.post(API_ENDPOINT, headers=HEADERS, data=json.dumps([ex])).json()
        if isinstance(r, list) and len(r) > 0 and "poses" in r[0]:
            return np.array(r[0]["poses"])
    except Exception:
        return None

    return None


# =========================================================
# DTW: best of 6 symmetry variants
# =========================================================
def best_variant_dtw(gt_curve: np.ndarray, pred_curve: np.ndarray) -> float:
    """
    Compute DTW between GT and predicted coupler curves, taking the best over
    6 symmetry variants (reverse, sign flips) and normalizing by combined scale.
    """
    global GLOBAL_DTW_BELOW_2

    variants = [
        pred_curve,
        pred_curve[::-1],
        np.column_stack([pred_curve[:, 0], -pred_curve[:, 1]]),
        np.column_stack([pred_curve[::-1, 0], -pred_curve[::-1, 1]]),
        np.column_stack([-pred_curve[:, 0], pred_curve[:, 1]]),
        np.column_stack([-pred_curve[::-1, 0], pred_curve[::-1, 1]]),
    ]

    best = 1e18

    for V in variants:
        path, dist = dtw_path(gt_curve, V)
        scale = np.sqrt(np.var(gt_curve) + np.var(V)) + 1e-12
        dtw_val = dist / (len(path) * scale)
        if dtw_val < best:
            best = dtw_val

    best *= 100.0

    if best < 2.0:
        GLOBAL_DTW_BELOW_2 += 1

    return best


# =========================================================
# Joint connections from B (not used for DTW, kept for completeness)
# =========================================================
def get_joint_connections(mech_name: str):
    if mech_name not in coupler_mapping:
        return []
    mech_entry = coupler_mapping[mech_name]
    if "B" not in mech_entry:
        return []
    B = mech_entry["B"]
    connections = []
    for row in B[1:]:
        joints = [j for j, val in enumerate(row) if val == 1]
        if len(joints) >= 2:
            for a in range(len(joints)):
                for b in range(a + 1, len(joints)):
                    connections.append((joints[a], joints[b]))
    return connections


# =========================================================
# Predict and ALIGN coupler + joints to GT coupler
# =========================================================
def predict_and_align_for_latent(
    latent_vec: np.ndarray,
    mech_idx: int,
    mech_name: str,
    orig_phi: float,
    orig_denom: float,
    ox_mean: float,
    oy_mean: float,
):
    """
    latent_vec: np.array (latent_dim,)

    Runs the model to predict coordinate tokens for the given mech_idx,
    simulates the mechanism, and aligns the predicted coupler and joints
    to match the GT coupler reference frame.

    Returns:
        aligned_curve: (T, 2) coupler curve aligned to GT
        aligned_joints: (J, 2) joints aligned to GT

    or (None, None) if something fails.
    """
    latent_t = torch.tensor(latent_vec, dtype=torch.float32, device=device)

    # Predict tokens
    pred_tokens = predict_autoregressive_latent(
        model,
        latent_t,
        mech_idx,
        tgt_seq_len,
        device,
        temperature=FIXED_TEMPERATURE,
        top_k=TOP_K,
    )

    coord_tokens = [t for t in pred_tokens if t >= BIN_OFFSET]
    if len(coord_tokens) < 4:
        return None, None

    coords = binner.bin_to_value_torch(
        torch.tensor(coord_tokens, device=device) - BIN_OFFSET
    ).cpu().numpy()

    if coords.size % 2:
        coords = coords[:-1]

    pred_points = coords.reshape(-1, 2)

    # Simulate predicted mechanism
    Pp = simulate_curve(pred_points, mech_name)
    if Pp is None or Pp.shape[0] < minsteps:
        return None, None

    coup_idx_pred = coupler_index_for(mech_name)
    if coup_idx_pred < 0:
        return None, None

    gen_x = Pp[:, coup_idx_pred, 0]
    gen_y = Pp[:, coup_idx_pred, 1]

    # Align predicted coupler to GT coupler frame
    gen_phi = -get_pca_inclination(gen_x, gen_y)
    rotation = gen_phi - orig_phi
    gen_x, gen_y = rotate_curve(gen_x, gen_y, rotation)

    gen_denom = np.sqrt(np.var(gen_x) + np.var(gen_y)) + 1e-8
    if gen_denom < 1e-12:
        return None, None

    scale = orig_denom / gen_denom
    gen_x *= scale
    gen_y *= scale

    # translate to GT center
    gen_x -= (np.mean(gen_x) - ox_mean)
    gen_y -= (np.mean(gen_y) - oy_mean)

    aligned_curve = np.column_stack([gen_x, gen_y])

    # Align joints by same transform
    px = pred_points[:, 0]
    py = pred_points[:, 1]
    px, py = rotate_curve(px, py, rotation)
    px *= scale
    py *= scale
    px -= (np.mean(px) - ox_mean)
    py -= (np.mean(py) - oy_mean)
    aligned_joints = np.column_stack([px, py])

    return aligned_curve, aligned_joints


# =========================================================
# kNN SETUP
# =========================================================
knn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS + 1)  # query + 100 neighbors
knn.fit(LATENTS_ALL)


# =========================================================
# PROCESS ONE QUERY CURVE
# =========================================================
def process_one_query():
    """
    Pick a random query latent, simulate its ground-truth coupler to define
    an alignment frame and normalized GT curve. Then find its 100 nearest
    neighbor latents. For each neighbor, use that neighbor's GT mechanism
    index to run prediction + simulation + alignment.

    For every successful neighbor prediction:
      - record the neighbor's mech_idx into PREDICTED_MECH_TYPES
      - compute DTW between normalized GT coupler and predicted coupler
        (using best of 6 symmetry variants) and store in GLOBAL_DTW_VALUES
    """

    global GLOBAL_DTW_VALUES

    # --------------------------
    # Pick random query index
    # --------------------------
    query_idx = np.random.randint(0, N_SAMPLES)
    query_latent = LATENTS_ALL[query_idx]
    gt_mech_idx = int(ENCODED_LABELS_ALL[query_idx])
    gt_mech_name = index_to_label[str(gt_mech_idx)]

    # --------------------------
    # Ground-truth mechanism params for the query
    # --------------------------
    gt_points = clean_and_reshape_label(LABELS_CONT_ALL[query_idx])
    if gt_points.shape[0] == 0:
        print(f"[{query_idx}] No valid GT points, skipping.")
        return

    # Simulate GT coupler
    P_gt = simulate_curve(gt_points, gt_mech_name)
    if P_gt is None or P_gt.shape[0] < minsteps:
        print(f"[{query_idx}] GT simulation failed, skipping.")
        return

    coup_idx_gt = coupler_index_for(gt_mech_name)
    if coup_idx_gt < 0:
        print(f"[{query_idx}] No coupler index for GT mech ({gt_mech_name}), skipping.")
        return

    gx, gy = P_gt[:, coup_idx_gt, 0], P_gt[:, coup_idx_gt, 1]

    # Raw GT coords (for alignment)
    gt_x, gt_y = gx.copy(), gy.copy()

    # Normalized GT curve (for DTW)
    gt_x_n, gt_y_n = normalize_curve(gx, gy)
    gt_curve = np.column_stack([gt_x_n, gt_y_n])

    # Alignment reference from raw GT coupler
    orig_phi = -get_pca_inclination(gt_x, gt_y)
    orig_denom = np.sqrt(np.var(gt_x) + np.var(gt_y)) + 1e-8
    ox_mean, oy_mean = np.mean(gt_x), np.mean(gt_y)

    # --------------------------
    # kNN neighbors (use first 100, skip self)
    # --------------------------
    _, idxs = knn.kneighbors(query_latent.reshape(1, -1))
    neighbor_idxs = idxs[0][1 : NUM_NEIGHBORS + 1]  # skip self at [0]

    print(
        f"Query idx={query_idx}, GT mech={gt_mech_name}, "
        f"using {len(neighbor_idxs)} kNN neighbors."
    )

    # --------------------------
    # For each neighbor, use its OWN GT mech_idx
    # --------------------------
    for nidx in neighbor_idxs:
        nidx = int(nidx)

        neigh_latent = LATENTS_ALL[nidx]
        neigh_mech_idx = int(ENCODED_LABELS_ALL[nidx])
        neigh_mech_name = index_to_label[str(neigh_mech_idx)]

        # Run prediction + simulation + alignment for this neighbor
        curve_n, joints_n = predict_and_align_for_latent(
            neigh_latent,
            neigh_mech_idx,   # <-- use neighbor's GT mech type
            neigh_mech_name,
            orig_phi,
            orig_denom,
            ox_mean,
            oy_mean,
        )

        # If prediction / simulation failed, skip it (no "prediction" to count)
        if curve_n is None or joints_n is None:
            continue

        # Successful prediction: record this mech_idx for histogram statistics
        PREDICTED_MECH_TYPES.append(neigh_mech_idx)

        # Compute DTW between normalized GT and normalized predicted coupler
        px_n, py_n = normalize_curve(curve_n[:, 0], curve_n[:, 1])
        pred_curve = np.column_stack([px_n, py_n])

        dtw_val = best_variant_dtw(gt_curve, pred_curve)
        GLOBAL_DTW_VALUES.append(dtw_val)


# =========================================================
# MAIN: RUN MULTIPLE QUERIES + PLOT HISTOGRAM + DTW STATS
# =========================================================
def main():

    # Process 100 random queries; for each, process up to 100 neighbors
    for _ in range(NUM_QUERIES):
        process_one_query()

    # =====================================================
    # HISTOGRAM OF ALL PREDICTED MECHANISM TYPES
    # =====================================================
    if not PREDICTED_MECH_TYPES:
        print("No successful predictions to build histogram or DTW stats from.")
        return

    print(f"Total successful predictions: {len(PREDICTED_MECH_TYPES)}")

    plt.figure(figsize=(8, 4))
    # bins as discrete mech indices [0, 1, ..., NUM_MECH_TYPES-1]
    plt.hist(PREDICTED_MECH_TYPES, bins=range(NUM_MECH_TYPES + 1), align="left", rwidth=0.8)
    plt.xticks(range(NUM_MECH_TYPES))
    plt.xlabel("Mechanism Type Index")
    plt.ylabel("Frequency")
    plt.title("Histogram of Predicted Mechanism Types (All kNN Predictions)")
    plt.tight_layout()
    plt.show()

    # =====================================================
    # DTW STATISTICS
    # =====================================================
    if GLOBAL_DTW_VALUES:
        avg_dtw = np.mean(GLOBAL_DTW_VALUES)
        total = len(GLOBAL_DTW_VALUES)
        below2 = GLOBAL_DTW_BELOW_2
        percentage = below2 / total * 100.0

        print("\n==============================================")
        print(f"Total DTW evaluations:       {total}")
        print(f"Average Best-DTW:            {avg_dtw:.6f}")
        print(f"DTW < 2.0 Count:             {below2}")
        print(f"DTW < 2.0 Percentage:        {percentage:.2f}%")
        print("==============================================\n")
    else:
        print("No DTW values computed.")

    print("DONE.")


if __name__ == "__main__":
    main()
