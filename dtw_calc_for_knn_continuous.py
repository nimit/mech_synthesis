# =========================================================
# IMPORTS
# =========================================================
import torch
from torch.utils.data import DataLoader
import numpy as np
from llama_latent_continuous import LatentLLaMA_Continuous
from dataset import BarLinkageDataset 
from sklearn.neighbors import NearestNeighbors

from dataset_generation.curve_plot import get_pca_inclination, rotate_curve
import scipy.spatial.distance as sciDist
from tqdm import tqdm
import requests
import time
import os
import json
import torch.nn.functional as F
from tslearn.metrics import dtw_path

torch.set_float32_matmul_precision('medium')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Headless simulator version
index = 0 # local server index 
API_ENDPOINT = f"http://localhost:4000/simulation"
HEADERS = {"Content-Type": "application/json"}
speedscale = 1
steps = 360
minsteps = int(steps*20/360)

checkpoint_path = "weights/LATENT_LLAMA_CONT_d512_nf512_h8_n6_bs512_lr0.0005.pth"
data_dir = "dataset_17mechs"
batch_size = 1

# Note: Dataset loading might not be strictly needed if we load NPY files directly, 
# but keeping existing structure.
dataset = BarLinkageDataset(data_dir=data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

checkpoint = torch.load(checkpoint_path, map_location=device)
model_config = checkpoint['model_config']
print("Model Config freqs:", model_config["num_freqs"])

# Initialize model
model = LatentLLaMA_Continuous(
    tgt_seq_len=model_config["tgt_seq_len"],
    d_model=model_config["d_model"],
    num_heads=model_config["num_heads"],
    num_layers=model_config["num_layers"],
    num_labels=model_config["num_labels"],
    latent_dim=model_config["latent_dim"],
    num_freqs=model_config["num_freqs"],
    log_scale=model_config.get("log_scale", False),
).to(device)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
model.eval()

# =========================================================
# CONFIG
# =========================================================
NUM_NEIGHBORS = 10
NUM_MECH_TYPES = 17
LATENT_DIM = 50
tgt_seq_len = 17

# Paths
latent_path = f"{data_dir}/vae_mu.npy"
labels_cont_path = f"{data_dir}/labels_continuous.npy"
encoded_labels_path = f"{data_dir}/encoded_labels.npy"

label_mapping_path = f"{data_dir}/label_mapping.json"
coupler_mapping_path = "BSIdict.json"

# GLOBAL statistics
GLOBAL_DTW_VALUES = []
GLOBAL_DTW_BELOW_2 = 0   

# =========================================================
# Load Label Mapping
# =========================================================
with open(label_mapping_path, "r") as f:
    label_mapping = json.load(f)

index_to_label = label_mapping["index_to_label"]

with open(coupler_mapping_path, "r") as f:
    coupler_mapping = json.load(f)


def coupler_index_for(mech_type: str) -> int:
    if mech_type in coupler_mapping and "c" in coupler_mapping[mech_type]:
        cvec = coupler_mapping[mech_type]["c"]
        if isinstance(cvec, list) and 1 in cvec:
            return cvec.index(1)
    return -1


def safe_name(name: str, max_len=30):
    return "".join([(c if c.isalnum() else "_") for c in name])[:max_len] or "unk"

@torch.no_grad()
def generate_mechanism(
    model, latent_vec, mech_idx, device="cuda", max_len=8, threshold=0.5
):
    model.eval()
    latent_vec = latent_vec.unsqueeze(0) if latent_vec.dim() == 1 else latent_vec
    latent_vec = latent_vec.to(device)
    mech_labels = torch.tensor([mech_idx], device=device, dtype=torch.long)
    B = latent_vec.shape[0]

    # Start with an empty coordinate sequence (B, 0, 2)
    generated_coords = torch.zeros((B, 0, 2), device=device)

    finished = torch.zeros(B, dtype=torch.bool, device=device)
    final_sequence = [None] * B

    for i in range(max_len):
        # 1. Create attention mask for joints generated so far
        # (B, current_length) - all ones because these are real joints
        attn_mask = torch.ones((B, generated_coords.shape[1]), device=device)

        # 2. Forward Pass
        # Note: We only pass the (x, y) parts of generated_coords
        preds = model(generated_coords, attn_mask, latent_vec, mech_labels)

        # 3. Get the prediction for the NEXT joint
        # Based on our training logic:
        # Input: [SOS, J1, J2...] -> Output predicts: [J1, J2, J3...]
        # So we always look at the LAST index of the output
        next_pred = preds[:, -1, :]  # Shape: (B, 3)

        new_xy = next_pred[:, :2]  # Predicted x, y
        stop_logit = next_pred[:, 2]  # Predicted stop bit
        stop_prob = torch.sigmoid(stop_logit)

        # 4. Update the sequence
        # Concatenate the new joint to the input for the next iteration
        generated_coords = torch.cat([generated_coords, new_xy.unsqueeze(1)], dim=1)

        # 5. Check for completion
        for batch_idx in range(B):
            if not finished[batch_idx]:
                if stop_prob[batch_idx] > threshold or i == max_len - 1:
                    finished[batch_idx] = True
                    # Slice the coordinates up to this joint
                    final_sequence[batch_idx] = (
                        generated_coords[batch_idx].cpu().numpy()
                    )

        if finished.all():
            break

    # If only 1 batch, return the first
    if B == 1:
        return final_sequence[0]
    return final_sequence


# =========================================================
# Normalization
# =========================================================
def get_pca_inclination(x, y):
    cx, cy = np.mean(x), np.mean(y)
    cov = np.cov(x - cx, y - cy)
    eigvals, eigvecs = np.linalg.eig(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    return np.arctan2(major[1], major[0])


def rotate_curve(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return x * c - y * s, x * s + y * c


def normalize_curve(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.var(x) + np.var(y))
    x /= denom
    y /= denom
    phi = -get_pca_inclination(x, y)
    return rotate_curve(x, y, phi)


# =========================================================
# Clean joint label rows
# =========================================================
def clean_and_reshape_label(row):
    mask = (row != 2.0) & (row != -1.0)
    row = row[mask]
    if row.size % 2:
        row = row[:-1]
    return row.reshape(-1, 2) if row.size else np.zeros((0, 2))


# =========================================================
# Simulate curve through API
# =========================================================
def simulate_curve(params, mech_type):
    ex = {
        "params": params.tolist(),
        "type": mech_type,
        "speedScale": speedscale,
        "steps": steps,
        "relativeTolerance": 0.1,
    }

    try:
        r = requests.post(API_ENDPOINT, headers=HEADERS, data=json.dumps([ex])).json()
        if isinstance(r, list) and "poses" in r[0]:
            return np.array(r[0]["poses"])
    except:
        return None

    return None


# =========================================================
# Compute best DTW among 6 transformations
# =========================================================
def best_variant_dtw(gt_curve, pred_curve):
    global GLOBAL_DTW_BELOW_2

    variants = [
        pred_curve,
        pred_curve[::-1],
        np.column_stack([ pred_curve[:,0], -pred_curve[:,1] ]),
        np.column_stack([ pred_curve[::-1,0], -pred_curve[::-1,1] ]),
        np.column_stack([ -pred_curve[:,0], pred_curve[:,1] ]),
        np.column_stack([ -pred_curve[::-1,0], pred_curve[::-1,1] ])
    ]

    best = 1e18

    for V in variants:
        path, dist = dtw_path(gt_curve, V)
        scale = np.sqrt(np.var(gt_curve) + np.var(V)) + 1e-12
        dtw_val = dist / (len(path) * scale)

        if dtw_val < best:
            best = dtw_val

    best *= 100.0

    # Track DTW < 2
    if best < 2.0:
        GLOBAL_DTW_BELOW_2 += 1

    return best


# =========================================================
# PROCESS ONE CURVE
# =========================================================
def process_one_curve():

    latents = np.load(latent_path)
    labels_cont = np.load(labels_cont_path)
    encoded_labels = np.load(encoded_labels_path)
    N = latents.shape[0]

    query_idx = np.random.randint(0, N)
    query_latent = latents[query_idx : query_idx + 1]

    # ground truth
    gt_points = clean_and_reshape_label(labels_cont[query_idx])
    gt_mech_idx = int(encoded_labels[query_idx])
    gt_mech_name = index_to_label[str(gt_mech_idx)]

    P_gt = simulate_curve(gt_points, gt_mech_name)
    if P_gt is None or P_gt.shape[0] < minsteps:
        return

    coup_idx_gt = coupler_index_for(gt_mech_name)
    if coup_idx_gt < 0:
        return

    gx, gy = P_gt[:, coup_idx_gt, 0], P_gt[:, coup_idx_gt, 1]
    gt_x, gt_y = normalize_curve(gx, gy)
    gt_curve = np.column_stack([gt_x, gt_y])

    orig_latent_tensor = torch.tensor(query_latent[0], dtype=torch.float32, device=device)

    # -----------------------------------------------
    # ORIGINAL LATENT
    # -----------------------------------------------
    for mech_idx in range(NUM_MECH_TYPES):

        mech_name = index_to_label[str(mech_idx)]

        # Continuous prediction returns (N, 2) array directly
        pred_points = generate_mechanism(
            model, orig_latent_tensor, mech_idx,
            device=device,
            max_len=tgt_seq_len,
        )

        # Minimum required points check
        if pred_points.shape[0] < 4:
            continue

        # simulate predicted curve
        Pp = simulate_curve(pred_points, mech_name)
        if Pp is None or Pp.shape[0] < minsteps:
            continue

        coup_idx_pred = coupler_index_for(mech_name)
        if coup_idx_pred < 0:
            continue

        px, py = Pp[:, coup_idx_pred, 0], Pp[:, coup_idx_pred, 1]
        px_n, py_n = normalize_curve(px, py)
        pred_curve = np.column_stack([px_n, py_n])

        best_dtw = best_variant_dtw(gt_curve, pred_curve)
        GLOBAL_DTW_VALUES.append(best_dtw)

    # -----------------------------------------------
    # NEIGHBOR LATENTS
    # -----------------------------------------------
    full_latents = np.load(latent_path)
    knn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS + 1)
    knn.fit(full_latents)
    _, idxs = knn.kneighbors(query_latent)

    neighbor_idxs = idxs[0][1:]

    for ng_idx in neighbor_idxs:
        neigh_latent = torch.tensor(full_latents[ng_idx], dtype=torch.float32, device=device)

        for mech_idx in range(NUM_MECH_TYPES):

            mech_name = index_to_label[str(mech_idx)]

            pred_points = generate_mechanism(
                model, neigh_latent, mech_idx,
                device=device,
                max_len=tgt_seq_len
            )

            if pred_points.shape[0] < 4:
                continue

            Pp = simulate_curve(pred_points, mech_name)
            if Pp is None or Pp.shape[0] < minsteps:
                continue

            coup_idx_pred = coupler_index_for(mech_name)
            if coup_idx_pred < 0:
                continue

            px, py = Pp[:, coup_idx_pred, 0], Pp[:, coup_idx_pred, 1]
            px_n, py_n = normalize_curve(px, py)
            pred_curve = np.column_stack([px_n, py_n])

            best_dtw = best_variant_dtw(gt_curve, pred_curve)
            GLOBAL_DTW_VALUES.append(best_dtw)


# =========================================================
# MAIN
# =========================================================
def main():

    NUM_SAMPLES = 100  # adjust as needed (e.g., 100)

    for _ in range(NUM_SAMPLES):
        process_one_curve()

    if GLOBAL_DTW_VALUES:
        avg_dtw = np.mean(GLOBAL_DTW_VALUES)
        total = len(GLOBAL_DTW_VALUES)
        below2 = GLOBAL_DTW_BELOW_2
        percentage = below2 / total * 100

        print("\n==============================================")
        print(f"Total Predictions Evaluated: {total}")
        print(f"Average Best-DTW: {avg_dtw:.6f}")
        print(f"DTW < 2.0 Count: {below2}")
        print(f"DTW < 2.0 Percentage: {percentage:.2f}%")
        print("==============================================")

        with open("final_dtw_stats_continuous.txt", "w") as f:
            f.write(f"Total Predictions: {total}\n")
            f.write(f"Average DTW: {avg_dtw}\n")
            f.write(f"DTW < 2 count: {below2}\n")
            f.write(f"Percent < 2: {percentage}\n")

    else:
        print("No DTW values computed.")


if __name__ == "__main__":
    main()