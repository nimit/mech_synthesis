# =========================================================
# IMPORTS
# =========================================================
import torch
import numpy as np
from llama_latent_continuous import LatentLLaMA_Continuous
from sklearn.neighbors import NearestNeighbors
from dataset_generation.curve_plot import get_pca_inclination, rotate_curve
from tqdm import tqdm
import requests
import os
import shutil
import json
from tslearn.metrics import dtw_path
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

matplotlib.use("Agg")
torch.set_float32_matmul_precision("medium")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Headless simulator version
index = 0  # local server index
API_ENDPOINT = "http://localhost:4000/simulation"
HEADERS = {"Content-Type": "application/json"}
speedscale = 1
steps = 360
minsteps = int(steps * 20 / 360)

import argparse

# =========================================================
# CONFIG
# =========================================================
NUM_SAMPLES = 100  # adjust as needed
NUM_PLOT = 5  # number of best/worst results to plot
NUM_NEIGHBORS = 32
OUTPUT_DIR = "knn_continuous_results"

# Parse CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
parser.add_argument("--num_plot", type=int, default=NUM_PLOT)
parser.add_argument("--num_neighbors", type=int, default=NUM_NEIGHBORS)
parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
args = parser.parse_args()

NUM_SAMPLES = args.num_samples
NUM_PLOT = args.num_plot
NUM_NEIGHBORS = args.num_neighbors
OUTPUT_DIR = args.output_dir

NUM_MECH_TYPES = 17
LATENT_DIM = 50
tgt_seq_len = 17
checkpoint_path = "weights/LATENT_LLAMA_CONT_d512_nf512_h8_n6_bs512_lr0.0005.pth"
data_dir = "dataset_17mechs"
batch_size = 1

checkpoint = torch.load(checkpoint_path, map_location=device)
model_config = checkpoint["model_config"]

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
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()


# Paths
latent_path = f"{data_dir}/vae_mu.npy"
labels_cont_path = f"{data_dir}/labels_continuous.npy"
encoded_labels_path = f"{data_dir}/encoded_labels.npy"

label_mapping_path = f"{data_dir}/label_mapping.json"
coupler_mapping_path = "BSIdict.json"

full_latents = np.load(latent_path)
knn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS + 1)
knn.fit(full_latents)

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
    model,
    latent_vec,
    mech_idx,
    device: torch.device | str = "cuda",
    max_len=8,
    threshold=0.5,
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


def compute_norm_params(x, y):
    mean_x, mean_y = np.mean(x), np.mean(y)
    x0, y0 = x - mean_x, y - mean_y
    denom = np.sqrt(np.var(x0) + np.var(y0)) + 1e-8
    phi = -get_pca_inclination(x0, y0)
    return mean_x, mean_y, denom, phi


def apply_norm(x, y, mean_x, mean_y, denom, phi):
    x0 = (x - mean_x) / denom
    y0 = (y - mean_y) / denom
    return rotate_curve(x0, y0, phi)


def normalize_curve(x, y):
    # Compatibility wrapper
    mean_x, mean_y, denom, phi = compute_norm_params(x, y)
    return apply_norm(x, y, mean_x, mean_y, denom, phi)


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
# PLOTTING & TRACKING HELPERS
# =========================================================
def apply_variant_to_points(points, variant_name):
    if points is None or points.size == 0:
        return points
    pts = points.copy()
    if variant_name == "orig":
        return pts
    elif variant_name == "orig_rev":
        return pts[::-1]
    elif variant_name == "mirror_x":
        pts[:, 1] *= -1
        return pts
    elif variant_name == "mirror_x_rev":
        pts = pts[::-1]
        pts[:, 1] *= -1
        return pts
    elif variant_name == "mirror_y":
        pts[:, 0] *= -1
        return pts
    elif variant_name == "mirror_y_rev":
        pts = pts[::-1]
        pts[:, 0] *= -1
        return pts
    return pts


def generate_curve_variants(curve):
    x = curve[:, 0]
    y = curve[:, 1]
    original = curve
    mirror_x = np.column_stack([x, -y])
    mirror_y = np.column_stack([-x, y])
    variants = [
        ("orig", original),
        ("orig_rev", original[::-1]),
        ("mirror_x", mirror_x),
        ("mirror_x_rev", mirror_x[::-1]),
        ("mirror_y", mirror_y),
        ("mirror_y_rev", mirror_y[::-1]),
    ]
    return variants


class PlotDataTracker:
    def __init__(self, k=5):
        self.k = k
        self.best_list = []  # Max-heap style (store smallest DTW)
        self.worst_list = []  # Min-heap style (store largest DTW)

    def update_with_best_entry(self, best_entry):
        # best_entry has structure: {'dtw', 'metadata', 'gt_curve', 'gt_joints', 'pred_curve', 'pred_joints', 'mech_name', 'variant_name'}

        # Update BEST list (keep smallest DTW)
        self.best_list.append(best_entry)
        self.best_list.sort(key=lambda x: x["dtw"])
        if len(self.best_list) > self.k:
            self.best_list = self.best_list[: self.k]

        # Update WORST list (keep largest DTW)
        self.worst_list.append(best_entry)
        self.worst_list.sort(key=lambda x: x["dtw"], reverse=True)
        if len(self.worst_list) > self.k:
            self.worst_list = self.worst_list[: self.k]


def plot_result_pair(data, save_path):
    gt_curve = data["gt_curve"]
    gt_joints = data["gt_joints"]
    pred_curve = data["pred_curve"]
    pred_joints = data["pred_joints"]
    mech = data["mech_name"]
    var = data["variant_name"]
    meta = data["metadata"]
    dtw = data["dtw"]

    label_str = f"{meta['gen_mech']} ({var})"

    plt.figure(figsize=(6, 6))
    plt.plot(gt_curve[:, 0], gt_curve[:, 1], "r-", label="GT Coupler")
    plt.plot(pred_curve[:, 0], pred_curve[:, 1], "b--", label=label_str)

    if gt_joints is not None and gt_joints.size > 0:
        plt.scatter(
            gt_joints[:, 0],
            gt_joints[:, 1],
            c="red",
            s=10,
            alpha=0.3,
            label="GT Joints",
        )

    if pred_joints is not None and pred_joints.size > 0:
        plt.scatter(
            pred_joints[:, 0], pred_joints[:, 1], c="blue", s=35, label="Pred Joints"
        )
        for i, (jx, jy) in enumerate(pred_joints):
            plt.text(jx, jy, f"{i + 1}", fontsize=9, color="black")

    title = f"Q{meta['query_idx']} | GT:{meta['gt_mech']} | Src:{meta['source']}\nDTW: {dtw:.4f}"
    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================================================
# Compute best DTW among 6 transformations
# =========================================================
def best_variant_dtw(gt_curve, pred_curve):
    variants = generate_curve_variants(pred_curve)

    best_dtw = 1e18
    best_name = "orig"
    best_curve = pred_curve

    for name, V in variants:
        path, dist = dtw_path(gt_curve, V)
        scale = np.sqrt(np.var(gt_curve) + np.var(V)) + 1e-12
        dtw_val = dist / (len(path) * scale)

        if dtw_val < best_dtw:
            best_dtw = dtw_val
            best_name = name
            best_curve = V

    best_dtw *= 100.0

    return best_dtw, best_name, best_curve


# =========================================================
# Train/Test Split Logic
# =========================================================
def get_test_indices(total_samples, split_ratio=0.8, seed=42):
    """
    Replicate the split logic from dataset_continuous.py
    Returns indices for the validation/test set.
    """
    indices = np.arange(total_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)
    split_idx = int(split_ratio * total_samples)
    val_idx = indices[split_idx:]
    return val_idx


# =========================================================
# PROCESS ONE CURVE
# =========================================================
def process_one_curve(tracker=None, test_indices=None):
    
    latents = np.load(latent_path)
    labels_cont = np.load(labels_cont_path)
    encoded_labels = np.load(encoded_labels_path)
    N = latents.shape[0]

    # Use test_indices if provided, otherwise sample random
    if test_indices is not None:
        # Sample a random index FROM the test set
        query_idx = int(np.random.choice(test_indices))
    else:
        query_idx = np.random.randint(0, N)

    query_latent = latents[query_idx : query_idx + 1]

    # ground truth
    gt_points = clean_and_reshape_label(labels_cont[query_idx])
    gt_mech_idx = int(encoded_labels[query_idx])
    gt_mech_name = index_to_label[str(gt_mech_idx)]

    P_gt = simulate_curve(gt_points, gt_mech_name)
    if P_gt is None or P_gt.shape[0] < minsteps:
        return []

    coup_idx_gt = coupler_index_for(gt_mech_name)
    if coup_idx_gt < 0:
        return []

    # Prepare GT for plotting/compare (one time per query)
    gx, gy = P_gt[:, coup_idx_gt, 0], P_gt[:, coup_idx_gt, 1]
    mean_x_gt, mean_y_gt, denom_gt, phi_gt = compute_norm_params(gx, gy)
    gt_x, gt_y = apply_norm(gx, gy, mean_x_gt, mean_y_gt, denom_gt, phi_gt)
    gt_curve = np.column_stack([gt_x, gt_y])

    # Normalize GT joints if present
    gt_joints_n = None
    if gt_points.size > 0:
        jx, jy = gt_points[:, 0], gt_points[:, 1]
        jx_n, jy_n = apply_norm(jx, jy, mean_x_gt, mean_y_gt, denom_gt, phi_gt)
        gt_joints_n = np.column_stack([jx_n, jy_n])

    # Track best for THIS query
    best_dtw_query = float("inf")
    best_mech_query = "None"
    best_source_query = "None"

    # We will store the full details of the best result here
    best_res_pkt = None

    # Count stats for THIS query
    count_1 = 0
    count_2 = 0
    count_3 = 0
    total_generated = 0

    # Helper to check validity and update stats
    # lat_tensor: tensor of shape (50,)
    def evaluate_latent(lat_tensor, source_label):
        nonlocal best_dtw_query, best_mech_query, best_source_query, best_res_pkt
        nonlocal count_1, count_2, count_3, total_generated

        lat_tensor = lat_tensor.to(device)  # ensure device

        for mech_idx in range(NUM_MECH_TYPES):
            mech_name = index_to_label[str(mech_idx)]

            pred_points = generate_mechanism(
                model,
                lat_tensor,
                mech_idx,
                device=device,
                max_len=tgt_seq_len,
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

            # Normalize Pred
            mean_x_p, mean_y_p, denom_p, phi_p = compute_norm_params(px, py)
            px_n, py_n = apply_norm(px, py, mean_x_p, mean_y_p, denom_p, phi_p)
            pred_curve = np.column_stack([px_n, py_n])

            # Normalize Pred Joints
            pred_joints_n = None
            if pred_points.size > 0:
                jx_p, jy_p = pred_points[:, 0], pred_points[:, 1]
                jx_p_n, jy_p_n = apply_norm(
                    jx_p, jy_p, mean_x_p, mean_y_p, denom_p, phi_p
                )
                pred_joints_n = np.column_stack([jx_p_n, jy_p_n])

            # Calculate DTW (returns 3 values)
            dtw_val, variant_name, pred_curve_best = best_variant_dtw(
                gt_curve, pred_curve
            )

            # Update Local Counts (for immediate printing)
            total_generated += 1
            if dtw_val < 1.0:
                count_1 += 1
            if dtw_val < 2.0:
                count_2 += 1
            if dtw_val < 3.0:
                count_3 += 1

            # Update Best for Query - Keep track of the full packet
            if dtw_val < best_dtw_query:
                best_dtw_query = dtw_val
                best_mech_query = mech_name
                best_source_query = source_label

                # Transform joints with best variant for storage
                pred_joints_transformed = apply_variant_to_points(
                    pred_joints_n, variant_name
                )

                res = {
                    "query_idx": int(query_idx),
                    "gt_mech": gt_mech_name,
                    "gen_mech": mech_name,
                    "source": source_label,
                    "dtw": dtw_val,
                }

                best_res_pkt = {
                    "dtw": dtw_val,
                    "metadata": res,
                    "gt_curve": gt_curve,
                    "gt_joints": gt_joints_n,
                    "pred_curve": pred_curve_best,
                    "pred_joints": pred_joints_transformed,
                    "mech_name": mech_name,
                    "variant_name": variant_name,
                }

    # 1. Original Latent
    orig_latent_tensor = torch.tensor(query_latent[0], dtype=torch.float32)
    evaluate_latent(orig_latent_tensor, "Original")

    # 2. Neighbor Latents
    _, idxs = knn.kneighbors(query_latent)
    neighbor_idxs = idxs[0][1:]  # skip self

    for ng_idx in neighbor_idxs:
        neigh_latent = torch.tensor(full_latents[ng_idx], dtype=torch.float32)
        evaluate_latent(neigh_latent, "Neighbor")

    # End of Query Processing
    if best_res_pkt is not None:
        tqdm.write(
            f"Query {query_idx} (GT: {gt_mech_name}) | Best: {best_dtw_query:.4f} ({best_mech_query}, {best_source_query}) | <1: {count_1}, <2: {count_2}, <3: {count_3} (Total: {total_generated})"
        )

        # Update Tracker with ONLY the best result for this query
        if tracker is not None:
            tracker.update_with_best_entry(best_res_pkt)

        # Return just the best result dictionary (inside a list for compatibility if needed, but we'll adapt main)
        return [best_res_pkt["metadata"]]

    return []


# =========================================================
# PLOTTING
# =========================================================
def plot_dtw_distribution(all_results, filename=None):
    # all_results here corresponds to BEST_PER_QUERY results (since process_one_curve returns only best)

    if filename is None:
        filename = "final_dtw_stats_continuous.png"

    all_dtws = [r["dtw"] for r in all_results]

    plt.figure(figsize=(10, 6))

    # Plot Histogram
    plt.hist(
        all_dtws,
        bins=50,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        label="Best Per Query",
    )

    # Add vertical lines for thresholds
    plt.axvline(x=1.0, color="r", linestyle="--", label="DTW=1.0")
    plt.axvline(x=2.0, color="g", linestyle="--", label="DTW=2.0")
    plt.axvline(x=3.0, color="b", linestyle="--", label="DTW=3.0")

    plt.title(f"Distribution of Best DTW per Query (N={len(all_dtws)})")
    plt.xlabel("DTW Distance")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Clip x-axis
    plt.xlim(0, 10)

    plt.savefig(filename)
    plt.close()
    print(f"Saved distribution plot to {filename}")


# =========================================================
# MAIN
# =========================================================
def main():
    # Create Output Directory
    if os.path.exists(OUTPUT_DIR) and os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir)

    ALL_GENERATED_RESULTS = []
    
    # Initialize Tracker
    tracker = PlotDataTracker(k=NUM_PLOT)
    
    # Pre-calculate test indices
    # We need to know total N first. We can get it from loading one npy file.
    full_latents_dummy = np.load(latent_path, mmap_mode='r')
    total_N = full_latents_dummy.shape[0]
    test_indices = get_test_indices(total_N, split_ratio=0.8, seed=42)
    print(f"Total Dataset Size: {total_N}")
    print(f"Test Set Size: {len(test_indices)} ({(len(test_indices)/total_N)*100:.1f}%)")
    
    print(f"Starting processing of {NUM_SAMPLES} queries...")
    for _ in tqdm(range(NUM_SAMPLES), desc="Processing Queries"):
        res_list = process_one_curve(tracker=tracker, test_indices=test_indices)
        ALL_GENERATED_RESULTS.extend(res_list)

    if not ALL_GENERATED_RESULTS:
        print("No results generated.")
        return

    # ---------------------------------------------------------
    # ANALYZE RESULTS
    # ---------------------------------------------------------
    # Note: ALL_GENERATED_RESULTS now contains only the BEST result for each query.

    total_queries = len(
        ALL_GENERATED_RESULTS
    )  # Should equal NUM_SAMPLES roughly (minus failed ones)
    best_dtw_arr = np.array([x["dtw"] for x in ALL_GENERATED_RESULTS])

    mean_best = np.mean(best_dtw_arr)
    median_best = np.median(best_dtw_arr)
    std_best = np.std(best_dtw_arr)
    min_best = np.min(best_dtw_arr)
    max_best = np.max(best_dtw_arr)

    count_best_below_1 = np.sum(best_dtw_arr < 1.0)
    percent_best_below_1 = (count_best_below_1 / total_queries) * 100

    count_best_below_2 = np.sum(best_dtw_arr < 2.0)
    percent_best_below_2 = (count_best_below_2 / total_queries) * 100

    count_best_below_3 = np.sum(best_dtw_arr < 3.0)
    percent_best_below_3 = (count_best_below_3 / total_queries) * 100

    # Best Mech
    best_mech_types = [x["gen_mech"] for x in ALL_GENERATED_RESULTS]
    mech_counts = Counter(best_mech_types)
    sorted_mechs = sorted(mech_counts.items(), key=lambda x: x[1], reverse=True)
    top_mech_str = ", ".join([f"{m}: {c}" for m, c in sorted_mechs[:3]])

    # Source Analysis
    sources = [x["source"] for x in ALL_GENERATED_RESULTS]
    source_counts = Counter(sources)
    orig_percent = (source_counts.get("Original", 0) / total_queries) * 100
    neigh_percent = (source_counts.get("Neighbor", 0) / total_queries) * 100

    # Per-Mechanism Performance
    mech_groups = defaultdict(list)
    for r in ALL_GENERATED_RESULTS:
        mech_groups[r["gen_mech"]].append(r["dtw"])

    mech_stats = []
    for m, dtws in mech_groups.items():
        arr = np.array(dtws)
        mean_d = np.mean(arr)
        c_1 = np.sum(arr < 1.0)
        c_2 = np.sum(arr < 2.0)
        c_3 = np.sum(arr < 3.0)
        total_m = len(arr)
        mech_stats.append(
            {
                "mech": m,
                "mean": mean_d,
                "count": total_m,
                "c_1": c_1,
                "c_2": c_2,
                "c_3": c_3,
            }
        )
    mech_stats.sort(key=lambda x: x["mean"])  # Sort by mean DTW

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------
    print("\n==============================================")
    print(" FINAL STATISTICS (Based on Best Result Per Query) ")
    print("==============================================")
    print(f"Total Queries Processed: {total_queries}")
    print("-" * 30)
    print(f"DTW < 1.0: {count_best_below_1} ({percent_best_below_1:.1f}%)")
    print(f"DTW < 2.0: {count_best_below_2} ({percent_best_below_2:.1f}%)")
    print(f"DTW < 3.0: {count_best_below_3} ({percent_best_below_3:.1f}%)")
    print(f"Mean DTW: {mean_best:.4f} (std: {std_best:.4f})")
    print(f"Median DTW: {median_best:.4f}")
    print(f"Min DTW: {min_best:.4f}")
    print(f"Max DTW: {max_best:.4f}")
    print("-" * 30)
    print("Source Analysis (Where did the best result come from?):")
    print(f"  Original: {source_counts.get('Original', 0)} ({orig_percent:.1f}%)")
    print(f"  Neighbor: {source_counts.get('Neighbor', 0)} ({neigh_percent:.1f}%)")
    print("-" * 30)
    print("Mechanism Performance (Mean Best DTW):")
    print(f"{'Mech':<15} | {'Mean':<6} | {'<1.0':<5} | {'<2.0':<5} | {'Total':<5}")
    for s in mech_stats:
        print(
            f"{s['mech']:<15} | {s['mean']:.3f}  | {s['c_1']:<5} | {s['c_2']:<5} | {s['count']:<5}"
        )
    print("==============================================")

    # File Output
    stats_path = os.path.join(OUTPUT_DIR, "final_dtw_stats_continuous.txt")
    with open(stats_path, "w") as f:
        f.write("=== Best-Per-Query Stats ===\n")
        f.write(f"Total Queries: {total_queries}\n")
        f.write(f"Mean Best DTW: {mean_best:.6f}\n")
        f.write(f"Std Dev Best DTW: {std_best:.6f}\n")
        f.write(f"Median Best DTW: {median_best:.6f}\n")
        f.write(f"Min Best DTW: {min_best:.6f}\n")
        f.write(f"Max Best DTW: {max_best:.6f}\n")
        f.write(f"Best < 1.0: {count_best_below_1} ({percent_best_below_1:.2f}%)\n")
        f.write(f"Best < 2.0: {count_best_below_2} ({percent_best_below_2:.2f}%)\n")
        f.write(f"Best < 3.0: {count_best_below_3} ({percent_best_below_3:.2f}%)\n")
        f.write(f"Top Mechanisms: {top_mech_str}\n\n")

        f.write("=== Source Analysis ===\n")
        f.write(f"Original: {source_counts.get('Original', 0)} ({orig_percent:.1f}%)\n")
        f.write(
            f"Neighbor: {source_counts.get('Neighbor', 0)} ({neigh_percent:.1f}%)\n\n"
        )

        f.write("=== Per-Mechanism Performance (Mean Best DTW) ===\n")
        f.write(f"{'Mech':<15} {'Mean':<8} {'<1.0':<6} {'<2.0':<6} {'Total':<6}\n")
        for s in mech_stats:
            f.write(
                f"{s['mech']:<15} {s['mean']:.4f}   {s['c_1']:<6} {s['c_2']:<6} {s['count']:<6}\n"
            )
        f.write("\n")

    # Plot Distribution
    plot_path = os.path.join(OUTPUT_DIR, "final_dtw_stats_continuous.png")
    plot_dtw_distribution(ALL_GENERATED_RESULTS, filename=plot_path)

    # Plot Best/Worst K

    # Plot Best
    for i, data in enumerate(tracker.best_list):
        fname = os.path.join(
            plots_dir,
            f"best_{data['dtw']:.2f}_Q{data['metadata']['query_idx']}.png",
        )
        plot_result_pair(data, fname)

    # Plot Worst
    for i, data in enumerate(tracker.worst_list):
        fname = os.path.join(
            plots_dir,
            f"worst_{data['dtw']:.2f}_Q{data['metadata']['query_idx']}.png",
        )
        plot_result_pair(data, fname)

    print(f"Saved results and plots to '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    main()
