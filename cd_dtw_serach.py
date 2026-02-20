# =========================
# IMPORTS
# =========================

import torch
from torch.utils.data import Subset, DataLoader
from llama_latent_continuous import LatentLLaMA_Continuous
from dataset_continuous import BarLinkageDataset
import matplotlib.pyplot as plt

import json
import numpy as np
import requests
from tqdm import tqdm
from scipy.spatial import KDTree

from path_decomposition import computeSolSteps, linkMajor
import os
import shutil

torch.set_float32_matmul_precision("medium")
# =========================
# DEVICE
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# PATHS
# =========================
checkpoint_path = "weights/LATENT_LLAMA_CONT_d512_nf128_h8_n6_bs512_lr0.0005.pth"
data_dir = "dataset_17mechs"

OUT_DIR = "results_gt_vs_pred_search_cd_and_dtw"
if os.path.exists(OUT_DIR) and os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

JSONL_LOG_PATH = os.path.join(OUT_DIR, "metrics.jsonl")

# =========================
# LOAD DATASET
# =========================
_, test_dataset = BarLinkageDataset.from_folder(data_dir)
# indices = torch.randperm(len(test_dataset))[:100]
# subset = Subset(test_dataset, indices)
# dataloader = DataLoader(subset, batch_size=1)
dataloader = DataLoader(test_dataset, batch_size=1)

# =========================
# LOAD MODEL
# =========================
checkpoint = torch.load(checkpoint_path, map_location=device)
model_config = checkpoint["model_config"]

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

model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

# =========================
# SERVER CONFIG
# =========================
API_ENDPOINT = "http://localhost:4000/simulation"
API_ENDPOINT_8BAR = "http://localhost:4000/simulation-8bar"
HEADERS = {"Content-Type": "application/json"}

speedscale = 1
steps = 360
minsteps = int(steps * 20 / 360)

# =========================
# CONFIG
# =========================
NUM_SAMPLES = 50
NUM_ROT_ANGLES = 32

RESAMPLE_N = 360
DTW_BAND = 24

CD_SUCCESS_THRESH = 0.01
DTW_SUCCESS_THRESH = 0.25

# SEARCH CONFIG
NUM_VARIATIONS = 50
MAX_ITER = 3
NUM_GOOD_THRESH = 3
SIGMA_INITIAL = 0.1
DECAY = 0.5


# =========================
# LOAD MAPPINGS
# =========================
label_mapping_path = os.path.join(data_dir, "label_mapping.json")
coupler_mapping_path = "BSIdict.json"

with open(label_mapping_path) as f:
    index_to_label = json.load(f)["index_to_label"]

with open(coupler_mapping_path) as f:
    coupler_mapping = json.load(f)


# =========================
# HELPERS
# =========================
def is_type8bar(name):
    return name.startswith("Type")


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


def safe_name(name, max_len=40):
    return "".join(c if c.isalnum() else "_" for c in name)[:max_len] or "unk"


def coupler_index_for(name):
    try:
        return 10 if is_type8bar(name) else coupler_mapping[name]["c"].index(1)
    except Exception:
        return -1


# =========================
# SAFE SIMULATION
# =========================
def simulate_safe(points, mech_name):
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
        print(
            f"misshapen points arr | {type(points)} | {points.shape if isinstance(points, np.ndarray) else len(points)}"
        )
        return None
    pts = np.round(points, 3)
    try:
        if is_type8bar(mech_name):
            raise ValueError("model not yet trained for 8 bar mechs")
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
                "params": pts.tolist()[:5] if "P" in mech_name else pts.tolist(),
                "type": mech_name,
                "speedScale": speedscale,
                "steps": steps,
                "relativeTolerance": 0.1,
            }
            url = API_ENDPOINT

        r = requests.post(
            url, headers=HEADERS, data=json.dumps([payload]), timeout=30
        ).json()
        if not r or "poses" not in r[0]:
            return None
        P = np.array(r[0]["poses"])
        if P.ndim < 3 or P.shape[0] < minsteps:
            return None
        return P
    except Exception as err:
        print(f"simulate_safe failed for mech {mech_name}\n{err}")
        return None


# =========================
# LATENT EVALUATION HELPER
# =========================
def evaluate_latent(latent_vec, gt_curve, gt_rs, seed_latent=None):
    """
    Evaluates a latent vector across all mechanism types.
    Returns a list of dicts: {'mech': name, 'cd': float, 'dtw': float, 'dist': float, 'curve': np.array, 'latent': tensor}
    """
    # Distance from seed (if provided)
    dist = 0.0
    if seed_latent is not None:
        dist = torch.norm(latent_vec - seed_latent).item()

    # Prepare batch inputs
    mech_items = list(index_to_label.items())

    results = []

    def process_mech_full(item):
        mech_idx_str, mech_name = item
        mech_idx = int(mech_idx_str)

        # Predict (Parallel execution)
        try:
            # predict_safe returns (B, T, 2). Here B=1.
            pred_norm = predict_safe(model, latent_vec, mech_idx)  # (1, T, 2)
            if pred_norm.ndim == 3:
                pred_norm = pred_norm[0]  # (T, 2)
        except Exception as e:
            # print(f"Prediction failed for {mech_name}: {e}")
            return None

        # Denormalize points (same padding logic as before)
        pred_padded = np.zeros((pred_norm.shape[0], 3), dtype=np.float32)
        pred_padded[:, :2] = pred_norm
        pred_pts = pred_padded[:, :2]

        # Simulate
        P = simulate_safe(pred_pts, mech_name)
        if P is None:
            return None

        ci = coupler_index_for(mech_name)
        if ci < 0:
            return None

        curve = P[:, ci, :]

        # Metrics
        cd, aligned, _ = chamfer_O2_align(gt_curve, curve)
        dtw = min(
            dtw_banded(gt_rs, aligned, DTW_BAND),
            dtw_banded(gt_rs, aligned[::-1], DTW_BAND),
        )

        return {
            "mech": mech_name,
            "params": pred_pts,
            "cd": cd,
            "dtw": dtw,
            "dist": dist,
            "curve": aligned,  # store aligned curve for plotting
            "latent": latent_vec,
            "mech_idx": mech_idx,
        }

    # Parallelize Prediction + Simulation
    from concurrent.futures import ThreadPoolExecutor

    # 17 mechanisms -> enough workers
    with ThreadPoolExecutor(max_workers=17) as executor:
        futures = executor.map(process_mech_full, mech_items)

        for res in futures:
            if res is not None:
                results.append(res)

    return results


# =========================
# CONTINUOUS GENERATION (AUTOREGRESSIVE)
# =========================
@torch.no_grad()
def predict_safe(model, latent, mech_idx, max_len=None):
    if max_len is None:
        max_len = 8  # Default to typical mechanism length

    model.eval()
    latent = latent.unsqueeze(0) if latent.dim() == 1 else latent
    latent = latent.to(device)
    mech_labels = torch.tensor([mech_idx], device=device, dtype=torch.long)
    B = latent.shape[0]

    # Start with an empty coordinate sequence (B, 0, 2)
    generated_coords = torch.zeros((B, 0, 2), device=device)

    for i in range(max_len):
        attn_mask = torch.ones((B, generated_coords.shape[1]), device=device)
        preds = model(generated_coords, attn_mask, latent, mech_labels)

        # Next prediction is the last element
        next_pred = preds[:, -1, :]  # (B, 3)
        new_xy = next_pred[:, :2]
        stop_logit = next_pred[:, 2]
        stop_prob = torch.sigmoid(stop_logit)

        generated_coords = torch.cat([generated_coords, new_xy.unsqueeze(1)], dim=1)

        if stop_prob.item() > 0.5:
            break

    return generated_coords.squeeze(0).cpu().numpy()


# =========================
# CURVE OPS
# =========================
def resample_by_arclength(curve, n):
    curve = np.asarray(curve, np.float64)
    d = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s = np.insert(np.cumsum(d), 0, 0)
    if s[-1] < 1e-8:
        return curve
    s /= s[-1]
    t = np.linspace(0, 1, n)
    out = np.zeros((n, 2))
    for k in range(2):
        out[:, k] = np.interp(t, s, curve[:, k])
    return out


def center_curve(c):
    return c - c.mean(axis=0)


def scale_to_match_variance(src, ref):
    return src * (np.sqrt(np.var(ref)) / (np.sqrt(np.var(src)) + 1e-8))


# =========================
# CHAMFER + ALIGNMENT
# =========================
def chamfer_distance(a, b):
    ta = KDTree(a)
    tb = KDTree(b)
    d_ab, _ = tb.query(a, k=1)
    d_ba, _ = ta.query(b, k=1)
    return float(np.mean(d_ab**2) + np.mean(d_ba**2))


def chamfer_O2_align(gt_curve, pred_curve):
    gt_rs = center_curve(resample_by_arclength(center_curve(gt_curve), RESAMPLE_N))
    pr_rs = scale_to_match_variance(
        center_curve(resample_by_arclength(pred_curve, RESAMPLE_N)), gt_rs
    )

    best_cd = 1e18
    best_curve = None

    mirrors = [lambda P: P, lambda P: np.column_stack([-P[:, 0], P[:, 1]])]
    angles = np.linspace(0, 2 * np.pi, NUM_ROT_ANGLES, endpoint=False)

    for mfn in mirrors:
        Pm = mfn(pr_rs)
        for theta in angles:
            c, s = np.cos(theta), np.sin(theta)
            Pr = np.column_stack(
                [Pm[:, 0] * c - Pm[:, 1] * s, Pm[:, 0] * s + Pm[:, 1] * c]
            )
            Pr = center_curve(Pr)
            cd = chamfer_distance(gt_rs, Pr)
            if cd < best_cd:
                best_cd = cd
                best_curve = Pr

    return best_cd, best_curve, gt_rs


# =========================
# FAST BANDED DTW
# =========================
try:
    from numba import njit

    @njit(cache=True, fastmath=True)
    def dtw_banded(a, b, band):
        n, m = a.shape[0], b.shape[0]
        INF = 1e30
        prev = np.full(m + 1, INF)
        curr = np.full(m + 1, INF)
        prev[0] = 0.0
        for i in range(1, n + 1):
            j0 = max(1, i - band)
            j1 = min(m, i + band)
            curr[:] = INF
            for j in range(j0, j1 + 1):
                dx = a[i - 1, 0] - b[j - 1, 0]
                dy = a[i - 1, 1] - b[j - 1, 1]
                curr[j] = dx * dx + dy * dy + min(prev[j], curr[j - 1], prev[j - 1])
            prev, curr = curr, prev
        return prev[m] / (n + m)
except Exception:

    def dtw_banded(a, b, band):
        return np.inf


# =========================
# METRICS
# =========================
QUERY_CD, QUERY_DTW = [], []
SEARCH_MEAN_CD, SEARCH_MIN_CD = [], []
SEARCH_MEAN_DTW, SEARCH_MIN_DTW = [], []

log_f = open(JSONL_LOG_PATH, "w")

# =========================
# MAIN LOOP
# =========================
for i, batch in enumerate(tqdm(dataloader)):
    if len(QUERY_DTW) >= NUM_SAMPLES:
        break

    latent = batch["vae_mu"][0].to(device)
    mech_idx = int(batch["encoded_labels"][0])
    mech_name = index_to_label[str(mech_idx)]

    sample_dir = os.path.join(OUT_DIR, f"{i:03d}_{safe_name(mech_name)}")
    os.makedirs(sample_dir, exist_ok=True)

    # GT: use labels_continuous
    gt_vals = batch["labels_continuous"][0].cpu().numpy()
    gt_pts = gt_vals[:, :2]  # (8, 2)

    # Filter padding
    attn_mask = batch["attn_mask"][0].cpu().numpy()
    gt_pts = gt_pts[attn_mask]

    Pgt = simulate_safe(gt_pts, mech_name)
    if Pgt is None:
        continue

    ci = coupler_index_for(mech_name)
    if ci < 0:
        continue

    gt_curve = Pgt[:, ci, :]

    # ---------- QUERY ----------
    pred_vals_norm = predict_safe(model, latent, mech_idx)
    pred_vals_padded = np.zeros((pred_vals_norm.shape[0], 3), dtype=np.float32)
    pred_vals_padded[:, :2] = pred_vals_norm
    pred_pts = pred_vals_padded[:, :2]  # (T, 2)

    Pp = simulate_safe(pred_pts, mech_name)
    if Pp is None:
        continue

    pred_curve = Pp[:, ci, :]

    # Alignment & Metrics (Original Method)
    cd, aligned_pred, gt_rs = chamfer_O2_align(gt_curve, pred_curve)
    dtw = min(
        dtw_banded(gt_rs, aligned_pred, DTW_BAND),
        dtw_banded(gt_rs, aligned_pred[::-1], DTW_BAND),
    )

    QUERY_CD.append(cd)
    QUERY_DTW.append(dtw)

    plt.figure(figsize=(6, 6))
    plt.plot(gt_rs[:, 0], gt_rs[:, 1], "k", lw=2, label="GT")
    plt.plot(aligned_pred[:, 0], aligned_pred[:, 1], "r--", lw=2, label="Pred")
    plt.legend()
    plt.axis("equal")
    plt.savefig(
        os.path.join(sample_dir, f"gt_vs_pred_query_CD_{cd:.6f}_DTW_{dtw:.6f}.png"),
        dpi=200,
    )
    plt.close()

    # ---------- LATENT SPACE EXPLORATION (SMART FAN) ----------
    all_search_metrics = []  # List of {'cd': float, 'dtw': float, 'dist': float}

    current_sigma = SIGMA_INITIAL

    # 1. Initial Candidates (Query + Noise)
    # Start with query (index 0)
    candidates_latents = [latent]
    for _ in range(NUM_VARIATIONS):
        noise = torch.randn_like(latent, device=device) * current_sigma
        candidates_latents.append(latent + noise)

    found_good_count = 0

    for iteration in range(MAX_ITER):
        iter_results = []  # Full objects for this iteration

        iter_dir = os.path.join(sample_dir, f"iter_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)

        # Evaluate batch
        for lat_cand in candidates_latents:
            res_list = evaluate_latent(lat_cand, gt_curve, gt_rs, seed_latent=latent)

            for res in res_list:
                res["iter"] = iteration

                # Metric tracking
                m = {"cd": res["cd"], "dtw": res["dtw"], "dist": res["dist"]}
                all_search_metrics.append(m)
                iter_results.append(res)

                if res["cd"] < CD_SUCCESS_THRESH:
                    found_good_count += 1

        # Sort iteration results by CD
        iter_results.sort(key=lambda x: x["cd"])
        top_winners = iter_results[:5]

        # Log & Plot Top 5
        print(f"  [Iter {iteration}] Best results:")
        for rank, res in enumerate(top_winners):
            print(
                f"    #{rank + 1}: {res['mech']} | CD={res['cd']:.6f} | Dist={res['dist']:.4f}"
            )

            # Plot
            fname = f"{rank + 1}_{safe_name(res['mech'])}_CD_{res['cd']:.4f}_D_{res['dist']:.3f}.png"
            plt.figure(figsize=(4, 4))
            plt.plot(gt_rs[:, 0], gt_rs[:, 1], "k", lw=2)
            plt.plot(res["curve"][:, 0], res["curve"][:, 1], "r--", lw=1)
            plt.axis("equal")
            plt.title(f"Rank {rank + 1} | CD={res['cd']:.4f}")
            plt.savefig(os.path.join(iter_dir, fname), dpi=100)
            plt.close()

        # Stop conditions
        if found_good_count >= NUM_GOOD_THRESH:
            break

        if iteration >= MAX_ITER:
            break

        # Refinement: Top 5 spawn new candidates
        candidates_latents = []
        current_sigma *= DECAY

        # Generate new variations
        for win in top_winners:
            base_lat = win["latent"]
            for _ in range(NUM_VARIATIONS):
                noise = torch.randn_like(base_lat) * current_sigma
                candidates_latents.append(base_lat + noise)

    # ---------- METRICS & RUGGEDNESS PLOT ----------
    search_cds, search_dtws = [], []

    if all_search_metrics:
        # Ruggedness Plot
        dists = [x["dist"] for x in all_search_metrics]
        cds = [x["cd"] for x in all_search_metrics]

        plt.figure(figsize=(6, 5))
        plt.scatter(dists, cds, alpha=0.5, s=10)
        plt.xlabel("Distance from Seed Latent")
        plt.ylabel("Chamfer Distance")
        plt.title(f"Latent Space Ruggedness (Query {i})")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(sample_dir, "latent_cd.png"), dpi=200)
        plt.close()

        # Collect Top 30 for stats
        all_search_metrics.sort(key=lambda x: x["cd"])
        top_k = all_search_metrics[:30]

        search_cds = [x["cd"] for x in top_k]
        search_dtws = [x["dtw"] for x in top_k]

    if len(search_cds) > 0:
        SEARCH_MEAN_CD.append(np.mean(search_cds))
        SEARCH_MIN_CD.append(np.min(search_cds))
        SEARCH_MEAN_DTW.append(np.mean(search_dtws))
        SEARCH_MIN_DTW.append(np.min(search_dtws))

    log_f.write(
        json.dumps(
            {
                "sample_i": i,
                "mech": mech_name,
                "query_cd": float(cd),
                "query_dtw": float(dtw),
                "search_mean_cd": None
                if len(search_cds) == 0
                else float(np.mean(search_cds)),
                "search_min_cd": None
                if len(search_cds) == 0
                else float(np.min(search_cds)),
                "search_mean_dtw": None
                if len(search_dtws) == 0
                else float(np.mean(search_dtws)),
                "search_min_dtw": None
                if len(search_dtws) == 0
                else float(np.min(search_dtws)),
            }
        )
        + "\n"
    )
    log_f.flush()

log_f.close()


# =========================
# FINAL SUMMARY
# =========================
def summarize(name, arr, thresh):
    arr = np.asarray(arr, float)
    print(
        f"{name} | count={len(arr)} | mean={arr.mean():.6f} | "
        f"median={np.median(arr):.6f} | success(<{thresh})={(arr < thresh).sum()}"
    )


print("\n=== FINAL METRIC SUMMARY ===")
summarize("QUERY CD", QUERY_CD, CD_SUCCESS_THRESH)
summarize("QUERY DTW", QUERY_DTW, DTW_SUCCESS_THRESH)
summarize("SEARCH MEAN CD", SEARCH_MEAN_CD, CD_SUCCESS_THRESH)
summarize("SEARCH MEAN DTW", SEARCH_MEAN_DTW, DTW_SUCCESS_THRESH)

# RESULTS

# NUM_SAMPLES = 10
# NUM_VARIATIONS = 50
# MAX_ITER = 10
# 139 mins
# === FINAL METRIC SUMMARY ===
# QUERY CD | count=10 | mean=0.127135 | median=0.089215 | success(<0.01)=1
# QUERY DTW | count=10 | mean=1.327858 | median=0.110028 | success(<0.25)=6
# SEARCH MEAN CD | count=10 | mean=0.023532 | median=0.013452 | success(<0.01)=3
# SEARCH MEAN DTW | count=10 | mean=0.830527 | median=0.157794 | success(<0.25)=7

# NUM_SAMPLES = 10
# NUM_VARIATIONS = 100
# MAX_ITER = 5
# 107 mins
# === FINAL METRIC SUMMARY ===
# QUERY CD | count=10 | mean=0.127135 | median=0.089215 | success(<0.01)=1
# QUERY DTW | count=10 | mean=1.327858 | median=0.110028 | success(<0.25)=6
# SEARCH MEAN CD | count=10 | mean=0.023224 | median=0.013570 | success(<0.01)=4
# SEARCH MEAN DTW | count=10 | mean=0.645297 | median=0.151527 | success(<0.25)=8
