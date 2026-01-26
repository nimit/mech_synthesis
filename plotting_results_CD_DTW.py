# =========================
# IMPORTS
# =========================

import torch
from torch.utils.data import DataLoader
from llama_latent_model import LatentLLaMA_SingleToken
from dataset import BarLinkageDataset
import matplotlib.pyplot as plt
torch.set_float32_matmul_precision("medium")

import json
import numpy as np
import requests
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree

from path_decomposition import computeSolSteps, linkMajor
import os

# =========================
# DEVICE
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# PATHS
# =========================
checkpoint_path = "./weights/WEIGHTS_33/LATENT_LLAMA_d1536_h32_n6_bs512_lr0.0001_best.pth"
data_dir = "/home/anurizada/Documents/processed_dataset_33"

OUT_DIR = "results_gt_vs_pred_knn_cd_and_dtw"
os.makedirs(OUT_DIR, exist_ok=True)

JSONL_LOG_PATH = os.path.join(OUT_DIR, "metrics.jsonl")

# =========================
# LOAD DATASET
# =========================
dataset = BarLinkageDataset(data_dir=data_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# =========================
# LOAD MODEL
# =========================
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

# =========================
# SERVER CONFIG
# =========================
API_ENDPOINT = "http://localhost:4001/simulation"
API_ENDPOINT_8BAR = "http://localhost:4001/simulation-8bar"
HEADERS = {"Content-Type": "application/json"}

speedscale = 1
steps = 360
minsteps = int(steps * 20 / 360)

# =========================
# CONFIG
# =========================
MAX_SAMPLES = 100
NUM_NEIGHBORS = 30
BIN_OFFSET = 3
NUM_BINS = 201
NUM_ROT_ANGLES = 32

RESAMPLE_N = 360
DTW_BAND = 24

CD_SUCCESS_THRESH = 0.01
DTW_SUCCESS_THRESH = 0.25

# =========================
# LOAD MAPPINGS
# =========================
label_mapping_path = os.path.join(data_dir, "label_mapping.json")
coupler_mapping_path = "/home/anurizada/Documents/transformer_gaussian/BSIdict.json"

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
# BINNING
# =========================
class CoordinateBinner:
    def __init__(self):
        edges = np.linspace(-1, 1, NUM_BINS)
        self.centers = (edges[:-1] + edges[1:]) / 2

    def decode(self, tokens):
        idx = tokens.astype(int) - BIN_OFFSET
        if len(idx) == 0 or np.any(idx < 0) or np.any(idx >= len(self.centers)):
            return None
        return self.centers[idx]

binner = CoordinateBinner()

# =========================
# SAFE SIMULATION
# =========================
def simulate_safe(points, mech_name):
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
        return None
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
                "params": pts.tolist()[:5] if "P" in mech_name else pts.tolist(),
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
# GREEDY DECODER
# =========================
def predict_safe(model, latent, mech_idx, max_len=25):
    latent = latent.unsqueeze(0).to(device)
    mech_labels = torch.tensor([mech_idx], device=device)
    tokens = torch.tensor([[0]], device=device)
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(tokens, None, latent, mech_labels)
            nxt = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            tokens = torch.cat([tokens, nxt], dim=1)
            if int(nxt.item()) == 1:
                break
    return tokens.squeeze().cpu().numpy()

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
    ta = cKDTree(a)
    tb = cKDTree(b)
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

    mirrors = [lambda P: P, lambda P: np.column_stack([-P[:,0], P[:,1]])]
    angles = np.linspace(0, 2*np.pi, NUM_ROT_ANGLES, endpoint=False)

    for mfn in mirrors:
        Pm = mfn(pr_rs)
        for theta in angles:
            c, s = np.cos(theta), np.sin(theta)
            Pr = np.column_stack([
                Pm[:,0]*c - Pm[:,1]*s,
                Pm[:,0]*s + Pm[:,1]*c
            ])
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
        prev = np.full(m+1, INF)
        curr = np.full(m+1, INF)
        prev[0] = 0.0
        for i in range(1, n+1):
            j0 = max(1, i-band)
            j1 = min(m, i+band)
            curr[:] = INF
            for j in range(j0, j1+1):
                dx = a[i-1,0] - b[j-1,0]
                dy = a[i-1,1] - b[j-1,1]
                curr[j] = dx*dx + dy*dy + min(prev[j], curr[j-1], prev[j-1])
            prev, curr = curr, prev
        return prev[m] / (n+m)
except Exception:
    def dtw_banded(a, b, band):
        return np.inf

# =========================
# BUILD KNN
# =========================
Z = np.stack([dataset[i]["vae_mu"].cpu().numpy().reshape(-1) for i in range(len(dataset))])
knn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS + 1).fit(Z)

# =========================
# METRICS
# =========================
QUERY_CD, QUERY_DTW = [], []
KNN_MEAN_CD, KNN_MIN_CD = [], []
KNN_MEAN_DTW, KNN_MIN_DTW = [], []

log_f = open(JSONL_LOG_PATH, "w")

# =========================
# MAIN LOOP
# =========================
for i, batch in enumerate(tqdm(dataloader)):
    if i >= MAX_SAMPLES:
        break

    latent = batch["vae_mu"][0]
    mech_idx = int(batch["encoded_labels"][0])
    mech_name = index_to_label[str(mech_idx)]

    sample_dir = os.path.join(OUT_DIR, f"{i:03d}_{safe_name(mech_name)}")
    os.makedirs(sample_dir, exist_ok=True)

    gt_tokens = batch["labels_discrete"][0].cpu().numpy()
    gt_tokens = gt_tokens[gt_tokens >= BIN_OFFSET]
    if len(gt_tokens) < 4:
        continue

    gt_vals = binner.decode(gt_tokens)
    gt_pts = gt_vals[:len(gt_vals)//2*2].reshape(-1,2)
    Pgt = simulate_safe(gt_pts, mech_name)
    if Pgt is None:
        continue

    ci = coupler_index_for(mech_name)
    gt_curve = Pgt[:,ci,:]

    # ---------- QUERY ----------
    pred_tokens = predict_safe(model, latent, mech_idx)
    pred_tokens = pred_tokens[pred_tokens >= BIN_OFFSET]
    if len(pred_tokens) < 4:
        continue

    pred_vals = binner.decode(pred_tokens)
    pred_pts = pred_vals[:len(pred_vals)//2*2].reshape(-1,2)
    Pp = simulate_safe(pred_pts, mech_name)
    if Pp is None:
        continue

    pred_curve = Pp[:,ci,:]

    cd, aligned_pred, gt_rs = chamfer_O2_align(gt_curve, pred_curve)
    dtw = min(
        dtw_banded(gt_rs, aligned_pred, DTW_BAND),
        dtw_banded(gt_rs, aligned_pred[::-1], DTW_BAND)
    )

    QUERY_CD.append(cd)
    QUERY_DTW.append(dtw)

    plt.figure(figsize=(6,6))
    plt.plot(gt_rs[:,0], gt_rs[:,1], "k", lw=2)
    plt.plot(aligned_pred[:,0], aligned_pred[:,1], "r--", lw=2)
    plt.axis("equal")
    plt.savefig(
        os.path.join(
            sample_dir,
            f"gt_vs_pred_query_CD_{cd:.6f}_DTW_{dtw:.6f}.png"
        ),
        dpi=200
    )
    plt.close()

    # ---------- KNN ----------
    knn_cds, knn_dtws = [], []

    _, idxs = knn.kneighbors(latent.cpu().numpy().reshape(1,-1))
    for k, zidx in enumerate(idxs[0][1:]):
        nb = dataset[int(zidx)]
        nb_latent = nb["vae_mu"].to(device)
        nb_mech_idx = int(nb["encoded_labels"])
        nb_mech = index_to_label[str(nb_mech_idx)]

        nb_tokens = predict_safe(model, nb_latent, nb_mech_idx)
        nb_tokens = nb_tokens[nb_tokens >= BIN_OFFSET]
        if len(nb_tokens) < 4:
            continue

        nb_vals = binner.decode(nb_tokens)
        nb_pts = nb_vals[:len(nb_vals)//2*2].reshape(-1,2)
        Pn = simulate_safe(nb_pts, nb_mech)
        if Pn is None:
            continue

        nb_curve = Pn[:,coupler_index_for(nb_mech),:]

        cd_nb, aligned_nb, _ = chamfer_O2_align(gt_curve, nb_curve)
        dtw_nb = min(
            dtw_banded(gt_rs, aligned_nb, DTW_BAND),
            dtw_banded(gt_rs, aligned_nb[::-1], DTW_BAND)
        )

        knn_cds.append(cd_nb)
        knn_dtws.append(dtw_nb)

        plt.figure(figsize=(6,6))
        plt.plot(gt_rs[:,0], gt_rs[:,1], "k", lw=2)
        plt.plot(aligned_nb[:,0], aligned_nb[:,1], "r--", lw=2)
        plt.axis("equal")
        plt.savefig(
            os.path.join(
                sample_dir,
                f"knn_{k:02d}_CD_{cd_nb:.6f}_DTW_{dtw_nb:.6f}.png"
            ),
            dpi=200
        )
        plt.close()

    if len(knn_cds) > 0:
        KNN_MEAN_CD.append(np.mean(knn_cds))
        KNN_MIN_CD.append(np.min(knn_cds))
        KNN_MEAN_DTW.append(np.mean(knn_dtws))
        KNN_MIN_DTW.append(np.min(knn_dtws))

    log_f.write(json.dumps({
        "sample_i": i,
        "mech": mech_name,
        "query_cd": float(cd),
        "query_dtw": float(dtw),
        "knn_mean_cd": None if len(knn_cds)==0 else float(np.mean(knn_cds)),
        "knn_min_cd": None if len(knn_cds)==0 else float(np.min(knn_cds)),
        "knn_mean_dtw": None if len(knn_dtws)==0 else float(np.mean(knn_dtws)),
        "knn_min_dtw": None if len(knn_dtws)==0 else float(np.min(knn_dtws)),
    }) + "\n")
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
summarize("KNN MEAN CD", KNN_MEAN_CD, CD_SUCCESS_THRESH)
summarize("KNN MEAN DTW", KNN_MEAN_DTW, DTW_SUCCESS_THRESH)

# === FINAL METRIC SUMMARY ===
# QUERY CD | count=100 | mean=0.000161 | median=0.000000 | success(<0.01)=100
# QUERY DTW | count=100 | mean=0.000051 | median=0.000000 | success(<0.25)=100
# KNN MEAN CD | count=100 | mean=0.007083 | median=0.002321 | success(<0.01)=79
# KNN MEAN DTW | count=100 | mean=0.057973 | median=0.001481 | success(<0.25)=94