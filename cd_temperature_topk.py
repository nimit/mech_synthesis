"""
CD (Chamfer) evaluation for FIXED sample indices on dataset_33 using LatentLLaMA,
with temperature + top-k decoding, and the SAME rigorous O(2)-invariant CD pipeline:

Pipeline per (GT curve, Pred curve):
  - simulate coupler curves from GT tokens + predicted tokens
  - O(2)-invariant Chamfer:
      * center -> arc-length resample -> center (GT)
      * resample -> center -> scale-to-GT (Pred)
      * try mirror (id / reflect y) + many rotations
      * re-center AFTER every transform (guards drift)
  - record CD, optionally save plot with filename containing CD

No KNN. Uses FIXED_SAMPLE_IDXS every run.

Requirements:
  - Local sim server at localhost:4001 with /simulation and /simulation-8bar
  - dataset_33 present
  - BSIdict.json present
"""

# =========================
# IMPORTS
# =========================
import os
import json
import math
import requests
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm

from llama_latent_model import LatentLLaMA_SingleToken
from dataset import BarLinkageDataset

from path_decomposition import computeSolSteps, linkMajor

# =========================
# DEVICE
# =========================
torch.set_float32_matmul_precision("medium")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# USER SETTINGS (sampling)
# =========================
TEMPERATURE = 0.0        # 0.0 => greedy; >0 => sampling
TOP_K = None             # None/0 => disabled; else int
SEED = 123               # for reproducible sampling when TEMPERATURE>0
DEBUG_PROBS = False      # prints entropy + top tokens each step

# Always evaluate these SAME dataset indices
FIXED_SAMPLE_IDXS = [
    37454, 95071, 73199, 59866, 15601, 15599, 86617, 60111, 70807, 2058,
    96990, 83244, 21233, 18340, 30424, 52475, 43195, 29123, 61185, 13949,
    29214, 36636, 45607, 78518, 19967, 51423, 59241, 4699, 45537, 36636,
    78518, 10000, 2058, 65088, 4505, 94888, 96564, 80840, 363, 28326,
    88053, 28319, 42515, 15602, 28918, 49851, 48947, 27126, 12811, 9905,
    66155, 16090, 54841, 69189, 22489, 40311, 34850, 70641, 9073, 79883,
    30424, 52192, 14223, 43135, 80219, 30505, 16949, 8258, 13261, 66060,
    34208, 82191, 93950, 19663, 61748, 2622, 97242, 84922, 56565, 6208,
    38063, 17701, 75094, 96895, 59789, 32533, 84123, 94356, 52185, 55517,
    12788, 58083, 54886, 97562, 8312, 2229, 80744, 31943
]

# =========================
# PATHS
# =========================
checkpoint_path = "./weights/WEIGHTS_33/LATENT_LLAMA_d1536_h32_n6_bs512_lr0.0001_best.pth"
data_dir = "/home/anurizada/Documents/processed_dataset_33"

OUT_DIR = "cd_fixedidx_temperature_topk_dataset33"
os.makedirs(OUT_DIR, exist_ok=True)
JSONL_LOG_PATH = os.path.join(OUT_DIR, "metrics.jsonl")

label_mapping_path = os.path.join(data_dir, "label_mapping.json")
coupler_mapping_path = "/home/anurizada/Documents/transformer_gaussian/BSIdict.json"

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
# TOKEN CONSTANTS
# =========================
SOS_TOKEN, EOS_TOKEN = 0, 1
BIN_OFFSET = 3
NUM_BINS = 201  # edges count, so centers count = 200

# =========================
# CD / ALIGNMENT CONFIG
# =========================
NUM_ROT_ANGLES = 32   # rotation marginalization resolution
RESAMPLE_N = 360      # arc-length resampling points (None to disable)
CD_SUCCESS_THRESH = 0.01
SAVE_PLOTS = True     # set False if you only want numbers (faster)

# =========================
# LOAD MAPPINGS
# =========================
with open(label_mapping_path, "r") as f:
    index_to_label = json.load(f)["index_to_label"]

with open(coupler_mapping_path, "r") as f:
    coupler_mapping = json.load(f)

def safe_name(name, max_len=50):
    return "".join(c if c.isalnum() else "_" for c in name)[:max_len] or "unk"

def is_type8bar(name: str) -> bool:
    return name.startswith("Type")

def coupler_index_for(name: str) -> int:
    """
    dataset_33 behavior:
      - Type* (8-bar): fixed index 10
      - otherwise: BSIdict.json "c" one-hot index
    """
    try:
        if is_type8bar(name):
            return 10
        return coupler_mapping[name]["c"].index(1)
    except Exception:
        return -1

# =========================
# BINNING (dataset_33)
# =========================
class CoordinateBinner:
    def __init__(self):
        edges = np.linspace(-1.0, 1.0, NUM_BINS)
        self.centers = (edges[:-1] + edges[1:]) / 2  # length 200

    def decode(self, tokens: np.ndarray):
        tokens = tokens.astype(int)
        idx = tokens - BIN_OFFSET
        if idx.size == 0 or np.any(idx < 0) or np.any(idx >= len(self.centers)):
            return None
        return self.centers[idx]

binner = CoordinateBinner()

# =========================
# 8-BAR helper
# =========================
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

# =========================
# SAFE SIMULATION (dataset_33 rules)
# =========================
def simulate_safe(points: np.ndarray, mech_name: str):
    """
    points: (N,2) float array (rounded)
    mech_name:
      - Type*: send to /simulation-8bar using B -> T and solSteps
      - else: send to /simulation with {type, params}; if "P" in name -> params[:5]
    Returns:
      P: (T, num_joints, 2) float array or None
    """
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
        return None

    pts = np.round(points.astype(np.float64), 3)

    try:
        if is_type8bar(mech_name):
            if mech_name not in coupler_mapping or "B" not in coupler_mapping[mech_name]:
                return None
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

        resp = requests.post(url, headers=HEADERS, data=json.dumps([payload]), timeout=30).json()
        if not isinstance(resp, list) or len(resp) == 0 or "poses" not in resp[0]:
            return None

        P = np.array(resp[0]["poses"], dtype=np.float64)

        # Guard the exact crash you saw + other common bad responses
        if P.ndim != 3:
            return None
        if P.shape[0] < minsteps:
            return None
        if P.shape[2] != 2:
            return None

        return P
    except Exception:
        return None

# =========================
# CAUSAL MASK
# =========================
def build_causal_mask(n: int, device: torch.device):
    m = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
    return m.unsqueeze(0).unsqueeze(0)

# =========================
# AUTOREGRESSIVE DECODER (temperature + top-k)
# =========================
def predict_autoregressive_latent(
    model,
    latent: torch.Tensor,
    mech_idx: int,
    max_len: int,
    temperature: float,
    top_k,
):
    """
    Returns token sequence (numpy int array).
    Behavior matches your DTW scripts:
      - temperature==0: greedy argmax
      - temperature>0: sample (optionally top-k truncated)
    """
    latent = latent.unsqueeze(0).to(device)               # (1, latent_dim)
    mech_labels = torch.tensor([mech_idx], device=device) # (1,)
    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)  # (1,1)

    with torch.no_grad():
        for step in range(max_len):
            mask = build_causal_mask(decoder_input.size(1), device)
            logits = model(decoder_input, mask, latent, mech_labels)  # (1, seq, vocab)
            logits = logits[:, -1, :]  # (1, vocab)

            if float(temperature) > 0.0:
                logits = logits / float(temperature)

            probs = F.softmax(logits, dim=-1)

            if DEBUG_PROBS:
                entropy = float(-(probs * (probs + 1e-12).log()).sum().item())
                top10_probs, top10_idx = torch.topk(probs, k=10, dim=-1)
                print(f"\n=== Step {step} ===")
                print(f"Entropy: {entropy:.4f}")
                print("Top-10 token IDs:", top10_idx[0].tolist())
                print("Top-10 probs   :", [float(p) for p in top10_probs[0]])

            # TOP-K sampling
            if top_k is not None and int(top_k) > 0:
                k = min(int(top_k), probs.size(-1))
                topk_probs, topk_idx = torch.topk(probs, k=k, dim=-1)
                topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-12)
                sampled = torch.multinomial(topk_probs, 1)
                next_token = topk_idx.gather(-1, sampled)
            else:
                # full distribution
                if float(temperature) == 0.0:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(probs, 1)

            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            if int(next_token.item()) == EOS_TOKEN:
                break

    return decoder_input.squeeze(0).detach().cpu().numpy().astype(int)

# =========================
# ARC-LENGTH RESAMPLING + CENTERING + SCALE
# =========================
def resample_by_arclength(curve, num_points):
    """
    Resample a 2D curve to a fixed number of points uniformly spaced by arc length.
    """
    if num_points is None:
        return curve

    curve = np.asarray(curve, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[0] < 2 or curve.shape[1] != 2:
        return curve

    diffs = np.diff(curve, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])

    if arc[-1] < 1e-8:
        return curve

    arc = arc / arc[-1]
    target = np.linspace(0.0, 1.0, num_points)

    out = np.zeros((num_points, 2), dtype=np.float64)
    for d in range(2):
        out[:, d] = np.interp(target, arc, curve[:, d])
    return out

def center_curve(curve):
    curve = np.asarray(curve, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 2:
        return curve
    return curve - curve.mean(axis=0)

def scale_to_match_variance(src_centered, ref_centered):
    """
    Both inputs MUST be centered already.
    Scales src so its total variance matches ref.
    """
    s = np.sqrt(np.var(ref_centered)) / (np.sqrt(np.var(src_centered)) + 1e-8)
    return src_centered * s

# =========================
# CHAMFER (O(2)-invariant)
# =========================
def chamfer_distance(a, b):
    ta = cKDTree(a)
    tb = cKDTree(b)
    d_ab, _ = tb.query(a, k=1)
    d_ba, _ = ta.query(b, k=1)
    return float(np.mean(d_ab**2) + np.mean(d_ba**2))

def chamfer_O2_invariant(gt_curve_raw, pred_curve_raw, num_angles=NUM_ROT_ANGLES):
    """
    Full O(2)-invariant Chamfer WITH arc-length resampling + correct centering:

      GT:
        center -> resample -> center

      Pred:
        resample -> center -> scale_to_GT

      Then:
        try mirrors (id, reflect_y) + rotations
        re-center AFTER every transform
    """
    # ---- GT: center -> resample -> center ----
    gt_c = center_curve(gt_curve_raw)
    gt_rs = resample_by_arclength(gt_c, RESAMPLE_N)
    gt_rs = center_curve(gt_rs)

    # ---- Pred: resample -> center -> scale ----
    pred_rs = resample_by_arclength(pred_curve_raw, RESAMPLE_N)
    pred_rs = center_curve(pred_rs)
    pred_rs = scale_to_match_variance(pred_rs, gt_rs)

    mirror_fns = [
        ("id", lambda P: P),
        ("my", lambda P: np.column_stack([-P[:, 0], P[:, 1]])),  # reflect across y-axis
    ]
    angles = np.linspace(0, 2 * np.pi, int(num_angles), endpoint=False)

    best = float("inf")
    best_curve = None
    best_info = None

    for mname, mfn in mirror_fns:
        Pm = mfn(pred_rs)
        for theta in angles:
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float64)
            Pr = Pm @ R.T

            # CRITICAL: re-center after transform (removes drift)
            Pr = center_curve(Pr)

            d = chamfer_distance(gt_rs, Pr)
            if d < best:
                best = d
                best_curve = Pr
                best_info = (mname, float(theta))

    return float(best), best_curve, best_info, gt_rs

# =========================
# LOAD MODEL + DATASET
# =========================
dataset = BarLinkageDataset(data_dir=data_dir)

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

MAX_LEN = int(model_config["tgt_seq_len"])

# =========================
# PROCESS ONE INDEX
# =========================
def process_index(idx: int, log_f):
    """
    Returns:
      dict row with cd (or None) + skip reason, etc.
    """
    if idx < 0 or idx >= len(dataset):
        return {"idx": int(idx), "ok": False, "reason": "idx_out_of_range"}

    sample = dataset[idx]
    latent = sample["vae_mu"].to(device)
    mech_idx = int(sample["encoded_labels"])
    mech_name = index_to_label[str(mech_idx)]
    ci = coupler_index_for(mech_name)

    # ---- GT tokens -> points ----
    gt_tokens = sample["labels_discrete"].cpu().numpy().astype(int)
    gt_coord_tokens = gt_tokens[gt_tokens >= BIN_OFFSET]
    if gt_coord_tokens.size < 4:
        return {"idx": int(idx), "ok": False, "mech": mech_name, "reason": "too_few_gt_tokens"}

    gt_vals = binner.decode(gt_coord_tokens)
    if gt_vals is None:
        return {"idx": int(idx), "ok": False, "mech": mech_name, "reason": "gt_decode_failed"}

    gt_vals = gt_vals[: (len(gt_vals) // 2) * 2]
    if gt_vals.size < 4:
        return {"idx": int(idx), "ok": False, "mech": mech_name, "reason": "gt_decode_too_short"}

    gt_pts = gt_vals.reshape(-1, 2)

    P_gt = simulate_safe(gt_pts, mech_name)
    if P_gt is None:
        return {"idx": int(idx), "ok": False, "mech": mech_name, "reason": "gt_sim_failed"}

    if ci < 0 or ci >= P_gt.shape[1]:
        return {"idx": int(idx), "ok": False, "mech": mech_name, "reason": f"bad_coupler_idx_gt:{ci}"}

    gt_curve_raw = P_gt[:, ci, :]  # (T,2)

    # ---- Predict tokens (temp/top-k) -> points ----
    tk = None if (TOP_K is None or int(TOP_K) <= 0) else int(TOP_K)
    pred_tokens = predict_autoregressive_latent(
        model=model,
        latent=latent,
        mech_idx=mech_idx,
        max_len=MAX_LEN,
        temperature=float(TEMPERATURE),
        top_k=tk,
    )

    pred_coord_tokens = pred_tokens[pred_tokens >= BIN_OFFSET]
    if pred_coord_tokens.size < 4:
        return {"idx": int(idx), "ok": False, "mech": mech_name, "reason": "too_few_pred_tokens"}

    pred_vals = binner.decode(pred_coord_tokens)
    if pred_vals is None:
        return {"idx": int(idx), "ok": False, "mech": mech_name, "reason": "pred_decode_failed"}

    pred_vals = pred_vals[: (len(pred_vals) // 2) * 2]
    if pred_vals.size < 4:
        return {"idx": int(idx), "ok": False, "mech": mech_name, "reason": "pred_decode_too_short"}

    pred_pts = pred_vals.reshape(-1, 2)

    P_pred = simulate_safe(pred_pts, mech_name)
    if P_pred is None:
        return {"idx": int(idx), "ok": False, "mech": mech_name, "reason": "pred_sim_failed"}

    if ci < 0 or ci >= P_pred.shape[1]:
        return {"idx": int(idx), "ok": False, "mech": mech_name, "reason": f"bad_coupler_idx_pred:{ci}"}

    pred_curve_raw = P_pred[:, ci, :]

    # ---- Rigorous CD ----
    cd_val, best_curve, (mname, theta), gt_rs = chamfer_O2_invariant(
        gt_curve_raw, pred_curve_raw, num_angles=NUM_ROT_ANGLES
    )

    # ---- Save plot ----
    if SAVE_PLOTS:
        sample_dir = os.path.join(OUT_DIR, f"{safe_name(mech_name)}")
        os.makedirs(sample_dir, exist_ok=True)
        plt.figure(figsize=(6, 6))
        plt.plot(gt_rs[:, 0], gt_rs[:, 1], "k", lw=2, label="GT (center+resample)")
        plt.plot(best_curve[:, 0], best_curve[:, 1], "r--", lw=2,
                 label=f"Pred (best)\nCD={cd_val:.6f} | {mname}, θ={theta*180/np.pi:.1f}°")
        plt.axis("equal")
        plt.legend()
        fname = f"idx_{idx:06d}_cd_{cd_val:.6f}.png"
        plt.savefig(os.path.join(sample_dir, fname), dpi=200)
        plt.close()

    row = {
        "idx": int(idx),
        "ok": True,
        "mech": mech_name,
        "mech_idx": int(mech_idx),
        "coupler_idx": int(ci),
        "temperature": float(TEMPERATURE),
        "top_k": tk,
        "cd": float(cd_val),
        "best_mirror": mname,
        "best_theta_deg": float(theta * 180.0 / np.pi),
    }
    log_f.write(json.dumps(row) + "\n")
    log_f.flush()
    return row

# =========================
# SUMMARY
# =========================
def summarize(values, thresh):
    v = np.array(values, dtype=float)
    return {
        "count": int(len(v)),
        "mean": float(v.mean()) if len(v) else None,
        "median": float(np.median(v)) if len(v) else None,
        "std": float(v.std()) if len(v) else None,
        "min": float(v.min()) if len(v) else None,
        "max": float(v.max()) if len(v) else None,
        "success_rate_pct": float((v < thresh).mean() * 100.0) if len(v) else None,
    }

# =========================
# MAIN
# =========================
def main():
    # reproducibility for sampling
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cds = []
    oks = 0
    total = 0
    skipped = {}

    with open(JSONL_LOG_PATH, "w") as log_f:
        for idx in tqdm(FIXED_SAMPLE_IDXS, desc="Fixed indices"):
            total += 1
            row = process_index(int(idx), log_f)
            if row.get("ok"):
                oks += 1
                cds.append(float(row["cd"]))
            else:
                r = row.get("reason", "unknown")
                skipped[r] = skipped.get(r, 0) + 1

    print("\n================ CD RESULTS ================")
    print(f"Temperature : {TEMPERATURE}")
    print(f"Top-K       : {None if (TOP_K is None or int(TOP_K) <= 0) else int(TOP_K)}")
    print(f"Total idxs  : {total}")
    print(f"Valid       : {oks}")
    if skipped:
        print("Skipped breakdown:")
        for k, v in sorted(skipped.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  - {k}: {v}")

    if len(cds) == 0:
        print("\nNo valid CD values (all skipped).")
        print("===========================================")
        return

    stats = summarize(cds, CD_SUCCESS_THRESH)
    print("-------------------------------------------")
    print(f"Count       : {stats['count']}")
    print(f"Mean        : {stats['mean']:.6f}")
    print(f"Median      : {stats['median']:.6f}")
    print(f"Std         : {stats['std']:.6f}")
    print(f"Min         : {stats['min']:.6f}")
    print(f"Max         : {stats['max']:.6f}")
    print(f"Success<{CD_SUCCESS_THRESH}: {stats['success_rate_pct']:.1f}%")
    print("-------------------------------------------")
    print(f"JSONL log   : {JSONL_LOG_PATH}")
    print(f"Plots dir   : {OUT_DIR}  (SAVE_PLOTS={SAVE_PLOTS})")
    print("===========================================")

if __name__ == "__main__":
    main()
