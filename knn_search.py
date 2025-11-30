import os
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import shutil

# =========================================================
# CONFIG
# =========================================================
NUM_NEIGHBORS = 10

latent_path = "/home/anurizada/Documents/processed_dataset_17/vae_mu.npy"
labels_cont_path = "/home/anurizada/Documents/processed_dataset_17/labels_continuous.npy"
encoded_labels_path = "/home/anurizada/Documents/processed_dataset_17/encoded_labels.npy"  # <--- NEW
label_mapping_path = "/home/anurizada/Documents/processed_dataset_17/label_mapping.json"
coupler_mapping_path = "/home/anurizada/Documents/transformer/BSIdict.json"

index = 0 # local server index 
API_ENDPOINT = f"http://localhost:4000/simulation"
HEADERS = {"Content-Type": "application/json"}
speedscale = 1
steps = 360
minsteps = int(steps*20/360)

# =========================================================
# Load mapping files
# =========================================================
with open(label_mapping_path, "r") as f:
    label_mapping = json.load(f)
index_to_label = label_mapping["index_to_label"]

with open(coupler_mapping_path, "r") as f:
    coupler_mapping = json.load(f)


def coupler_index_for(mech_type: str) -> int:
    """Return coupler curve index from BSIdict.json."""
    if mech_type in coupler_mapping and "c" in coupler_mapping[mech_type]:
        cvec = coupler_mapping[mech_type]["c"]
        if isinstance(cvec, list) and 1 in cvec:
            return cvec.index(1)
    return -1


# =========================================================
# Clean + reshape continuous labels → (M,2)
# =========================================================
def clean_and_reshape_label(row: np.ndarray) -> np.ndarray:
    mask = (row != 2.0) & (row != -1.0)
    filtered = row[mask]
    if filtered.size % 2 == 1:
        filtered = filtered[:-1]
    if filtered.size == 0:
        return np.zeros((0, 2))
    return filtered.reshape(-1, 2)


# =========================================================
# Run simulator
# =========================================================
def simulate_curve(joint_coords: np.ndarray, mech_name: str):
    ex_pred = {
        "params": joint_coords.tolist(),
        "type": mech_name,
        "speedScale": speedscale,
        "steps": steps,
        "relativeTolerance": 0.1,
    }

    try:
        temp_resp = requests.post(API_ENDPOINT, headers=HEADERS, data=json.dumps([ex_pred])).json()
        if not isinstance(temp_resp, list) or "poses" not in temp_resp[0]:
            return None
        Pp = np.array(temp_resp[0]["poses"])
        return Pp
    except:
        return None


# =========================================================
# MAIN
# =========================================================
def main():

    # Load npy files
    latents = np.load(latent_path)               # (N,50)
    labels_cont = np.load(labels_cont_path)      # (N,D)
    encoded_labels = np.load(encoded_labels_path) # (N,) <-- important
    N = latents.shape[0]

    print(f"Loaded latents: {latents.shape}")
    print(f"Loaded continuous labels: {labels_cont.shape}")
    print(f"Loaded encoded labels: {encoded_labels.shape}")

    # -----------------------------------------
    # Pick random latent
    # -----------------------------------------
    global_query_index = np.random.randint(0, N)
    query_latent = latents[global_query_index : global_query_index + 1]

    print(f"\nSelected query latent index = {global_query_index}")

    # -----------------------------------------
    # Build KNN on full dataset
    # -----------------------------------------
    knn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS + 1, metric="euclidean")
    knn.fit(latents)

    distances, indices = knn.kneighbors(query_latent)
    indices = indices[0]
    distances = distances[0]

    neighbor_global_indices = indices[1:NUM_NEIGHBORS + 1]
    neighbor_distances = distances[1:NUM_NEIGHBORS + 1]

    print("Nearest neighbors:", neighbor_global_indices)

    # -----------------------------------------
    # Output folder
    # -----------------------------------------
    out_dir = "latent_knn_results"

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)   # delete entire folder
    os.makedirs(out_dir, exist_ok=True)  # recreate empty folder

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------
    # Process query (GT)
    # -----------------------------------------
    query_label = labels_cont[global_query_index]
    query_mech_idx = int(encoded_labels[global_query_index])
    query_mech_name = index_to_label.get(str(query_mech_idx), "UNKNOWN")

    query_coords = clean_and_reshape_label(query_label)

    Pq = simulate_curve(query_coords, query_mech_name)
    if Pq is None:
        print("❌ Query simulation failed.")
        return

    coup_idx_q = coupler_index_for(query_mech_name)
    if coup_idx_q < 0:
        print(f"No coupler index for mech={query_mech_name}")
        return

    original_x = Pq[:, coup_idx_q, 0]
    original_y = Pq[:, coup_idx_q, 1]

    # -----------------------------------------
    # Process neighbors
    # -----------------------------------------
    for k, (gidx, dist) in enumerate(zip(neighbor_global_indices, neighbor_distances)):

        neigh_label_row = labels_cont[gidx]
        neigh_mech_idx = int(encoded_labels[gidx])
        neigh_mech_name = index_to_label.get(str(neigh_mech_idx), "UNKNOWN")

        neigh_coords = clean_and_reshape_label(neigh_label_row)

        Pn = simulate_curve(neigh_coords, neigh_mech_name)
        if Pn is None:
            print(f"Neighbor {k} simulation failed.")
            continue

        coup_idx_n = coupler_index_for(neigh_mech_name)
        if coup_idx_n < 0:
            continue

        nx = Pn[:, coup_idx_n, 0]
        ny = Pn[:, coup_idx_n, 1]

        # -----------------------------------------
        # PLOT
        # -----------------------------------------
        plt.figure(figsize=(6, 6))
        plt.plot(original_x, original_y, "r-", label=f"Original ({query_mech_name})")
        plt.plot(nx, ny, "g--", label=f"Neighbor {k} ({neigh_mech_name})")

        plt.scatter(original_x[0], original_y[0], c="red", s=40)
        plt.scatter(nx[0], ny[0], c="green", s=40)

        plt.title(f"KNN {k} | dist={dist:.3f}")
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(out_dir, f"query_{global_query_index}_neighbor_{k}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved {save_path}")

    print("\n✅ DONE.")


if __name__ == "__main__":
    main()
