#!/usr/bin/env python3
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from llama_latent_model import LatentLLaMA_SingleToken
from dataset import BarLinkageDataset


# =========================================================
# CONFIG
# =========================================================
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
NUM_SPECIAL_TOKENS = 3
NUM_BINS = 201
BIN_OFFSET = NUM_SPECIAL_TOKENS
LATENT_DIM = 50

CHECKPOINT_PATH = "./weights/LATENT_LLAMA_d512_h8_n6_bs512_lr0.001_best.pth"   # <<< CHANGE THIS


# =========================================================
# CoordinateBinner (same as training)
# =========================================================
class CoordinateBinner:
    def __init__(self, kappa=1.0, num_bins=NUM_BINS):
        self.kappa = kappa
        self.num_bins = num_bins
        self.bin_edges = np.linspace(-kappa, kappa, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def bin_to_value_torch(self, bin_index_tensor):
        bin_index_tensor = torch.clamp(bin_index_tensor, 0, self.num_bins - 1)
        centers = torch.tensor(
            self.bin_centers,
            dtype=torch.float32,
            device=bin_index_tensor.device,
        )
        return centers[bin_index_tensor]


# =========================================================
# Build causal mask
# =========================================================
def build_causal_mask(T, device):
    mask = torch.tril(torch.ones((T, T), device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)   # (1,1,T,T)


# =========================================================
# FULL-SEQUENCE INFERENCE
# =========================================================
def predict_full_sequence(model, latent, mech_idx, decoder_input_gt, device):
    model.eval()

    latent = latent.unsqueeze(0).to(device)                 # (1, latent_dim)
    mech_labels = torch.tensor([mech_idx], device=device)   # (1,)

    decoder_input = decoder_input_gt.unsqueeze(0).to(device)  # (1, T)
    T = decoder_input.size(1)

    causal_mask = build_causal_mask(T, device)

    with torch.no_grad():
        logits = model(decoder_input, causal_mask, latent, mech_labels)
        preds = logits.argmax(dim=-1)   # (1, T)

    return preds.squeeze(0)              # (T,)


# =========================================================
# MAIN
# =========================================================
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------ Load Dataset ------------------
    dataset = BarLinkageDataset("/home/anurizada/Documents/processed_dataset_17")

    # ------------------ Load Model ------------------
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model_config = checkpoint["model_config"]

    model = LatentLLaMA_SingleToken(
        latent_dim=model_config["latent_dim"],
        tgt_seq_len=model_config["tgt_seq_len"],
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        h=model_config["h"],
        N=model_config["N"],
        num_labels=model_config["num_labels"],
        dropout=model_config["dropout"],
        pad_token_id=model_config["pad_token_id"],
        debug=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Model loaded.")

    # ------------------ Binner ------------------
    binner = CoordinateBinner(kappa=1.0, num_bins=NUM_BINS)

    # ------------------ Run inference ------------------
    NUM_SAMPLES = 20

    for idx in tqdm(range(NUM_SAMPLES), desc="Evaluating"):

        batch = dataset[idx]

        latent = batch["vae_mu"].to(device).squeeze(-1)
        mech_idx = int(batch["encoded_labels"])
        decoder_input_gt = batch["decoder_input_discrete"]
        gt_tokens = batch["labels_discrete"]

        # ---- FULL-SEQUENCE PREDICTION ----
        pred_tokens = predict_full_sequence(
            model=model,
            latent=latent,
            mech_idx=mech_idx,
            decoder_input_gt=decoder_input_gt,
            device=device,
        )

        pred_tokens_np = pred_tokens.cpu().numpy()
        gt_tokens_np = gt_tokens.numpy()

        print("\n==================================================")
        print(f"SAMPLE {idx}")
        print("GT tokens:   ", gt_tokens_np.tolist())
        print("PRED tokens: ", pred_tokens_np.tolist())

        # ---- Extract coordinate tokens ----
        gt_coord_bins = [t for t in gt_tokens_np if t >= BIN_OFFSET]
        pred_coord_bins = [t for t in pred_tokens_np if t >= BIN_OFFSET]

        if len(gt_coord_bins) < 4:
            print("GT has insufficient coordinate tokens.")
            continue

        if len(pred_coord_bins) < 4:
            print("Pred has insufficient coordinate tokens.")
            continue

        gt_bins_tensor = torch.tensor(gt_coord_bins, device=device) - BIN_OFFSET
        pred_bins_tensor = torch.tensor(pred_coord_bins, device=device) - BIN_OFFSET

        gt_xy = binner.bin_to_value_torch(gt_bins_tensor).cpu().numpy()
        pred_xy = binner.bin_to_value_torch(pred_bins_tensor).cpu().numpy()

        # reshape
        gt_xy = gt_xy[: (len(gt_xy) // 2) * 2].reshape(-1, 2)
        pred_xy = pred_xy[: (len(pred_xy) // 2) * 2].reshape(-1, 2)

        print("GT XY:")
        print(gt_xy)
        print("PRED XY:")
        print(pred_xy)

    print("\nDONE.")


if __name__ == "__main__":
    main()
