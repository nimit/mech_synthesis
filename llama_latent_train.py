import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split, DistributedSampler, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb

from llama_latent_model import LatentLLaMA_SingleToken
from dataset import BarLinkageDataset


# =========================================================
# CoordinateBinner (for continuous MSE monitoring)
# =========================================================
class CoordinateBinner:
    def __init__(self, kappa=1.0, num_bins=200):
        self.kappa = kappa
        self.num_bins = num_bins
        self.bin_edges = np.linspace(-kappa, kappa, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def bin_to_value_torch(self, bin_index_tensor):
        bin_index_tensor = torch.clamp(bin_index_tensor, 0, self.num_bins - 1)
        bin_centers_tensor = torch.tensor(
            self.bin_centers, device=bin_index_tensor.device, dtype=torch.float32
        )
        return bin_centers_tensor[bin_index_tensor]


# =========================================================
# BinDistanceLoss (CE on all tokens + extra bin loss on bin tokens)
# =========================================================
class BinDistanceLoss(nn.Module):
    def __init__(
        self,
        vocab_size,
        bin_start_id,
        bin_end_id,
        ignore_index=2,
        temperature=2.0,
        bin_weight=1.0,
        debug=False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.bin_start_id = bin_start_id
        self.bin_end_id = bin_end_id
        # inclusive range: e.g. 3..203 → 201 bins
        self.num_bins = (bin_end_id - bin_start_id + 1)
        print(
            f"BinDistanceLoss: bin IDs {bin_start_id} to {bin_end_id} "
            f"({self.num_bins} bins)"
        )
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.bin_weight = bin_weight
        self.debug = debug
        self.mse_weight = 1

        self.ce_loss = nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=ignore_index
        )

    # Gaussian soft targets over [0 .. num_bins-1]
    def _create_soft_targets(self, true_bins):
        """
        true_bins: (N,) integer in [0, num_bins-1]
        returns: (N, num_bins)
        """
        N = true_bins.size(0)
        C = self.num_bins
        device = true_bins.device

        indices = torch.arange(C, device=device).unsqueeze(0)  # (1, C)
        true_exp = true_bins.unsqueeze(1).float()              # (N, 1)
        dist = torch.abs(indices - true_exp)                   # (N, C)

        soft = torch.exp(-dist.pow(2) / (2 * (self.temperature ** 2)))
        soft = soft / soft.sum(dim=1, keepdim=True)
        return soft

    def forward(self, logits_flat, targets_flat):
        """
        logits_flat: (N, V)
        targets_flat: (N,)
        """
        device = logits_flat.device
        N, V = logits_flat.shape

        torch.set_printoptions(sci_mode=False, precision=6)

        # ------------------------------------------------------------
        # Masks
        # ------------------------------------------------------------
        valid_mask = (targets_flat != self.ignore_index)

        # BIN tokens = within [bin_start_id, bin_end_id]
        is_bin = (
            (targets_flat >= self.bin_start_id)
            & (targets_flat <= self.bin_end_id)
            & valid_mask
        )

        # ============================================================
        # 1) CE loss on ALL valid tokens (bin + non-bin)
        # ============================================================
        ce_all = self.ce_loss(logits_flat, targets_flat)  # (N,)
        loss = ce_all.clone()  # base loss = CE everywhere

        # ============================================================
        # 2) Gaussian + MSE on BIN tokens only
        # ============================================================
        if is_bin.any():
            idx_bin = torch.where(is_bin)[0]

            # logits restricted to bin range [bin_start_id .. bin_end_id]
            sel_logits = logits_flat[idx_bin][:,
                        self.bin_start_id : self.bin_end_id + 1]  # (Nb, num_bins)

            # Map absolute IDs → relative bin IDs: 0..num_bins-1
            sel_targets = targets_flat[idx_bin] - self.bin_start_id  # (Nb,)

            # ---------- 2a) Gaussian soft loss ----------
            soft_targets = self._create_soft_targets(sel_targets)    # (Nb, num_bins)
            log_probs = F.log_softmax(sel_logits, dim=1)             # (Nb, num_bins)

            gaussian_vec = -(soft_targets * log_probs).sum(dim=1)    # (Nb,)
            gaussian_vec = gaussian_vec * float(self.bin_weight)

            # Add Gaussian loss on top of CE
            loss[idx_bin] = loss[idx_bin] + gaussian_vec

        # ============================================================
        # 3) Final loss reduction
        # ============================================================
        if valid_mask.any():
            final_loss = loss[valid_mask].mean()
        else:
            final_loss = loss.mean()

        if self.debug:
            print("\n====== FINAL REDUCED LOSS ======")
            print(format(final_loss.item(), ".6f"))

        return final_loss



# =========================================================
# Debug Function (first batch dump)
# =========================================================
def debug_batch(
    logits,           # (B, T, V)
    targets,          # (B, T)
    BIN_START,
    BIN_END,
    PAD_TOKEN,
    loss_fn,
    max_print=20
):
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    B, T, V = logits.shape
    logits_flat = logits.reshape(B*T, V)
    targets_flat = targets.reshape(B*T)

    print("\n================ DEBUG BATCH ================\n")
    print(f"Logits shape:        {logits.shape}")
    print(f"Targets shape:       {targets.shape}")
    print(f"Flattened N = {B*T}")

    # Masks
    is_pad = targets_flat == PAD_TOKEN
    is_bin = (
        (targets_flat >= BIN_START)
        & (targets_flat <= BIN_END)
        & (~is_pad)
    )
    is_not_bin = (~is_bin) & (~is_pad)

    print(f"PAD tokens:          {is_pad.sum().item()}")
    print(f"BIN tokens:          {is_bin.sum().item()}")
    print(f"NON-BIN tokens:      {is_not_bin.sum().item()}")

    print("\n--- First tokens ---")
    for i in range(min(max_print, len(targets_flat))):
        tok = targets_flat[i].item()
        if is_pad[i]:
            tag = "PAD"
        elif is_bin[i]:
            tag = "BIN"
        else:
            tag = "NON-BIN"
        print(f"[{i:03d}] token={tok:4d} | {tag}")

    # -------- Detailed BIN token info --------
    bin_idx = torch.where(is_bin)[0]
    if len(bin_idx) > 0:
        print("\n--- BIN TOKEN DETAILS ---")
        sel_logits = logits_flat[bin_idx][:, BIN_START:BIN_END+1]
        sel_targets = targets_flat[bin_idx] - BIN_START

        soft = loss_fn._create_soft_targets(sel_targets)
        log_probs = torch.log_softmax(sel_logits, dim=1)
        bin_loss = -(soft * log_probs).sum(dim=1)

        for j in range(min(5, len(bin_idx))):
            idx = bin_idx[j].item()
            print(f"\n[Flat idx {idx}]")
            print(f"  raw target id     = {targets_flat[idx].item()}")
            print(f"  rel bin index     = {sel_targets[j].item()}")
            print(f"  soft[:5]          = {soft[j].cpu().numpy()}")
            print(f"  loss              = {bin_loss[j].item():.6f}")

    # -------- Non-bin token CE details --------
    non_idx = torch.where(is_not_bin)[0]
    if len(non_idx) > 0:
        print("\n--- NON-BIN TOKEN CE DETAILS ---")
        ce_all = loss_fn.ce_loss(logits_flat, targets_flat)
        for j in range(min(5, len(non_idx))):
            idx = non_idx[j].item()
            print(
                f"[Flat idx {idx}] token={targets_flat[idx].item()} "
                f"loss={ce_all[idx].item():.6f}"
            )

    print("\n============= END DEBUG BATCH =============\n")


# =========================================================
# DDP UTILS
# =========================================================
def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


# =========================================================
# Checkpoint (with config + best loss)
# =========================================================
def save_best_checkpoint(model, optimizer, epoch, best_loss, batch_size, lr, model_config,
                         save_dir="./weights"):
    os.makedirs(save_dir, exist_ok=True)
    d_model = model_config["d_model"]
    n_heads = model_config["h"]
    n_layers = model_config["N"]

    path = os.path.join(
        save_dir,
        f"LATENT_LLAMA_d{d_model}_h{n_heads}_n{n_layers}_bs{batch_size}_lr{lr}_best.pth"
    )

    torch.save(
        {
            "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "model_config": model_config,
        },
        path,
    )

    print(f"[Rank {get_rank()}] Saved best model at {path} (Val Loss: {best_loss:.6f})")


# =========================================================
# Metric helper (acc + discrete & continuous MSE + EOS/BIN stats)
# =========================================================
def compute_metrics(logits, targets, BIN_START, NUM_BINS, PAD_TOKEN, binner, EOS_TOKEN=1):
    """
    logits:  (B, T, V)
    targets: (B, T)
    """
    preds = logits.argmax(dim=-1)         # (B, T)
    valid_mask = targets != PAD_TOKEN

    # ---------------- Token accuracy (all non-PAD) ----------------
    correct = ((preds == targets) & valid_mask).sum().float()
    token_acc = correct / (valid_mask.sum().float() + 1e-12)

    # ---------------- Coordinate bins (BIN_START .. BIN_START+NUM_BINS-1) ----------------
    pred_bin_rel = preds - BIN_START
    target_bin_rel = targets - BIN_START

    coord_mask = (
        (target_bin_rel >= 0) &
        (target_bin_rel < NUM_BINS) &
        (targets != PAD_TOKEN)
    )

    if coord_mask.sum() == 0:
        bin_mse = 0.0
        coord_mse = 0.0
    else:
        # Discrete MSE
        bin_diff = (pred_bin_rel[coord_mask] - target_bin_rel[coord_mask]).float()
        bin_mse = (bin_diff ** 2).mean().item()

        # Continuous MSE
        pred_cont = binner.bin_to_value_torch(pred_bin_rel.clamp(0, NUM_BINS - 1))
        target_cont = binner.bin_to_value_torch(target_bin_rel.clamp(0, NUM_BINS - 1))
        coord_mse = F.mse_loss(pred_cont[coord_mask], target_cont[coord_mask]).item()

    # ---------------- EOS / BIN per-class accuracies & frequencies ----------------
    eos_mask = (targets == EOS_TOKEN) & valid_mask
    bin_mask = coord_mask  # targets inside bin range & not PAD

    # EOS accuracy
    if eos_mask.any():
        eos_correct = ((preds == targets) & eos_mask).sum().float()
        eos_acc = (eos_correct / (eos_mask.sum().float() + 1e-12)).item()
    else:
        eos_acc = 0.0

    # BIN accuracy
    if bin_mask.any():
        bin_correct = ((preds == targets) & bin_mask).sum().float()
        bin_acc = (bin_correct / (bin_mask.sum().float() + 1e-12)).item()
    else:
        bin_acc = 0.0

    # Fractions in targets
    valid_targets = targets[valid_mask]
    if valid_targets.numel() > 0:
        frac_eos_target = (valid_targets == EOS_TOKEN).float().mean().item()
        frac_bin_target = (
            ((valid_targets >= BIN_START) & (valid_targets < BIN_START + NUM_BINS))
            .float()
            .mean()
            .item()
        )
    else:
        frac_eos_target = 0.0
        frac_bin_target = 0.0

    # Fractions in predictions
    valid_preds = preds[valid_mask]
    if valid_preds.numel() > 0:
        frac_eos_pred = (valid_preds == EOS_TOKEN).float().mean().item()
        frac_bin_pred = (
            ((valid_preds >= BIN_START) & (valid_preds < BIN_START + NUM_BINS))
            .float()
            .mean()
            .item()
        )
    else:
        frac_eos_pred = 0.0
        frac_bin_pred = 0.0

    return {
        "token_acc": token_acc.item(),
        "bin_mse": bin_mse,
        "coord_mse": coord_mse,
        "eos_acc": eos_acc,
        "bin_acc": bin_acc,
        "frac_eos_target": frac_eos_target,
        "frac_eos_pred": frac_eos_pred,
        "frac_bin_target": frac_bin_target,
        "frac_bin_pred": frac_bin_pred,
    }


# =========================================================
# Training Loop
# =========================================================
def train(
    checkpoint_path=None,
    use_strict_resume=False,  # True = exact resume; False = fine-tune from weights only
):
    local_rank = setup_ddp()
    rank = get_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {rank}] Using {device}")
    torch.set_float32_matmul_precision("medium")

    # ---------------- CONFIG ----------------
    batch_size = 512
    num_epochs = 1000
    lr = 5e-4
    seq_len = 17

    NUM_BINS = 201
    NUM_SPECIAL = 3
    SOS_TOKEN = 0
    EOS_TOKEN = 1
    PAD_TOKEN = 2
    VOCAB_SIZE = NUM_SPECIAL + NUM_BINS
    LATENT_DIM = 50

    BIN_START = NUM_SPECIAL             # 3
    BIN_END   = BIN_START + NUM_BINS - 1  # 3 + 201 - 1 = 203  (FIXED)

    # ---------------- DATA ----------------
    dataset = BarLinkageDataset("/home/anurizada/Documents/processed_dataset_17")
    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    val_sampler   = DistributedSampler(
        val_dataset,   rank=rank, num_replicas=world_size, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        sampler=val_sampler, num_workers=4, pin_memory=True
    )

    # ---------------- MODEL ----------------
    model_config = {
        "tgt_seq_len": seq_len,
        "d_model": 512,
        "h": 8,
        "N": 6,
        "num_labels": 17,
        "vocab_size": VOCAB_SIZE,
        "latent_dim": LATENT_DIM,
        "dropout": 0.1,
        "pad_token_id": PAD_TOKEN,
        "debug": False,
    }

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
        debug=model_config["debug"],
    ).to(device)

    model = DDP(model, device_ids=[local_rank])

    # ---------------- LOSS ----------------
    loss_fn = BinDistanceLoss(
        vocab_size=VOCAB_SIZE,
        bin_start_id=BIN_START,
        bin_end_id=BIN_END,
        ignore_index=PAD_TOKEN,
        temperature=2.0,
        bin_weight=1.0,
        debug=False,   # set True if you want the detailed prints
    )

    # ---------------- Binner for continuous MSE ----------------
    binner = CoordinateBinner(kappa=1.0, num_bins=NUM_BINS)

    # ---------------- OPTIMIZER ----------------
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0001)

    # (optional) scheduler: warmup + cosine
    warmup_epochs = 0
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                          total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                 milestones=[warmup_epochs])
    else:
        scheduler = None

    # ---------------- WandB ----------------
    if rank == 0:
        wandb.init(
            project="bar-linkage-transformer",
            name=f"LATENT_LLaMA_BinLoss_d{model_config['d_model']}_h{model_config['h']}_n{model_config['N']}_bs{batch_size}_lr{lr}",
            config=model_config,
        )

    best_loss = float("inf")
    start_epoch = 0

    # ---------------- Optional: resume vs fine-tune ----------------
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)

        if use_strict_resume:
            # Exact resume (same loss, same optimizer state)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_loss = ckpt.get("best_loss", float("inf"))

            if rank == 0:
                print(f"[Rank {rank}] 🔁 Strict resume from {checkpoint_path}, epoch {start_epoch}")

    # =========================================================
    # EPOCH LOOP
    # =========================================================
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_bin_mse = 0.0
        epoch_coord_mse = 0.0
        epoch_eos_acc = 0.0
        epoch_bin_acc = 0.0
        epoch_frac_eos_pred = 0.0
        epoch_frac_eos_target = 0.0
        epoch_frac_bin_pred = 0.0
        epoch_frac_bin_target = 0.0

        pbar = tqdm(train_loader, disable=(rank != 0), desc=f"[Rank {rank}] Train {epoch}")

        for batch_i, batch in enumerate(pbar):

            latents = batch["vae_mu"].to(device).squeeze(-1)
            decoder_input = batch["decoder_input_discrete"].to(device)
            decoder_mask  = batch["causal_mask"].to(device).bool()
            mech_labels   = batch["encoded_labels"].to(device)
            targets       = batch["labels_discrete"].to(device)

            optimizer.zero_grad()
            logits = model(decoder_input, decoder_mask, latents, mech_labels)

            B, T, V = logits.shape
            logits_flat = logits.reshape(B*T, V)
            targets_flat = targets.reshape(B*T)

            # 🔍 Debug only on very first batch
            if epoch == 0 and batch_i == 0 and rank == 0:
                debug_batch(
                    logits,
                    targets,
                    BIN_START,
                    BIN_END,
                    PAD_TOKEN,
                    loss_fn,
                    max_print=30
                )

            loss = loss_fn(logits_flat, targets_flat)
            loss.backward()

            # Gradient norm (before optimizer.step)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # ---- Metrics: accuracy + discrete MSE + continuous MSE + EOS/BIN stats ----
            metrics = compute_metrics(
                logits,
                targets,
                BIN_START,
                NUM_BINS,
                PAD_TOKEN,
                binner,
                EOS_TOKEN=EOS_TOKEN,
            )

            epoch_loss += loss.item()
            epoch_acc += metrics["token_acc"]
            epoch_bin_mse += metrics["bin_mse"]
            epoch_coord_mse += metrics["coord_mse"]
            epoch_eos_acc += metrics["eos_acc"]
            epoch_bin_acc += metrics["bin_acc"]
            epoch_frac_eos_pred += metrics["frac_eos_pred"]
            epoch_frac_eos_target += metrics["frac_eos_target"]
            epoch_frac_bin_pred += metrics["frac_bin_pred"]
            epoch_frac_bin_target += metrics["frac_bin_target"]

            if rank == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                wandb.log({
                    "train/total_loss": loss.item(),
                    "train/token_acc": metrics["token_acc"],
                    "train/bin_mse": metrics["bin_mse"],
                    "train/coord_mse": metrics["coord_mse"],
                    "train/eos_acc": metrics["eos_acc"],
                    "train/bin_acc": metrics["bin_acc"],
                    "train/frac_eos_pred": metrics["frac_eos_pred"],
                    "train/frac_eos_target": metrics["frac_eos_target"],
                    "train/frac_bin_pred": metrics["frac_bin_pred"],
                    "train/frac_bin_target": metrics["frac_bin_target"],
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                    "train/lr": current_lr,
                    "train/epoch": epoch,
                })

            pbar.set_postfix({
                "loss": loss.item(),
                "acc": metrics["token_acc"],
                "eos_acc": metrics["eos_acc"],
                "bin_acc": metrics["bin_acc"],
            })

        pbar.close()

        if scheduler is not None:
            scheduler.step()

        # ---------------- VAL ----------------
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_bin_mse = 0.0
        val_coord_mse = 0.0
        val_eos_acc = 0.0
        val_bin_acc = 0.0
        val_frac_eos_pred = 0.0
        val_frac_eos_target = 0.0
        val_frac_bin_pred = 0.0
        val_frac_bin_target = 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, disable=(rank != 0), desc=f"[Rank {rank}] Val {epoch}")
            for batch in pbar:
                latents = batch["vae_mu"].to(device).squeeze(-1)
                decoder_input = batch["decoder_input_discrete"].to(device)
                decoder_mask  = batch["causal_mask"].to(device).bool()
                mech_labels   = batch["encoded_labels"].to(device)
                targets       = batch["labels_discrete"].to(device)

                logits = model(decoder_input, decoder_mask, latents, mech_labels)

                B, T, V = logits.shape
                logits_flat = logits.reshape(B*T, V)
                targets_flat = targets.reshape(B*T)

                loss = loss_fn(logits_flat, targets_flat)
                val_loss += loss.item()

                metrics = compute_metrics(
                    logits,
                    targets,
                    BIN_START,
                    NUM_BINS,
                    PAD_TOKEN,
                    binner,
                    EOS_TOKEN=EOS_TOKEN,
                )
                val_acc += metrics["token_acc"]
                val_bin_mse += metrics["bin_mse"]
                val_coord_mse += metrics["coord_mse"]
                val_eos_acc += metrics["eos_acc"]
                val_bin_acc += metrics["bin_acc"]
                val_frac_eos_pred += metrics["frac_eos_pred"]
                val_frac_eos_target += metrics["frac_eos_target"]
                val_frac_bin_pred += metrics["frac_bin_pred"]
                val_frac_bin_target += metrics["frac_bin_target"]

                pbar.set_postfix({
                    "val_loss": loss.item(),
                    "val_acc": metrics["token_acc"],
                    "val_eos_acc": metrics["eos_acc"],
                    "val_bin_acc": metrics["bin_acc"],
                })

            pbar.close()

        # --------- Aggregate epoch metrics ---------
        n_train = len(train_loader)
        n_val = len(val_loader)

        avg_train_loss = epoch_loss / n_train
        avg_train_acc = epoch_acc / n_train
        avg_train_bin_mse = epoch_bin_mse / n_train
        avg_train_coord_mse = epoch_coord_mse / n_train
        avg_train_eos_acc = epoch_eos_acc / n_train
        avg_train_bin_acc = epoch_bin_acc / n_train
        avg_train_frac_eos_pred = epoch_frac_eos_pred / n_train
        avg_train_frac_eos_target = epoch_frac_eos_target / n_train
        avg_train_frac_bin_pred = epoch_frac_bin_pred / n_train
        avg_train_frac_bin_target = epoch_frac_bin_target / n_train

        avg_val_loss = val_loss / n_val
        avg_val_acc = val_acc / n_val
        avg_val_bin_mse = val_bin_mse / n_val
        avg_val_coord_mse = val_coord_mse / n_val
        avg_val_eos_acc = val_eos_acc / n_val
        avg_val_bin_acc = val_bin_acc / n_val
        avg_val_frac_eos_pred = val_frac_eos_pred / n_val
        avg_val_frac_eos_target = val_frac_eos_target / n_val
        avg_val_frac_bin_pred = val_frac_bin_pred / n_val
        avg_val_frac_bin_target = val_frac_bin_target / n_val

        if rank == 0:
            print(
                f"[Epoch {epoch}] "
                f"TrainLoss={avg_train_loss:.4f} | TrainAcc={avg_train_acc:.4f} | "
                f"TrainBinMSE={avg_train_bin_mse:.6f} | TrainCoordMSE={avg_train_coord_mse:.6f} | "
                f"TrainEOSAcc={avg_train_eos_acc:.4f} | TrainBinAcc={avg_train_bin_acc:.4f} || "
                f"ValLoss={avg_val_loss:.4f} | ValAcc={avg_val_acc:.4f} | "
                f"ValBinMSE={avg_val_bin_mse:.6f} | ValCoordMSE={avg_val_coord_mse:.6f} | "
                f"ValEOSAcc={avg_val_eos_acc:.4f} | ValBinAcc={avg_val_bin_acc:.4f}"
            )

            wandb.log({
                "epoch/train_loss": avg_train_loss,
                "epoch/train_acc": avg_train_acc,
                "epoch/train_bin_mse": avg_train_bin_mse,
                "epoch/train_coord_mse": avg_train_coord_mse,
                "epoch/train_eos_acc": avg_train_eos_acc,
                "epoch/train_bin_acc": avg_train_bin_acc,
                "epoch/train_frac_eos_pred": avg_train_frac_eos_pred,
                "epoch/train_frac_eos_target": avg_train_frac_eos_target,
                "epoch/train_frac_bin_pred": avg_train_frac_bin_pred,
                "epoch/train_frac_bin_target": avg_train_frac_bin_target,

                "epoch/val_loss": avg_val_loss,
                "epoch/val_acc": avg_val_acc,
                "epoch/val_bin_mse": avg_val_bin_mse,
                "epoch/val_coord_mse": avg_val_coord_mse,
                "epoch/val_eos_acc": avg_val_eos_acc,
                "epoch/val_bin_acc": avg_val_bin_acc,
                "epoch/val_frac_eos_pred": avg_val_frac_eos_pred,
                "epoch/val_frac_eos_target": avg_val_frac_eos_target,
                "epoch/val_frac_bin_pred": avg_val_frac_bin_pred,
                "epoch/val_frac_bin_target": avg_val_frac_bin_target,

                "epoch": epoch,
            })

            # Save best checkpoint based on val loss
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_best_checkpoint(
                    model, optimizer, epoch, best_loss, batch_size, lr, model_config
                )

    cleanup_ddp()


if __name__ == "__main__":
    train(
        checkpoint_path=None,
        use_strict_resume=False
    )
