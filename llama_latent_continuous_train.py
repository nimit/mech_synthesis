import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb
from datetime import datetime

from llama_latent_continuous import LatentLLaMA_Continuous
from dataset_continuous import BarLinkageDataset


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


def get_name(cfg, bs, lr):
    d_model = cfg["d_model"]
    n_heads = cfg["num_heads"]
    n_layers = cfg["num_layers"]
    num_freqs = cfg["num_freqs"]
    return f"LATENT_LLAMA_CONT_d{d_model}_nf{num_freqs}_h{n_heads}_n{n_layers}_bs{bs}_lr{lr}"


# =========================================================
# Checkpoint
# =========================================================
def save_best_checkpoint(
    model,
    optimizer,
    epoch,
    best_loss,
    batch_size,
    lr,
    model_config,
    save_dir="./weights",
):
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(
        save_dir,
        f"{get_name(model_config, batch_size, lr)}.pth",
    )
    torch.save(
        {
            "model_state_dict": model.module.state_dict()
            if isinstance(model, DDP)
            else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "model_config": model_config,
        },
        path,
    )
    if get_rank() == 0:
        print(f"Saved best model at {path} (Val Loss: {best_loss:.6f})")


# =========================================================
# Metrics & Loss
# =========================================================
def compute_euclidean_error(preds, targets, mask):
    """
    Calculates average joint distance error in raw units for valid joints only.
    preds/targets: (B, 8, 3)
    """

    p_raw = preds[..., :2]
    t_raw = targets[..., :2]

    # Euclidean distance
    dist = torch.sqrt(torch.sum((p_raw - t_raw) ** 2, dim=-1))  # (B, 8)

    # Mean only over valid joints
    return dist[mask].mean().item()


class MechanismRegressionLoss(nn.Module):
    def __init__(self, stop_weight=1.0):
        super().__init__()
        self.huber = nn.HuberLoss(reduction="none", delta=1.0)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.stop_weight = stop_weight

    def forward(self, preds, targets, mask):
        # preds: (B, 8, 3), targets: (B, 8, 3), mask: (B, 8)

        # 1. Coordinate Loss (Huber)
        coord_loss = self.huber(preds[..., :2], targets[..., :2]).mean(dim=-1)  # (B, 8)
        coord_loss = (coord_loss * mask).sum() / mask.sum()

        # 2. Stop Bit Loss (BCE)
        stop_loss = self.bce(preds[..., 2], targets[..., 2])  # (B, 8)
        stop_loss = (stop_loss * mask).sum() / mask.sum()

        return coord_loss + (self.stop_weight * stop_loss), coord_loss, stop_loss


# =========================================================
# Training Loop
# =========================================================
def train(checkpoint_path=None, use_strict_resume=False):
    # local_rank = setup_ddp()
    # rank = get_rank()
    local_rank, rank = 0, 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}")

    torch.set_float32_matmul_precision("medium")

    # ---------------- CONFIG ----------------
    num_epochs = 2000
    warmup_epochs = 100
    batch_size = 512
    lr = 5e-4
    LATENT_DIM = 50

    # ---------------- DATA ----------------
    train_dataset, val_dataset = BarLinkageDataset.from_folder(
        data_dir="dataset_17mechs", split_ratio=0.8
    )

    train_sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, rank=rank, num_replicas=world_size, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )

    # ---------------- MODEL ----------------
    model_config = {
        "latent_dim": LATENT_DIM,
        "tgt_seq_len": 8,  # 8 joint pairs
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "num_labels": 17,
        "dropout": 0.1,
        "num_freqs": 128,
    }

    model = LatentLLaMA_Continuous(**model_config).to(device)
    # model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ---------------- OPTIMIZER & LOSS ----------------
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    criterion = MechanismRegressionLoss(stop_weight=1.0)

    # Scheduler with Warmup
    scheduler_warmup = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs],
    )

    # ---------------- WandB ----------------
    if rank == 0:
        wandb.init(
            project="path_synthesis",
            name=f"LATENT_LLaMA_Continuous_{datetime.now().isoformat(timespec='minutes')}",
            config=model_config,
            notes=f"num_epochs={num_epochs}, warmup={warmup_epochs}, batch_size={batch_size}, lr: {lr}",
        )

    best_loss = float("inf")
    start_epoch = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.module.load_state_dict(ckpt["model_state_dict"])
        if use_strict_resume:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_loss = ckpt.get("best_loss", float("inf"))

    # =========================================================
    # EPOCH LOOP
    # =========================================================
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss, epoch_bce = 0.0, 0.0

        pbar = tqdm(train_loader, disable=(rank != 0), desc=f"Train {epoch}")

        for batch in pbar:
            # Prepare Data
            latents = batch["vae_mu"].to(device).squeeze(-1)
            # Take only x,y for model input projection (B, 8, 2)
            dec_in = batch["decoder_input_continuous"][..., :2].to(device)
            attn_mask = batch["attn_mask"].to(device)
            mech_labels = batch["encoded_labels"].to(device)
            targets = batch["labels_continuous"].to(device)  # (B, 8, 3)

            optimizer.zero_grad()

            # Forward
            # model prepends SOS internally, so output length matches targets length
            preds = model(dec_in, attn_mask, latents, mech_labels)
            preds = preds[:, :8, :]  # Match target length

            # Loss
            loss, reg_loss, bce_part = criterion(preds, targets, attn_mask)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += reg_loss.item()
            epoch_bce += bce_part.item()

            if rank == 0:
                wandb.log(
                    {
                        "train/total_loss": loss.item(),
                        "train/reg_loss": reg_loss.item(),
                        "train/stop_bce": bce_part.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                )
                pbar.set_postfix(
                    {"LOSS": f"{reg_loss.item():.5f}", "BCE": f"{bce_part.item():.5f}"}
                )

        scheduler.step()

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss, val_euc = 0.0, 0.0

        with torch.no_grad():
            for batch in val_loader:
                latents = batch["vae_mu"].to(device).squeeze(-1)
                dec_in = batch["decoder_input_continuous"][..., :2].to(device)
                attn_mask = batch["attn_mask"].to(device)
                mech_labels = batch["encoded_labels"].to(device)
                targets = batch["labels_continuous"].to(device)

                preds = model(dec_in, attn_mask, latents, mech_labels)
                preds = preds[:, :8, :]

                _, reg_loss, _ = criterion(preds, targets, attn_mask)

                # Denormalize for Euclidean Error (Raw mm units)
                preds_denorm = val_dataset.denormalize(preds)
                targets_denorm = val_dataset.denormalize(targets)

                euc_err = compute_euclidean_error(
                    preds_denorm, targets_denorm, attn_mask
                )

                val_loss += reg_loss.item()
                val_euc += euc_err

        avg_val_loss = val_loss / len(val_loader)
        avg_val_euc = val_euc / len(val_loader)

        if rank == 0:
            print(
                f"Epoch {epoch} | Train loss: {epoch_loss / len(train_loader):.6f} | Val loss: {avg_val_loss:.6f} | Val Euc Err: {avg_val_euc:.4f}"
            )
            wandb.log(
                {"epoch/val_loss": avg_val_loss, "epoch/val_euc_err": avg_val_euc}
            )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_best_checkpoint(
                    model, optimizer, epoch, best_loss, batch_size, lr, model_config
                )

    # cleanup_ddp()


if __name__ == "__main__":
    train()
