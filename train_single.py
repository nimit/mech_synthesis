import torch
import os
from torch.utils.data import random_split
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import torch.nn as nn

from dataset import SingleTransformerDataset
from model import SingleTransformer

def mse_loss(predictions, targets, mask_value=-1.0):
    mask = ~(targets == mask_value).all(dim=-1)
    mask = mask.unsqueeze(-1).expand_as(predictions)
    # Zero out masked predictions and targets
    masked_predictions = predictions.clone()
    masked_predictions[~mask] = 0.0
    # targets = targets.clone()
    targets[~mask] = 0.0

    # ---- Debug checks ----
    nan_in_preds = torch.isnan(masked_predictions).any()
    inf_in_preds = torch.isinf(masked_predictions).any()
    nan_in_labels = torch.isnan(targets).any()
    inf_in_labels = torch.isinf(targets).any()

    if nan_in_preds or inf_in_preds or nan_in_labels or inf_in_labels:
        # print(mask.shape, masked_predictions.shape, targets.shape)

        # # print(f"❌ Invalid values detected (Rank {rank}):")
        # print(f"Predictions - NaN: {nan_in_preds}, Inf: {inf_in_preds}")
        # print(f"Labels - NaN: {nan_in_labels}, Inf: {inf_in_labels}")
        # print("Sample predictions:", predictions[0])
        # print("Sample masekd predictions:", masked_predictions[0])
        # print("Sample labels:", targets[0])
        raise ValueError("Stopping due to NaN/Inf in validation data.")
    
    # print(predictions)
    # print(targets)

    return F.mse_loss(predictions[mask], targets[mask], reduction="mean")

class CLIPContrastiveLoss(nn.Module):
    def __init__(self, init_scale=1/0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(init_scale)))

    def forward(self, image_embeddings, text_embeddings):
        # print(image_embeddings.shape)
        image_embeddings = F.normalize(image_embeddings.squeeze(), p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings.squeeze(), p=2, dim=1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_embeddings @ text_embeddings.t()

        N = logits.shape[0]
        targets = torch.arange(N, device=logits.device)

        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.t(), targets)
        return (loss_i2t + loss_t2i) / 2

def setup_ddp():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup_ddp():
    dist.destroy_process_group()

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def save_best_checkpoint(model, optimizer, epoch, best_loss, batch_size, lr, clip_loss_fn, model_config, save_dir="./weights"):
    os.makedirs(save_dir, exist_ok=True)
    d_model = model_config['d_model']
    n_heads = model_config['h']
    n = model_config['N']

    checkpoint_path = os.path.join(save_dir, f"d{d_model}_h{n_heads}_n{n}_bs{batch_size}_lr{lr}_best.pth")
    torch.save({
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'clip_loss_state_dict': clip_loss_fn.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
        'batch_size': batch_size,
        'learning_rate': lr,
        'model_config': model_config
    }, checkpoint_path)
    print(f"[Rank {get_rank()}] Saved best model at {checkpoint_path} with loss {best_loss:.6f}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_gradient_flow(model, epoch):
    """Log detailed gradient flow information"""
    total_grad_norm = 0.0
    layer_grads = {}
    layer_weights = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            weight_norm = param.data.norm().item()
            layer_grads[name] = grad_norm
            layer_weights[name] = weight_norm
            total_grad_norm += grad_norm ** 2
            
            wandb.log({
                f"grad/{name}": grad_norm,
                f"weight/{name}": weight_norm,
                # f"grad_weight_ratio/{name}": grad_norm / (weight_norm + 1e-8),
            })
    
    total_grad_norm = total_grad_norm ** 0.5
    wandb.log({
        f"grad/total": total_grad_norm,
        "epoch": epoch
    })
    
    return layer_grads, layer_weights

def log_learning_rates(optimizer, epoch, prefix="train"):
    """Log learning rates for all parameter groups"""
    for i, group in enumerate(optimizer.param_groups):
        wandb.log({
            f"{prefix}/lr/group_{i}": group['lr'],
            "epoch": epoch
        })

def train():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    
    device = torch.device(f'cuda:{local_rank}')
    print(f"Using device: {device}")

    setup_ddp()
    torch.set_float32_matmul_precision('medium')

    # Hyperparameters
    batch_size = 1024
    max_mech_size = 10
    num_epochs = 500
    lr = 5e-4
    clip_loss_weight = 1.0

    # Load Dataset
    dataset = SingleTransformerDataset(data_dir='/home/anurizada/Documents/nobari_10_transformer')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)

    # Model Configuration
    model_config = {
        'output_size': 2,
        'tgt_seq_len': max_mech_size,
        'd_model': 256,
        'h': 4,
        'N': 12
    }

    # Initialize Model
    model = SingleTransformer(
        output_size=model_config['output_size'],
        tgt_seq_len=model_config['tgt_seq_len'],
        d_model=model_config['d_model'],
        h=model_config['h'],
        N=model_config['N']
    ).to(local_rank)

    total_params = count_parameters(model)
    print(f"[Rank {rank}] Model created with {total_params:,} trainable parameters")

    model = DDP(model, device_ids=[local_rank])
    clip_loss_fn = CLIPContrastiveLoss().to(local_rank)

    # Initialize WandB
    if rank == 0:
        wandb.init(
            project="distributed-training",
            name=f"d{model_config['d_model']}_h{model_config['h']}_n{model_config['N']}_bs{batch_size}_lr{lr}",
            config={
                "output_size": model_config['output_size'],
                "tgt_seq_len": model_config['tgt_seq_len'],
                "d_model": model_config['d_model'],
                "n_heads": model_config['h'],
                "n_layers": model_config['N'],
                "batch_size": batch_size,
                "lr": lr,
                "clip_loss_weight": clip_loss_weight,
                "total_params": total_params
            }
        )

    optimizer = Adam([
        {'params': model.parameters()},
        {'params': clip_loss_fn.parameters()}
    ], lr=lr)

    scheduler = StepLR(optimizer, step_size=200, gamma=0.1)
    best_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Rank {rank} Train Epoch {epoch}", leave=False) as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # Get batch data
                decoder_input = batch["decoder_input"].to(local_rank)
                decoder_mask = batch["decoder_mask"].to(local_rank)
                curve_data = batch["curve_numerical"].to(local_rank)
                adj_data = batch["adjacency"].to(local_rank)
                labels = batch["label"].to(local_rank)

                # Forward pass
                optimizer.zero_grad()
                predictions, curve_emb, adj_emb = model(decoder_input, decoder_mask, curve_data, adj_data)

                # Calculate losses
                prediction_loss = mse_loss(predictions, labels)
                clip_loss = clip_loss_fn(curve_emb, adj_emb)
                total_loss = prediction_loss + clip_loss_weight * clip_loss

                # Backward pass and optimization
                total_loss.backward()
                
                # Log gradients and learning rates
                if rank == 0:
                    log_gradient_flow(model, epoch)
                    log_learning_rates(optimizer, epoch)
                
                optimizer.step()
                epoch_loss += total_loss.item()

                if rank == 0:
                    wandb.log({
                        "train/mse_loss": prediction_loss.item(),
                        "train/clip_loss": clip_loss.item(),
                        "train/clip_logit_scale": clip_loss_fn.logit_scale.exp().item(),
                        "train/total_loss": total_loss.item(),
                        "epoch": epoch,
                    })

                pbar.set_postfix({"Loss": total_loss.item()})
                pbar.update(1)

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)

        if avg_train_loss < best_loss and rank == 0:
            best_loss = avg_train_loss
            save_best_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_loss=best_loss,
                batch_size=batch_size,
                lr=lr,
                clip_loss_fn=clip_loss_fn,
                model_config=model_config
            )

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Rank {rank} Val Epoch {epoch}", leave=False) as pbar:
                for batch_idx, batch in enumerate(val_loader):
                    decoder_input = batch["decoder_input"].to(local_rank)
                    decoder_mask = batch["decoder_mask"].to(local_rank)
                    curve_data = batch["curve_numerical"].to(local_rank)
                    adj_data = batch["adjacency"].to(local_rank)
                    labels = batch["label"].to(local_rank)

                    # Forward pass
                    predictions, curve_emb, adj_emb = model(decoder_input, decoder_mask, curve_data, adj_data)
                    
                    # Calculate losses
                    prediction_loss = mse_loss(predictions, labels)
                    clip_loss = clip_loss_fn(curve_emb, adj_emb)
                    total_loss = prediction_loss + clip_loss_weight * clip_loss

                    val_loss += total_loss.item()

                    if rank == 0:
                        wandb.log({
                            "val/mse_loss": prediction_loss.item(),
                            "val/clip_loss": clip_loss.item(),
                            "val/clip_logit_scale": clip_loss_fn.logit_scale.exp().item(),
                            "val/total_loss": total_loss.item(),
                            "epoch": epoch,
                        })

                    pbar.set_postfix({"Val Loss": total_loss.item()})
                    pbar.update(1)

        if rank == 0:
            print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss/len(val_loader):.6f}")

    cleanup_ddp()

if __name__ == "__main__":
    train()
