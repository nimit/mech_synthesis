import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel


def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        if getattr(m, "weight", None) is not None:
            nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)  # type: ignore


class FourierFeatureEmbedding(nn.Module):
    """
    Fourier Features using LINEAR frequency bands (1, 2, 3, ..., n).
    """
    def __init__(self, num_freqs=8, scale=1.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.scale = scale
        # Linear frequencies: 1, 2, 3, ..., num_freqs
        self.register_buffer("freq_bands", torch.arange(1, num_freqs + 1).float())

    def forward(self, x):
        # x: (..., 2)
        # (..., 2, 1) * (num_freqs,) -> (..., 2, num_freqs)
        x_proj = x.unsqueeze(-1) * self.freq_bands * self.scale
        
        # Concatenate sin and cos: (..., 2, num_freqs*2)
        x_enc = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
        # Flatten: (..., 2 * num_freqs * 2) = (..., 128)
        return x_enc.flatten(start_dim=-2)


class ContinuousInputEmbeddings(nn.Module):
    def __init__(self, d_model: int, num_freqs: int = 32, **kwargs):
        super().__init__()
        # Deterministic Fourier Features: (x,y) -> 128 dim
        self.fourier = FourierFeatureEmbedding(num_freqs=num_freqs)
        input_dim = 2 * num_freqs * 2  # 2 coords * num_freqs * 2 (sin/cos)

        # Projects high-dim features to d_model via MLP
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        # x shape: (B, T, 2)
        features = self.fourier(x)
        return self.proj(features)


# ------------------------------------------------------------
# SINGLE LATENT TOKEN PROJECTOR
# ------------------------------------------------------------
class SingleLatentToken(nn.Module):
    """
    Takes the whole latent vector (B, latent_dim)
    -> produces ONE token (B, 1, d_model)
    """

    def __init__(self, latent_dim, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Optional modality embedding
        self.latent_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.apply(_init_weights)

    def forward(self, latent_vec):
        """
        latent_vec: (B, latent_dim)
        """
        if latent_vec.dim() == 3:  # safety
            latent_vec = latent_vec.squeeze(-1)  # (B, 50)

        x = self.proj(latent_vec)  # (B, d_model)
        x = x.unsqueeze(1)  # (B, 1, d_model)
        x = x + self.latent_type
        return self.norm(x)


class LatentLLaMA_Continuous(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        tgt_seq_len: int = 8,  # 8 coordinate pairs
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        num_labels: int = 17,
        dropout: float = 0.1,
        num_freqs: int = 256,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Input Projection for (x, y) pairs using Gaussian RFF
        self.tgt_embed = ContinuousInputEmbeddings(d_model, num_freqs=num_freqs, sigma=sigma)

        # 2. Learnable SOS token (represents the start of a mechanism)
        self.sos_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.mech_embed = nn.Embedding(num_labels, d_model)
        self.tgt_pos = nn.Embedding(tgt_seq_len + 2, d_model)  # +2 for Latent/Mech
        self.latent_token = SingleLatentToken(latent_dim, d_model, dropout)

        llama_cfg = LlamaConfig(
            vocab_size=1,  # Not used, but required by config
            hidden_size=d_model,
            intermediate_size=4 * d_model,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            use_cache=False,
            attention_dropout=dropout,
        )
        self.llama = LlamaModel(llama_cfg)

        # 3. Output Head: predicts x, y, and stop_logit
        self.proj = nn.Linear(d_model, 3)

    def forward(self, decoder_input_coords, attn_mask, latent_vec, mech_labels):
        """
        decoder_input_coords: (B, T, 2) -> Raw (x, y) float values
        attn_mask: (B, T) -> 1 for valid joints, 0 for padding
        """
        B, T, _ = decoder_input_coords.shape
        device = decoder_input_coords.device

        # Embeddings
        lat_tok = self.latent_token(latent_vec)  # (B, 1, D)
        mech_tok = self.mech_embed(mech_labels).unsqueeze(1)  # (B, 1, D)

        # Project raw (x, y) coordinates
        tgt_emb = self.tgt_embed(decoder_input_coords)  # (B, T, D)

        # Prepend the learnable SOS token to the coordinate sequence
        # Sequence becomes: [SOS, Joint1, Joint2, ...]
        sos_tok = self.sos_token.expand(B, -1, -1)
        tgt_seq = torch.cat([sos_tok, tgt_emb], dim=1)  # (B, T+1, D)

        # Apply Positional Encoding
        # Indices: Latent(-), Mech(0), SOS(1), Joints(2...T+1)
        pos_ids = torch.arange(T + 2, device=device).unsqueeze(0)
        mech_tok = mech_tok + self.tgt_pos(pos_ids[:, :1])
        tgt_seq = tgt_seq + self.tgt_pos(pos_ids[:, 1:])

        # Combine: [Latent, Mech, SOS, Joint1, ...]
        full_seq = torch.cat([lat_tok, mech_tok, tgt_seq], dim=1)  # (B, T+3, D)

        # Update Attention Mask for added tokens (Latent, Mech, SOS)
        # prefix_mask is for [Latent, Mech, SOS]
        prefix_mask = torch.ones((B, 3), device=device, dtype=attn_mask.dtype)
        full_attn_mask = torch.cat([prefix_mask, attn_mask], dim=1)

        # LLaMA forward pass
        out = self.llama(inputs_embeds=full_seq, attention_mask=full_attn_mask)

        # Slice output: We want to predict the NEXT joint and STOP bit
        # input [SOS] -> predicts [Joint1, Stop1]
        # input [Joint1] -> predicts [Joint2, Stop2]
        dec_out = out.last_hidden_state[:, 2:, :]  # Skip Latent & Mech

        preds = self.proj(dec_out)  # (B, T+1, 3)
        return preds
