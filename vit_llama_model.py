import math
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel


# ------------------------------------------------------------
# Utils / init
# ------------------------------------------------------------
def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Embedding)):
        if getattr(m, "weight", None) is not None:
            nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.zeros_(m.bias)


# ------------------------------------------------------------
# Token & Positional embeddings
# ------------------------------------------------------------
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x) -> torch.Tensor:
        # (B, T) -> (B, T, D)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


# ------------------------------------------------------------
# ViT-style Image Encoder (sequence output)
# For 64x64 grayscale images, patch_size=8 -> N=64 tokens
# ------------------------------------------------------------
class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder that outputs patch embeddings (B, N, D).
    Designed for 64x64 grayscale images by default.
    """

    def __init__(
        self,
        emb_size: int = 512,
        in_channels: int = 1,
        image_size: int = 64,
        patch_size: int = 8,
        num_layers: int = 6,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_patches = (image_size // patch_size) ** 2
        self.emb_size = emb_size

        # Patch embedding via strided conv
        self.patch_embed = nn.Conv2d(
            in_channels,
            emb_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=False,
        )

        # Learnable positional embeddings for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, emb_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder over patch tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=4 * emb_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(emb_size)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, N, D)
        """
        patches = self.patch_embed(x)  # (B, D, H/ps, W/ps)
        tokens = patches.flatten(2).transpose(1, 2)  # (B, N, D)
        tokens = tokens + self.pos_embed
        encoded = self.encoder(tokens)  # (B, N, D)
        return self.norm(encoded)


# ------------------------------------------------------------
# ViT + LLaMA decoder-only model
# ------------------------------------------------------------
class SingleImageTransformerCLIP_LLaMA(nn.Module):
    """
    ViT image encoder + LLaMA decoder-only transformer.

    Sequence into LLaMA:
        full_seq = [ image_tokens (from ViT) , mech_token , target_token_embeddings ]

    We then take the last T positions (the target tokens) and project to logits.
    """

    def __init__(
        self,
        tgt_seq_len: int = 25,
        vocab_size: int = 204,
        d_model: int = 512,
        h: int = 8,
        N: int = 6,
        num_labels: int = 17,  # mechanism types
        dropout: float = 0.1,
        img_in_channels: int = 1,
        img_size: int = 64,
        img_patch: int = 8,
        pad_token_id: int = 2,
        debug: bool = False,
    ):
        super().__init__()
        self.tgt_seq_len = tgt_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.debug = debug

        # 1) Embeddings
        self.tgt_embed = InputEmbeddings(d_model, vocab_size)
        self.mech_embedding = nn.Embedding(num_labels, d_model)

        # Optional positional encoding for mech+target (you can keep or drop this;
        # LLaMA already has RoPE, but ViT tokens have their own positional info.)
        self.decoder_positional_encoding = PositionalEncoding(
            d_model, seq_len=tgt_seq_len + 1, dropout=dropout
        )

        # 2) Image encoder (ViT) -> image tokens
        self.image_encoder = ViTEncoder(
            emb_size=d_model,
            in_channels=img_in_channels,
            image_size=img_size,
            patch_size=img_patch,
            num_layers=N,
            nhead=h,
            dropout=dropout,
        )

        # 3) LLaMA model as decoder stack
        llama_cfg = LlamaConfig(
            vocab_size=vocab_size,  # dummy > 0 (unused if we use inputs_embeds)
            hidden_size=d_model,
            intermediate_size=4 * d_model,
            num_attention_heads=h,
            num_hidden_layers=N,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            use_cache=False,
        )
        self.llama = LlamaModel(llama_cfg)
        
        # ✅ freeze unused text token embeddings (not used when passing inputs_embeds)
        for param in self.llama.embed_tokens.parameters():
            param.requires_grad = False

        # 4) LM head to your own vocab
        self.projection = nn.Linear(d_model, vocab_size)

        self.apply(_init_weights)
        print("✅ Initialized SingleImageTransformerCLIP_LLaMA (ViT + LLaMA)")

    # ----------------------
    # Debug helper
    # ----------------------
    def _dprint(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    # ----------------------
    # Encode image -> tokens
    # ----------------------
    def encode(self, image_data: torch.Tensor) -> torch.Tensor:
        """
        image_data: (B, C, H, W)
        returns: image_tokens: (B, N_img, D)
        """
        image_tokens = self.image_encoder(image_data)
        self._dprint(f"[ENCODE] image_tokens: {image_tokens.shape}")
        return image_tokens

    # ----------------------
    # Decode with LLaMA
    # ----------------------
    def decode(
        self,
        image_tokens: torch.Tensor,  # (B, N_img, D)
        tgt_ids: torch.Tensor,       # (B, T)
        mech_labels: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """
        Build full sequence [image_tokens, mech_token, tgt_embeds] and pass it
        as inputs_embeds to LLaMA.
        """
        B, T = tgt_ids.shape
        device = tgt_ids.device
        B_img, N_img, D = image_tokens.shape
        assert B == B_img, "Batch size mismatch between image tokens and tgt_ids"

        # Mech token
        mech_emb = self.mech_embedding(mech_labels).unsqueeze(1)  # (B, 1, D)

        # Target tokens
        tgt_emb = self.tgt_embed(tgt_ids)                         # (B, T, D)

        # Optionally add sinusoidal PE to mech + targets
        mech_plus_tgt = torch.cat([mech_emb, tgt_emb], dim=1)     # (B, 1+T, D)
        mech_plus_tgt = self.decoder_positional_encoding(mech_plus_tgt)

        # Full sequence for LLaMA
        full_seq = torch.cat([image_tokens, mech_plus_tgt], dim=1)  # (B, N_img+1+T, D)
        B_full, L_full, _ = full_seq.shape
        self._dprint(f"[DECODE] full_seq: {full_seq.shape} (N_img={N_img}, T={T})")

        # Attention mask: all ones (no padding, LLaMA will apply causal masking internally)
        attention_mask = torch.ones(B_full, L_full, dtype=torch.long, device=device)
        self._dprint(f"[DECODE] attention_mask: {attention_mask.shape}")

        # Run through LLaMA decoder stack
        outputs = self.llama(
            inputs_embeds=full_seq,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )
        hidden = outputs.last_hidden_state  # (B, L_full, D)
        self._dprint(f"[DECODE] LLaMA hidden: {hidden.shape}")

        # Extract only target token positions:
        # positions: 0..N_img-1 -> image, N_img -> mech, N_img+1..N_img+T -> targets
        dec_out = hidden[:, N_img + 1 :, :]  # (B, T, D)
        self._dprint(f"[DECODE] dec_out (targets): {dec_out.shape}")

        return dec_out

    # ----------------------
    # Forward
    # ----------------------
    def forward(
        self,
        decoder_input_ids: torch.Tensor,  # (B, T)
        _decoder_mask_unused: torch.Tensor,  # kept for compatibility
        image_data: torch.Tensor,         # (B, C, H, W)
        mech_labels: torch.Tensor,        # (B,)
    ) -> torch.Tensor:
        """
        Returns logits: (B, T, vocab_size)
        """
        self._dprint("\n===== FORWARD START =====")
        self._dprint(f"decoder_input_ids: {decoder_input_ids.shape}")
        self._dprint(f"image_data:        {image_data.shape}")
        self._dprint(f"mech_labels:       {mech_labels.shape}")

        # Encode image
        image_tokens = self.encode(image_data)  # (B, N_img, D)

        # Decode with LLaMA
        dec_out = self.decode(image_tokens, decoder_input_ids, mech_labels)  # (B, T, D)

        # Project to logits
        logits = self.projection(dec_out)  # (B, T, V)
        self._dprint(f"[FORWARD] logits: {logits.shape}")
        self._dprint("===== FORWARD END =====\n")

        return logits
