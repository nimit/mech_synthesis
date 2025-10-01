import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from torch.nn import MultiheadAttention

def _init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        if not any(s in m.__class__.__name__.lower() for s in ["resnet", "batchnorm"]):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale

class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        d_model: int, 
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dim = 4 * d_model

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return self.dropout(x)

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, bias: bool=True) -> None:  # Changed from input_size to vocab_size
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)  # Use Embedding layer instead of Linear

    def forward(self, x) -> torch.Tensor:
        # (batch, seq_len) --> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)

class LabelEmbedding(nn.Module):
    def __init__(self, d_model: int, num_labels: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_labels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch,) --> (batch, d_model)
        embedded = self.embedding(x)
        return embedded * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float=0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    
class Encoder(nn.Module):
    def __init__(self, dim: int, n_heads: int, seq_len: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.seq_len = seq_len
        
        self.attention = MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.n_heads,
            dropout=self.dropout,
            bias=True,
            batch_first=True
        )

        self.feed_forward = FeedForwardBlock(self.dim)

        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(self.dim, eps=1e-5)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(self.dim, eps=1e-5)
        self.final_norm = RMSNorm(dim, eps=1e-5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x_n = self.attention_norm(x)
        
        attn_output, _ = self.attention(
            query=x_n,
            key=x_n,
            value=x_n,
            key_padding_mask=mask.squeeze(-1) if mask is not None else None
        )
        
        h = x + attn_output
        out = h + self.feed_forward(self.ffn_norm(h))
        return self.final_norm(out)

class Decoder(nn.Module):
    def __init__(self, dim: int, n_heads: int, tgt_seq_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.dropout = dropout

        # Self-attention
        self.self_attention = MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            bias=True,
            batch_first=True
        )
        
        # Cross-attention
        self.cross_attention = MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            bias=True,
            batch_first=True
        )

        # Normalization layers
        self.self_attention_norm = RMSNorm(dim, eps=1e-5)
        self.cross_attention_norm = RMSNorm(dim, eps=1e-5)
        self.ffn_norm = RMSNorm(dim, eps=1e-5)
        self.final_norm = RMSNorm(dim, eps=1e-5)
        self.feed_forward = FeedForwardBlock(dim)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        
        # ===== 1. Self-Attention =====
        x_n = self.self_attention_norm(x)
        
        attn_mask = None
        if tgt_mask is not None:
            if tgt_mask.dim() == 3:  # [B, T, T]
                # Convert to float and set masked positions to -inf
                attn_mask = torch.zeros_like(tgt_mask, dtype=torch.float)
                attn_mask.masked_fill_(~tgt_mask, float('-inf'))
                # Expand for multi-head attention
                attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0)

        # print(attn_mask)
        if not hasattr(self, "_mask_debug_done"):
            self._mask_debug_done = True  # Print only once
            if tgt_mask is not None:
                print("\n=== Attention Mask Debug ===")
                print("tgt_mask shape:", tgt_mask.shape)
                print("Sample tgt_mask[0]:\n", tgt_mask[0].int().cpu().numpy())

                if 'attn_mask' in locals() and attn_mask is not None:
                    print("attn_mask shape:", attn_mask.shape)
                    # Print first head’s mask
                    num_heads = self.n_heads
                    seq_len = tgt_mask.shape[-1]
                    first_head_mask = attn_mask[:seq_len, :seq_len].cpu().numpy()
                    print("attn_mask (first head) [0:5,0:5]:\n", first_head_mask[:5, :5])

        self_attn_out, _ = self.self_attention(
            query=x_n,
            key=x_n,
            value=x_n,
            attn_mask=attn_mask,
            need_weights=False
        )
        
        h = x + self_attn_out

        # ===== 2. Cross-Attention =====
        out_n = self.cross_attention_norm(h)
        
        cross_attn_out, _ = self.cross_attention(
            query=out_n,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=src_mask,
            need_weights=False
        )
        h = h + cross_attn_out

        # ===== 3. Feed Forward =====
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return self.final_norm(out)
    
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.rm1 = RMSNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rm2 = RMSNorm(hidden_dim)
        self.act2 = nn.SiLU()
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale1 = nn.Parameter(torch.ones(hidden_dim))
        self.layer_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        x = self.fc1(x)
        x = self.rm1(x)
        x = self.act1(x) * self.layer_scale1
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.rm2(x)
        x = self.act2(x) * self.layer_scale2
        x = self.dropout(x)
        return self.fc_out(x)

class ContrastiveEncoder(nn.Module):
    def __init__(self, in_channels=3, emb_size=128):
        super().__init__()
        self.convnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # ResNet18 with ResNet18 weights
        self.convnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.convnet.fc = nn.Identity()
        self.projector = ProjectionHead(512, 1024, emb_size)  # Change 2048 to 512 for ResNet18

    def forward(self, x):
        features = self.convnet(x)
        return self.projector(features)

import torch.nn as nn

class SingleImageTransformer(nn.Module):
    def __init__(self, tgt_seq_len=25, vocab_size=204, d_model=512, 
                 h=8, N=6, num_labels=105, dropout=0.1):
        super().__init__()
        self.tgt_seq_len = tgt_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_labels = num_labels
        self.n_heads = h

        # Embeddings
        self.tgt_embed = InputEmbeddings(d_model, vocab_size)
        self.label_embed = LabelEmbedding(d_model, num_labels)

        # Positional encodings
        self.encoder_positional_encoding = PositionalEncoding(d_model, seq_len=2)
        self.decoder_positional_encoding = PositionalEncoding(d_model, seq_len=tgt_seq_len)

        # Swap in PyTorch Transformer encoder/decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=h,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # closer to your RMSNorm-first design
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=h,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=N)

        # Projection layer
        self.projection = nn.Linear(d_model, vocab_size)

        # Image encoder
        self.image_encoder = ContrastiveEncoder(in_channels=1, emb_size=d_model)

    def encode(self, image_data, labels, src_mask=None):
        image_embedding = self.image_encoder(image_data).unsqueeze(1)   # (B,1,D)
        label_embedding = self.label_embed(labels).unsqueeze(1)         # (B,1,D)
        combined_embedding = torch.cat([image_embedding, label_embedding], dim=1)
        src = self.encoder_positional_encoding(combined_embedding)
        memory = self.encoder(src, src_key_padding_mask=src_mask)
        return memory, image_embedding, label_embedding

    def decode(self, memory, tgt, tgt_mask=None, memory_mask=None):
        tgt = self.tgt_embed(tgt)
        tgt = self.decoder_positional_encoding(tgt)
        
        attn_mask = None
        if tgt_mask is not None:
            if tgt_mask.dim() == 3:  # [B, T, T]
                # Convert to float and set masked positions to -inf
                attn_mask = torch.zeros_like(tgt_mask, dtype=torch.float)
                attn_mask.masked_fill_(~tgt_mask, float('-inf'))
                # Expand for multi-head attention
                attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0)

        output = self.decoder(
            tgt,
            memory,
            tgt_mask=attn_mask,
            memory_mask=memory_mask
        )
        return output

    def forward(self, decoder_input, decoder_mask, image_data, labels):
        memory, image_embedding, label_embedding = self.encode(image_data, labels)
        decoder_output = self.decode(memory, decoder_input, tgt_mask=decoder_mask)
        logits = self.projection(decoder_output)
        return logits, image_embedding, label_embedding

# class SingleImageTransformer(nn.Module):
#     def __init__(self, tgt_seq_len: int = 25, vocab_size: int = 204, d_model: int = 512, 
#                  h: int = 8, N: int = 6, num_labels: int = 105):
#         super(SingleImageTransformer, self).__init__()
#         self.tgt_seq_len = tgt_seq_len
#         self.vocab_size = vocab_size
#         self.d_model = d_model
#         self.num_labels = num_labels

#         # Positional Encoding for encoder and decoder
#         self.encoder_positional_encoding = PositionalEncoding(d_model, seq_len=2)
#         self.decoder_positional_encoding = PositionalEncoding(d_model, seq_len=tgt_seq_len)
        
#         # Embeddings - FIXED: Use vocab_size for target embedding
#         self.tgt_embed = InputEmbeddings(d_model, vocab_size, bias=False)  # Changed to vocab_size
#         self.label_embed = LabelEmbedding(d_model, num_labels)
        
#         # Encoder and Decoder
#         self.encoder = nn.ModuleList([Encoder(d_model, h, seq_len=2) for _ in range(N)])
#         self.decoder = nn.ModuleList([Decoder(d_model, h, tgt_seq_len) for _ in range(N)])

#         # Projection layer - output vocab probabilities
#         self.projection = nn.Linear(d_model, vocab_size, bias=False)
#         self.projection_norm = RMSNorm(d_model, eps=1e-5)

#         # Image encoder
#         self.image_encoder = ContrastiveEncoder(in_channels=1, emb_size=d_model)

#         # Initialize weights
#         for m in [self.encoder, self.decoder, self.projection, self.projection_norm, self.tgt_embed, self.label_embed]:
#             m.apply(_init_weights)
    
#     def encode(self, image_data, labels):
#         print("="*40)
#         print("[ENCODE] Starting encoding")
#         print("="*40)

#         # Process image data
#         print(f"[ENCODE] image_data shape: {image_data.shape}")
#         image_embedding = self.image_encoder(image_data).unsqueeze(1)  # Shape: (B,1,D)
#         print(f"[ENCODE] image_embedding shape: {image_embedding.shape}, "
#               f"mean={image_embedding.mean().item():.6f}, std={image_embedding.std().item():.6f}")

#         # Process labels
#         print(f"[ENCODE] labels shape: {labels.shape}, dtype={labels.dtype}")
#         label_embedding = self.label_embed(labels).unsqueeze(1)  # Shape: (B,1,D)
#         print(f"[ENCODE] label_embedding shape: {label_embedding.shape}, "
#               f"mean={label_embedding.mean().item():.6f}, std={label_embedding.std().item():.6f}")

#         # Combine
#         combined_embedding = torch.cat([image_embedding, label_embedding], dim=1)
#         print(f"[ENCODE] combined_embedding shape: {combined_embedding.shape}")

#         # Positional encoding
#         src = self.encoder_positional_encoding(combined_embedding)
#         print(f"[ENCODE] after positional encoding: {src.shape}, "
#               f"mean={src.mean().item():.6f}, std={src.std().item():.6f}")

#         # Encoder layers
#         for i, layer in enumerate(self.encoder):
#             src = layer(src)
#             print(f"[ENCODE] after encoder layer {i}: {src.shape}, "
#                   f"mean={src.mean().item():.6f}, std={src.std().item():.6f}")

#         return src, image_embedding, label_embedding

#     def decode(self, encoder_output, src_mask, tgt, tgt_mask):
#         print("="*40)
#         print("[DECODE] Starting decoding")
#         print("="*40)

#         print(f"[DECODE] tgt input shape: {tgt.shape}, dtype={tgt.dtype}")
#         tgt = self.tgt_embed(tgt)  # (B,T,D)
#         print(f"[DECODE] after tgt_embed: {tgt.shape}, "
#               f"mean={tgt.mean().item():.6f}, std={tgt.std().item():.6f}")

#         tgt = self.decoder_positional_encoding(tgt)
#         print(f"[DECODE] after positional encoding: {tgt.shape}, "
#               f"mean={tgt.mean().item():.6f}, std={tgt.std().item():.6f}")

#         for i, layer in enumerate(self.decoder):
#             tgt = layer(tgt, encoder_output, src_mask, tgt_mask)
#             print(f"[DECODE] after decoder layer {i}: {tgt.shape}, "
#                   f"mean={tgt.mean().item():.6f}, std={tgt.std().item():.6f}")

#         return tgt
    
#     def forward(self, decoder_input, decoder_mask, image_data, labels):
#         print("="*40)
#         print("[FORWARD] Starting forward pass")
#         print("="*40)

#         # Encode
#         encoder_output, image_embedding, label_embedding = self.encode(image_data, labels)
#         print(f"[FORWARD] encoder_output shape: {encoder_output.shape}")

#         # Decode
#         decoder_output = self.decode(
#             encoder_output,
#             None,
#             decoder_input,
#             decoder_mask
#         )
#         print(f"[FORWARD] decoder_output before norm: {decoder_output.shape}, "
#               f"mean={decoder_output.mean().item():.6f}, std={decoder_output.std().item():.6f}")

#         # Norm + projection
#         decoder_output = self.projection_norm(decoder_output)
#         print(f"[FORWARD] decoder_output after norm: {decoder_output.shape}, "
#               f"mean={decoder_output.mean().item():.6f}, std={decoder_output.std().item():.6f}")

#         final_output = self.projection(decoder_output)
#         print(f"[FORWARD] final_output logits: {final_output.shape}, "
#               f"logit-range=({final_output.min().item():.6f}, {final_output.max().item():.6f})")

#         return final_output, image_embedding, label_embedding
