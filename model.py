import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from utils import preprocess_curves
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

    def __init__(self, d_model: int, input_size: int, bias: bool=True) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.embedding = nn.Linear(self.input_size, self.d_model, bias=bias)

    def forward(self, x) -> torch.Tensor:
        # (batch, seq_len, input_size) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)

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
        
        # Replace custom Attention with PyTorch's MultiheadAttention
        self.attention = MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.n_heads,
            dropout=self.dropout,
            bias=True,  # Keep bias (unlike your original)
            batch_first=True  # (B, Seq_Len, Dim) format
        )

        # self.attention = Attention(self.dim, self.n_heads, self.seq_len)
        self.feed_forward = FeedForwardBlock(self.dim)

        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(self.dim, eps=1e-5)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(self.dim, eps=1e-5)
        self.final_norm = RMSNorm(dim, eps=1e-5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x_n = self.attention_norm(x)
        
        # PyTorch's MultiheadAttention expects mask shape:
        # (B, Seq_Len) for key_padding_mask (1 for padded positions)
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
        
        # Cross-attention (now using MultiheadAttention)
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
        
        # tgt_mask = ~tgt_mask

        attn_mask = None
        if tgt_mask is not None:
            if tgt_mask.dim() == 3:  # [B, T, T]
                # Convert to float and set masked positions to -inf
                attn_mask = torch.zeros_like(tgt_mask, dtype=torch.float)
                attn_mask.masked_fill_(tgt_mask, float('-inf'))
                # Expand for multi-head attention
                attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0)

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
        
        # Cross-attention: query=decoder, key/value=encoder

        cross_attn_out, _ = self.cross_attention(
            query=out_n,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=src_mask,  # src_mask should be [B,S] if used
            need_weights=False
        )
        h = h + cross_attn_out

        # ===== 3. Feed Forward =====
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return self.final_norm(out)
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

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
    def __init__(self, in_channels=1, emb_size=128):
        super().__init__()
        self.convnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.convnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convnet.fc = nn.Identity()
        self.projector = ProjectionHead(2048, 1024, emb_size)

    def forward(self, x):
        features = self.convnet(x)
        return self.projector(features)

class SingleImageTransformer(nn.Module):
    def __init__(self, tgt_seq_len: int, output_size: int, d_model: int=1024, h: int=32, N: int=6):
        super(SingleImageTransformer, self).__init__()
        self.tgt_seq_len = tgt_seq_len

        # Positional Encoding for encoder and decoder
        self.encoder_positional_encoding = PositionalEncoding(d_model, seq_len=2)
        self.decoder_positional_encoding = PositionalEncoding(d_model, seq_len=tgt_seq_len)
        
        # Embeddings and Encoder
        self.tgt_embed = InputEmbeddings(d_model, output_size, bias=False)
        self.encoder = nn.ModuleList([Encoder(d_model, h, seq_len=2) for _ in range(N)])

        # Single decoder
        self.decoder = nn.ModuleList([Decoder(d_model, h, tgt_seq_len) for _ in range(N)])

        # Projection layer
        self.projection = nn.Linear(d_model, output_size, bias=False)
        self.projection_norm = RMSNorm(d_model, eps=1e-5)

        # Add Contrastive_Curve and GraphHop models
        self.contrastive_curve = ContrastiveEncoder(in_channels=1, emb_size=d_model)
        self.contrastive_adj = ContrastiveEncoder(in_channels=1, emb_size=d_model)

        # Initialize weights
        for m in [
            self.encoder,
            self.decoder,
            self.projection,
            self.tgt_embed,
        ]:
            m.apply(_init_weights)
    
    def encode(self, curve_data, adj_data):
        # Process curve data with Contrastive_Curve
        curve_embedding = self.contrastive_curve(curve_data).unsqueeze(1)  # Shape: (batch_size, d_model)
        # Process graph data with GraphHop
        adj_embedding = self.contrastive_adj(adj_data).unsqueeze(1)  # Shape: (batch_size, d_model)

        combined_embedding = torch.cat([curve_embedding, adj_embedding], dim=1)

        # Add positional encoding to the combined embeddings
        src = self.encoder_positional_encoding(combined_embedding)  # Shape: (batch_size, 1, d_model)

        # Pass through encoder layers
        for layer in self.encoder:
            src = layer(src)  # Shape: (batch_size, 1, d_model)

        return src, curve_embedding, adj_embedding

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # Embed the target sequence and add positional encoding
        tgt = self.tgt_embed(tgt)
        tgt = self.decoder_positional_encoding(tgt)

        # Apply decoder layers
        for layer in self.decoder:
            tgt = layer(tgt, encoder_output, src_mask, tgt_mask)

        return tgt
    
    def forward(self, decoder_input, decoder_mask, curve_data, adj_data):
        # Encode source input with curve and graph data
        encoder_output, curve_embedding, adj_embedding = self.encode(curve_data, adj_data)
        
        # Decoder pass
        decoder_output = self.decode(
            encoder_output, 
            None, 
            decoder_input, 
            decoder_mask
        )
        
        decoder_output = self.projection_norm(decoder_output)
        final_output = self.projection(decoder_output)

        return final_output, curve_embedding, adj_embedding

class SingleTransformer(nn.Module):
    def __init__(self, tgt_seq_len: int, output_size: int, d_model: int=1024, h: int=32, N: int=6):
        super(SingleTransformer, self).__init__()
        self.tgt_seq_len = tgt_seq_len

        # Positional Encoding for encoder and decoder
        self.encoder_positional_encoding = PositionalEncoding(d_model, seq_len=2)
        self.decoder_positional_encoding = PositionalEncoding(d_model, seq_len=tgt_seq_len)
        
        # Embeddings and Encoder
        self.tgt_embed = InputEmbeddings(d_model, output_size, bias=False)
        self.encoder = nn.ModuleList([Encoder(d_model, h, seq_len=2) for _ in range(N)])

        # Single decoder
        self.decoder = nn.ModuleList([Decoder(d_model, h, tgt_seq_len) for _ in range(N)])

        # Projection layer
        self.projection = nn.Linear(d_model, output_size, bias=False)
        self.projection_norm = RMSNorm(d_model, eps=1e-5)

        # Add Contrastive_Curve and GraphHop models
        self.contrastive_curve = ContrastiveEncoder(in_channels=1, emb_size=d_model)
        self.contrastive_adj = ContrastiveEncoder(in_channels=1, emb_size=d_model)

        # Initialize weights
        for m in [
            self.encoder,
            self.decoder,
            self.projection,
            self.tgt_embed,
        ]:
            m.apply(_init_weights)
    
    def encode(self, curve_data, adj_data):
        curve_data = preprocess_curves(curves=curve_data).unsqueeze(1)

        # Process curve data with Contrastive_Curve
        curve_embedding = self.contrastive_curve(curve_data).unsqueeze(1)  # Shape: (batch_size, d_model)
        # Process graph data with GraphHop
        adj_embedding = self.contrastive_adj(adj_data).unsqueeze(1)  # Shape: (batch_size, d_model)

        combined_embedding = torch.cat([curve_embedding, adj_embedding], dim=1)

        # Add positional encoding to the combined embeddings
        src = self.encoder_positional_encoding(combined_embedding)  # Shape: (batch_size, 1, d_model)

        # Pass through encoder layers
        for layer in self.encoder:
            src = layer(src)  # Shape: (batch_size, 1, d_model)

        return src, curve_embedding, adj_embedding

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # Embed the target sequence and add positional encoding
        tgt = self.tgt_embed(tgt)
        tgt = self.decoder_positional_encoding(tgt)

        # Apply decoder layers
        for layer in self.decoder:
            tgt = layer(tgt, encoder_output, src_mask, tgt_mask)

        return tgt
    
    def forward(self, decoder_input, decoder_mask, curve_data, adj_data):
        # Encode source input with curve and graph data
        encoder_output, curve_embedding, adj_embedding = self.encode(curve_data, adj_data)
        
        # Decoder pass
        decoder_output = self.decode(
            encoder_output, 
            None, 
            decoder_input, 
            decoder_mask
        )
        
        decoder_output = self.projection_norm(decoder_output)
        final_output = self.projection(decoder_output)

        return final_output, curve_embedding, adj_embedding
