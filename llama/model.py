# Copyright (c) Meta Platforms, Inc. and affiliates.
# Llama-like transformer implementation for Windows, using pure PyTorch (no fairscale/model_parallel).
# Extensive comments included to make this a self-contained tutorial for building, understanding, and training small LLMs!

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. Model Hyperparameters Container

@dataclass
class ModelArgs:
    dim: int = 4096                # Model hidden size (embedding dim)
    n_layers: int = 32             # Number of transformer blocks
    n_heads: int = 32              # Number of attention heads
    n_kv_heads: Optional[int] = None # (not used in basic models, kept for completeness)
    vocab_size: int = -1           # Number of tokens (to be set after tokenizer is loaded)
    multiple_of: int = 256         # Make FFN size a multiple of this (efficient for hardware)
    ffn_dim_multiplier: Optional[float] = None  # Custom FFN multiplier (optional)
    norm_eps: float = 1e-5         # Epsilon for RMSNorm

    max_batch_size: int = 32       # Not used in this pure PyTorch version
    max_seq_len: int = 2048        # Max sequence length the model will see

# -----------------------------------------------------------------------------
# 2. RMSNorm (Root Mean Square Layer Norm)

class RMSNorm(nn.Module):
    """
    RMSNorm: Like LayerNorm but normalizes by RMS (root mean square), not mean/var.
    Keeps scale of activations stable, helps training.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Normalize x over last dimension (features)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Apply RMSNorm and scale
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# -----------------------------------------------------------------------------
# 3. Rotary Position Embeddings (for Llama, GPT-NeoX, etc.)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute rotary frequencies for each position and head dimension.
    Returns a (end, dim // 2) complex tensor representing rotation angles.
    """
    # log-spaced frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)  # positions
    freqs = torch.outer(t, freqs)  # (end, dim // 2)
    # Convert to polar representation (radius=1, angle=freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Prepare freqs_cis for broadcasting over the batch/head dims of x.
    """
    # For rotary embedding: freq_cis shape = (seq_len, head_dim), x shape = (batch, seq_len, n_heads, head_dim)
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings (RoPE) to query and key tensors.
    xq, xk: (batch, seq, n_heads, head_dim)
    freqs_cis: (seq, head_dim)
    """
    # View as complex numbers (each pair is real+imag part)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# -----------------------------------------------------------------------------
# 4. Multi-head Attention (Standard PyTorch)

class Attention(nn.Module):
    """
    Multi-head attention module (pure PyTorch version).
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.dim = args.dim

        # Instead of ColumnParallelLinear, just use nn.Linear and split heads manually.
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Inference-time key/value cache (optional: for training, you may omit)
        # self.cache_k = ...
        # self.cache_v = ...

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        x: (batch, seq, dim)
        start_pos: (not used for training, only for incremental generation)
        freqs_cis: (seq, head_dim)
        mask: (seq, seq) or None (causal mask)
        """
        bsz, seqlen, _ = x.shape

        # 1. Linear projections, shape (batch, seq, n_heads, head_dim)
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim)

        # 2. Apply rotary positional embedding to queries and keys
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 3. Transpose to (batch, n_heads, seq, head_dim) for attention
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 4. Attention scores: (batch, n_heads, seq, seq)
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 5. Apply mask (causal: lower triangle only)
        if mask is not None:
            scores = scores + mask  # mask should broadcast over (batch, n_heads, seq, seq)

        # 6. Softmax over "past" tokens (last dim)
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)

        # 7. Weighted sum of values: (batch, n_heads, seq, head_dim)
        attn_output = torch.matmul(attn_weights, xv)

        # 8. Merge heads back (batch, seq, n_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 9. Final output projection
        return self.wo(attn_output)

# -----------------------------------------------------------------------------
# 5. Feedforward Network (MLP/SwiGLU variant)

class FeedForward(nn.Module):
    """
    Llama MLP: SiLU activation, "gated" feedforward network.
    """
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        # For Llama, FFN size is 2/3 of 4*dim, rounded to multiple_of
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * ffn_dim_multiplier)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU: w2(silu(w1(x)) * w3(x))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# -----------------------------------------------------------------------------
# 6. Transformer Block: Attention + MLP + RMSNorm + Residuals

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        # Residual connection: output = input + F(...)
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

# -----------------------------------------------------------------------------
# 7. Top-level Transformer Language Model

class Transformer(nn.Module):
    """
    Full Llama-style transformer model.
    """
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # Token embedding layer (input)
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(i, params) for i in range(params.n_layers)
        ])

        # Final normalization and output projection to vocab
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # Precompute rotary embedding frequencies (enough for max_seq_len*2)
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        tokens: (batch, seq)
        start_pos: used for incremental decoding; for full forward, set 0
        """
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # Build (causal) attention mask: [batch, 1, seq, seq]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # for broadcasting over batch, heads

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output  # logits, shape (batch, seq, vocab_size)
