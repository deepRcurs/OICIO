"""
OICIO Core: Ternary Simple Attention Network
Credits: deepRcurs Labs, @deeprcurs / Mzed Imamkh, @mzedimamkh

Menggabungkan:
- BitNet b1.58: ternary {-1,0,1} absmean quantization
- Needle2 Simple Attention Network: Hadamard MLP + Engram + Multi-lane hyper-connections
- Sandwich norm + gated residuals

Toy version: 2 layers, d=128, bisa jalan di CPU 1.9GB RAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def hadamard_transform(x):
    """Fast Walsh-Hadamard Transform (FWHT) - fixed matrix, no weights, O(n log n)
    Dari Needle2: orthonormal Walsh-Hadamard transform
    Preserves leading dims, operates on last dim
    Correct implementation like Rust: butterfly with add/sub only
    """
    orig_shape = x.shape
    n = orig_shape[-1]
    # Clone to avoid in-place modification of original
    x_2d = x.reshape(-1, n).clone()
    batch = x_2d.shape[0]

    # pad to power of 2 if needed
    if n & (n-1) != 0:
        next_pow2 = 1 << (n-1).bit_length()
        pad = next_pow2 - n
        x_2d = F.pad(x_2d, (0, pad))
        n_padded = next_pow2
    else:
        n_padded = n
        pad = 0

    # Correct FWHT like Rust: iterative butterfly
    h = 1
    while h < n_padded:
        for i in range(0, n_padded, h*2):
            for j in range(h):
                a = x_2d[:, i+j].clone()
                b = x_2d[:, i+j+h].clone()
                x_2d[:, i+j] = a + b
                x_2d[:, i+j+h] = a - b
        h *= 2

    x_2d = x_2d / math.sqrt(n_padded)

    if pad > 0:
        x_2d = x_2d[:, :n]

    return x_2d.view(orig_shape)

class BitLinear(nn.Module):
    """BitNet b1.58 BitLinear: ternary weights {-1,0,1} via absmean
    Reference: https://github.com/microsoft/BitNet
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Full precision shadow weights for training, quantized on forward
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # Activation quant to 8-bit (BitNet a4.8 target)
        self.activation_bits = 8

    def absmean_quant(self, w):
        """Quantize to {-1,0,1} via absmean"""
        # scale = 1 / mean(abs(w))
        scale = w.abs().mean().clamp(min=1e-5)
        w_scaled = w / scale
        # Round to nearest in {-1,0,1}
        w_ternary = w_scaled.round().clamp(-1, 1)
        return w_ternary, scale

    def forward(self, x):
        # Weight ternary
        w_ternary, w_scale = self.absmean_quant(self.weight)
        # Activation 8-bit quant (simulated)
        # x_quant = quantize activation to 8-bit
        # For POC, use simple scaling
        x_scale = x.abs().max().clamp(min=1e-5) / 127.0
        x_q = (x / x_scale).round().clamp(-128, 127) * x_scale

        # Matmul becomes addition only: since w in {-1,0,1}, it's sum/subtract
        # We simulate with standard matmul but with ternary weights for correctness
        # In real hardware, this would be INT8 add only, no multiplication
        out = F.linear(x_q, w_ternary * w_scale)
        if self.bias is not None:
            out = out + self.bias
        return out

class HadamardMLP(nn.Module):
    """Needle2 style Hadamard MLP: replaces FFN
    x_hat is RMSNorm(flatten(4 residual streams)), H is fixed Walsh-Hadamard
    """
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.dim = dim
        self.hidden_dim = hidden_dim
        # Gate projections are ternary
        self.gate_proj = BitLinear(dim, hidden_dim)
        self.up_proj = BitLinear(dim, hidden_dim)
        self.down_proj = BitLinear(hidden_dim, dim)
        self.rms_norm = nn.RMSNorm(dim)

    def forward(self, x):
        # x: [B, S, D]
        residual = x
        x_norm = self.rms_norm(x)
        # Hadamard transform (fixed, no weights)
        x_h = hadamard_transform(x_norm)
        # Gated
        gate = F.silu(self.gate_proj(x_h))
        up = self.up_proj(x_h)
        x = gate * up
        x = self.down_proj(x)
        # Sandwich norm + gated residual (Needle2 style)
        return residual + x * 0.5

class EngramMemory(nn.Module):
    """Needle2 Engram: hashed n-gram tables as key-value memory
    Innovation OICIO: Surprise-Gated Engram - only fires on high surprise
    """
    def __init__(self, dim, num_engrams=1024, ngram=3):
        super().__init__()
        self.dim = dim
        self.num_engrams = num_engrams
        self.ngram = ngram
        # Hashed tables: k_t, v_t rows gathered from hashed n-gram tables
        self.engram_k = nn.Embedding(num_engrams, dim)
        self.engram_v = nn.Embedding(num_engrams, dim)
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x, surprise_mask=None):
        # x: [B, S, D]
        # Simple hash: sum of token ids mod num_engrams (for POC, use random hash from x)
        B, S, D = x.shape
        # Hash from x mean
        hash_ids = (x.mean(dim=-1) * 1000).long() % self.num_engrams
        hash_ids = hash_ids.clamp(0, self.num_engrams-1)

        k = self.engram_k(hash_ids)  # [B, S, D]
        v = self.engram_v(hash_ids)

        # Attention over engram
        scores = (x * k).sum(dim=-1, keepdim=True) / math.sqrt(D)  # [B, S, 1]
        # Surprise gating: if surprise_mask provided, only fire where surprise high
        if surprise_mask is not None:
            # surprise_mask: [B, S] bool
            gate_factor = surprise_mask.float().unsqueeze(-1)  # [B, S, 1]
            scores = scores * gate_factor

        out = torch.sigmoid(scores) * v
        # Input-dependent gating
        out = out * torch.sigmoid(self.gate)
        return out

class TernarySANBlock(nn.Module):
    """Single OICIO block: Attention + HadamardMLP + Engram + Hyper-connections"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = BitLinear(dim, dim)
        self.k_proj = BitLinear(dim, dim)
        self.v_proj = BitLinear(dim, dim)
        self.o_proj = BitLinear(dim, dim)

        self.rms_norm1 = nn.RMSNorm(dim)
        self.rms_norm2 = nn.RMSNorm(dim)

        self.mlp = HadamardMLP(dim)
        self.engram = EngramMemory(dim)

        # Multi-lane hyper-connections: 4 residual streams (Needle2)
        self.lane_weights = nn.Parameter(torch.ones(4) / 4)

    def forward(self, x, surprise_mask=None):
        B, S, D = x.shape
        residual = x

        # Pre-norm
        x_norm = self.rms_norm1(x)

        # Ternary QKV
        Q = self.q_proj(x_norm).view(B, S, self.num_heads, self.head_dim).transpose(1,2)
        K = self.k_proj(x_norm).view(B, S, self.num_heads, self.head_dim).transpose(1,2)
        V = self.v_proj(x_norm).view(B, S, self.num_heads, self.head_dim).transpose(1,2)

        # Attention (simplified, no RoPE for POC)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1,2).contiguous().view(B, S, D)
        attn_out = self.o_proj(attn_out)

        # Engram with surprise gating (OICIO Innovation #1)
        engram_out = self.engram(x_norm, surprise_mask)

        x = residual + attn_out * 0.5 + engram_out * 0.3

        # MLP with Hadamard
        x = x + self.mlp(self.rms_norm2(x)) * 0.5

        return x

class TernarySAN(nn.Module):
    """OICIO Core Model: Ternary Simple Attention Network
    Toy version: 2-4 layers, 128 dim, ~0.5M params ternary
    Real version would be 8B ternary = 1.75GB
    """
    def __init__(self, vocab_size=32000, dim=128, num_layers=2, num_heads=4, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TernarySANBlock(dim, num_heads) for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(dim)
        self.lm_head = BitLinear(dim, vocab_size, bias=False)

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, surprise_mask=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, surprise_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def count_ternary_params(self):
        """Hitung kompresi"""
        total = sum(p.numel() for p in self.parameters())
        # Ternary = 1.58 bit vs 16 bit = 10.1x compression
        fp16_size_mb = total * 2 / 1024 / 1024
        ternary_size_mb = total * 1.58 / 8 / 1024 / 1024
        return {
            "total_params": total,
            "fp16_mb": fp16_size_mb,
            "ternary_mb": ternary_size_mb,
            "compression": fp16_size_mb / ternary_size_mb if ternary_size_mb > 0 else 0
        }
