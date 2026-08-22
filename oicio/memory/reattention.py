"""
OICIO Memory Fabric: ReAttention - Training-Free Infinite Context with Finite Attention Scope
Credits: deepRcurs Labs, @deeprcurs / Mzed Imamkh @mzedimamkh

Berdasarkan:
- ReAttention paper 2407.15176v3
- 3 syarat infinite context: pos emb not OOD, stable entropy, effective awareness

Core:
1. Position-agnostic top-k attention BEFORE position-aware attention
2. q_t * K_middle^T tanpa RoPE untuk cari critical info
3. Concat [K_global 32 + K_select 127*32 + K_local 4096] = 8192 max, baru kasih RoPE
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

class ReAttention:
    """
    ReAttention: finite attention scope, infinite context
    """
    def __init__(self,
                 global_tokens: int = 32,
                 local_tokens: int = 4096,
                 select_span: int = 32,
                 top_k: int = 4,
                 top_k_prime: int = 127):
        self.global_tokens = global_tokens
        self.local_tokens = local_tokens
        self.select_span = select_span
        self.top_k = top_k
        self.top_k_prime = top_k_prime

        # Max attention scope = global + local + k' * span
        self.max_scope = global_tokens + local_tokens + top_k_prime * select_span
        print(f"[ReAttention] Max attention scope: {self.max_scope} (global={global_tokens} + local={local_tokens} + {top_k_prime}*{select_span})")

    def split_cache(self, kv_cache: np.ndarray):
        """
        Split KV cache into global, middle, local
        kv_cache: [seq_len, dim]
        """
        seq_len = kv_cache.shape[0]

        if seq_len <= self.global_tokens + self.local_tokens:
            # Not enough to split, return all as local
            return kv_cache[:0], kv_cache, kv_cache[:0]

        global_part = kv_cache[:self.global_tokens]
        local_part = kv_cache[-self.local_tokens:]
        middle_part = kv_cache[self.global_tokens:-self.local_tokens]

        return global_part, middle_part, local_part

    def position_agnostic_selection(self, 
                                   query: np.ndarray, 
                                   middle_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Position-agnostic top-k selection
        query: [dim] or [1, dim]
        middle_k: [middle_len, dim]
        Returns: indices of top-k' spans and selected K/V

        Real paper uses Triton kernel fused top-k
        POC uses numpy
        """
        if query.ndim == 1:
            query = query[None, :]

        # Dot product without position embedding
        # q_t * K_middle^T
        scores = query @ middle_k.T  # [1, middle_len]
        scores = scores[0]  # [middle_len]

        # Top-k' selection (k' = 127)
        # But we also want to consider multiple heads/queries voting
        # For POC, simple top-k

        # To ensure semantic coherence, not only top-k' elements but also m neighboring entries
        # Overlapping parts deduplicated
        top_indices = np.argsort(scores)[-self.top_k_prime*2:][::-1]  # get more, then dedup spans

        # Expand each index to span
        selected_indices = set()
        for idx in top_indices:
            start = max(0, idx - self.select_span//2)
            end = min(len(middle_k), start + self.select_span)
            for j in range(start, end):
                selected_indices.add(j)
            if len(selected_indices) >= self.top_k_prime * self.select_span:
                break

        selected_indices = sorted(list(selected_indices))[:self.top_k_prime * self.select_span]

        return np.array(selected_indices), scores[selected_indices] if len(selected_indices) > 0 else np.array([])

    def reconstruct_cache(self,
                         global_k: np.ndarray,
                         select_k: np.ndarray,
                         local_k: np.ndarray,
                         global_v: np.ndarray = None,
                         select_v: np.ndarray = None,
                         local_v: np.ndarray = None):
        """
        Concatenate selected segments between global and local
        Preserves relative order while ignoring absolute distance
        Then apply RoPE sequentially (simulated)
        """
        # Concatenate K
        if len(select_k) > 0:
            k_concat = np.concatenate([global_k, select_k, local_k], axis=0)
        else:
            k_concat = np.concatenate([global_k, local_k], axis=0)

        # V similarly
        if global_v is not None:
            if select_v is not None and len(select_v) > 0:
                v_concat = np.concatenate([global_v, select_v, local_v], axis=0)
            else:
                v_concat = np.concatenate([global_v, local_v], axis=0)
        else:
            v_concat = None

        # Apply position embedding sequentially (RoPE simulation)
        # In real ReAttention, PE is separated from KV cache and performed AFTER selection
        # This ensures PE never OOD because concat length <= pretrain window
        # For POC, we just return concat, PE would be applied in attention

        return k_concat, v_concat

    def forward(self, query: np.ndarray, kv_cache: np.ndarray, v_cache: np.ndarray = None):
        """
        Full ReAttention forward
        query: [dim] current query
        kv_cache: [seq_len, dim] full cache
        v_cache: [seq_len, dim] optional V cache

        Returns: selected K,V for attention
        """
        global_k, middle_k, local_k = self.split_cache(kv_cache)

        if v_cache is not None:
            global_v, middle_v, local_v = self.split_cache(v_cache)
        else:
            global_v, middle_v, local_v = None, None, None
            middle_v = middle_k  # for simplicity

        if len(middle_k) == 0:
            # No middle, just global + local
            k_concat = np.concatenate([global_k, local_k], axis=0) if len(global_k) > 0 else local_k
            v_concat = np.concatenate([global_v, local_v], axis=0) if global_v is not None and len(global_v) > 0 else local_v
            return k_concat, v_concat, np.array([])

        # Position-agnostic selection
        select_indices, select_scores = self.position_agnostic_selection(query, middle_k)

        if len(select_indices) > 0:
            select_k = middle_k[select_indices]
            select_v = middle_v[select_indices] if middle_v is not None else select_k
        else:
            select_k = np.zeros((0, kv_cache.shape[1]))
            select_v = np.zeros((0, kv_cache.shape[1])) if v_cache is not None else None

        # Reconstruct
        k_final, v_final = self.reconstruct_cache(global_k, select_k, local_k, global_v, select_v, local_v)

        return k_final, v_final, select_indices

    def attention(self, query: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray):
        """
        Self-attention with selected cache
        query: [dim]
        k_cache, v_cache: selected caches [selected_len, dim]
        """
        # Apply RoPE sequentially (simulate as no-op for POC, but ensure length within window)
        assert len(k_cache) <= self.max_scope, f"Cache {len(k_cache)} exceeds max scope {self.max_scope}, would be OOD!"

        # Standard attention
        scores = query @ k_cache.T  # [selected_len]
        scores = scores / np.sqrt(query.shape[0])
        attn_weights = np.exp(scores - np.max(scores))
        attn_weights = attn_weights / np.sum(attn_weights)

        # Output
        out = attn_weights @ v_cache  # [dim]

        return out, attn_weights

# Demo
if __name__ == "__main__":
    print("=== ReAttention POC ===")
    dim = 64
    seq_len = 100000  # 100K context

    # Simulate KV cache
    kv_cache = np.random.randn(seq_len, dim).astype(np.float32)
    v_cache = np.random.randn(seq_len, dim).astype(np.float32)

    # Current query
    query = np.random.randn(dim).astype(np.float32)

    reatt = ReAttention(global_tokens=32, local_tokens=128, select_span=32, top_k_prime=10)  # small for POC

    k_final, v_final, indices = reatt.forward(query, kv_cache, v_cache)

    print(f"Original cache: {seq_len}")
    print(f"Selected cache: {len(k_final)} (global 32 + select {len(indices)} + local 128)")
    print(f"Compression: {seq_len} -> {len(k_final)} = {seq_len/len(k_final):.1f}x")
    print(f"Within max scope {reatt.max_scope}? {len(k_final) <= reatt.max_scope}")

    out, weights = reatt.attention(query, k_final, v_final)
    print(f"Attention output shape: {out.shape}")
    print(f"Attention entropy: {-np.sum(weights * np.log(weights + 1e-8)):.3f} (should be stable, not grow with seq_len)")
