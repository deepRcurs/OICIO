"""
OICIO Memory Fabric: TurboQuant - Data-Oblivious Vector Quantization
Credits: deepRcurs Labs, @deeprcurs / Mzed Imamkh @mzedimamkh

Berdasarkan:
- Google Research TurboQuant (ICLR 2026) - paper 2504.19874
- RyanCodrai/turbovec - Rust implementation

Core idea:
1. Normalize vectors to unit hypersphere, store norm as float
2. Random orthogonal rotation (Walsh-Hadamard) -> predictable Beta -> Gaussian distribution
3. Lloyd-Max scalar quantization to 2-4 bit per coordinate
4. Bit-packing for compression
5. Search: rotate query once, score directly against quantized codes via SIMD

POC Python version, no Rust, but same math.
"""

import numpy as np
import math
from typing import Tuple

class TurboQuant:
    """
    Data-oblivious quantizer: no training, no codebook calibration
    """
    def __init__(self, dim: int, bit_width: int = 4, use_hadamard: bool = True):
        assert bit_width in [2, 3, 4, 8], "bit_width must be 2,3,4,8"
        self.dim = dim
        self.bit_width = bit_width
        self.num_levels = 2 ** bit_width
        self.use_hadamard = use_hadamard

        # Random orthogonal rotation matrix (fixed, data-oblivious)
        # For POC, use random Gaussian then QR decomposition to get orthogonal
        # Real TurboQuant uses Walsh-Hadamard + random diagonal
        np.random.seed(42)  # deterministic for reproducibility
        if use_hadamard:
            # Approximate Hadamard-like rotation via random orthogonal
            rand_mat = np.random.randn(dim, dim).astype(np.float32)
            q, _ = np.linalg.qr(rand_mat)
            self.rotation = q.astype(np.float32)  # [dim, dim]
        else:
            self.rotation = np.eye(dim, dtype=np.float32)

        # Lloyd-Max quantizer for Gaussian distribution
        # For Gaussian N(0,1), optimal quantization boundaries
        # We precompute for 2-bit and 4-bit
        self.codebook = self._build_lloyd_max_codebook()

        self.compressed = None
        self.norms = None
        self.num_vectors = 0

    def _build_lloyd_max_codebook(self):
        """Build Lloyd-Max codebook for Gaussian distribution"""
        # For POC, use simple uniform quant for Gaussian with known variance
        # Real Lloyd-Max would iterate, but we approximate
        if self.bit_width == 2:
            # 4 levels for Gaussian: approx -1.5, -0.5, 0.5, 1.5 (scaled)
            # These are optimal for Gaussian with 2-bit
            return np.array([-1.510, -0.4528, 0.4528, 1.510], dtype=np.float32)
        elif self.bit_width == 4:
            # 16 levels uniform in range [-2, 2] for POC
            # Real would be non-uniform Lloyd-Max
            return np.linspace(-2.0, 2.0, 16, dtype=np.float32)
        elif self.bit_width == 3:
            return np.linspace(-2.0, 2.0, 8, dtype=np.float32)
        else:  # 8-bit
            return np.linspace(-3.0, 3.0, 256, dtype=np.float32)

    def _hadamard_transform_numpy(self, x: np.ndarray) -> np.ndarray:
        """Fast Walsh-Hadamard Transform for numpy"""
        # x: [N, D] or [D]
        # For simplicity, use matrix multiplication with rotation (already orthogonal)
        # Real FWHT is O(n log n) without matrix
        return x @ self.rotation

    def compress(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compress vectors
        Input: [N, D] float32
        Output: compressed codes [N, D] uint8 + norms [N]
        """
        assert vectors.shape[1] == self.dim
        N = vectors.shape[0]

        # 1. Store L2 norm
        norms = np.linalg.norm(vectors, axis=1).astype(np.float32)  # [N]
        # Avoid div by zero
        norms = np.maximum(norms, 1e-8)

        # 2. Normalize to unit sphere
        normalized = vectors / norms[:, None]  # [N, D]

        # 3. Random orthogonal rotation -> makes coordinates ~ Gaussian
        rotated = normalized @ self.rotation  # [N, D]

        # 4. Lloyd-Max scalar quantization per coordinate
        # Quantize each coordinate to nearest codebook entry
        # For POC, simple nearest neighbor
        # Expand for broadcasting: [N, D, 1] vs [num_levels]
        # Use vectorized search
        quantized_indices = np.zeros((N, self.dim), dtype=np.uint8)

        # For each level, compute distance (could be optimized)
        # For 4-bit, 16 levels
        for i in range(N):
            # For each vector, quantize each dim
            # Use digitize or argmin
            # Reshape for broadcasting
            diff = np.abs(rotated[i, :, None] - self.codebook[None, :])  # [D, L]
            indices = np.argmin(diff, axis=1)  # [D]
            quantized_indices[i] = indices.astype(np.uint8)

        # 5. Bit-packing (for POC, keep as uint8 indices, real would pack bits)
        # 4-bit: 2 indices per byte, 2-bit: 4 indices per byte
        # For simplicity, we store as uint8 but calculate compression ratio as if packed

        self.compressed = quantized_indices
        self.norms = norms
        self.num_vectors = N

        return quantized_indices, norms

    def decompress(self, indices: np.ndarray = None, norms: np.ndarray = None) -> np.ndarray:
        """Decompress back to approximate vectors"""
        if indices is None:
            indices = self.compressed
        if norms is None:
            norms = self.norms

        # Dequantize
        dequant = self.codebook[indices]  # [N, D]

        # Inverse rotation
        # Since rotation is orthogonal, inverse = transpose
        unrotated = dequant @ self.rotation.T  # [N, D]

        # Restore norm
        reconstructed = unrotated * norms[:, None]

        return reconstructed.astype(np.float32)

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search: rotate query once, score directly against quantized codes
        No decompression of database vectors needed for scoring (SIMD friendly)
        """
        # query: [1, D] or [D]
        if query.ndim == 1:
            query = query[None, :]

        # Normalize query
        q_norm = np.linalg.norm(query, axis=1, keepdims=True)
        q_norm = np.maximum(q_norm, 1e-8)
        q_normalized = query / q_norm

        # Rotate query once
        q_rotated = q_normalized @ self.rotation  # [1, D]

        # Dequantize database for scoring (in real turbovec, scoring directly against codes via LUT)
        # For POC, decompress
        db_dequant = self.codebook[self.compressed]  # [N, D]

        # Cosine similarity (since both normalized and rotated, dot product = cosine)
        scores = q_rotated @ db_dequant.T  # [1, N]
        scores = scores[0]  # [N]

        # Top-k
        top_k_idx = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_idx]

        return top_k_scores, top_k_idx

    def get_compression_stats(self, num_vectors: int = None):
        """Hitung kompresi"""
        if num_vectors is None:
            num_vectors = self.num_vectors

        fp32_size = num_vectors * self.dim * 4  # 4 bytes per float32
        # Packed size: bit_width bits per coordinate + 4 bytes for norm per vector
        packed_bits = num_vectors * self.dim * self.bit_width
        packed_bytes = packed_bits // 8
        norms_bytes = num_vectors * 4
        total_packed = packed_bytes + norms_bytes

        return {
            "num_vectors": num_vectors,
            "dim": self.dim,
            "bit_width": self.bit_width,
            "fp32_bytes": fp32_size,
            "fp32_mb": fp32_size / 1024 / 1024,
            "packed_bytes": total_packed,
            "packed_mb": total_packed / 1024 / 1024,
            "compression_ratio": fp32_size / total_packed if total_packed > 0 else 0,
            "example": f"{fp32_size/1024/1024:.1f}MB -> {total_packed/1024/1024:.1f}MB ({fp32_size/total_packed:.1f}x)"
        }

# Demo / test
if __name__ == "__main__":
    print("=== TurboQuant POC ===")
    dim = 128
    n_docs = 10000

    # Simulate embeddings
    vectors = np.random.randn(n_docs, dim).astype(np.float32)

    for bw in [2, 4]:
        tq = TurboQuant(dim=dim, bit_width=bw)
        codes, norms = tq.compress(vectors)
        stats = tq.get_compression_stats()
        print(f"\n{bw}-bit: {stats['example']}")

        # Test search
        query = np.random.randn(dim).astype(np.float32)
        scores, indices = tq.search(query, k=5)
        print(f"Top-5 scores: {scores[:3]}...")

        # Test reconstruction error
        recon = tq.decompress()
        mse = np.mean((vectors - recon) ** 2)
        print(f"MSE reconstruction: {mse:.6f}")
