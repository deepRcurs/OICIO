"""
OICIO Memory Fabric: EM-LLM Surprise-based Event Segmentation
Credits: deepRcurs Labs, @deeprcurs / Mzed Imamkh @mzedimamkh

Berdasarkan:
- EM-LLM ICLR 2025: Human-inspired Episodic Memory for Infinite Context LLMs
- Bayesian surprise + graph-theoretic boundary refinement

POC: Implementasi surprise segmentation sederhana
"""

import numpy as np
import torch
from typing import List, Tuple

class SurpriseSegmenter:
    """
    Segment sequence into episodic events based on surprise
    """
    def __init__(self, 
                 gamma: float = 1.0,  # std scaling factor (paper: surprisal_threshold_gamma)
                 min_block_size: int = 8,
                 max_block_size: int = 128,
                 use_refinement: bool = True):
        self.gamma = gamma
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.use_refinement = use_refinement

    def compute_surprise(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute surprise per token
        For POC, we use simple methods:
        - Option 1: distance to previous token (prediction error proxy)
        - Option 2: -log prob proxy via embedding norm change

        Real EM-LLM uses LLM's own next-token prediction loss

        Input: [seq_len, dim]
        Output: [seq_len] surprise scores
        """
        seq_len, dim = embeddings.shape
        surprise = np.zeros(seq_len)

        # Surprise as L2 distance to previous token (simple proxy)
        # High distance = high prediction error = event boundary
        for i in range(1, seq_len):
            dist = np.linalg.norm(embeddings[i] - embeddings[i-1])
            surprise[i] = dist

        # Also add norm change surprise
        norms = np.linalg.norm(embeddings, axis=1)
        norm_change = np.abs(np.diff(norms, prepend=norms[0]))
        surprise = surprise * 0.7 + norm_change * 0.3

        return surprise

    def initial_segmentation(self, surprise: np.ndarray) -> List[int]:
        """
        Initial segmentation via surprise threshold
        Returns list of boundary indices
        """
        # Threshold = mean + gamma * std (from paper)
        mean_surprise = np.mean(surprise)
        std_surprise = np.std(surprise)
        threshold = mean_surprise + self.gamma * std_surprise

        boundaries = [0]  # start
        current_block_size = 0

        for i, s in enumerate(surprise):
            current_block_size += 1

            # If surprise high and block size >= min, create boundary
            if s > threshold and current_block_size >= self.min_block_size:
                boundaries.append(i)
                current_block_size = 0
            # If block too big, force split
            elif current_block_size >= self.max_block_size:
                boundaries.append(i)
                current_block_size = 0

        boundaries.append(len(surprise))  # end
        return sorted(list(set(boundaries)))

    def refinement(self, embeddings: np.ndarray, boundaries: List[int]) -> List[int]:
        """
        Graph-theoretic boundary refinement (simplified)
        Real paper uses modularity/conductance optimization

        POC: Use similarity within vs across blocks
        """
        if not self.use_refinement or len(boundaries) <= 2:
            return boundaries

        # Compute similarity matrix (cosine)
        # For POC, use small window refinement
        refined = [boundaries[0]]
        
        for i in range(1, len(boundaries)-1):
            prev_bound = boundaries[i-1]
            curr_bound = boundaries[i]
            next_bound = boundaries[i+1]

            # Current block: [prev_bound, curr_bound)
            # Next block: [curr_bound, next_bound)
            # Try shifting boundary by +/- min_block_size//2 to maximize cohesion

            best_boundary = curr_bound
            best_score = -1

            # Search window
            search_start = max(prev_bound + self.min_block_size, curr_bound - self.min_block_size//2)
            search_end = min(next_bound - self.min_block_size, curr_bound + self.min_block_size//2)

            for candidate in range(search_start, search_end+1):
                # Compute within-block similarity vs cross-block
                block1 = embeddings[prev_bound:candidate]
                block2 = embeddings[candidate:next_bound]

                if len(block1) == 0 or len(block2) == 0:
                    continue

                # Within similarity (cohesion)
                # Mean pairwise cosine within block1 + within block2
                def mean_sim(block):
                    if len(block) <= 1:
                        return 0
                    # Normalize
                    normed = block / (np.linalg.norm(block, axis=1, keepdims=True) + 1e-8)
                    sim_matrix = normed @ normed.T
                    # Upper triangle mean
                    triu = np.triu(sim_matrix, k=1)
                    return np.mean(triu[triu != 0]) if np.any(triu != 0) else 0

                within1 = mean_sim(block1)
                within2 = mean_sim(block2)

                # Cross similarity (separation) - should be low
                normed1 = block1 / (np.linalg.norm(block1, axis=1, keepdims=True) + 1e-8)
                normed2 = block2 / (np.linalg.norm(block2, axis=1, keepdims=True) + 1e-8)
                cross = np.mean(normed1 @ normed2.T) if len(block1) > 0 and len(block2) > 0 else 0

                # Modularity-like score: within - cross
                score = (within1 + within2) - 2 * cross

                if score > best_score:
                    best_score = score
                    best_boundary = candidate

            refined.append(best_boundary)

        refined.append(boundaries[-1])
        return sorted(list(set(refined)))

    def segment(self, embeddings: np.ndarray) -> Tuple[List[int], np.ndarray, List[Tuple[int,int]]]:
        """
        Full segmentation pipeline
        Returns: boundaries, surprise scores, blocks as (start, end) tuples
        """
        surprise = self.compute_surprise(embeddings)
        boundaries = self.initial_segmentation(surprise)
        if self.use_refinement:
            boundaries = self.refinement(embeddings, boundaries)

        blocks = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

        return boundaries, surprise, blocks

    def get_representative_tokens(self, embeddings: np.ndarray, blocks: List[Tuple[int,int]], topk: int = 4) -> List[np.ndarray]:
        """
        Get representative tokens per block (like InfLLM/EM-LLM)
        Select topk tokens with highest influence (e.g., highest norm or attention score proxy)
        """
        representatives = []
        for start, end in blocks:
            block_emb = embeddings[start:end]
            if len(block_emb) == 0:
                continue
            # For POC: select tokens with highest L2 norm as most influential
            norms = np.linalg.norm(block_emb, axis=1)
            topk_idx = np.argsort(norms)[-topk:][::-1]
            rep = block_emb[topk_idx]
            representatives.append(rep)

        return representatives

# Demo
if __name__ == "__main__":
    print("=== EM-LLM Surprise Segmentation POC ===")
    seq_len = 1000
    dim = 64

    # Simulate document with 3 topics (events)
    # Topic 1: embeddings around [1,0,0...], Topic 2: [0,1,0...], Topic 3: [0,0,1...]
    embeddings = []
    for i in range(seq_len):
        if i < 300:
            emb = np.random.randn(dim) * 0.1
            emb[0] += 2.0
        elif i < 700:
            emb = np.random.randn(dim) * 0.1
            emb[1] += 2.0
        else:
            emb = np.random.randn(dim) * 0.1
            emb[2] += 2.0
        embeddings.append(emb)
    embeddings = np.array(embeddings)

    segmenter = SurpriseSegmenter(gamma=1.0, min_block_size=8, max_block_size=128)
    boundaries, surprise, blocks = segmenter.segment(embeddings)

    print(f"Found {len(blocks)} events, boundaries: {boundaries[:10]}...")
    print(f"Surprise stats: mean={np.mean(surprise):.3f}, std={np.std(surprise):.3f}, max={np.max(surprise):.3f}")
    print(f"Block sizes: {[end-start for start,end in blocks[:5]]}...")

    reps = segmenter.get_representative_tokens(embeddings, blocks, topk=4)
    print(f"Representative tokens per block: {[r.shape for r in reps[:3]]}")
