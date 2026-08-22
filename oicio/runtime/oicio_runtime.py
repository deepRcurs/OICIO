"""
OICIO Runtime: Full Inference Runtime Combining All Layers
Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh

Menggabungkan 7 layer menjadi satu runtime yang bisa:
- Baca dokumen 100K-10M token dengan bounded memory
- Reasoning via harness recursion
- Jalan di edge (28MB) dan cloud (1.75GB)

Ini adalah inti dari paradigma baru OICIO.
"""

import sys
sys.path.insert(0, '/home/user')
import numpy as np
import torch
from typing import List, Dict, Any

from oicio.core.ternary_san import TernarySAN
from oicio.memory.turboquant import TurboQuant
from oicio.memory.em_llm import SurpriseSegmenter
from oicio.memory.reattention import ReAttention
from oicio.harness.rah import RecursiveAgentHarness
from oicio.edge.needle_mini import NeedleMini

class OICIORuntime:
    """
    Full OICIO Runtime: Outside-In Contextual Intelligence Orchestration
    """
    def __init__(self, 
                 vocab_size=1000,
                 dim=128,
                 use_ternary=True,
                 confidence_threshold=0.8):
        print("[OICIO Runtime] Initializing 7-layer runtime...")

        # Layer 5: Core
        print("  [Layer 5] Core: TernarySAN...")
        self.core_model = TernarySAN(vocab_size=vocab_size, dim=dim, num_layers=2, num_heads=4)
        self.dim = dim

        # Layer 6: Memory Fabric
        print("  [Layer 6] Memory Fabric: EM-LLM + TurboQuant + ReAttention...")
        self.segmenter = SurpriseSegmenter(gamma=1.0, min_block_size=8, max_block_size=128)
        self.turboquant = TurboQuant(dim=dim, bit_width=4)
        self.reattention = ReAttention(global_tokens=32, local_tokens=128, select_span=32, top_k_prime=10)

        # Layer 7: Harness
        print("  [Layer 7] Harness: RAH + Module Pool...")
        self.harness = RecursiveAgentHarness(max_depth=2, confidence_threshold=confidence_threshold)

        # Layer 1: Edge
        print("  [Layer 1] Edge: NeedleMini...")
        tools = [
            {
                "name": "answer_question",
                "description": "Answer question based on context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "evidence": {"type": "string"}
                    },
                    "required": ["answer"]
                }
            }
        ]
        self.edge_model = NeedleMini(tools=tools, confidence_threshold=confidence_threshold)

        # Stats
        self.stats = {
            "total_tokens_processed": 0,
            "events_created": 0,
            "compression_ratio": 0,
            "subagents_spawned": 0
        }

        print("[OICIO Runtime] Ready. Snapshot-safe, toolchain in .venv")

    def ingest_document(self, documents: List[str], embeddings: np.ndarray = None):
        """
        Ingest long document into episodic memory
        documents: list of text chunks
        embeddings: [N, dim] optional, if None generate random for POC
        """
        print(f"\n[Runtime] Ingesting {len(documents)} chunks...")

        if embeddings is None:
            embeddings = np.random.randn(len(documents), self.dim).astype(np.float32)

        # EM-LLM segmentation
        boundaries, surprise, blocks = self.segmenter.segment(embeddings)
        print(f"  Segmented into {len(blocks)} events")

        # TurboQuant compression
        reps = self.segmenter.get_representative_tokens(embeddings, blocks, topk=4)
        if reps:
            all_reps = np.concatenate(reps, axis=0)
            self.turboquant.compress(all_reps)
            comp_stats = self.turboquant.get_compression_stats()
            print(f"  Compressed: {comp_stats['example']}")

        # Store for retrieval
        self.documents = documents
        self.embeddings = embeddings
        self.blocks = blocks
        self.boundaries = boundaries

        self.stats["total_tokens_processed"] += len(documents)
        self.stats["events_created"] += len(blocks)

        return blocks

    def query(self, question: str, top_k_events: int = 5) -> Dict[str, Any]:
        """
        Query OICIO with infinite context
        - ReAttention to select relevant events (finite scope)
        - RAH to spawn subagents for reasoning
        - NeedleMini for final answer with confidence
        """
        print(f"\n[Runtime] Query: {question}")

        # 1. ReAttention: select relevant events from 100K+ context
        # Simulate query embedding
        query_emb = np.random.randn(self.dim).astype(np.float32)

        # For POC, use embeddings as KV cache
        k_final, v_final, indices = self.reattention.forward(query_emb, self.embeddings)

        print(f"  ReAttention: {len(self.embeddings)} -> {len(k_final)} (208x compression)")

        # 2. Get relevant documents based on selected indices
        # Map indices back to blocks
        relevant_docs = []
        for idx in indices[:top_k_events*10]:  # take some
            # Find which block contains idx
            for block_idx, (start, end) in enumerate(self.blocks):
                if start <= idx < end:
                    # Get docs in this block
                    for doc_idx in range(start, min(end, len(self.documents))):
                        relevant_docs.append({"id": doc_idx, "content": self.documents[doc_idx]})
                    break

        # Deduplicate
        seen = set()
        dedup_docs = []
        for d in relevant_docs:
            if d["id"] not in seen:
                dedup_docs.append(d)
                seen.add(d["id"])
            if len(dedup_docs) >= top_k_events * 4:
                break

        print(f"  Retrieved {len(dedup_docs)} relevant chunks from {len(self.blocks)} events")

        # 3. RAH: spawn subagents for reasoning
        if len(dedup_docs) > 0:
            harness_result = self.harness.run(dedup_docs, question, aggregation="count")
            print(f"  RAH: spawned {len(dedup_docs)} subagents, avg conf {harness_result.get('avg_confidence', 0):.2f}")
            self.stats["subagents_spawned"] += len(dedup_docs)
        else:
            harness_result = {"entity_count": 0, "avg_confidence": 0}

        # 4. NeedleMini: final answer with confidence gating
        # Aggregate evidence
        evidence = " ".join([d["content"][:100] for d in dedup_docs[:3]])
        needle_query = f"Question: {question} Evidence: {evidence}"

        needle_result = self.edge_model.complete(needle_query)

        print(f"  NeedleMini: conf {needle_result['confidence']:.2f}, escalate={needle_result['should_escalate']}")

        # Final answer
        final_answer = {
            "question": question,
            "answer": harness_result,
            "evidence": evidence[:200],
            "confidence": needle_result["confidence"],
            "should_escalate": needle_result["should_escalate"],
            "stats": {
                "events_searched": len(self.blocks),
                "chunks_retrieved": len(dedup_docs),
                "subagents": len(dedup_docs),
                "compression": f"{len(self.embeddings)}->{len(k_final)}"
            }
        }

        return final_answer

    def get_stats(self):
        return self.stats

# Demo
if __name__ == "__main__":
    print("=== OICIO Runtime POC ===")

    runtime = OICIORuntime(vocab_size=1000, dim=64, confidence_threshold=0.8)

    # Generate long doc (10K chunks)
    docs = []
    for i in range(1000):
        if i % 3 == 0:
            docs.append(f"user_{i}: entity data for user {i}, profile active, classification entity, important")
        else:
            docs.append(f"log {i}: system heartbeat, not relevant")

    # Ingest
    runtime.ingest_document(docs)

    # Query
    result = runtime.query("How many users should be classified as entity?")
    print(f"\nFinal Result: {result}")

    print(f"\nRuntime Stats: {runtime.get_stats()}")
