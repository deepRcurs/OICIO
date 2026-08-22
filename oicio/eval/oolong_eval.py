"""
OICIO Eval: OOLONG Benchmark Evaluation
Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh

Berdasarkan:
- OOLONG: Order-Oriented Long-Context benchmark
- Oolong-Synthetic: 199 samples, 13 buckets 1K-4M tokens
- Task: semantic aggregation across thousands of entries, not needle retrieval

Evaluasi OICIO vs baselines:
- Full-context baseline: 59.22%
- RLM: 64.38%
- Codex: 71.75%
- RAH GPT-5: 81.36%
- RAH Sonnet 4.5: 89.77%

Target OICIO 8B: 78-80% with 1.75GB
"""

import sys
sys.path.insert(0, '/home/user')
import numpy as np
import json
from typing import List, Dict

from oicio.runtime.oicio_runtime import OICIORuntime

class OOLONGEval:
    def __init__(self):
        self.buckets = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1000000, 2000000, 4000000]
        self.results = []

    def generate_synthetic_sample(self, context_length: int, num_entries: int = None) -> Dict:
        """
        Generate Oolong-Synthetic-like sample
        - context_length: total tokens
        - num_entries: number of key-value pairs (if None, derive from context_length)
        """
        if num_entries is None:
            # Approx: avg entry ~ 100 tokens, so num_entries = context_length / 100
            num_entries = max(10, context_length // 100)

        # Generate entries
        entries = []
        ground_truth = {"entity": 0, "not_entity": 0, "total": num_entries}

        for i in range(num_entries):
            # Simulate label distribution
            # For OOLONG, labels are not pre-labeled, model must infer
            if i % 3 == 0:
                # Entity
                content = f"user_{i}: profile data, user_id {i}, active, entity classification required, evidence for entity"
                label = "entity"
                ground_truth["entity"] += 1
            else:
                content = f"log_{i}: system event {i}, heartbeat, timestamp, not entity relevant"
                label = "not_entity"
                ground_truth["not_entity"] += 1

            entries.append({"id": i, "content": content, "label": label})

        # Question types: USER, COMPARISON, NUMERIC, etc
        question_type = np.random.choice(["USER", "COMPARISON", "NUMERIC"])
        if question_type == "USER":
            question = f"Among instances from users {', '.join([str(e['id']) for e in entries[:5]])}... how many should be classified as 'entity'?"
        elif question_type == "COMPARISON":
            question = f"Compare entity vs non-entity counts in this document"
        else:
            question = f"How many entries total?"

        return {
            "context_length": context_length,
            "num_entries": num_entries,
            "entries": entries,
            "question": question,
            "question_type": question_type,
            "ground_truth": ground_truth
        }

    def evaluate_sample(self, runtime: OICIORuntime, sample: Dict) -> Dict:
        """Evaluate single sample"""
        # Ingest
        docs = [e["content"] for e in sample["entries"]]
        runtime.ingest_document(docs)

        # Query
        result = runtime.query(sample["question"])

        # Calculate accuracy (simplified)
        # For entity counting task
        pred_entity = result["answer"].get("entity_count", 0)
        true_entity = sample["ground_truth"]["entity"]

        # Accuracy: 1 - |pred-true|/true
        if true_entity > 0:
            accuracy = max(0, 1 - abs(pred_entity - true_entity) / true_entity)
        else:
            accuracy = 1.0 if pred_entity == 0 else 0

        return {
            "context_length": sample["context_length"],
            "question_type": sample["question_type"],
            "true_entity": true_entity,
            "pred_entity": pred_entity,
            "accuracy": accuracy,
            "confidence": result["confidence"],
            "compression": result["stats"]["compression"]
        }

    def run_eval(self, num_samples_per_bucket: int = 3):
        """Run evaluation across all buckets"""
        print(f"=== OOLONG Evaluation: {len(self.buckets)} buckets, {num_samples_per_bucket} samples each ===")

        all_results = []

        for bucket in self.buckets[:5]:  # For POC, only first 5 buckets (1K-16K)
            print(f"\n[Bucket] Context length: {bucket} tokens")
            bucket_results = []

            for i in range(num_samples_per_bucket):
                sample = self.generate_synthetic_sample(context_length=bucket)

                # Fresh runtime per sample (to avoid contamination)
                runtime = OICIORuntime(vocab_size=1000, dim=64, confidence_threshold=0.8)

                result = self.evaluate_sample(runtime, sample)
                bucket_results.append(result)

                print(f"  Sample {i+1}: true={result['true_entity']}, pred={result['pred_entity']}, acc={result['accuracy']:.2f}, conf={result['confidence']:.2f}")

            avg_acc = np.mean([r["accuracy"] for r in bucket_results])
            print(f"  Bucket {bucket} Avg Accuracy: {avg_acc:.2f}")

            all_results.extend(bucket_results)

        # Overall stats
        overall_acc = np.mean([r["accuracy"] for r in all_results])
        print(f"\n=== Overall OOLONG Score: {overall_acc*100:.2f}% ===")
        print(f"Baseline comparison:")
        print(f"  Full-context baseline: 59.22%")
        print(f"  RLM: 64.38%")
        print(f"  Codex: 71.75%")
        print(f"  RAH GPT-5: 81.36%")
        print(f"  RAH Sonnet 4.5: 89.77%")
        print(f"  OICIO POC (toy 0.5M): {overall_acc*100:.2f}%")

        # By question type
        for qtype in ["USER", "COMPARISON", "NUMERIC"]:
            type_results = [r for r in all_results if r["question_type"] == qtype]
            if type_results:
                avg = np.mean([r["accuracy"] for r in type_results])
                print(f"  {qtype}: {avg*100:.1f}%")

        return all_results

# Demo
if __name__ == "__main__":
    eval = OOLONGEval()
    results = eval.run_eval(num_samples_per_bucket=2)
