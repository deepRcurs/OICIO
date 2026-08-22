"""
OICIO Harness: Recursive Agent Harness (RAH)
Credits: deepRcurs Labs, @deeprcurs / Mzed Imamkh @mzedimamkh

Berdasarkan:
- MIT RLM (2512.24601): Recursive Language Models, context as external variable
- PwC RAH (2606.13643): Harness recursion, code-first spawning

Core:
- Parent agent generates executable script that spawns subagent harnesses in parallel
- Subagents carry same spawning capability (recursive)
- Code-execution path bypasses per-turn tool-call limit
- JSON tool-call path for small subtasks (1-5 entries)

OICIO Innovation: Confidence-Gated Rollback (from MLREF)
- Each subagent returns confidence
- Parent does hybrid credit assignment + rollback
- Module pool persistent
"""

import asyncio
import json
import os
import tempfile
import subprocess
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import random

@dataclass
class TaskResult:
    task_id: int
    entry_id: int
    answer: Any
    confidence: float
    reasoning: str
    success: bool

class SubAgentHarness:
    """
    Full agent harness with filesystem tools, code execution, planning
    Each subagent has isolated workspace
    """
    def __init__(self, agent_id: int, tools: List[str] = None):
        self.agent_id = agent_id
        self.tools = tools or ["read_file", "write_file", "grep", "execute", "reasoning"]
        self.workspace = tempfile.mkdtemp(prefix=f"oicio_subagent_{agent_id}_")

    def read_file(self, path: str) -> str:
        try:
            with open(path, 'r') as f:
                return f.read()
        except:
            return ""

    def write_file(self, path: str, content: str):
        full_path = os.path.join(self.workspace, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)

    def grep(self, pattern: str, text: str) -> List[str]:
        import re
        return re.findall(pattern, text)

    def reasoning(self, instruction: str, context_slice: str) -> Dict[str, Any]:
        """
        Simulate LLM reasoning over context slice
        For POC, we simulate with heuristic + confidence

        Real would be: llm.query(prompt=instruction, context=context_slice)
        """
        # Simulate reasoning: if instruction asks to label entity, check keywords
        # This is where Needle2 45M would run
        confidence = random.uniform(0.6, 0.99)

        # Simple heuristic for demo
        if "entity" in instruction.lower():
            # Check if context contains entity-like patterns
            if "user_id" in context_slice or "entity" in context_slice.lower():
                answer = "entity"
                confidence = random.uniform(0.85, 0.99)
                reasoning = f"'{context_slice[:50]}' contains user_id -> entity"
            else:
                answer = "not_entity"
                confidence = random.uniform(0.6, 0.85)
                reasoning = f"'{context_slice[:50]}' no entity pattern"
        else:
            answer = f"processed: {context_slice[:20]}"
            reasoning = f"Processed {len(context_slice)} chars"

        # Simulate failure for low confidence
        success = confidence > 0.5

        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning": reasoning,
            "success": success
        }

    def run(self, entry_id: int, instruction: str, context_slice: str) -> TaskResult:
        result = self.reasoning(instruction, context_slice)
        return TaskResult(
            task_id=self.agent_id,
            entry_id=entry_id,
            answer=result["answer"],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            success=result["success"]
        )

class ModulePool:
    """
    MLREF-inspired Module Pool: persistent repository of reusable components
    Evolves across iterations by accumulating successful modules
    """
    def __init__(self):
        self.modules = {}  # name -> {code, success_count, failure_count, avg_confidence}
        self.history = []

    def add_module(self, name: str, code: str, confidence: float, success: bool):
        if name not in self.modules:
            self.modules[name] = {"code": code, "success": 0, "failure": 0, "confidences": []}

        if success:
            self.modules[name]["success"] += 1
        else:
            self.modules[name]["failure"] += 1

        self.modules[name]["confidences"].append(confidence)
        self.history.append({"name": name, "confidence": confidence, "success": success})

    def get_best_modules(self, top_k: int = 5):
        # Sort by success rate and avg confidence
        scored = []
        for name, data in self.modules.items():
            total = data["success"] + data["failure"]
            if total == 0:
                continue
            success_rate = data["success"] / total
            avg_conf = sum(data["confidences"]) / len(data["confidences"]) if data["confidences"] else 0
            score = success_rate * 0.7 + avg_conf * 0.3
            scored.append((name, score, data))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def should_rollback(self, recent_results: List[TaskResult], threshold: float = 0.7) -> bool:
        # If recent success rate < threshold, rollback
        if not recent_results:
            return False
        success_rate = sum(1 for r in recent_results if r.success) / len(recent_results)
        avg_conf = sum(r.confidence for r in recent_results) / len(recent_results)
        return success_rate < threshold or avg_conf < 0.6

class RecursiveAgentHarness:
    """
    RAH: Parent agent that spawns subagents via code execution
    """
    def __init__(self, max_depth: int = 3, confidence_threshold: float = 0.8):
        self.max_depth = max_depth
        self.confidence_threshold = confidence_threshold
        self.module_pool = ModulePool()
        self.depth = 0

    def select_spawning_path(self, num_entries: int) -> str:
        """Select spawning path based on entry count"""
        if num_entries <= 5:
            return "json_tool_call"  # structured function call
        else:
            return "code_execution"  # write executable script

    def spawn_via_json(self, entries: List[Dict], instruction: str) -> List[TaskResult]:
        """JSON tool-call spawning for 1-5 entries"""
        results = []
        for i, entry in enumerate(entries):
            agent = SubAgentHarness(agent_id=i)
            result = agent.run(entry_id=entry["id"], instruction=instruction, context_slice=entry["content"])
            results.append(result)
            self.module_pool.add_module(f"json_task_{i}", instruction, result.confidence, result.success)
        return results

    def spawn_via_code(self, entries: List[Dict], instruction: str, parallel_limit: int = 50) -> List[TaskResult]:
        """
        Code-execution spawning for fine-grained workloads
        Parent writes self-contained script that instantiates Task() objects and runs them via asyncio.gather
        This bypasses per-turn tool-call cap
        """
        results = []

        # Simulate code generation
        # Real RAH would generate Python code and execute via shell tool
        # For POC, we simulate parallel execution

        # Batch entries to avoid OOM
        for batch_start in range(0, len(entries), parallel_limit):
            batch = entries[batch_start:batch_start+parallel_limit]
            batch_results = []

            # Simulate asyncio.gather
            for j, entry in enumerate(batch):
                agent_id = batch_start + j
                agent = SubAgentHarness(agent_id=agent_id)
                result = agent.run(entry_id=entry["id"], instruction=instruction, context_slice=entry["content"])
                batch_results.append(result)

            results.extend(batch_results)

            # Check if need rollback (MLREF innovation)
            if self.module_pool.should_rollback(batch_results):
                print(f"[RAH] Rollback triggered at batch {batch_start}, low confidence. Retrying with best modules...")
                best_modules = self.module_pool.get_best_modules(top_k=3)
                # In real, would re-run with best module code
                # For POC, just log
                print(f"[RAH] Best modules: {[m[0] for m in best_modules]}")

            # Add to pool
            for r in batch_results:
                self.module_pool.add_module(f"code_task_{r.entry_id}", instruction, r.confidence, r.success)

        return results

    def aggregate_results(self, results: List[TaskResult], aggregation: str = "count") -> Dict[str, Any]:
        """Aggregate subagent results"""
        if aggregation == "count":
            # Count entity vs not_entity
            entity_count = sum(1 for r in results if r.answer == "entity")
            total = len(results)
            avg_conf = sum(r.confidence for r in results) / len(results) if results else 0
            success_rate = sum(1 for r in results if r.success) / len(results) if results else 0

            # Confidence-gated: only count high confidence
            high_conf_results = [r for r in results if r.confidence >= self.confidence_threshold]
            high_conf_entity = sum(1 for r in high_conf_results if r.answer == "entity")

            return {
                "total_entries": total,
                "entity_count": entity_count,
                "high_conf_entity_count": high_conf_entity,
                "high_conf_total": len(high_conf_results),
                "avg_confidence": avg_conf,
                "success_rate": success_rate,
                "low_confidence_escalated": total - len(high_conf_results)
            }
        else:
            return {"results": results}

    def run(self, context: List[Dict], instruction: str, aggregation: str = "count") -> Dict[str, Any]:
        """
        Main RAH run
        context: list of entries [{"id": 0, "content": "..."}, ...]
        instruction: task description
        """
        num_entries = len(context)
        path = self.select_spawning_path(num_entries)

        print(f"[RAH] Parent agent: {num_entries} entries, selected path: {path}, depth: {self.depth}")

        if path == "json_tool_call":
            results = self.spawn_via_json(context, instruction)
        else:
            results = self.spawn_via_code(context, instruction)

        aggregated = self.aggregate_results(results, aggregation)

        # If depth < max_depth and there are low confidence results, recurse
        if self.depth < self.max_depth and aggregated.get("low_confidence_escalated", 0) > 0:
            low_conf_entries = [e for e, r in zip(context, results) if r.confidence < self.confidence_threshold]
            if low_conf_entries:
                print(f"[RAH] Recursing depth {self.depth+1} for {len(low_conf_entries)} low confidence entries")
                self.depth += 1
                # Recursive call
                recursed = self.run(low_conf_entries, instruction + " (re-evaluate carefully)", aggregation)
                # Merge
                aggregated["recursed"] = recursed

        return aggregated

# Demo
if __name__ == "__main__":
    print("=== RAH POC ===")

    # Simulate Oolong-Synthetic: 1772 entries, 536K tokens
    # For POC, 100 entries
    num_entries = 100
    context = []
    for i in range(num_entries):
        # Simulate entry content
        if i % 3 == 0:
            content = f"user_id: {i}, data: entity information for user {i}, profile..."
        else:
            content = f"log entry {i}: system event, not relevant"
        context.append({"id": i, "content": content})

    instruction = "Among these entries, how many should be classified as 'entity'? Check if contains user_id and entity information."

    rah = RecursiveAgentHarness(max_depth=2, confidence_threshold=0.8)
    result = rah.run(context, instruction, aggregation="count")

    print(f"\nResult: {json.dumps(result, indent=2)}")
    print(f"\nModule Pool best: {rah.module_pool.get_best_modules(top_k=3)}")
