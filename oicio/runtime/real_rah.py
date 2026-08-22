"""
OICIO Real RAH: Actual Code-Execution Spawning
Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh

Real implementation where parent writes executable Python script that spawns subagents via asyncio.gather
This bypasses per-turn tool-call limit (Anthropic dynamic workflows pattern)
"""

import os
import sys
import tempfile
import subprocess
import json
import asyncio
from typing import List, Dict

class RealRAH:
    """
    Parent agent that WRITES CODE and EXECUTES it
    """
    def __init__(self, parallel_limit=20):
        self.parallel_limit = parallel_limit

    def generate_spawning_script(self, entries: List[Dict], instruction: str) -> str:
        """
        Generate executable Python script that spawns subagents
        This is the core RAH innovation: code as action
        """
        script = f'''
import asyncio
import json
import os
import sys
sys.path.insert(0, '/home/user')

from oicio.harness.rah import SubAgentHarness

async def run_subagent(agent_id, entry_id, content, instruction):
    # Each subagent is full harness with tools
    agent = SubAgentHarness(agent_id=agent_id)
    result = agent.run(entry_id=entry_id, instruction=instruction, context_slice=content)
    return {{
        "agent_id": agent_id,
        "entry_id": entry_id,
        "answer": result.answer,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "success": result.success
    }}

async def main():
    entries = {json.dumps(entries)}
    instruction = {json.dumps(instruction)}

    # Create tasks for all entries (bypasses tool-call budget, scales to thousands)
    tasks = []
    for i, entry in enumerate(entries):
        task = run_subagent(i, entry["id"], entry["content"], instruction)
        tasks.append(task)

    # Run in parallel with asyncio.gather (RAH pattern)
    results = await asyncio.gather(*tasks)

    # Write aggregated output to shared file (no IPC overhead)
    with open("aggregated_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    entity_count = sum(1 for r in results if r["answer"] == "entity")
    avg_conf = sum(r["confidence"] for r in results) / len(results) if results else 0
    print(f"RAH Results: {{len(results)}} entries, {{entity_count}} entity, avg_conf {{avg_conf:.2f}}")

    # Return via stdout
    print(json.dumps({{"entity_count": entity_count, "total": len(results), "avg_confidence": avg_conf}}))

if __name__ == "__main__":
    asyncio.run(main())
'''
        return script

    def execute_script(self, script_content: str) -> Dict:
        """Execute generated script via shell tool (like coding agent)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "spawn_subagents.py")
            with open(script_path, 'w') as f:
                f.write(script_content)

            # Execute via shell (parent's execute tool)
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=30
            )

            print(f"[RealRAH] Script stdout:\n{result.stdout}")
            if result.stderr:
                print(f"[RealRAH] Script stderr:\n{result.stderr}")

            # Read aggregated file
            agg_path = os.path.join(tmpdir, "aggregated_results.json")
            if os.path.exists(agg_path):
                with open(agg_path, 'r') as f:
                    detailed = json.load(f)
            else:
                detailed = []

            # Try parse last line as JSON summary
            try:
                lines = result.stdout.strip().split("\n")
                summary = json.loads(lines[-1])
            except:
                summary = {"entity_count": 0, "total": 0}

            return {"summary": summary, "detailed": detailed, "stdout": result.stdout}

    def run(self, entries: List[Dict], instruction: str):
        print(f"[RealRAH] Generating spawning script for {len(entries)} entries...")
        script = self.generate_spawning_script(entries, instruction)
        print(f"[RealRAH] Script generated ({len(script)} chars), executing via shell tool...")

        # Save script for audit (snapshot-safe, small)
        with open("/home/user/oicio/data/last_spawn_script.py", "w") as f:
            f.write(script)

        result = self.execute_script(script)
        return result

if __name__ == "__main__":
    print("=== Real RAH: Code-Execution Spawning POC ===")

    entries = [{"id": i, "content": f"user_{i}: entity data" if i%3==0 else f"log {i}: system"} for i in range(20)]
    instruction = "Count entity entries"

    rah = RealRAH()
    result = rah.run(entries, instruction)

    print(f"\nFinal: {result['summary']}")
    print(f"Detailed count: {len(result['detailed'])}")
