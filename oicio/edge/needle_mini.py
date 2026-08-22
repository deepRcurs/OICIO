"""
OICIO Edge: Needle Mini - 45M param model simulation
Credits: deepRcurs Labs, @deeprcurs / Mzed Imamkh @mzedimamkh

Berdasarkan:
- Cactus-Compute/needle2: 45M, 14MB binary, 28MB RAM, 500 tok/s Pi5
- Simple Attention Network + Confidence-gated + Grammar-constrained

POC: Simulate tool calling with confidence gating and bounded memory
"""

import json
import re
from typing import List, Dict, Any, Optional
import random

class NeedleMini:
    """
    Needle 2 simulation: text in, JSON out, confidence-gated, bounded memory
    """
    def __init__(self, tools: List[Dict], confidence_threshold: float = 0.8):
        self.tools = tools
        self.confidence_threshold = confidence_threshold
        # Bounded memory: 256-token sliding window
        self.max_window = 256
        self.kv_cache = []  # list of tokens, bounded
        # Tools pinned as KV sinks (never evicted)
        self.tool_sinks = [t["name"] for t in tools]
        # Grammar compiled from schemas
        self.grammar = self._compile_grammar(tools)

    def _compile_grammar(self, tools: List[Dict]) -> Dict:
        """Compile JSON schema into decode grammar (byte-level)"""
        grammar = {}
        for tool in tools:
            name = tool["name"]
            # For POC, just store required fields and patterns
            props = tool.get("parameters", {}).get("properties", {})
            required = tool.get("parameters", {}).get("required", [])
            grammar[name] = {"properties": props, "required": required}
        return grammar

    def _enforce_grammar(self, tool_name: str, arguments: Dict) -> Dict:
        """Enforce grammar: only allow valid values"""
        if tool_name not in self.grammar:
            return {}

        schema = self.grammar[tool_name]
        enforced = {}

        for field, value in arguments.items():
            if field not in schema["properties"]:
                continue
            prop = schema["properties"][field]
            # Check type
            expected_type = prop.get("type")
            if expected_type == "string" and not isinstance(value, str):
                continue
            if expected_type == "integer" and not isinstance(value, int):
                continue
            if expected_type == "number" and not isinstance(value, (int, float)):
                continue

            # Check pattern, min/max, etc (Field constraints)
            if "pattern" in prop:
                if not re.match(prop["pattern"], str(value)):
                    continue
            if "minimum" in prop and isinstance(value, (int, float)):
                if value < prop["minimum"]:
                    continue
            if "maximum" in prop and isinstance(value, (int, float)):
                if value > prop["maximum"]:
                    continue
            if "maxLength" in prop and isinstance(value, str):
                if len(value) > prop["maxLength"]:
                    continue

            enforced[field] = value

        return enforced

    def _calculate_confidence(self, query: str, tool_name: str, arguments: Dict) -> float:
        """
        Confidence = min(calibrated head + decoding prob)
        For POC, simulate based on evidence in query
        """
        # If arguments have evidence in query, high confidence
        # If guessing, low confidence
        confidence = 0.5

        # Check if argument values appear in query
        evidence_count = 0
        for v in arguments.values():
            if str(v).lower() in query.lower():
                evidence_count += 1

        if evidence_count > 0:
            confidence = 0.85 + random.uniform(0, 0.14)
        else:
            confidence = 0.4 + random.uniform(0, 0.3)

        # Tool retrieval: only top 5 tools per turn (built-in retrieval head)
        # For POC, assume tool is in top 5 if name appears in query or high similarity
        if tool_name.lower() in query.lower():
            confidence += 0.05

        return min(confidence, 0.99)

    def complete(self, query: str) -> Dict[str, Any]:
        """
        Complete: text in, JSON out
        Returns: {type, function_calls, reasoning, confidence, ...}
        """
        # Bounded memory: add to KV cache, evict oldest if >256, but keep tool sinks
        self.kv_cache.append(query)
        if len(self.kv_cache) > self.max_window:
            # Evict oldest non-sink
            self.kv_cache = self.kv_cache[-self.max_window:]

        # Tool retrieval: embed query, get top 5 tools (simulated)
        # For POC, simple keyword matching
        relevant_tools = []
        for tool in self.tools:
            if any(kw in query.lower() for kw in tool["name"].split("_")) or len(relevant_tools) < 5:
                relevant_tools.append(tool)
            if len(relevant_tools) >= 5:
                break

        if not relevant_tools:
            relevant_tools = self.tools[:5]

        # Simulate model picking tool
        # For POC, pick first relevant tool
        # Real Needle uses contrastive head to score tools

        # Check if query is off-topic (no tool can serve)
        # Return empty call []
        off_topic_keywords = ["quantum", "philosophy", "meaning of life"]
        if any(kw in query.lower() for kw in off_topic_keywords):
            return {
                "type": "call",
                "function_calls": [],
                "reasoning": "No tool can serve this request",
                "confidence": 0.95,
                "success": True
            }

        # Try to extract arguments from query
        # For demo, handle specific tools
        tool_name = relevant_tools[0]["name"]
        arguments = {}

        # Simple extraction heuristics
        if tool_name == "set_lights":
            # Extract room, brightness
            if "living room" in query.lower():
                arguments["room"] = "living room"
            elif "bedroom" in query.lower():
                arguments["room"] = "bedroom"
            else:
                arguments["room"] = "living room"

            # Extract brightness
            m = re.search(r"(\d+)\s*%", query)
            if m:
                arguments["brightness"] = int(m.group(1))
            elif "dim" in query.lower():
                arguments["brightness"] = 30
                arguments["on"] = True
            elif "bright" in query.lower():
                arguments["brightness"] = 100
                arguments["on"] = True

            if "on" not in arguments:
                arguments["on"] = True

        elif tool_name == "get_weather":
            m = re.search(r"in\s+([A-Za-z\s]+)", query)
            if m:
                arguments["city"] = m.group(1).strip()
            else:
                arguments["city"] = "Lagos"

        # Enforce grammar
        arguments = self._enforce_grammar(tool_name, arguments)

        # Calculate confidence
        confidence = self._calculate_confidence(query, tool_name, arguments)

        # Reasoning: short derivation of each arg from source span
        reasoning_parts = []
        for k, v in arguments.items():
            reasoning_parts.append(f"'{v}' -> {k}")
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No args"

        # If confidence below threshold, we would escalate (not execute)
        # For POC, still return but mark

        return {
            "type": "call",
            "success": True,
            "error": None,
            "function_calls": [{"name": tool_name, "arguments": arguments}] if arguments else [],
            "reasoning": reasoning,
            "confidence": confidence,
            "peak_ram_mb": 28.0,
            "should_escalate": confidence < self.confidence_threshold
        }

    def run(self, query: str, tools_impl: Dict[str, callable] = None) -> Dict[str, Any]:
        """
        Run: complete loop - model picks call, execute function, feed result back
        """
        response = self.complete(query)

        if response["type"] == "call" and response["function_calls"]:
            if tools_impl:
                # Execute tool
                for call in response["function_calls"]:
                    tool_name = call["name"]
                    if tool_name in tools_impl:
                        result = tools_impl[tool_name](**call["arguments"])
                        # Feed result back (for POC, just return)
                        response["tool_result"] = result

        return response

# Demo
if __name__ == "__main__":
    print("=== Needle Mini POC ===")

    tools = [
        {
            "name": "set_lights",
            "description": "Turn a room's lights on or off and set brightness",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {"type": "string", "description": "which room"},
                    "on": {"type": "boolean"},
                    "brightness": {"type": "integer", "minimum": 0, "maximum": 100}
                },
                "required": ["room", "on"]
            }
        },
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    ]

    needle = NeedleMini(tools=tools, confidence_threshold=0.8)

    queries = [
        "dim the living room to 30",
        "what's it like in Lagos right now?",
        "set bedroom brightness to 150",  # invalid, should be enforced
        "explain quantum physics"  # off-topic
    ]

    def set_lights(room: str, on: bool, brightness: int = 100):
        return {"room": room, "on": on, "brightness": brightness}

    def get_weather(city: str):
        return {"city": city, "temp_c": 27, "sky": "clear"}

    for q in queries:
        print(f"\nQuery: {q}")
        res = needle.run(q, tools_impl={"set_lights": set_lights, "get_weather": get_weather})
        print(f"Response: {json.dumps(res, indent=2)}")
