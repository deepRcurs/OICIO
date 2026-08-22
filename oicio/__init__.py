"""
OICIO — Optimized Infinite Context Intelligence Orchestration
Credits: deepRcurs Labs, @deeprcurs
Author: Mzed Imamkh, @mzedimamkh
Version: v0.1 POC

Paradigma baru: Frontier quality at 1.58-bit with harness recursion.
Snapshot-safe: all code <128MB, dependencies in .venv (excluded)
"""

__version__ = "0.1.0-poc"
__author__ = "Mzed Imamkh @mzedimamkh"
__lab__ = "deepRcurs Labs @deeprcurs"

from .core.ternary_san import TernarySAN, BitLinear
from .memory.turboquant import TurboQuant
from .memory.em_llm import SurpriseSegmenter
from .memory.reattention import ReAttention
from .harness.rah import RecursiveAgentHarness
from .edge.needle_mini import NeedleMini

__all__ = ["TernarySAN", "BitLinear", "TurboQuant", "SurpriseSegmenter", "ReAttention", "RecursiveAgentHarness", "NeedleMini"]
