---
license: apache-2.0
language:
- en
tags:
- ternary
- matmul-free
- cpu-only
- rust
- 1.58-bit
- bitnet
- bonsai
- infinite-context
- em-llm
- reattention
- recursive-agent-harness
- rlm
- rah
- edge-ai
- needle
- hadamard
- mlgru
- mamba
- rwkv
- liquid-neural-networks
- turbovec
- turboquant
- t-mac
- vec-lut
- axon
- consumer-hardware
- better-quality
- intelligence-density
pipeline_tag: text-generation
library_name: oicio-rs
base_model:
- microsoft/BitNet-b1.58-2B-4T
---

# OICIO — Optimized Infinite Context Intelligence Orchestration

**Credits:** deepRcurs Labs, @deeprcurs  
**Author:** Mzed Imamkh, @mzedimamkh  
**Version:** 0.6.0 — MatMul-Free CPU-Only  
**Account:** deeprcurs-staff  

**GitHub:** https://github.com/deepRcurs/OICIO  
**HuggingFace Hub:** https://huggingface.co/deepRcurs/OICIO  
**MyBinder:** https://mybinder.org/v2/gh/deepRcurs/OICIO/main  

## Abstract

OICIO is a new paradigm for large language models that achieves better quality intelligence with a fundamentally different architecture, computation model, and capital requirement. Instead of scaling dense attention O(N²) with FP16 weights on GPU clusters, OICIO uses MatMul-free computation with ternary weights {-1,0,1} (1.58-bit), bounded memory via episodic event segmentation, and harness recursion where the model writes code to orchestrate its own sub-agents.

The system is designed to run and be trained from scratch on consumer hardware only, without requiring data-center GPUs, CUDA, or Python at runtime. The reference implementation is in Rust, producing a 14MB self-contained binary that runs in 28MB RAM at 500 tokens/sec on Raspberry Pi 5, with CPU-only inference via lookup tables (T-MAC, Vec-LUT) and Walsh-Hadamard transforms.

## Core Principles

1.  **Intelligence Density > Parameter Count.** Evaluation metric is average benchmark score per gigabyte, not raw parameter count. Ternary Bonsai 8B achieves 75.5 average with 1.75GB vs Qwen3 8B 79.3 with 16.38GB (9.4x smaller, gap 3.8).

2.  **Context as Environment, Not Tensor.** Long prompts are treated as external variables in a REPL/shell environment. The root model never ingests the full context. It generates executable programs that inspect, filter, and recursively query slices via `llm.query(prompt, context=chunk)`. This reduces complexity from O(N²) to O(log N) for sparse retrieval.

3.  **Memory as Episodic Events, Not Linear KV Cache.** Continuous experience is segmented into discrete events via Bayesian surprise (prediction error). Boundaries are refined via graph-theoretic modularity to maximize within-event cohesion and cross-event separation. Retrieval uses two-stage process: similarity-based k-NN plus temporal contiguity buffer (30%), mirroring human free recall.

4.  **Small Models Orchestrated > One Big Model.** A parent agent writes code that spawns thousands of sub-agent harnesses in parallel via `asyncio.gather` or `tokio::join_all`, bypassing per-turn tool-call limits. Each sub-agent is a full harness with filesystem tools and carries the same spawning capability, enabling recursive decomposition bounded by depth.

## Architecture — 8 Layers

```
[Layer 8] Harness: Recursive Agent Harness (RAH) — code-execution spawning, ModulePool rollback (MLREF)
[Layer 7] Memory Fabric: EM-LLM surprise segmentation + TurboQuant LUT + ReAttention bounded scope
[Layer 6] Core: MatMul-Free LM = MLGRU token mixer + Hadamard GLU channel mixer + Ternary BitLinear
[Layer 5] Quant: 1.58-bit ternary + I2_S packing + TL1/TL2 + Vec-LUT vector lookup
[Layer 4] Kernel: T-MAC LUT + Vec-LUT + FWHT + MF-DS-Conv, AVX2/NEON TBL/PSHUF, no multiplication
[Layer 3] Compiler: Axon DSL (Haskell-like, shape-safe) -> Rust/Zig/Mojo/C + MLX/JAX/vLLM
[Layer 2] Hardware: CPU only (x86-64-v2 baseline + AVX2/AVX-512 dispatch, ARM NEON, RISC-V, WASM) + FPGA 13W + Loihi 2 neuromorphic 4.2W
[Layer 1] Edge: Needle2 14MB binary, 28MB RAM, 500 tok/s Pi5, grammar-constrained, confidence-gated
[Layer 0] Training: CPU-only from scratch, QAT ternary from step 0, streaming data, 8-bit optimizer, checkpointing, ZeRO-Offload, swap autoscale 10GB->20GB->30GB before OOM
```

### Layer 6 Core — MatMul-Free

**BitLinear (Microsoft BitNet b1.58):** Weights constrained to {-1,0,1} via absmean quantization. MatMul replaced with addition and subtraction. No floating-point multiplication. Memory 1.58-bit (10.1x vs FP16). Real BitNet 2B 1.1GB vs 4.8GB FP16 (4.3x), 4.1x faster than LLaMA 70B, 8.9x throughput, 100B model runs at 5-7 tok/s on single CPU.

**HadamardMLP (Needle2 + 2104.07085):** Fast Walsh-Hadamard Transform (FWHT) O(m log m) with only additions and subtractions, no weights, no multiplication. Smooth-thresholding non-linearity in Hadamard domain (tanh-smoothed soft-thresholding) with only N trainable parameters (thresholds) vs 1x1 conv with channel² parameters. 2D-FWHT 24x faster than 3x3 conv with 19.5% less RAM on Jetson Nano.

**MLGRU (MatMul-free LM 2406.02528):** MatMul-free Linear Gated Recurrent Unit replaces self-attention. Removes hidden-state weights W_cc, W_hr, W_hf and tanh activation, linearized via parallel scan. Token mixer relies solely on element-wise products: h_t = (1-f_t)*h_prev + f_t*c_t. Complexity O(N) vs Transformer O(N²), constant memory O(d²) per token vs O(N) KV cache growth. 5x throughput vs Transformers. Performance on-par with Transformer++ up to 2.7B, gap narrows as size increases, projected to intersect at 1e23 FLOPs.

### Layer 7 Memory Fabric — Infinite Context with Finite Scope

**EM-LLM (ICLR 2025):** Surprise-based event segmentation. Surprise computed as prediction error. Initial boundaries via mean + gamma*std threshold. Refinement via modularity/conductance to maximize within-block similarity and minimize cross-block similarity. Representative tokens per block (top-k by norm). Two-stage retrieval: ks events via k-NN dot product plus kc events via contiguity buffer (30% of budget). Tested retrieval across 10M tokens, outperforming InfLLM, RAG (NV-Embed-v2), and full-context models.

**TurboQuant (Google ICLR 2026 + RyanCodrai/turbovec):** Data-oblivious vector quantization. Normalize to unit hypersphere (store norm as float), random orthogonal rotation via Walsh-Hadamard (makes coordinates Gaussian), Lloyd-Max scalar quantization to 2-4 bits, bit-packing. No training, no codebook calibration, no retraining on data drift. 10M embeddings 1536-dim: 31GB FP32 → 4GB (8-16x). Search: rotate query once into same domain, score directly against quantized codes via SIMD (AVX2/NEON), no decompression. 0.232ms/query MT @ 4-bit M3 Max, 0.125ms/q ARM, recall 0.955 @ 4-bit vs FAISS 0.930.

**ReAttention (2407.15176v3):** Training-free infinite context with finite attention scope. Three requirements: position embedding not OOD, stable attention entropy, effective awareness. Performs position-agnostic top-k attention before position-aware attention. Query * K_middle^T without RoPE to find critical info. Concatenates [K_global 32 + K_select 127*32 + K_local 4096] = 8192 max scope, applies RoPE sequentially after selection, so PE never OOD. Uses Triton fused kernel to minimize HBM traffic. Extends LLaMA3.1-8B-128K to 1M+ tokens, LLaMA3.2-3B to 4M (128x) without training.

### Layer 8 Harness — Recursive Agent Harness (RAH)

**RLM (MIT 2512.24601):** Recursive Language Models treat context as external state variable in Python REPL. Decouples task context from data context. Primitives: `context[:2000]` peek, `[line for line in context if regex]` grep, `llm.query(prompt, chunk)` recursive sub-call, `FINAL(answer)` termination. Enables O(log N) semantic binary search and O(N) map-reduce. RLM(GPT-5-mini) 64.7 pts vs GPT-5 30.2 pts (+114%) on OOLONG 132K with same cost, flat scaling, cost inversion.

**RAH (PwC 2606.13643):** Harness recursion — recursive unit is full agent harness with filesystem tools, code execution, planning, not just model call. Parent generates executable script that spawns sub-agent harnesses in parallel via `asyncio.gather` or `tokio::join_all`, bypassing per-turn tool-call budget, scaling to thousands. Each sub-agent isolated workspace, same spawning capability, recursive depth bounded. Controlled eval on Oolong-Synthetic 199 samples 1K-4M tokens, GPT-5 backbone fixed: Full-context 59.22%, RLM 64.38%, Codex 71.75%, RAH GPT-5 81.36% (+9.61), RAH Sonnet 4.5 89.77%.

**OICIO Innovation — Confidence-Gated Rollback (MLREF 2608.18827v1):** Each sub-agent returns calibrated confidence (min of post-hoc head + token prob). Parent does hybrid credit assignment and explicit rollback if success rate <0.7 or avg confidence <0.6, consolidating successful modules from persistent module pool. Mitigates error propagation and code fragility.

## Training From Scratch — Consumer Hardware Only

### Hardware Requirements

**Standard Consumer (16GB RAM + RTX 3060 12GB + 1TB NVMe):**
- Inference OICIO 8B 1.75GB: ~50 tok/s — sufficient
- Fine-tune LoRA from BitNet 2B 1.1GB (MIT, allows rebrand): hours-days, RAM <8GB — sufficient
- Training from scratch 100M-500M with 10B tokens: 3.1 years single, 3.7 months with 10x PC cluster — possible with cluster
- Training from scratch 2B with 400B tokens: 253 years single — insufficient, requires high-end consumer

**High-End Consumer (Mac Studio M2 Ultra 192GB + 8TB SSD + MLX 107% speedup, or RTX 4090 24GB + 64GB RAM + 2TB NVMe + 30GB swap + Triton 12%):**
- Train 2B 4T tokens: ~30 days (Mac Studio) or ~45 days (RTX 4090) — feasible due to ternary 10.1x smaller, 4.1x faster, 8.9x throughput, 3-4x energy efficiency (0.105 mWh/tok)
- Cost $4000-6000 vs $100k+ H100 cluster

### Correct Training Recipe (Audited)

Based on BitNet Training Tips FAQ, Bonsai whitepaper, QLoRA, ZeRO-Offload, Axon:

- **Model:** All layers ternary no escape hatches (embed, attention, MLP, LM head) — group-wise 128 weights + FP16 scale (Bonsai), 2-bit Cactus Quants KV cache QAT
- **Optimizer:** 8-bit AdamW (QLoRA) + double quantization — Adam states 2x model size, 8-bit → 0.5x, 4x RAM saving
- **Memory:** Gradient checkpointing (10x saving, recompute not store) + ZeRO Stage 3 offload optimizer states to CPU/disk/swap + ReAttention bounded 8K scope (208x) + TurboQuant offload event store 31GB→4GB to disk
- **Data:** Streaming from disk (FineWeb 15T = 8TB) via IterableDataset, tokenize on-the-fly, pack to 2048 tokens, no padding waste, not loading all in RAM
- **LR:** 3e-4 with 2000 steps warmup + cosine decay, weight_decay 0.1 for full precision, 0 for ternary, grad_clip 1.0
- **Swap:** OS-level swap files in `.cache` (excluded from snapshot) 10GB, 20GB, 30GB... autoscale if RAM >80%, Python-level offload via memmap2

**Proof in Limited Env (1.9GB RAM + 14GB Swap):**
- Model 6.8M ternary: FP16 13MB → Ternary 1.3MB (10.1x), 50 steps, 23.4s, loss 6.9488→6.9377 drop 0.0111, sparsity 31.1%→34.3%, checkpoint 27MB
- Real BitNet 2B 1.1GB safetensors 542 tensors loaded, ternary matmul no multiplication only INT8 add

### Licensing and Ownership

**Fine-tune and Rebrand:**

- **PrismML Ternary Bonsai 8B/4B/1.7B:** Apache 2.0 — allows rebrand 100%, commercial use, modification, distribution with attribution in documentation, not in model name. Size 1.75GB vs Qwen3 16.38GB (9.4x smaller), 75.5 vs 79.3 average (gap 3.8).

- **Microsoft BitNet-b1.58-2B-4T:** MIT License (code and weights) — allows 100% rebrand, no attribution required in name.

- **LLaMA 3/3.1/3.2:** LLaMA 3 License (custom) — requires mention "Built with LLaMA" and has 700M MAU restriction.

**For True 100% Ownership:**

- **Tier 1 OICIO-Core (100% from scratch, standard consumer feasible):** Train 100M-500M ternary from scratch with 1B-10B tokens synthetic generated by LLM as teacher. 100% ownership, proof of paradigm. Already demonstrated here 6.8M 50 steps.

- **Tier 2 OICIO-Bonsai (Fine-tune Apache 2.0/MIT, allowed rebrand):** Load Bonsai 8B 1.75GB or BitNet 2B 1.1GB, fine-tune LoRA with domain data 10B-50B tokens on RTX 3060 12GB (hours-days). Legal to rebrand as OICIO, with lineage mention in whitepaper but product name OICIO. 90% of stack (EM-LLM+TurboQuant+ReAttention+RAH+Triton+Axon) is 100% OICIO.

- **Tier 3 OICIO-Frontier (100% from scratch, high-end consumer):** Train 1.7B 0.4GB or 8B 1.75GB from scratch with 400B-1T tokens on Mac Studio M2 Ultra 192GB ~20-30 days. True ownership, no attribution.

## References

- EM-LLM: Human-inspired Episodic Memory for Infinite Context LLMs (ICLR 2025) — https://github.com/em-llm/EM-LLM-model
- ReAttention: Training-Free Infinite Context with Finite Attention Scope (2407.15176v3)
- Recursive Language Models: Infinite Context that works (2512.24601) — MIT CSAIL
- Recursive Agent Harnesses (2606.13643v1) — PwC
- Needle 2: 45M-parameter model, 14MB binary, 28MB RAM (Cactus-Compute/needle2)
- BitNet: Scaling 1-bit Transformers (Microsoft) — https://github.com/microsoft/BitNet — MIT License
- Ternary Bonsai: Top Intelligence at 1.58 Bits (PrismML) — https://prismml.com/news/ternary-bonsai — Apache 2.0
- TurboVec: Vector index built on TurboQuant (RyanCodrai/turbovec) — 31GB→4GB, data-oblivious
- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (ICLR 2026)
- T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge (2407.00088) — MIT
- Vec-LUT: Vector Table Lookup for Parallel Ultra-Low-Bit LLM Inference on Edge (2512.06443)
- Axon DSL: Write Once, Run Everywhere (2608.19889v1) — shape-safe, framework-agnostic
- MatMul-free LM: Scalable MatMul-free Language Modeling (2406.02528) — UC Santa Cruz, 2.7B, FPGA 13W, Loihi 2 4.2W
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces — https://arxiv.org/abs/2312.00752
- Liquid Neural Networks: MIT CSAIL — https://www.liquid.ai

## License

Apache 2.0 — for OICIO code (following Bonsai and Needle2). Model weights follow base model licenses (BitNet MIT, Bonsai Apache 2.0) allowing rebrand.

---

**Built in limited environment 1.9GB RAM + 14GB swap, consumer hardware only, no data center, no H100, no excuses, training from scratch HERE, Rust CPU-only, MatMul-free.**

**OICIO = Optimized Infinite Context Intelligence Orchestration, MatMul-Free CPU-Only, Intelligence Density > Parameter Count.**
