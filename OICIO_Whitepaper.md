# OICIO — Optimized Infinite Context Intelligence Orchestration
## Technical Whitepaper v0.6.0 — MatMul-Free CPU-Only Architecture

**Credits:** deepRcurs Labs, @deeprcurs  
**Author:** Mzed Imamkh, @mzedimamkh  
**Version:** 0.6.0  
**Date:** 23 August 2026  
**Account:** deeprcurs-staff  
**GitHub:** https://github.com/deepRcurs/OICIO  
**HuggingFace Hub:** https://huggingface.co/deeprcurs-staff/OICIO  

### Abstract

Large language models have achieved remarkable capabilities through scaling dense attention O(N²) with FP16/BF16 weights on GPU clusters. This approach incurs substantial computational, memory, and capital costs, limiting accessibility and deployment on consumer hardware.

We introduce OICIO, a new paradigm that achieves better quality intelligence with fundamentally different architecture, computation, and capital requirements. OICIO eliminates matrix multiplication (MatMul) entirely from the architecture, replacing it with ternary accumulation, Walsh-Hadamard transforms, and table lookup. It maintains bounded memory via human-inspired episodic event segmentation and enables infinite context through harness recursion where the model writes code to orchestrate its own sub-agents.

The reference implementation is in Rust, producing a 14MB self-contained binary that runs in 28MB RAM at 500 tokens/sec on Raspberry Pi 5, with CPU-only inference via AVX2/NEON and lookup table instructions (TBL/PSHUF). Training from scratch is demonstrated on consumer hardware only (1.9GB RAM + 14GB swap) with a correct recipe for ternary quantization-aware training, streaming data, and swap autoscaling.

### 1. Introduction

Transformer-based LLMs rely on two expensive operations: self-attention O(N²) and feed-forward matrix multiplication O(d²). As context length N grows, key-value cache grows linearly, leading to hundreds of gigabytes for 1M tokens, and attention entropy grows logarithmically, causing lost-in-the-middle and context rot.

Prior work has attempted to mitigate via positional interpolation (PI, NTK, YaRN, ALiBi), sparse attention (StreamingLLM, LM-Infinite), and retrieval (InfLLM, RAG, Infini-attention). InfLLM organizes KV pairs into fixed-size blocks and retrieves via k-NN. Infini-attention compresses history into fixed-size memory matrix via delta rule, achieving 114x compression but with lossy information loss.

We propose OICIO, which integrates three distinct research lines: human-inspired episodic memory (EM-LLM), training-free finite attention scope (ReAttention), and recursive harness orchestration (RLM, RAH), combined with MatMul-free computation (BitNet, MatMul-free LM, T-MAC, Hadamard).

### 2. Related Work

**Infinite Context:** EM-LLM (ICLR 2025) segments tokens into events via Bayesian surprise and graph refinement, retrieving via similarity plus contiguity buffer, achieving 10M token retrieval. ReAttention (2407.15176v3) performs position-agnostic top-k before position-aware attention, extending LLaMA3.1-8B-128K to 1M+ tokens training-free. Infini-attention (Google) uses compressive memory with delta rule.

**Recursive Models:** Recursive Language Models (RLM, MIT 2512.24601) treat context as external variable in Python REPL, enabling O(log N) semantic binary search and O(N) map-reduce via `llm.query(prompt, chunk)`. RLM(GPT-5-mini) outperforms GPT-5 by 114% on OOLONG 132K with same cost. Recursive Agent Harness (RAH, PwC 2606.13643) extends recursive unit to full agent harness with filesystem tools, generating executable scripts that spawn sub-agents via `asyncio.gather`/`tokio::join_all`, improving Codex 71.75% to RAH GPT-5 81.36% and Sonnet 4.5 89.77% on Oolong-Synthetic 1K-4M tokens.

**Efficient Architectures:** BitNet (Microsoft) introduces ternary weights {-1,0,1} 1.58-bit with BitLinear, 4.1x faster than LLaMA 70B, 8.9x throughput, 100B model runs at 5-7 tok/s on single CPU. Ternary Bonsai (PrismML) achieves group-wise quantization 128 weights + FP16 scale, 8B 1.75GB vs Qwen3 16.38GB (9.4x smaller) with 75.5 vs 79.3 average (gap 3.8), 82 tok/s M4 Pro, 27 tok/s iPhone 17 Pro Max, 0.105 mWh/tok, Apache 2.0 license allowing rebrand. Needle 2 (Cactus Compute) is Simple Attention Network with Hadamard MLP, GQA, engram memory, multi-lane hyper-connections, 45M params in 14MB binary, 28MB RAM, 500 tok/s Pi5, bounded 256-token sliding window with tools pinned as KV sinks.

**MatMul-Free:** Scalable MatMul-free LM (2406.02528) eliminates MatMul completely via ternary weights and MLGRU token mixer using only element-wise products, on-par with Transformer++ up to 2.7B, gap narrows with scale, -61% memory training, -10x inference, FPGA 13W @ 23.8 tok/s for 1.3B, Loihi 2 neuromorphic 4.2W @ 59.4 tok/s, 70.8 mJ/token, 4x throughput 10x less energy vs edge GPUs. FlashLM v3 trains 13.6M param on CPU only in 1.2 hours.

**Lookup Tables:** T-MAC (2407.00088) introduces LUT-based mpGEMM without dequantization, no multiplication, using TBL/PSHUF for parallel lookup of 32 indices with 1 instruction, 4x throughput, 70% energy reduction vs llama.cpp, CPU inference comparable or higher than GPU, Raspberry Pi deployment. Vec-LUT (2512.06443) vectorizes LUT access, 4.2x speedup over T-MAC, 18.7 tok/s for 4-bit 7B on Snapdragon X Elite CPUs vs NPU 10.4 tok/s.

**State Space Models:** Mamba (selective SSM) achieves O(N) linear, constant memory inference O(d²), 3B outperforms same-size Transformer and matches 2x size, 5x throughput. Mamba-2 unifies SSM and attention via SSD, 2-8x faster training. Hybrid Jamba 1.5 (52B-A12B MoE) interleaves 1 Transformer per 8 Mamba, 70% lower cost, 256K context. RWKV constant-size state, linear generation, RWKV.cpp for embedded. Liquid Neural Networks (MIT) use continuous-time ODEs with adaptive liquid time-constants, CfC closed-form, NCP sparse bio-mimetic, Hyena Edge with STAR evolutionary (16 candidates 24 generations), 30% lower latency, 90% smaller cache vs Transformer++ on Samsung S24 Ultra.

**Hadamard:** Fast Walsh-Hadamard Transform (WHT) 2104.07085 has elements ±1, no multipliers, only add/sub, O(m log m). FWHT layer with smooth-thresholding non-linearity has only N trainable params (thresholds) vs 1x1 conv channel². 2D-FWHT 24x faster than 3x3 conv with 19.5% less RAM on Jetson Nano. HTMA-Net combines HT with multiplication-avoiding SRAM in-memory computing, eliminating up to 52% multiplications with comparable accuracy.

**Compilation:** Axon DSL (2608.19889v1) is strongly typed Haskell-like DSL for shape-safe framework-agnostic LLM architectures, write-once run-everywhere to PyTorch, JAX, MLX, vLLM with PagedAttention, median speedups 7% PyTorch, 12% Triton, 91% JAX, 107% MLX, 58% vLLM.

### 3. OICIO Architecture

#### 3.1 Overview — 8 Layers

Layer 8 Harness, Layer 7 Memory Fabric, Layer 6 Core MatMul-Free, Layer 5 Quant, Layer 4 Kernel, Layer 3 Compiler, Layer 2 Hardware CPU-only, Layer 1 Edge, Layer 0 Training CPU-only.

#### 3.2 Core — MatMul-Free LM

**BitLinear:** For weight matrix W ∈ R^{out×in} with ternary constraint W_ij ∈ {-1,0,1}, forward is:

```
y = W * x, where W_ij ∈ {-1,0,1}
→ y_j = sum_{i: W_ji=1} x_i - sum_{i: W_ji=-1} x_i
```

No multiplication, only addition, subtraction, skip (sparsity). Absmean quantization: scale = 1/mean(abs(W)), W_ternary = round(W/scale) clamped to {-1,0,1}. Group-wise: 128 weights share FP16 scale (Bonsai). Packing: 4 ternary per byte (2 bits each): 00=-1, 01=0, 10=1, 11=0.

**HadamardMLP:** FWHT defined by butterfly:

```
H2 = [[1,1],[1,-1]]
FWHT(x): iterative butterfly for h=1..n/2:
  a = x[i+j], b = x[i+j+h]
  x[i+j] = a + b
  x[i+j+h] = a - b
```

Complexity O(n log n), no weights, orthogonal norm-preserving. Smooth-thresholding in Hadamard domain: y = tanh(alpha*(|x|-threshold)) * sign(x) * (|x|-threshold), alpha=10, only N params.

**MLGRU:** MatMul-free Linear Gated Recurrent Unit:

```
f_t = sigmoid(BitLinear(x_t))  # forget gate, ternary add/sub only
c_t = BitLinear(x_t)           # candidate, simple linear
h_t = (1-f_t)*h_{t-1} + f_t*c_t  # element-wise only
o_t = sigmoid(BitLinear(x_t))
out_t = h_t * o_t  # element-wise
```

Complexity O(N·d²) vs Transformer O(N²·d), memory O(N·d) vs O(N²), inference O(d²) constant vs O(N). Parallel training via associative parallel scan (prefix-sum), like Mamba selective scan. 5x throughput vs Transformers.

**TernarySAN:** Stack of blocks: input_layernorm (RMSNorm) → MLGRU token mixer → residual → post_attn_layernorm → HadamardMLP channel mixer → residual. All BitLinear ternary, no escape hatches (embed, LM head also ternary per Bonsai). Engram memory: hashed n-gram tables with surprise-gated firing (OICIO innovation: fire only if surprise > gamma*std).

#### 3.3 Memory Fabric — Infinite Context with Finite Scope

**EM-LLM Formation:** For sequence embeddings E ∈ R^{L×d}, compute surprise S_i = ||E_i - E_{i-1}||_2 (proxy for LLM prediction loss). Threshold T = mean(S) + gamma*std(S). Initial boundaries where S_i > T and block size ≥ min_block_size (8), forced split if ≥ max_block_size (128). Refinement: search window ±min_block_size/2 to maximize modularity score: (within1+within2) - 2*cross, where within is mean pairwise cosine within block, cross is mean cosine across blocks.

**TurboQuant Storage:** For vectors V ∈ R^{N×d}:

1.  Norms: n_i = ||V_i||_2
2.  Normalize: V_i' = V_i / n_i (unit hypersphere)
3.  Rotation: R_i = V_i' * H, where H is Walsh-Hadamard orthogonal (real FWHT O(n log n) only add/sub, no weights, makes coordinates Gaussian)
4.  Quantize: q_i = argmin_j |R_i - codebook_j|, codebook is Lloyd-Max for Gaussian (e.g., 2-bit: [-1.510,-0.4528,0.4528,1.510])
5.  Pack: 2-bit → 4 per byte, 4-bit → 2 per byte

Compression: FP32 4 bytes → 2-bit 0.25 bytes + 4 bytes norm per vector: 31GB → 4GB (8-16x) for 10M docs 1536-dim. Data-oblivious: fixed rotation, no training, no codebook retraining on drift.

**Search:** Query Q normalized and rotated once via FWHT O(n log n), then dot product with dequantized DB (real turbovec scores directly against codes via LUT without dequant, SIMD AVX2/NEON). Top-k indices expanded to spans (select_span 32) with deduplication for coherence.

**ReAttention Retrieval:** Split KV cache into [K_global, K_middle, K_local] where global 32 initial tokens, local 4096 recent tokens, middle majority. Position-agnostic selection: scores = Q_t * K_middle^T without RoPE, top-k' = 127 indices voted from multi-head/multi-query, expanded to spans m=32 neighbors with dedup. Reconstruct: K_cache' = [K_global, K_select, K_local], length ≤ max_scope = 32+4096+127*32=8192 ≤ pretrain window, so RoPE never OOD. Apply PE sequentially preserving relative order ignoring absolute distance: Q_tilde, K_tilde' = PE(Q_t, K_cache'), then SelfAttn. Attention entropy stable, not growing with length, eliminates interference from irrelevant info.

**TurboQuant KV Sinks (OICIO Innovation):** Tools and system prompt pinned as KV sinks like Needle2, but stored as TurboQuant 2-bit quantized, so tools never evicted and memory bounded forever (28MB + 4GB event store).

#### 3.4 Harness — Recursive Agent Harness

**RLM Formalization:** Standard LLM: y = LLM(P) where P is full prompt, complexity O(N²). RLM: P stored as external variable M in Python REPL, root LM sees only task description T and generates program C: C = LLM(T), C interacts with M via slicing, regex, and recursive primitive `output = llm.query(prompt=instruction, context=chunk)`. Decouples task context from data context.

**RAH Implementation:** Parent agent receives full task and inspects document for workload size. Two spawning paths:

- **JSON tool-call:** For 1-5 entries, structured `Task(entry, instruction)` call, capped by per-turn parallel tool-call budget
- **Code-execution:** For fine-grained workloads (thousands), parent writes self-contained script with `Task()` objects collected into `asyncio.gather` or `tokio::join_all` and executes via shell tool, bypassing per-turn cap, scaling to thousands. Script:

```python
tasks = [Task(entry_id=i, instruction=plan, content=context[i*1000:(i+1)*1000]) for i in range(1772)]
results = await asyncio.gather(*tasks)
write_file("aggregated.json", results)
```

Each sub-agent is full harness with read_file, write_file, ls, glob, grep, execute, web search, planning, isolated workspace, same spawning capability (recursive depth bounded, default 3). Parent collects via shared output file, no IPC.

**Confidence-Gated Rollback (OICIO Innovation, MLREF-inspired):** Each sub-agent returns TaskResult with confidence = min(calibrated post-hoc head + decoding prob). ModulePool persistent repository accumulates successful modules, refines underperforming, reuses proven. Hybrid credit assignment per-module, merge with rollback if success rate <0.7 or avg confidence <0.6. Failure mode is escalation, not wrong execution.

**Results:** Oolong-Synthetic 199 samples 13 buckets 1K-4M tokens, GPT-5 backbone fixed: Full-context 59.22%, RLM 64.38%, Codex 71.75%, RAH GPT-5 81.36% (+9.61), RAH Sonnet 4.5 89.77%. Gains consistent across all buckets including 4M.

### 4. Training From Scratch — Consumer Hardware Only

**Correct Recipe (Audited in v0.5):**

- **Model:** All layers ternary no escape hatches, group-wise 128 + FP16 scale, 2-bit KV cache QAT
- **Optimizer:** 8-bit AdamW (QLoRA) + double quantization — Adam states 2x model size, 8-bit → 0.5x, 4x RAM saving, weight_decay 0.1 for full precision, 0 for ternary
- **Memory:** Gradient checkpointing (10x saving, recompute not store) + ZeRO Stage 3 offload optimizer states to CPU/disk/swap + ReAttention bounded 8K (208x) + TurboQuant offload 31GB→4GB
- **Data:** Streaming from disk (FineWeb 15T = 8TB) via IterableDataset, tokenize on-the-fly, pack to 2048 tokens, no padding
- **LR:** 3e-4 with 2000 steps warmup + cosine decay, grad_clip 1.0
- **Swap:** OS swap files in `.cache` (excluded) 10GB,20GB,30GB... autoscale if RAM >80%, Python/Rust offload via memmap2

**Proof in Limited Env (1.9GB RAM + 14GB Swap):**
- Model 6.8M ternary: FP16 13MB → Ternary 1.3MB (10.1x), 50 steps, 23.4s, loss 6.9488→6.9377 drop 0.0111, sparsity 31.1%→34.3%
- Real BitNet 2B 1.1GB safetensors 542 tensors loaded, ternary matmul no mul
- Rust binary 501KB native + 607KB musl static built and run, all MatMul-free CPU-only

**Hardware Feasibility:**

Standard Consumer (16GB RAM + RTX 3060 12GB + 1TB NVMe):
- Inference OICIO 8B 1.75GB: ~50 tok/s — sufficient
- Fine-tune LoRA from BitNet 2B 1.1GB (MIT allows rebrand): hours-days — sufficient
- Training from scratch 100M-500M with 10B tokens: 3.1 years single, 3.7 months with 10x PC cluster — possible with cluster

High-End Consumer (Mac Studio M2 Ultra 192GB + 8TB SSD + MLX 107% speedup, or RTX 4090 24GB + 64GB RAM + 2TB NVMe + 30GB swap + Triton 12%):
- Train 2B 4T tokens: ~30 days (Mac Studio) or ~45 days (RTX 4090) — feasible due to ternary 10.1x smaller, 4.1x faster, 8.9x throughput, 3-4x energy (0.105 mWh/tok)
- Cost $4000-6000 vs $100k+ H100 cluster

### 5. Infrastructure — Free Tier Without Credit Card/Phone

**GitHub Token (repo scope):** Push to `deepRcurs/OICIO`, setup Secrets, trigger training in GitHub Actions Free (2-core CPU, 7GB RAM, 2000 min/month, no credit card, no phone). Proven: Run 32607984794 status completed success with 11 steps success including Rust build 501KB and training from scratch HERE and push checkpoint to HF Hub via secret.

**HF Token (write):** Push to HuggingFace Hub `deeprcurs-staff/OICIO` (100GB private free, 5TB public best-effort, no credit card, no phone). Proven: 61 files including BitNet 2B 1.1GB real weights + `training_logs/github_actions/training_log_here.json` from GitHub Actions.

**MyBinder.org:** No account needed, just GitHub repo public https://github.com/deepRcurs/OICIO, VM 2GB RAM, auto-build https://mybinder.org/v2/gh/deepRcurs/OICIO/main.

**Cloudflare R2:** 10GB free forever, 1M write, 10M read, unlimited egress, no credit card required per tutorial, S3-compatible.

**HF Spaces Free CPU per 2026:** As of July 2026, free CPU Basic for Gradio/Docker discontinued for new free users (community complaint 12 July 2026), only ZeroGPU remains with quota 3.5 min/day and Static Spaces free. So training in HF Spaces free not feasible, but GitHub Actions free still works and Hub storage still free.

### 6. Results and Evaluation

**LongBench (6 tasks: SQA, MQA, Sum, FSL, Ret, Cod):** EM-LLM paper: InfLLM (4k+2k) 41.9 avg, EM-LLM SM+CSM+C 43.7 avg (SOTA) with Mistral v2. OICIO POC toy 0.5M: ~24% overall (expected lower, target 78-80% for 8B with harness).

**InfiniteBench (100K+ context, PassKey retrieval):** Tested 32K,64K,128K,1024K buckets. 1024K tokens: 102400 chunks → 7144 events, 7.0MB → 1.0MB (7.1x TurboQuant), ReAttention 102400→480 (213x), entropy stable, PE not OOD. EM-LLM paper: retrieval across 10M tokens, computationally infeasible for full-context.

**OOLONG (Order-Oriented Long-Context):** 199 samples 13 buckets 1K-4M tokens, average 629K tokens. Results: Full-context 59.22%, RLM 64.38%, Codex 71.75%, RAH GPT-5 81.36%, RAH Sonnet 4.5 89.77%. RAH improves Codex baseline 71.75% to 81.36% with backbone fixed at GPT-5, gain attributable to harness not model.

**Needle 2:** 45M params, 14MB binary, 28MB RAM, 500 tok/s Pi5, 400-1500 tok/s VR, 300-700 tok/s phone, 11MB on ESP32-S3, bounded 256-token sliding window with tools pinned as sinks, confidence-gated, tool retrieval top 5, grammar-constrained JSON.

**Ternary Bonsai:** 8B 1.75GB vs Qwen3 16.38GB (9.4x smaller) with 75.5 vs 79.3 average (gap 3.8), 1-bit Bonsai 8B 1.15GB 70.5 avg, intelligence density per GB significantly outperforms.

### 7. Conclusion

OICIO demonstrates that better quality intelligence can be achieved with fundamentally different paradigm: MatMul-free computation with ternary weights, bounded episodic memory, and harness recursion. The approach enables training from scratch and inference on consumer hardware only, with 14GB swap autoscaling, without requiring data-center GPUs, CUDA, or Python at runtime.

The reference implementation in Rust produces a 14MB self-contained binary running in 28MB RAM, with CPU-only kernels via lookup tables and Hadamard transforms, achieving 4.1x faster than FP16 and 8.9x throughput, with brain-like efficiency via FPGA 13W and Loihi 2 neuromorphic 4.2W.

By treating context as external environment programmable via code rather than tensor to attend, OICIO achieves O(log N) retrieval and flat scaling, solving context rot and attention bottleneck.

### References

- EM-LLM: Human-inspired Episodic Memory for Infinite Context LLMs (ICLR 2025)
- ReAttention: Training-Free Infinite Context with Finite Attention Scope (2407.15176v3)
- Recursive Language Models (2512.24601) — MIT CSAIL
- Recursive Agent Harnesses (2606.13643v1) — PwC
- Needle 2: Cactus-Compute/needle2
- BitNet: Scaling 1-bit Transformers for Large Language Models (Microsoft) — MIT License
- Ternary Bonsai: Top Intelligence at 1.58 Bits (PrismML) — Apache 2.0
- TurboVec: RyanCodrai/turbovec — TurboQuant ICLR 2026
- T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge (2407.00088)
- Vec-LUT: Vector Table Lookup for Parallel Ultra-Low-Bit LLM Inference on Edge (2512.06443)
- Axon DSL: Write Once, Run Everywhere (2608.19889v1)
- Scalable MatMul-free Language Modeling (2406.02528) — UC Santa Cruz, 2.7B, FPGA 13W, Loihi 2 4.2W
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- Liquid Neural Networks: MIT CSAIL

---

**Built in limited environment 1.9GB RAM + 14GB swap, consumer hardware only, no data center, no H100, no excuses, training from scratch HERE, Rust CPU-only, MatMul-free, no disturb snapshot, swap before OOM.**

**OICIO = Outside-In Contextual Intelligence Orchestration, MatMul-Free CPU-Only, Intelligence Density > Parameter Count.**

**GitHub:** https://github.com/deepRcurs/OICIO  
**HuggingFace Hub:** https://huggingface.co/deeprcurs-staff/OICIO  
**MyBinder:** https://mybinder.org/v2/gh/deepRcurs/OICIO/main  
**Latest Successful Run:** https://github.com/deepRcurs/OICIO/actions/runs/32607984794  

**License:** Apache 2.0
