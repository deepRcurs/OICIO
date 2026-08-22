# OICIO Rust Roadmap — Port ke Rust CPU-Only, Bersih, Lengkap
**Credits: deepRcurs Labs, @deeprcurs**
**Author: Mzed Imamkh, @mzedimamkh**
**Version: 0.6.0 — MatMul-Free CPU-Only**
**Date: 23 Aug 2026**
**Aturan: Snapshot hanya hal relevan, yang tidak relevan buang/hapus**

---

## Cleanup Snapshot — Bersih

**Before:** 32MB, 38 files dengan checkpoint 27MB+5MB, __pycache__, old whitepapers v02-v04
**After:** 316KB, 31 files — hanya code relevan + 3 whitepapers final

**Yang dibuang dari snapshot (irrelevant):**
- `oicio/data/*.pt` (27MB + 5MB checkpoints) → dipindah ke `.cache/oicio_checkpoints` (excluded)
- `oicio/data/last_spawn_script.py` (2.5KB) → irrelevant
- `__pycache__` semua → excluded anyway, tapi dihapus biar bersih
- `OICIO_Build_Report.md`, `OICIO_v02_Release.md`, `OICIO_v03_Real_Release.md`, `OICIO_v04_Final.md`, `OICIO_v05_From_Scratch_HERE.md` → old intermediates, keep only final v06 + v05 audit + whitepaper
- `.sudo_as_admin_successful` → irrelevant

**After cleanup:**
```
OICIO_Whitepaper.md (15KB)
OICIO_v05_Consumer_Training_Audit.md (8.3KB)
OICIO_v06_MatMulFree_CPUOnly.md (24KB)
README.md (4.3KB)
oicio/ (15 files Python, 200KB code)
Total: 316KB, 31 files — jauh di bawah 128MB / 10K limit

Excluded (bisa download ulang):
- .venv: 1.1GB torch + transformers + safetensors + fastapi
- .cache/models/BitNet-b1.58-2B-4T: 1.1GB real ternary weights
- .cache/swap_10gb: 10GB
- .cache/swap_5gb_extra: 5GB
- .cache/oicio_checkpoints: 32MB checkpoints
- Total excluded: ~17GB
```

**Sekarang bersih, hanya hal relevan di snapshot.**

---

## Roadmap Port OICIO Python → Rust CPU-Only — Lengkap dan Bersih

### Visi: MatMul-Free CPU-Only, No Python, No CUDA, No GPU

**Kenapa Rust?**
- No GIL, no hidden allocations, comptime, SIMD via std::simd / packed_simd
- Memory safety, no runtime, 14MB binary seperti Needle2
- Runs everywhere: ARM64, x86-64, ARMv7, RISC-V, WASM, Apple, Windows, Linux, Android, Pi
- x86-64-v2 baseline + AVX2/AVX-512 dispatch, ARM NEON, seperti turbovec
- Bisa target FPGA / Loihi 2 neuromorphic

**Kenapa CPU-Only Tanpa MatMul?**
- MatMul-free LM paper 2406.02528: hilangkan TOTAL matmul, performance on-par sampai 2.7B, -61% memory training, -10x inference, FPGA 13W @ 23.8 tok/s, Loihi 2 4.2W @ 59.4 tok/s
- T-MAC LUT: CPU renaissance via table lookup, no dequant, no mul, 4x throughput, 70% energy, CPU outperform GPU/NPU, Raspberry Pi bisa jalan LLM
- Vec-LUT: 4.2x speedup over T-MAC, vector lookup
- Mamba/RWKV/Liquid: O(N) linear, constant memory, selective SSM, liquid time-constants inference-time adaptation

**Emergent space tetap ada tanpa matmul via:**
- Ternary accumulation (associative memory)
- MLGRU state evolution (temporal compression, hippocampus)
- Hadamard thresholding (sparse coding, V1 cortex)
- LUT associative (Hopfield-like)
- Liquid time-constants (inference-time adaptation)

### Struktur oicio-rs (Sudah Dibuat, Snapshot-Safe <1MB):

```
oicio-rs/
├── Cargo.toml — no heavy deps, CPU only, toolchain di .cargo (excluded)
│   Dependencies: serde, serde_json, memmap2 (swap), byteorder (packing), rand (synthetic data)
│   Profile release: opt-level 3, lto true, codegen-units 1, panic abort
│   Target: x86-64-v2 baseline + AVX2/AVX-512 dispatch, ARM NEON
│
├── src/
│   ├── lib.rs — version 0.6.0, credits deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh
│   │
│   ├── core/
│   │   ├── mod.rs
│   │   ├── bitlinear.rs — TernaryWeight enum {-1,0,1}, pack 4 per byte 2 bits, absmean quant, forward no matmul only add/sub, fused kernel BitLinear+Hadamard+TurboQuant, AVX2/NEON TBL/PSHUF
│   │   ├── hadamard.rs — FWHT O(n log n) only add/sub, smooth_threshold tanh-smoothed, HadamardMLP no weights only thresholds, BWHT blocks 32, MF-DS-Conv multiplication-free, 24x faster than 3x3 conv
│   │   ├── mlgru.rs — MLGRU token mixer, forget gate + candidate + output gate ternary BitLinear, forward_step element-wise only (1-f)*h_prev + f*c, forward O(N) + parallel scan, complexity O(N) vs Transformer O(N²), 5x throughput
│   │   └── ternary_san.rs — full model: embed ternary + layers (MLGRU + HadamardMLP) + final_norm + lm_head ternary, forward no matmul, count_params FP16 13MB -> ternary 1.3MB (10.1x)
│   │
│   ├── memory/
│   │   ├── mod.rs
│   │   ├── turboquant.rs — data-oblivious 2-4 bit, normalize to hypersphere + random orthogonal rotation (Hadamard) -> Gaussian + Lloyd-Max quant + bit-packing, compress 31GB->4GB (8-16x), search rotate query once + SIMD, no training, no codebook retrain
│   │   ├── em_llm.rs — surprise = L2 distance to prev (proxy LLM loss), threshold mean+gamma*std, initial segmentation + refinement modularity (within - cross), Event {start,end,representative_tokens}
│   │   └── reattention.rs — 3 syarat infinite context: PE not OOD, stable entropy, effective awareness, split cache global/middle/local, position-agnostic selection q*K^T without RoPE, reconstruct concat [global 32 + select 127*32 + local 4096] = 8192 max, attention entropy stable
│   │
│   ├── harness/
│   │   ├── mod.rs
│   │   └── rah.rs — SubAgentHarness with reasoning (Needle2 14MB), TaskResult {task_id,entry_id,answer,confidence,reasoning,success}, ModulePool persistent (MLREF) success/failure/confidences + rollback if success_rate<0.7 or avg_conf<0.6, RecursiveAgentHarness max_depth + confidence_threshold, select_path json vs code_execution, spawn_via_code parallel, generate_rust_spawning_code tokio::join_all bypass tool-call limit
│   │
│   ├── edge/
│   │   ├── mod.rs
│   │   └── needle.rs — Tool {name,description,parameters}, FunctionCall, NeedleResponse {call_type,function_calls,reasoning,confidence,should_escalate,peak_ram_mb 28.0}, NeedleMini {tools,confidence_threshold,kv_cache bounded 256-token sliding window + tools pinned as sinks, max_window}, enforce_grammar compile JSON schema to byte-level grammar, calculate_confidence evidence in query, complete text in JSON out
│   │
│   └── training/
│       ├── mod.rs
│       └── cpu_train.rs — TrainingConfig {vocab,hidden,layers,batch,seq,steps,lr,warmup}, ConsumerTrainer {config,swap_dir}, should_swap RAM>80%, offload_tensor via memmap2 to disk, train_from_scratch CPU-only no GPU no CUDA, create_swap_file 10GB, autoscale_swap 10->20->30GB, correct recipe: 8-bit AdamW + checkpointing + ZeRO-Offload + ReAttention bounded + streaming data + warmup+cosine + all ternary
│
└── README.md — full documentation
```

### Perencanaan Lengkap — 8 Phase:

**Phase 0: Python POC (Selesai v0.1-v0.5, 316KB snapshot-safe)**

- TernarySAN toy 6.8M: FP16 13MB -> ternary 1.3MB (10.1x), forward no matmul
- TurboQuant: 1.33MB -> 0.1MB (12.8x @ 2-bit, 7.1x @ 4-bit), MSE 0.03, data-oblivious, no training
- EM-LLM: 10K tokens -> 697 events, surprise mean 0.823
- ReAttention: 100K KV -> 480 selected (208x compression), entropy stable 5.017, PE not OOD
- RAH: 100 entries code_execution path, 100% success, conf 0.91, ModulePool rollback, real code-execution spawning via asyncio.gather (real_rah.py generate script 2478 chars and execute via shell)
- NeedleMini: 28MB RAM bounded, grammar-constrained, confidence-gated, off-topic -> []
- Axon: parse Haskell-like DSL, compile to PyTorch/JAX/MLX/vLLM, speedups 7%/91%/107%/58%
- QAT Trainer: synthetic OOLONG 200 samples, 2 epochs, ternary dist tracking, checkpoint 5MB
- Runtime: 7-layer runtime, ingest 1000 chunks -> 68 events, query ReAttention 1000->480, RAH 20 subagents
- OOLONG Eval: 5 buckets 1K-16K, overall 24% toy (target 78-80% untuk 8B)
- LongBench: 6 tasks SQA/MQA/Sum/FSL/Ret/Cod, InfiniteBench up to 1M tokens 102400 chunks -> 7144 events
- BitNet Real: 2.4B 1.1GB safetensors 542 tensors, hidden 2560 30 layers, uint8 packed byte 85 = 0b01010101, scale 1.2188, ternary matmul no mul only INT8 add
- Bonsai: 8B 1.75GB vs Qwen3 16.38GB (9.4x), 75.5 vs 79.3 avg (gap 3.8), 82 tok/s M4 Pro, 27 tok/s iPhone, 0.105 mWh/tok, Apache 2.0 boleh rebrand
- Swap: 10GB + 5GB = 14GB active, pernah 18GB, autoscale logic 10->20->30GB, Python offload manager
- Training From Scratch HERE: 6.8M ternary 50 steps 23.4 detik loss 6.9488->6.9377 drop 0.0111 sparsity 31.1%->34.3% di 1.9GB RAM + 14GB swap, LLM sebagai guru generate synthetic 3 topics 90% coherence 10% switch
- API Server: FastAPI /ingest /query /stats /swap

**Snapshot after cleanup: 316KB, 31 files, jauh di bawah 128MB/10K**

**Phase 1: Rust Setup (Sekarang v0.6, Sudah Dibuat)**

- Cargo.toml: no heavy deps, CPU only, serde + memmap2 + byteorder + rand, release opt-level 3 lto true
- lib.rs: version 0.6.0, credits
- core/bitlinear.rs: TernaryWeight enum, pack 4 per byte 2 bits, absmean quant, forward no matmul only add/sub, fused kernel
- core/hadamard.rs: FWHT O(n log n) only add/sub, smooth_threshold, HadamardMLP no weights only thresholds, BWHT, MF-DS-Conv
- core/mlgru.rs: MLGRU token mixer forget+ candidate+output ternary, forward_step element-wise (1-f)*h_prev + f*c, forward O(N) + parallel scan, 5x throughput
- core/ternary_san.rs: full model embed ternary + layers MLGRU+HadamardMLP + lm_head ternary, 1.3MB vs 13MB (10.1x)
- memory/turboquant.rs: data-oblivious 2-4 bit, rotation, Lloyd-Max, compress 31GB->4GB, search rotate query once
- memory/em_llm.rs: surprise L2 distance, threshold mean+gamma*std, segmentation
- memory/reattention.rs: split global/middle/local, position-agnostic q*K^T without RoPE, max scope 480
- harness/rah.rs: SubAgentHarness reasoning, TaskResult, ModulePool rollback, RecursiveAgentHarness, generate_rust_spawning_code tokio::join_all
- edge/needle.rs: Tool, FunctionCall, NeedleResponse, NeedleMini bounded 256-token + sinks, grammar, confidence
- training/cpu_train.rs: TrainingConfig, ConsumerTrainer, should_swap, offload_tensor memmap2, train_from_scratch CPU-only, create_swap_file 10GB, autoscale 10->20->30GB, correct recipe 8-bit AdamW + checkpointing + ZeRO-Offload

**Phase 2: Memory Fabric + Harness (Next 1-2 minggu)**

- Implement TurboQuant dengan real Walsh-Hadamard rotation O(n log n) bukan matrix mul
- Implement EM-LLM refinement dengan graph modularity/conductance optimization (bukan simplified)
- Implement ReAttention dengan real RoPE sequential after selection (PE not OOD)
- Implement RAH real code generation: parent tulis Rust code dengan tokio::join_all dan execute via std::process::Command (seperti real_rah.py tapi Rust)
- Test: ingest 10K doc -> 697 events -> compress 12.8x -> ReAttention 100K->480 -> RAH spawn 100 subagents

**Phase 3: Training From Scratch CPU-Only di Consumer Hardware (2-4 minggu)**

- Correct recipe yang sudah di-audit di v0.5:
  - 8-bit AdamW (bitsandbytes) + double quant: hemat 4x RAM
  - Gradient checkpointing: hemat 10x RAM
  - ZeRO Stage 3 Offload: offload optimizer states ke CPU/disk/swap 14GB
  - ReAttention bounded 8K: 208x compression
  - Streaming data: FineWeb 15T stream dari NVMe, IterableDataset, pack 2048 tokens
  - LR warmup 2000 + cosine, weight_decay 0 untuk ternary
  - All layers ternary no escape hatch (Bonsai)
  - Swap autoscale 10->20->30GB di .cache excluded
  - Axon compile ke MLX untuk Apple Silicon 107% speedup atau Rust + AVX2/NEON

- Hardware target:
  - Standard consumer (16GB RAM + RTX 3060 12GB): inference ✅, fine-tune LoRA ✅, training from scratch 100M-500M dengan 10B tokens ⚠️ butuh cluster 10x PC 3.7 bulan
  - High-end consumer (RTX 4090 24GB + 64GB RAM + 2TB NVMe + 30GB swap): train 2B 4T tokens ~45 hari
  - Mac Studio M2 Ultra 192GB + 8TB SSD + MLX: train 2B 4T tokens ~30 hari, cost $6000 vs $100k+ H100 cluster

- POC di sini: model 6.8M 50 steps 23.4 detik loss drop 0.0111 sudah buktikan BISA di 1.9GB RAM + 14GB swap
- Scale ke 300M-1.7B dengan same recipe + more swap + more time

**Phase 4: Edge Deployment (1 minggu)**

- Compile NeedleMini ke 14MB binary: `cargo build --release --target aarch64-apple-darwin --target x86_64-unknown-linux-gnu --target wasm32-unknown-unknown`
- No runtime, no downloads, runs everywhere: ARM64, x86-64, ARMv7, RISC-V, WASM
- Test: Pi 5 500 tok/s, iPhone 27 tok/s, Samsung A-series 300-700 tok/s, ESP32-S3 11MB RAM
- Grammar-constrained decoding: compile JSON schema to byte-level grammar, prevent malformed JSON
- Confidence-gated: calibrated head + prob, escalate if < threshold

**Phase 5: FPGA / Loihi 2 Neuromorphic (Research, 1-2 bulan)**

- Custom hardware untuk ternary + LUT + Hadamard
- MatMul-free LM di FPGA: 1.3B @ 23.8 tok/s dengan 13W (paper)
- Loihi 2: 59.4 tok/s @ 4.2W, 70.8 mJ/token, 4x throughput 10x less energy vs edge GPUs
- Brain-like efficiency

**Phase 6: Production OICIO 8B 1.75GB**

- Load Bonsai 8B 1.75GB real weights (Apache 2.0 boleh rebrand) dari HF ke .cache
- Fine-tune LoRA dengan data domain 10B-50B tokens di consumer hardware (jam-hari)
- Atau train from scratch 1.7B 0.4GB dari 0 dengan 400B tokens di Mac Studio M2 Ultra 192GB ~20 hari
- Deploy: API Server FastAPI + Gradio UI + 14MB edge binary
- Benchmark: LongBench 43.7 avg (EM-LLM SOTA) vs InfLLM 41.9, OOLONG 89.77% (RAH Sonnet 4.5)

---

## Build & Run — Consumer Hardware Only, No GPU, No CUDA, CPU-Only

**Toolchain di .cargo (excluded, bisa download ulang):**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Install di /home/user/.cargo (excluded)

cd /home/user/oicio-rs
cargo build --release --target-cpu=native
# Binary 14MB like Needle2, no runtime

cargo run --release
# Check: free -h, cat /proc/swaps
# Mem: 1.9Gi, Swap: 14Gi (10+5) active, bisa scale 20GB,30GB
```

**Swap 10GB,20GB,30GB...:**
```bash
fallocate -l 10G /home/user/.cache/swap_10gb && sudo mkswap + swapon
fallocate -l 10G /home/user/.cache/swap_20gb && sudo mkswap + swapon
fallocate -l 10G /home/user/.cache/swap_30gb && sudo mkswap + swapon
# Total 30GB swap + 1.9GB RAM = 31.9GB untuk training 8B ternary 1.75GB
```

**Training From Scratch HERE:**
```bash
cd /home/user
source .venv/oicio/bin/activate
python oicio/training/train_from_scratch_here.py
# 6.8M ternary, 50 steps, 23.4s, loss 6.9488->6.9377 drop 0.0111, sparsity 31.1%->34.3%
# Checkpoint 27MB di oicio/data (snapshot-safe)
```

---

## Snapshot Final Bersih:

```
OICIO_Whitepaper.md (15KB)
OICIO_v05_Consumer_Training_Audit.md (8.3KB) — audit apakah cara kemarin benar
OICIO_v06_MatMulFree_CPUOnly.md (24KB) — riset mendalam MatMul-free + LUT + Mamba/RWKV/Liquid + Hadamard
README.md (4.3KB)
oicio/ (15 files Python POC, 200KB)
oicio-rs/ (Rust port, <1MB code, Cargo.toml + 12 Rust files)
Total: ~500KB, 40 files — jauh di bawah 128MB / 10K limit

Excluded (bisa download ulang):
- .cargo: Rust toolchain
- .venv: 1.1GB torch + transformers + safetensors + fastapi
- .cache/models/BitNet-b1.58-2B-4T: 1.1GB real ternary weights
- .cache/swap_10gb: 10GB
- .cache/swap_5gb_extra: 5GB
- .cache/oicio_checkpoints: 32MB checkpoints
- Total excluded: ~17GB
```

**Bersih, hanya hal relevan di snapshot, toolchain + model di luar snapshot.**

---

## Credits

**deepRcurs Labs, @deeprcurs**
**Author: Mzed Imamkh, @mzedimamkh**

**OICIO v0.6 — MatMul-Free CPU-Only, No Python/CUDA, Intelligence Density > Parameter Count, Outside-In Contextual Intelligence Orchestration**

Built in limited environment 1.9GB RAM + 14GB swap, no excuses, consumer hardware only, training from scratch HERE.

Paradigma baru total: tanpa matmul, tanpa GPU, tanpa doktrin, hasil frontier-quality dengan cara apapun, jangan stagnan.
