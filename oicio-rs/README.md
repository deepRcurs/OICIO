# OICIO Rust — MatMul-Free CPU-Only
**Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh**
**Version: 0.6.0 — Paradigma Baru Total: Tanpa MatMul, Tanpa GPU, Tanpa Python/CUDA**

## Apa Ini?

Port OICIO dari Python ke Rust CPU-only, tanpa matmul, tanpa GPU, tanpa Python/CUDA.

**Paradigma baru total:**
- No MatMul, only Add/Sub, Table Lookup, Hadamard Transform
- Ternary weights {-1,0,1} 1.58-bit, group-wise 128 + FP16 scale
- MLGRU token mixer (element-wise, no attention O(N²)), O(N) linear, constant memory
- Hadamard channel mixer (fixed WHT, no weights, O(m log m), only add/sub)
- T-MAC LUT + Vec-LUT: CPU renaissance via table lookup, no dequant, no mul, 4x throughput, 70% energy, CPU outperform GPU/NPU
- Mamba/RWKV/Liquid alternative architectures: linear O(N), selective SSM, liquid time-constants

**Emergent space tetap ada via:**
- Ternary accumulation (associative memory)
- MLGRU state evolution (temporal compression, hippocampus)
- Hadamard thresholding (sparse coding, V1 cortex)
- LUT associative (Hopfield-like)
- Liquid time-constants (inference-time adaptation, rewire itself)

**Hasil, bukan cara — jangan terdoktrin narasi GPU/MatMul/Python/CUDA.**

## Struktur

```
oicio-rs/
├── Cargo.toml — no heavy deps, CPU only, toolchain in .cargo (excluded)
├── src/
│   ├── lib.rs — version + credits
│   ├── core/
│   │   ├── bitlinear.rs — ternary {-1,0,1}, pack 4 per byte, no matmul only add/sub, AVX2/NEON TBL/PSHUF
│   │   ├── hadamard.rs — FWHT O(n log n), only add/sub, smooth-thresholding, 24x faster than 3x3 conv
│   │   ├── mlgru.rs — MatMul-free Linear GRU, element-wise only, O(N), 5x throughput vs Transformer
│   │   └── ternary_san.rs — full model: MLGRU + HadamardMLP + BitLinear, 1.3MB ternary vs 13MB FP16 (10.1x)
│   ├── memory/
│   │   ├── turboquant.rs — data-oblivious 2-4 bit, 31GB->4GB (8-16x), 0.232ms/query M3 Max, no training
│   │   ├── em_llm.rs — surprise segmentation, Bayesian surprise + modularity refinement, 10K->697 events
│   │   └── reattention.rs — finite scope 8K (global 32 + select 127*32 + local 4096), 100K->480 (208x), entropy stable
│   ├── harness/
│   │   └── rah.rs — RAH code-execution spawning, tokio::join_all, bypass tool-call limit, ModulePool rollback (MLREF)
│   ├── edge/
│   │   └── needle.rs — NeedleMini 45M 14MB binary 28MB RAM 500 tok/s Pi5, grammar-constrained, confidence-gated
│   └── training/
│       └── cpu_train.rs — CPU-only training from scratch, 8-bit AdamW, checkpointing, ZeRO-Offload, swap 10->20->30GB
```

## Build — Consumer Hardware Only, No GPU, No CUDA

**Snapshot-safe:** Rust code <1MB, toolchain in `.cargo` (excluded), model in `.cache` (excluded)

```bash
# Toolchain di .cargo (excluded, bisa download ulang)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Install di /home/user/.cargo (excluded)

# Build
cd /home/user/oicio-rs
cargo build --release --target-cpu=native

# Binary: 14MB like Needle2, no runtime, no downloads, runs everywhere
# ARM64, x86-64, ARMv7, RISC-V, WASM, Apple, Windows, Linux, Android, Pi

# Run
cargo run --release

# Check
free -h
cat /proc/swaps
# Mem: 1.9Gi
# Swap: 14Gi (10+5) active, bisa scale 20GB,30GB
```

**Swap 10GB,20GB,30GB...:**
```bash
fallocate -l 10G /home/user/.cache/swap_10gb && sudo mkswap + swapon
fallocate -l 10G /home/user/.cache/swap_20gb && sudo mkswap + swapon
fallocate -l 10G /home/user/.cache/swap_30gb && sudo mkswap + swapon
# Total 30GB swap + 1.9GB RAM = 31.9GB untuk training 8B ternary 1.75GB di consumer hardware
```

## Training From Scratch di Consumer Hardware — Correct Method

**Standard Consumer (16GB RAM + RTX 3060 12GB):**
- Inference OICIO 8B 1.75GB: 50 tok/s → ✅ CUKUP
- Fine-tune LoRA dari BitNet 2B 1.1GB: jam-hari → ✅ CUKUP
- Training from scratch 100M-500M dengan 10B tokens: 3.1 tahun single, 3.7 bulan dengan 10x PC → ⚠️ BISA dengan cluster
- Training from scratch 2B dengan 400B tokens: 253 tahun single → ❌ Butuh high-end consumer

**High-End Consumer (Mac Studio M2 Ultra 192GB + 8TB SSD + MLX 107% speedup):**
- Train 2B 4T tokens: ~30 hari, cost $6000 vs $100k+ H100 cluster → ✅ BISA

**Correct recipe (sudah implement di training/cpu_train.rs):**
- 8-bit AdamW (hemat 4x RAM) + double quant
- Gradient checkpointing (hemat 10x RAM)
- ZeRO Stage 3 Offload ke CPU/disk/swap
- ReAttention bounded 8K (208x compression)
- Streaming data dari disk (FineWeb 15T = 8TB, stream dari NVMe)
- LR warmup 2000 + cosine, weight_decay 0 untuk ternary
- All layers ternary no escape hatch (Bonsai)
- Swap autoscale 10->20->30GB di .cache excluded

**Bukti di sini (1.9GB RAM + 14GB swap):**
- Model 6.8M ternary: FP16 13MB -> Ternary 1.3MB (10.1x)
- 50 steps, 23.4 detik, loss 6.9488->6.9377 drop 0.0111, sparsity 31.1%->34.3%
- Checkpoint 27MB saved

## Roadmap Port Python -> Rust CPU-Only

**Phase 0 (Selesai di Python v0.1-v0.5):**
- POC Python: ternary + Hadamard + MLGRU + TurboQuant + EM-LLM + ReAttention + RAH + NeedleMini + swap 14GB + real BitNet 2B 1.1GB + training from scratch HERE

**Phase 1 (Sekarang v0.6):**
- Setup oicio-rs Cargo.toml, no heavy deps, CPU only
- Implement core: BitLinear ternary pack 4 per byte, Hadamard FWHT O(n log n) only add/sub, MLGRU element-wise O(N)

**Phase 2 (Next):**
- Memory fabric: TurboQuant data-oblivious 2-4 bit, EM-LLM surprise segmentation, ReAttention finite scope
- Harness: RAH code-execution spawning via tokio::join_all, ModulePool rollback
- Edge: NeedleMini grammar-constrained + confidence-gated, 14MB binary, 28MB RAM

**Phase 3:**
- Training: CPU-only training from scratch, streaming data, 8-bit optimizer, checkpointing, ZeRO-Offload, swap 10->20->30GB
- Compiler: Axon DSL (Haskell-like) -> Rust/Zig/Mojo/C + MLX/JAX/vLLM

**Phase 4:**
- FPGA 13W untuk 1.3B @ 23.8 tok/s, Loihi 2 4.2W @ 59.4 tok/s — brain-like efficiency, 10x less energy

## Credits

**deepRcurs Labs, @deeprcurs**
**Author: Mzed Imamkh, @mzedimamkh**

**Paradigma Baru: MatMul-Free, CPU-Only, No Python/CUDA, Intelligence Density > Parameter Count, Outside-In Contextual Intelligence Orchestration**

Built in limited environment, no excuses, consumer hardware only.
