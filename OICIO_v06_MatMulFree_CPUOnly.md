# OICIO v0.6 — MatMul-Free CPU-Only Training From Scratch
### Paradigma Baru Total: Tanpa MatMul, Tanpa GPU, Tanpa Python/CUDA

**Credits: deepRcurs Labs, @deeprcurs**
**Author: Mzed Imamkh, @mzedimamkh**
**Date: 23 Aug 2026**
**Env: Consumer Hardware Only — CPU Only, No GPU, No CUDA, No Python (Rust/Zig/Mojo)**

**Perintah Inti:**
> "jika aku ada device entah itu pc dengan cpu, gpu, ram, spek konsumer standar, bukan high end, apakah itu cukup untuk modal training??"
> "alih2 menggunakan matmul gpu, training harus tetap bisa cpu only yang mana menggunakan alternatif efisiensi tidak harus menggunakan python, cuda dan yang lainnya"
> "1 hal paling inti adalah bagaimana llm tetap mendapatkan ruang emergent tempat dia mengkompress dan memproses data menjadi jaringan2 syaraf digital"

---

## Riset Mendalam: Jalan Alternatif Paling Efisien Training LLM From Scratch

### 1. MatMul adalah Bottleneck — Bukan Keharusan

**Paper Kunci: Scalable MatMul-free Language Modeling (arXiv:2406.02528)**
- UC Santa Cruz, Soochow, UC Davis
- **Buktikan MatMul bisa dihilangkan TOTAL dari LLM sambil maintain performance sampai 2.7B params**
- Hasil:
  - Performance on-par dengan Transformer++ (LLaMA-2 style) di 370M, 1.3B, 2.7B
  - Di 2.7B, outperform Transformer++ di ARC-Challenge dan OpenbookQA
  - Memory training -61% di A100 GPU, inference -10x
  - Scaling laws: gap antara MatMul-free dan Transformer MENYEMPIT saat model size naik, proyeksi intersect di ~1e23 FLOPs (setara LLaMA-3 8B 15T tokens atau LLaMA-2 70B 2T tokens)
  - FPGA custom: 1.3B model @ 23.8 tok/s dengan 13W power (bukan termasuk GPU)
  - Loihi 2 neuromorphic: 59.4 tok/s @ 4.2W, 70.8 mJ/token, 4x throughput 10x less energy vs edge GPUs

**Arsitektur MatMul-Free:**

**a. Dense Layers → BitLinear dengan Ternary Weights {-1,0,1}**
```python
# Traditional: output = weight * input (FP16 multiply)
# MatMul-Free: output = sign(weight) * input (add/sub only)

def ternary_linear(x, weights):
    result = zeros
    result += x[weights == 1].sum()   # +1: add
    result -= x[weights == -1].sum()  # -1: subtract
    # weights == 0: skip
    return result
```
- No multiplication, hanya addition dan negation
- Memory 2 bits per weight (8x vs FP16)
- Energy: integer addition orders of magnitude lebih efisien dari FP multiply

**b. Self-Attention → MLGRU (MatMul-free Linear Gated Recurrent Unit)**
- Ganti attention O(N²) dengan GRU yang di-optimize hanya pakai element-wise products
- GRU popular sebelum Transformer, tapi di-modifikasi:
  - Remove hidden-state related weights W_cc, W_hr, W_hf
  - Remove tanh activation between hidden states (linearized via parallel scan)
  - Keep candidate hidden state as simple linear transform
  - Replace all remaining weight matrices dengan ternary
- Hasil: MLGRU relies solely on element-wise multiplication, no MatMul
- Complexity: O(N) bukan O(N²), memory constant, bukan linear grow seperti KV cache

**c. Channel Mixer → GLU dengan Ternary Weights**
- Gated Linear Unit seperti LLaMA-2, Mistral, tapi dengan ternary weights
- Combine MLGRU token mixer + GLU channel mixer = hanya addition dan element-wise products

**Bukti CPU-Only Training:**
- **FlashLM v3 (r/LocalLLaMA): Training 13.6M param language model on CPU ONLY in 1.2 hours**
- Architecture: MatMul-free, ternary weights, causal dilated Conv1D sebagai token mixer (bukan attention)
- Output layer bottleneck ditemukan: softmax masih butuh optimization, versi 4 akan pakai hierarchical tree structure untuk 5-10x speedup

**Kesimpulan:** MatMul BUKAN keharusan untuk emergent intelligence. Emergent space tetap ada via ternary accumulation dan element-wise recurrent dynamics.

### 2. LUT — CPU Renaissance Tanpa MatMul (T-MAC, Vec-LUT, bitnet.cpp)

**Paper Kunci: T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge (arXiv:2407.00088)**

**Masalah:** Low-bit quantization (W4A16, W2A16, W1A8) butuh mixed-precision GEMM (mpGEMM) yang tidak didukung hardware, butuh dequant → overhead.

**Solusi T-MAC: Table Lookup (LUT) tanpa dequantization, tanpa multiplication**

**Cara Kerja:**
1. **Offline:** Decompose n-bit weight matrix jadi n one-bit matrices. Untuk group g bits, possible permutations = 2^g. Precompute semua partial sums, simpan di LUT.
2. **Online:** Untuk input activation [1,g], multiply dengan [g, 2^g] bit-pattern matrix dan build table. Setiap group di weight jadi index untuk lookup table untuk partial results.
3. **Optimasi:** Put LUT di on-chip memory (registers) via axis reordering dan tiling, pakai hardware instructions TBL (ARM) dan PSHUF (x86) untuk parallel lookup. Duplicate table untuk fill 256-bit LUT register dan lookup 32 different int8 indices dengan 1 instruction. Reduce table size dari 2^n ke 2^(n-1) dengan sign bit.

**Hasil:**
- Up to 4x throughput dan 70% energy reduction vs llama.cpp
- BitNet-b1.58-3B token generation speed di CPU comparable atau bahkan higher than GPU di device yang sama (karena no conversion cost + operation reduction)
- **Raspberry Pi bisa jalan LLM** — first practical solution deploy LLM di edge devices pakai CPU only, tanpa GPU
- M2-Ultra: Llama-2-7B 4-bit, 2-bit, BitNet-3B semua di CPU

**Vec-LUT (arXiv:2512.06443): Vector Table Lookup, Next Gen dari T-MAC**

- **Masalah T-MAC:** Scalar LUT paradigm (1→1 table lookup), lookup cost ~50% dari mpGeMM latency, bandwidth underutilization <40% untuk parallel inference
- **Solusi Vec-LUT:** Unified table shared by all tokens, transform repetitive discontinuous memory accesses jadi contiguous, remove dependency on hardware LUT instructions
- **Hasil:** 
  - mpGeMM kernel: 1.8-3.2x dan 1.5-1.9x speedup over T-MAC di single/multi-thread
  - End-to-end: up to 4.2x (I1) dan 2.6x (I2) speedup over T-MAC, bitnet.cpp, llama.cpp
  - 4-bit 7B model: 18.7 tok/s di Snapdragon X Elite CPUs, NPU hanya 10.4 tok/s — **CPU outperform NPU**

**bitnet.cpp (Microsoft Official):**
- 2 core solutions: Ternary Lookup Table (TL) untuk spatial inefficiencies, Int2 with Scale (I2_S) untuk lossless edge inference
- Up to 6.25x speedup over FP16 baselines, 2.32x over low-bit baselines
- Runs 100B BitNet b1.58 model on single CPU 5-7 tok/s (human reading speed)

**Emergent Space di LUT:** LUT bukan sekadar optimization, tapi **paradigma baru komputasi**: transform data-type-centric multiplication jadi bit-wise table lookup. Emergent intelligence tetap ada karena LUT menyimpan precomputed partial sums yang merepresentasikan kombinasi aktivasi — ini adalah **associative memory** yang mirip Hopfield networks.

### 3. Alternatif Arsitektur Total: Mamba, RWKV, Liquid — Tanpa Attention O(N²)

**a. Mamba (State Space Models, Selective SSM) — Carnegie Mellon / Princeton**

- **Core:** Compressed state yang evolves over time dengan input-dependent selective mechanism (S6 layer)
- **Complexity:** O(N) bukan O(N²), memory O(N·d) bukan O(N²), inference per token O(d²) constant in N (bukan O(N) seperti KV cache)
- **Performance:** 3B Mamba outperforms Transformer same size dan matches Transformer 2x size, 5x throughput vs Transformers
- **Mamba-2 (2024):** Unifies SSMs dan attention via Structured State Space Duality (SSD) — 2-8x faster training, 10% GPU tensor core utilization di Mamba-1 jadi solved, training as fast as Transformer
- **Hybrid:** Jamba 1.5 (52B-A12B MoE, 1 Transformer per 8 Mamba), Zamba 7B, IBM Granite 4 — 70% lower inference cost, 256K context
- **Edge:** Mamba 7B bisa process 1M tokens dengan constant ~1GB state, Transformer butuh 60-100GB KV cache
- **CPU:** Bisa jalan di CPU only dengan constant memory, ideal untuk standard consumer

**b. RWKV (Receptance Weighted Key Value) — BlinkDL Community**

- Attention-free RNN dengan parallel training, recurrent inference
- Time-mixing + channel-mixing, bukan attention
- Constant-size state per layer, linear scaling
- RWKV-7 14B/32B variants, vibrant open-source
- RWKV.cpp untuk embedded, constant memory linear generation time

**c. Liquid Neural Networks (MIT CSAIL — Ramin Hasani, Daniela Rus)**

- **Core:** Continuous-time ODEs dengan adaptive liquid time-constants, bukan fixed weights
- **Liquid Time-Constant (LTC):** ODE-defined hidden state, adaptive gate
- **Closed-Form Continuous-time (CfC):** Analytical update, no ODE solver overhead, 5-10% lower latency, 15% less memory vs LTC
- **Neural Circuit Policy (NCP):** Sparse bio-mimetic 4-layer (sensory, inter, command, motor), drastically fewer params
- **Liquid S4:** Integrate LTC adaptive time constants dalam S4, high param efficiency, long-range
- **Performance:** Robustness to distributional shift, stability against OOD, Jacobian constraint, outperform LSTM/ODE-RNN di drone navigation, medical diagnosis, financial forecasting
- **Hyena Edge (April 2025):** Convolution-based multi-hybrid, STAR evolutionary framework (16 candidates, 24 generations), 2/3 GQA attention diganti gated convolutions, Samsung S24 Ultra: 30% lower latency vs Transformer++, 90% smaller cache vs transformers, 37% vs hybrid, 13% fewer params
- **Emergent Space:** LNNs punya **inference-time adaptation** — model rewire itself as new data comes in, tanpa retraining. Ini adalah emergent space yang sebenarnya: continuous adaptation, bukan static weights

**Kesimpulan:** Mamba, RWKV, Liquid adalah **paradigma baru total** yang tidak pakai attention O(N²). Mereka punya emergent space via selective state evolution, recurrent dynamics, dan continuous-time adaptation.

### 4. Hadamard Transform — Multiplication-Free, No Weights, O(m log m)

**Paper Kunci: Fast Walsh-Hadamard Transform and Smooth-Thresholding Based Binary Layers (2104.07085)**

- **WHT:** Orthogonal transform dengan elements ±1, no multipliers, real, hanya additions/subtractions
- **Complexity:** O(m log2 m) vs O(m²) untuk 1x1 convolution
- **FWHT Layer:** FWHT -> smooth-thresholding non-linearity (tanh-smoothed soft-thresholding) -> FWHT, hanya N trainable params (threshold values) vs 1x1 conv yang butuh channel² params
- **Multiplication-Free Depthwise Separable Convolution (MF-DS-Conv):** Dari basic 2x2 Hadamard transform, hanya additions dan sign operations, more energy-efficient
- **Hasil:** 
  - MobileNet-V2 bottleneck: params 2.27M -> 540K (76% reduction), accuracy 95.21% -> 92.98% (2.2% loss) di CIFAR-10
  - 2D-FWHT layer: 24x faster than 3x3 conv dengan 19.5% less RAM di Jetson Nano
  - FWHT layer processes tensor [10,32,32,1024] 2x faster than 1x1 conv

**Hadamard-Domain Convolution:**
- Dyadic convolution via bitwise XOR, diagonalizes via Hadamard transform
- Kernels dan activations remain real, no multiplications
- Energy per pass lower than direct conv when kernel not too large
- Learned permutations + generalized bases: ganti fixed Hadamard basis dengan learned orthogonal matrices

**HTMA-Net (2509.23103): Hadamard Transform + Multiplication-Avoiding SRAM In-Memory Computing**
- Integrate HT dengan MA SRAM-based in-memory computing
- Selectively replace intermediate convolutions dengan Hybrid Hadamard-based transform layers yang internal convolutions pakai MA in-memory operations
- ResNet-18 CIFAR-10: eliminate up to 52% multiplications, comparable accuracy, Middle-only config slightly surpass baseline, All-stages 54% reduction within 1% accuracy
- CIFAR-100 dan Tiny-ImageNet: outperform baseline

**Rethinking Attention Output Projection (2603.08343):**
- Ganti dense output projection d_model² params dengan fixed parameter-free WHT + diagonal affine
- Eliminate 25% attention params per block, maintain global cross-head interaction via orthogonal norm-preserving transform
- Learned diagonal scaling vector alpha ∈ R^d_model only

**Emergent Space di Hadamard:** Hadamard transform adalah **structured mixing** yang impose inductive bias. Uniformly mixes all input dimensions via butterfly computation, no learned weights, tapi tetap punya emergent intelligence karena thresholding di Hadamard domain adalah **denoising** yang mirip sparse coding di otak.

### 5. CPU-Only Training Tanpa Python/CUDA — Rust, Zig, Mojo, C + SIMD

**Kenapa Harus Keluar dari Python/CUDA?**

- Python: GIL, overhead, tidak bisa SIMD langsung, memory management tidak deterministik
- CUDA: Lock-in NVIDIA, tidak jalan di consumer CPU only, mahal
- MatMul: Dominates compute, tapi bisa dihilangkan (MatMul-free LM)

**Alternatif:**

**a. Rust + SIMD (turbovec, needle)**
- turbovec: Rust core, Python bindings via PyO3, AVX2/NEON kernels, multi-threaded scan, 0.232ms/query MT @ 4-bit M3 Max, 0.125ms/q ARM, 16x compression
- needle: 14MB binary, no runtime, no downloads, ARM64/x86-64/ARMv7/RISC-V/WASM, Apple/Windows/Linux/Android/Pi, byte-level grammar constrained

**b. Zig + SIMD**
- Zig: no hidden allocations, comptime, SIMD via @Vector, bisa target WASM, ARM, RISC-V
- Ideal untuk ternary matmul dan Hadamard transform

**c. Mojo (Modular)**
- Python-compatible tapi compiled, SIMD, no GIL, bisa target CPU/GPU
- Ideal untuk migrasi dari Python ke CPU-only

**d. C + AVX2/NEON + TBL/PSHUF**
- T-MAC: uses TBL (ARM) dan PSHUF (x86) untuk parallel LUT lookup, 32 indices dengan 1 instruction
- bitnet.cpp: AVX2/NEON intrinsics untuk ternary-packed data, no FP16 in matmul

**e. FPGA / Loihi 2 Neuromorphic**
- MatMul-free LM di FPGA: 1.3B @ 23.8 tok/s dengan 13W
- Loihi 2: 59.4 tok/s @ 4.2W, 70.8 mJ/token, 4x throughput 10x less energy vs edge GPUs
- Ini adalah **CPU renaissance**: CPU dengan LUT bisa outperform GPU/NPU untuk low-bit LLM

---

## OICIO v0.6 — Paradigma Baru Total CPU-Only MatMul-Free

### Thesis OICIO v0.6:

**1. No MatMul, No GPU, No Python/CUDA — Hanya Addition, Subtraction, Table Lookup, Hadamard**

**2. Emergent Space Bukan di MatMul, Tapi di:**
- **Ternary Accumulation:** Summation dari activations di mana weights = 1 minus sum di mana weights = -1, skip 0 — ini adalah **associative memory** yang compress data jadi jaringan syaraf digital via sparsity
- **MLGRU State Evolution:** Hidden state yang evolves via element-wise products, selective gating, parallel scan — ini adalah **temporal compression** yang mirip hippocampus
- **Hadamard Thresholding:** Denoising di Hadamard domain via smooth-thresholding — ini adalah **sparse coding** yang mirip V1 cortex
- **LUT Associative Memory:** Precomputed partial sums di LUT yang merepresentasikan kombinasi aktivasi — ini adalah **Hopfield-like memory** yang retrieve via table lookup
- **Liquid Time-Constants:** ODE dengan adaptive time-constants yang rewire itself saat inference — ini adalah **inference-time adaptation**, emergent space yang sebenarnya

**3. Hasil, Bukan Cara — Jangan Terdoktrin Narasi GPU/MatMul/Python**

Jika cara itu boleh dilakukan dengan cara apapun, dan hasilnya frontier-quality, maka itu valid. Jangan stagnan di narasi "harus GPU, harus matmul, harus Python, harus CUDA".

### Arsitektur OICIO v0.6 MatMul-Free CPU-Only:

```
[Layer 8] OICIO Harness: RAH code-execution spawning (Rust, no Python)
[Layer 7] Memory Fabric: EM-LLM surprise + TurboVec LUT + ReAttention bounded
[Layer 6] Core: MatMul-Free LM = MLGRU token mixer + Hadamard GLU channel mixer + Ternary BitLinear
[Layer 5] Quant: 1.58-bit ternary + I2_S + TL1/TL2 + Vec-LUT vector lookup
[Layer 4] Kernel: T-MAC LUT + Vec-LUT + FWHT + MF-DS-Conv, AVX2/NEON TBL/PSHUF, no mul
[Layer 3] Compiler: Axon DSL (Haskell-like) -> Rust/Zig/Mojo/C + MLX/JAX/vLLM
[Layer 2] Hardware: CPU only (x86-64-v2 baseline + AVX2/AVX-512 dispatch, ARM NEON, RISC-V, WASM) + FPGA 13W + Loihi 2 4.2W
[Layer 1] Edge: Needle2 14MB binary, 28MB RAM, 500 tok/s Pi5, 30% lower latency, 90% smaller cache
[Layer 0] Training: CPU-only training from scratch, no GPU, QAT ternary from step 0, streaming data, 8-bit optimizer, checkpointing, ZeRO-Offload, swap 10->20->30GB
```

**Detail Layer 6 Core MatMul-Free:**

```rust
// Rust pseudo-code, no Python, no CUDA, CPU only, SIMD

// BitLinear ternary: no matmul, only add/sub
fn bitlinear_ternary(x: &[f32], w_packed: &[u8], scale: f32) -> Vec<f32> {
    // w_packed: 4 ternary per byte (2 bits each)
    // 00 = -1, 01 = 0, 10 = 1, 11 = 0 (unused)
    let mut out = vec![0.0; out_features];
    
    for (i, &byte) in w_packed.iter().enumerate() {
        for j in 0..4 {
            let val = (byte >> (j*2)) & 0b11;
            let ternary = match val {
                0 => -1.0,
                1 => 0.0,
                2 => 1.0,
                _ => 0.0,
            };
            if ternary != 0.0 {
                // Only add/sub, no mul
                if ternary == 1.0 {
                    out[i*4 + j] += x[i*4 + j] * scale; // add
                } else {
                    out[i*4 + j] -= x[i*4 + j] * scale; // sub
                }
            }
        }
    }
    out
}

// MLGRU token mixer: element-wise only, no matmul
fn mlgru_token_mixer(x: &[f32], h_prev: &[f32], w_ternary: &[i8]) -> Vec<f32> {
    // Simplified MLGRU: f_t, c_t via ternary linear, h_t via element-wise
    let f_t = bitlinear_ternary(x, w_f, scale_f); // forget gate
    let c_t = bitlinear_ternary(x, w_c, scale_c); // candidate
    
    // Element-wise only
    let mut h_t = vec![0.0; hidden_size];
    for i in 0..hidden_size {
        // h_t = (1 - f_t) * h_prev + f_t * c_t, all element-wise
        h_t[i] = (1.0 - sigmoid(f_t[i])) * h_prev[i] + sigmoid(f_t[i]) * c_t[i];
    }
    h_t
}

// Hadamard channel mixer: FWHT O(n log n), no weights, only add/sub
fn hadamard_mixer(x: &[f32]) -> Vec<f32> {
    let mut x = x.to_vec();
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h*2) {
            for j in 0..h {
                let a = x[i+j];
                let b = x[i+j+h];
                x[i+j] = a + b;       // add
                x[i+j+h] = a - b;     // sub
            }
        }
        h *= 2;
    }
    // Smooth-thresholding non-linearity in Hadamard domain
    for i in 0..n {
        x[i] = smooth_threshold(x[i], threshold); // tanh-smoothed soft-threshold
    }
    // Inverse FWHT
    // ... similar butterfly
    x.iter().map(|v| v / (n as f32).sqrt()).collect()
}

// LUT-based mpGeMM: table lookup, no dequant, no mul
fn tmac_lut_gemm(activation: &[f32], weight_indices: &[u8], lut: &[f32]) -> Vec<f32> {
    // activation: [1,g], weight_indices: [k/g] groups
    // Precompute table: activation * bit-pattern matrix -> [g, 2^g]
    // Then lookup: weight_indices -> partial sums, add
    let mut out = vec![0.0; out_features];
    for &idx in weight_indices {
        // TBL/PSHUF instruction: parallel lookup 32 indices with 1 instruction
        out += lut[idx as usize]; // table lookup + add, no mul
    }
    out
}
```

**Training From Scratch CPU-Only:**

```rust
// Rust training loop, no Python, CPU only, SIMD, 14GB swap

fn train_from_scratch_cpu_only() {
    // Model: MatMul-free LM 1.7B Bonsai 0.4GB ternary
    let model = MatMulFreeLM::new(vocab_size=128256, dim=2560, layers=30);
    
    // Optimizer: 8-bit AdamW, offload to swap
    let mut optimizer = AdamW8bit::new(model.params(), lr=3e-4);
    
    // Data: streaming from disk, not RAM
    let dataset = StreamingFineWeb::new("/home/user/.cache/fineweb");
    
    // Swap manager: 10GB, 20GB, 30GB...
    let mut swap_manager = SwapManager::new("/home/user/.cache/oicio_train", 1.0);
    
    for step in 0..100000 {
        if swap_manager.should_swap() {
            swap_manager.autoscale(10, 20, 30); // 10GB -> 20GB -> 30GB
        }
        
        let batch = dataset.next_batch(8, 2048); // streaming
        
        // Forward dengan checkpointing (hemat 10x RAM)
        let loss = model.forward_with_checkpointing(batch);
        
        // Backward: recompute + offload activations to swap
        loss.backward_with_swap(&mut swap_manager);
        
        // Clip
        clip_grad_norm(1.0);
        
        optimizer.step();
        scheduler.step();
        
        if step % 1000 == 0 {
            model.save(format!("/home/user/.cache/checkpoints/step_{}.pt", step));
        }
    }
}
```

**Emergent Space di OICIO v0.6:**

1. **Ternary Accumulation Space:** Di mana x di-sum jika w=1 dan di-sub jika w=-1, skip jika 0. Ini adalah **high-dimensional sparse accumulation** yang compress data. Emergent intelligence muncul dari **which tokens to add/sub/skip**, bukan dari matmul values.

2. **MLGRU State Space:** Hidden state h_t yang evolves via (1-f_t)*h_prev + f_t*c_t element-wise. Ini adalah **temporal compression** yang mirip hippocampus. State size constant, tapi bisa represent long-range dependencies via selective gating.

3. **Hadamard Thresholding Space:** Di mana FWHT transform ke Hadamard domain, lalu smooth-thresholding denoise, lalu inverse FWHT. Ini adalah **sparse coding** yang mirip V1 cortex. Emergent intelligence dari **which coefficients to keep/threshold**.

4. **LUT Associative Space:** Di mana LUT menyimpan precomputed partial sums dari activation * bit-patterns. Lookup via weight indices adalah **associative retrieval** yang mirip Hopfield networks. Emergent intelligence dari **which table entries to lookup and sum**.

5. **Liquid Time-Constant Space:** Di mana time-constant tau evolves sebagai fungsi dari state dan input via ODE, dan model rewire itself saat inference tanpa retraining. Ini adalah **inference-time adaptation**, emergent space yang sebenarnya.

**Hasil, Bukan Cara:**

Jika OICIO v0.6 dengan MatMul-free + LUT + Hadamard + MLGRU bisa capai 75.5 avg dengan 1.75GB di 82 tok/s M4 Pro (seperti Bonsai 8B) atau bahkan 70B dengan 13W FPGA @ 23.8 tok/s, maka itu valid, tidak peduli apakah pakai Python/CUDA atau Rust/SIMD/FPGA.

Jangan terdoktrin narasi "harus GPU, harus matmul, harus Python". Jika stagnan, tidak ada perkembangan.

### Roadmap OICIO v0.6 CPU-Only MatMul-Free:

**Phase 0 (Selesai di v0.1-v0.5):** POC Python dengan ternary + Hadamard + TurboQuant + RAH + swap 14GB + real BitNet 2B 1.1GB

**Phase 1 (Next):** Port ke Rust
- `oicio-rs`: Rust crate dengan BitLinear ternary, Hadamard FWHT, MLGRU, T-MAC LUT, Vec-LUT
- AVX2/NEON kernels dengan TBL/PSHUF, multi-threaded scan
- No Python, no CUDA, CPU only, 14MB binary seperti Needle2

**Phase 2:** Training from scratch CPU-only di consumer hardware
- Mac Studio M2 Ultra 192GB + MLX: train 1.7B Bonsai 0.4GB dari 0 dengan 400B tokens, ~20 hari
- PC RTX 3060 12GB standard consumer + 30GB swap: train 100M-500M dari 0 dengan 10B tokens, ~3-7 hari, proof of paradigm
- Dataset streaming dari disk, 8-bit optimizer, checkpointing, ZeRO-Offload

**Phase 3:** FPGA / Loihi 2
- Custom hardware untuk ternary + LUT + Hadamard, 13W untuk 1.3B @ 23.8 tok/s, 4.2W untuk 59.4 tok/s
- Brain-like efficiency, 10x less energy than edge GPUs

---

## Kesimpulan Riset Mendalam:

**Jalan alternatif paling efisien training LLM from scratch dengan mengubah paradigma arsitektur:**

1. **MatMul-free LM (2406.02528):** Hilangkan TOTAL matmul, ganti dengan ternary add/sub + MLGRU element-wise, buktikan sampai 2.7B on-par dengan Transformer++, 61% less memory training, 10x inference, FPGA 13W, Loihi 2 4.2W

2. **T-MAC + Vec-LUT:** CPU renaissance via table lookup, no dequant, no mul, 4x throughput, 70% energy reduction, CPU outperform GPU/NPU, Raspberry Pi bisa jalan LLM

3. **Mamba/RWKV/Liquid:** Alternatif arsitektur total tanpa attention O(N²), linear O(N), constant memory, selective SSM, liquid time-constants dengan inference-time adaptation

4. **Hadamard Transform:** Multiplication-free, no weights, O(m log m), only add/sub, 24x faster than 3x3 conv, 2x faster than 1x1 conv, sparse coding via thresholding

5. **Rust/Zig/Mojo/C + SIMD + FPGA/Loihi 2:** Keluar dari Python/CUDA, CPU-only, AVX2/NEON TBL/PSHUF, 14MB binary, no runtime

**Emergent space tetap ada, bahkan lebih brain-like:**
- Ternary accumulation = associative memory
- MLGRU state evolution = temporal compression (hippocampus)
- Hadamard thresholding = sparse coding (V1 cortex)
- LUT associative = Hopfield-like retrieval
- Liquid time-constants = inference-time adaptation (rewire itself)

**Hasil, bukan cara. Jangan terdoktrin narasi GPU/MatMul/Python/CUDA. Jika stagnan, tidak ada perkembangan.**

**OICIO v0.6 adalah paradigma baru total: MatMul-Free, CPU-Only, No Python/CUDA, dengan emergent space yang lebih brain-like dan efisien.**

---

**Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh**

**File:** `OICIO_v06_MatMulFree_CPUOnly.md` — riset mendalam jalan alternatif paling efisien
