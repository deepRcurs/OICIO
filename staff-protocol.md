---
license: apache-2.0
---

# Staff Protocol — OICIO Internal Operations

**Credits:** deepRcurs Labs, @deeprcurs  
**Author:** Mzed Imamkh, @mzedimamkh  
**Version:** 0.6.0 — MatMul-Free CPU-Only  
**Account:** deeprcurs-staff  

**GitHub:** https://github.com/deepRcurs/OICIO  
**HuggingFace Hub:** https://huggingface.co/deepRcurs/OICIO  

This document contains internal operational rules for snapshot management, swap handling, and free-tier infrastructure automation. It is separated from the public README to keep the public documentation professional, academic, and focused on architecture.

## Implementation — Snapshot Rules

**Snapshot limit:** 128MB / 10K files

**Snapshot-safe (<1MB):** Code only `oicio/` Python POC + `oicio-rs/` Rust CPU-only + whitepapers + README + Dockerfile + app.py + .github/workflows

**Excluded (can re-download, outside snapshot):**

- `.cache/`: Rust toolchain (Cargo, rustup), Python venv (torch 191MB CPU, transformers, safetensors, fastapi, gradio), models (BitNet 2B 1.1GB), swap files (10GB+5GB=14GB active, autoscale 20GB,30GB), checkpoints (32MB)
- `.venv/`: Python venv
- `.cargo/`, `target/`, `oicio-rs/target/`: Rust build artifacts
- `__pycache__/`, `*.pt`, `*.safetensors`: Cache and weights
- Total excluded: ~17GB

**Rules:**

- Do not disturb snapshot: keep code <128MB / 10K files, toolchain in `.cache` excluded
- If RAM insufficient by calculation, swap before OOM: OS swap 10GB,20GB,30GB... in `.cache` + Python/Rust offload via memmap2

**Proof:**

- Snapshot: 64 files, 526KB total after cleanup, 57 files 466KB after Rust port
- Swap: 14GB active (10+5), autoscale logic to 20GB demonstrated
- Training: 6.8M model 50 steps 23.4s loss drop 0.0111 sparsity 31->34% in 1.9GB RAM + 14GB swap
- Real weights: BitNet 2B 1.1GB safetensors 542 tensors loaded, ternary matmul no mul
- Rust binary: 501KB native + 607KB musl static (like Needle2 14MB) + 4.5MB generated via rustc CPU-only, all MatMul-free CPU-only

## Infrastructure — Free Tier Without Credit Card/Phone

**For automation without manual steps, using 2 tokens (GH + HF) shared:**

- **GitHub Token `ghp_...` (repo scope):** Push to `deepRcurs/OICIO`, setup Actions Secrets, trigger training in GitHub Actions Free (2-core CPU, 7GB RAM, 2000 min/month, no credit card, no phone verification). Already proven: Run 32607984794 status completed success with 11 steps success including Rust build 501KB and training from scratch HERE and push checkpoint to HF Hub via secret.

- **HF Token `hf_...` (write):** Push to HuggingFace Hub `deepRcurs/OICIO` (100GB private free, 5TB public best-effort, no credit card, no phone). Already proven: 68 files including BitNet 2B 1.1GB real weights + training logs pushed from GitHub Actions.

- **MyBinder.org:** No account needed, just GitHub repo public https://github.com/deepRcurs/OICIO, VM 2GB RAM, auto-build https://mybinder.org/v2/gh/deepRcurs/OICIO/main, no credit card, no phone.

- **Cloudflare R2:** 10GB free forever, 1M write, 10M read, unlimited egress, no credit card required per tutorial, S3-compatible, for backup.

- **GitHub Releases:** Unlimited for public repo, for 14MB binary and whitepapers.

**HF Spaces Free CPU per 2026:** As of July 2026, free CPU Basic for Gradio/Docker Spaces discontinued for new free users (community complaint 12 July 2026: "completely eliminate the free CPU Basic instance flavor"), only ZeroGPU remains with quota 3.5 min/day and Static Spaces free. So training in HF Spaces free is not feasible, but GitHub Actions free still works and Hub storage still free.

**Final URLs:**
- GitHub: https://github.com/deepRcurs/OICIO
- HF Hub: https://huggingface.co/deepRcurs/OICIO
- MyBinder: https://mybinder.org/v2/gh/deepRcurs/OICIO/main
- Latest Successful Run: https://github.com/deepRcurs/OICIO/actions/runs/32607984794

**Swap Management:**
- OS-level swap files in `.cache` (excluded): 10GB, 20GB, 30GB... autoscale if RAM >80%
- Python-level offload via memmap2: offload KV cache, gradients, optimizer states to disk before OOM
- Rust-level: memmap2 for tensor offloading

**Training Automation:**
- GitHub Actions workflow `.github/workflows/train.yml` triggers on push to main
- Steps: setup swap 10GB+5GB=14GB, build Rust 501KB, training from scratch HERE 6.8M 50 steps, push checkpoint to HF Hub via HF_TOKEN secret
- Proven SUCCESS Run 32607984794 with 11 steps success

**Storage:**
- HF Hub: 100GB private free, 5TB public best-effort, no credit card, no phone, just email
- R2: 10GB free forever, no credit card per tutorial
- GitHub Releases: unlimited for public repo

---

**This file is internal staff protocol, separated from public README to keep public docs professional and academic.**

**Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh**
