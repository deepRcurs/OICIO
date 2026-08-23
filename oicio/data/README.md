# OICIO Data — Training Data and Checkpoints

**Credits:** deepRcurs Labs, @deeprcurs  
**Author:** Mzed Imamkh, @mzedimamkh  

## Overview

This directory contains training data and checkpoints for OICIO, following snapshot rules: code in snapshot-safe (<128MB), toolchain, dependencies, and large artifacts in `.cache` (excluded, can be re-downloaded).

As per requirements: dataset and trainer is LLM itself (LLM as teacher, source of knowledge, dataset, and auditor).

## Dataset Generation — LLM as Teacher

No large datasets are downloaded to snapshot (would exceed 128MB limit). Synthetic data is generated on-the-fly in RAM with swap offloading if needed.

**Synthetic datasets:**

- **OOLONG Synthetic:** Generates entries with user_id and entity classification, 3 topics with 90% coherence and 10% switch (surprise event boundary), mimicking Oolong-Synthetic benchmark (199 samples, 13 buckets 1K-4M tokens, average 629K tokens)

- **LongBench-like:** Generates QA, summarization, code tasks across 6 categories (SQA, MQA, Sum, FSL, Ret, Cod)

- **InfiniteBench-like:** Generates PassKey retrieval with hidden passkey at random position, tested up to 1M tokens (102400 chunks → 7144 events)

All generated on-the-fly in 1.9GB RAM + 14GB swap, not stored permanently (snapshot-safe).

## Checkpoints

- `training_log_here.json` — Training log from scratch HERE: 6.8M ternary, 50 steps, 23.4s, loss 6.9488→6.9377 drop 0.0111, sparsity 31.1%→34.3%, FP16 13MB → Ternary 1.3MB (10.1x), swap 14GB active, consumer hardware only

- Real checkpoints (BitNet 2B 1.1GB, Bonsai 8B 1.75GB) stored in `/home/user/.cache/models` (excluded from snapshot, can re-download via `hf download`)

- Large checkpoints (e.g., `oicio_from_scratch_here.pt` 27MB, `ternary_san_qat.pt` 5MB) moved to `/home/user/.cache/oicio_checkpoints` (excluded) to keep snapshot clean (316KB → 510KB after cleanup)

## Usage

```python
from oicio.training.qat_trainer import SyntheticOOLONGDataset
dataset = SyntheticOOLONGDataset(num_samples=1000, seq_len=128)

from oicio.training.train_from_scratch_here import LLMasTeacherDataset
dataset = LLMasTeacherDataset(vocab_size=1024, seq_len=128, num_samples=10000)
# Generates synthetic with 3 topics, LLM as teacher
```

LLM is teacher: generates data, trains, audits, repeats.

## Storage — Free Tier Without Credit Card/Phone

- **HuggingFace Hub:** Public best-effort up to 5TB, private 100GB free, no credit card, no phone verification, just email. Already proven push of BitNet 2B 1.1GB real weights + training logs via HF token.

- **Cloudflare R2:** 10GB free forever, 1M write, 10M read, unlimited egress, no credit card required per tutorial, S3-compatible.

- **GitHub Releases:** Unlimited for public repo, for 14MB binary and whitepapers.

- **MyBinder.org:** No account needed, just GitHub repo public, VM 2GB RAM, auto-build.

## Snapshot Compliance

Code in `oicio/data/` is snapshot-safe: README.md 1.2KB + training_log_here.json 570 bytes = ~2KB.

Large artifacts (*.pt, *.safetensors) excluded via `.gitignore` and stored in `.cache` (excluded from snapshot).
