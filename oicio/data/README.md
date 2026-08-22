# OICIO Data — LLM sebagai Dataset, Trainer, dan Guru

**Credits:** deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh

Sesuai aturan:
- Dataset dan trainer adalah kamu (LLM sumber pengetahuan)
- Sebagai guru yang membimbing, memproses dan mengaudit
- Semua data yang bisa di-download ulang taruh di .cache (excluded)

## Dataset Generation (LLM as Teacher)

Kita tidak download dataset besar (akan melebihi 128MB snapshot).
Kita generate synthetic data sebagai guru:

1. **OOLONG Synthetic:** Generate entries user_id, entity classification
2. **LongBench-like:** Generate QA, summarization, code tasks
3. **InfiniteBench-like:** Generate passkey retrieval, long dialog

Semua di-generate on-the-fly di RAM 1.9GB, tidak disimpan permanen (snapshot-safe).

## Checkpoints

- `ternary_san_qat.pt` — toy model 0.25MB ternary (snapshot-safe)
- Real checkpoints (2B, 8B) akan di .cache (excluded) atau HuggingFace

## Cara Generate

```python
from oicio.training.qat_trainer import SyntheticOOLONGDataset
dataset = SyntheticOOLONGDataset(num_samples=1000, seq_len=128)
```

LLM adalah guru: generate data, train, audit, repeat.
