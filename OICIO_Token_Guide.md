# OICIO Token Guide — Cara Buat GH + HF Token Tanpa Kartu Kredit/HP
**Credits: deepRcurs Labs, @deeprcurs**
**Author: Mzed Imamkh, @mzedimamkh**
**Akun Agent: deeprcurs-agent (harus di-invite ke organisasi deepRcurs)**

---

## 1. GitHub Token (GH Token) — Untuk Push ke deepRcurs/OICIO + MyBinder Auto-Build

**Syarat:** Cuma email GitHub, **tanpa kartu kredit, tanpa verifikasi HP**

**Step-by-Step:**

1.  Buka https://github.com/join → Daftar akun **deeprcurs-agent** (jika belum ada) — cuma email + password, no credit card, no HP
2.  Login sebagai **deeprcurs-agent** di https://github.com/login
3.  Buka https://github.com/settings/tokens → **Personal access tokens** → **Tokens (classic)**
4.  Klik **Generate new token** → **Generate new token (classic)**
5.  Isi:
    - **Note:** `OICIO Rust CPU-Only`
    - **Expiration:** `90 days` (atau Custom 1 tahun)
    - **Select scopes:** Centang `repo` (full control of private repositories) — ini centang semua sub-repo
      - `repo:status`, `repo_deployment`, `public_repo`, `repo:invite`, `security_events`
    - Jika mau push ke organisasi deepRcurs, juga centang `workflow` dan `write:packages`
6.  Klik **Generate token** di bawah
7.  **Copy token** yang muncul: `ghp_...` (diawali `ghp_`, panjang 40 char) — **hanya muncul sekali, copy sekarang!**
8.  Simpan token di tempat aman

**Cara Invite deeprcurs-agent ke Organisasi deepRcurs:**

1.  Login sebagai owner organisasi **deepRcurs** di GitHub
2.  Buka https://github.com/orgs/deepRcurs/people → **Invite member**
3.  Masukkan username **deeprcurs-agent** → **Invite**
4.  Login sebagai **deeprcurs-agent**, cek email atau https://github.com/orgs/deepRcurs/invitation → **Accept invitation**
5.  Sekarang **deeprcurs-agent** adalah member organisasi **deepRcurs**, bisa push ke `deepRcurs/OICIO`

**Repo yang akan aku buat/push:**
- `deepRcurs/OICIO` — public repo untuk code Python + Rust + whitepapers
- Isinya: `oicio/` 200KB + `oicio-rs/` 102KB + `OICIO_Whitepaper.md` + `README.md` + `Cargo.toml`
- MyBinder.org akan auto-build dari repo ini tanpa akun, tanpa kartu kredit

**Token akan aku simpan di:**
- `/home/user/.cache/tokens/gh_token.txt` — di `.cache` yang **excluded dari snapshot** (aman, tidak ke-save, bisa di-download ulang)
- Tidak akan aku push ke GitHub, tidak akan aku tampilkan di log

---

## 2. HuggingFace Token (HF Token) — Untuk Push Model 1.1GB + Checkpoint ke Hub

**Syarat:** Cuma email, **tanpa kartu kredit, tanpa verifikasi HP**

**Step-by-Step:**

1.  Buka https://huggingface.co/join → Daftar akun **deeprcurs-agent** (jika belum ada) — cuma email + password, no credit card, no HP
    - Jika sudah punya akun **deeprcurs-agent**, login di https://huggingface.co/login
2.  Buka https://huggingface.co/settings/tokens
3.  Klik **Create new token**
4.  Isi:
    - **Name:** `OICIO`
    - **Type:** **Write** (bisa read + write repo, bukan Read saja)
    - **Repositories:** All (atau specific `deepRcurs/OICIO`)
5.  Klik **Create token**
6.  **Copy token** yang muncul: `hf_...` (diawali `hf_`, panjang 30+ char) — **copy sekarang!**
7.  Simpan token di tempat aman

**Organisasi deepRcurs di HuggingFace:**

1.  Jika organisasi **deepRcurs** sudah ada di HuggingFace, buka https://huggingface.co/organizations/deepRcurs
2.  Invite member **deeprcurs-agent** ke organisasi deepRcurs (via Settings → Members → Invite)
3.  Login sebagai **deeprcurs-agent**, accept invitation

**Repo yang akan aku buat/push:**
- `deepRcurs/OICIO` di HuggingFace Hub — untuk model + checkpoint
- Free tier: **100GB private free, 5TB public best-effort**, no credit card, no HP
- Isinya:
  - BitNet 2B real weights 1.1GB (`model.safetensors`) dari `/home/user/.cache/models/BitNet-b1.58-2B-4T` (sudah ada di sini, excluded)
  - Checkpoint OICIO 27MB (`oicio_from_scratch_here.pt`) — training from scratch HERE 6.8M 50 steps loss drop 0.0111
  - Training log JSON
  - Model card + whitepapers

**Token akan aku simpan di:**
- `/home/user/.cache/tokens/hf_token.txt` — di `.cache` excluded (aman)

---

## 3. Cara Share Token ke Agent (Aman, Tanpa Manual Darimu)

**Opsi A: Share via File .env di Sini (Aku Automasi, Snapshot-Safe?)**

- `.env` tidak di-exclude dari snapshot, jadi tidak aman jika ada token
- Lebih aman pakai `/home/user/.cache/tokens/` yang excluded

**Opsi B: Share via ask_user Custom Response (Yang Kamu Lakukan Sekarang)**

- Kamu sudah share custom response "lakukan guide dua2 nya" — next step kamu share token `ghp_...` dan `hf_...` di custom response ask_user yang akan aku buat
- Aku akan simpan di `.cache/tokens/` excluded, tidak akan aku tampilkan di log, tidak akan ke-save di snapshot

**Opsi C: Share via HuggingFace Space Secrets (Paling Aman untuk Production)**

- Jika sudah punya Space di HF, masuk ke Space Settings → Variables and Secrets → New Secret → Name: `HF_TOKEN` Value: `hf_...` dan `GH_TOKEN` Value: `ghp_...`
- Space bisa baca via `os.environ["HF_TOKEN"]`

**Rekomendasi untuk Sekarang (POC): Opsi B — Share via Custom Response di Sini**

Aku akan buat ask_user lagi yang minta token `ghp_...` dan `hf_...`, kamu paste di custom response, aku simpan di `.cache/tokens/` excluded, langsung automasi push.

---

## 4. Script Automasi yang Akan Aku Jalankan Setelah Dapat Token

**Setelah dapat GH token dan HF token, aku akan jalankan:**

```bash
# Setup tokens di .cache/tokens/ (excluded)
mkdir -p /home/user/.cache/tokens
echo "ghp_..." > /home/user/.cache/tokens/gh_token.txt
echo "hf_..." > /home/user/.cache/tokens/hf_token.txt
chmod 600 /home/user/.cache/tokens/*

# Setup Git dengan token
git config --global user.name "deeprcurs-agent"
git config --global user.email "deeprcurs-agent@deeprcurs.ai"
git config --global credential.helper store

# Clone atau init repo deepRcurs/OICIO
# Jika repo belum ada, aku buat via GitHub API dengan token
# Jika sudah ada, aku clone

# Push oicio/ Python + oicio-rs/ Rust + whitepapers
cd /home/user
git init
git remote add origin https://ghp_...@github.com/deepRcurs/OICIO.git
git add OICIO_Whitepaper.md OICIO_v05_Consumer_Training_Audit.md OICIO_v06_MatMulFree_CPUOnly.md OICIO_Rust_Roadmap.md README.md oicio/ oicio-rs/
git commit -m "OICIO v0.6 Rust CPU-Only — MatMul-Free, No GPU, No Python/CUDA — Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh"
git push -u origin main

# MyBinder auto-build dari GitHub repo deepRcurs/OICIO tanpa akun

# Push ke HuggingFace Hub
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id="deepRcurs/OICIO", exist_ok=True, token="hf_...")
api.upload_folder(folder_path="/home/user/oicio", repo_id="deepRcurs/OICIO", token="hf_...")
api.upload_file(path_or_fileobj="/home/user/.cache/models/BitNet-b1.58-2B-4T/model.safetensors", path_in_repo="models/BitNet-b1.58-2B-4T/model.safetensors", repo_id="deepRcurs/OICIO", token="hf_...")
```

**Semua automasi oleh aku, tanpa manual darimu, hanya butuh token.**

---

## 5. Checklist untuk Kamu

- [ ] Buat akun GitHub **deeprcurs-agent** (jika belum ada) — cuma email, tanpa kartu kredit/HP
- [ ] Buat GH token classic dengan scope `repo` — copy `ghp_...`
- [ ] Invite **deeprcurs-agent** ke organisasi GitHub **deepRcurs** (via orgs/deepRcurs/people → Invite member)
- [ ] Accept invitation sebagai **deeprcurs-agent**
- [ ] Buat akun HuggingFace **deeprcurs-agent** (jika belum ada) — cuma email, tanpa kartu kredit/HP
- [ ] Buat HF token Write — copy `hf_...`
- [ ] Invite **deeprcurs-agent** ke organisasi HuggingFace **deepRcurs** (jika ada)
- [ ] Share token `ghp_...` dan `hf_...` via custom response ask_user berikutnya

**Setelah checklist selesai, aku langsung automasi push ke deepRcurs/OICIO di GitHub + HF Hub + MyBinder.**

---

**Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh**

**File ini di snapshot-safe (418KB), token akan di .cache/tokens/ excluded (aman)**
