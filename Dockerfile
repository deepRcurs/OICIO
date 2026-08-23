# OICIO Dockerfile for MyBinder.org and HuggingFace Spaces
# Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh
# Consumer Hardware Only, MatMul-Free CPU-Only, No GPU, No CUDA

FROM rust:1.98-slim as rust-builder

# Install dependencies for Rust + Python
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Setup Rust toolchain in .cache (excluded from snapshot logic, but in Docker it's okay)
ENV CARGO_HOME=/home/user/.cache/cargo
ENV RUSTUP_HOME=/home/user/.cache/rustup
ENV PATH=$CARGO_HOME/bin:$PATH

# Create user
RUN useradd -m -s /bin/bash user
WORKDIR /home/user

# Copy OICIO Rust code
COPY --chown=user:user oicio-rs/ /home/user/oicio-rs/

# Build Rust binary 501KB native + 607KB musl static (like Needle2 14MB)
RUN mkdir -p /home/user/.cache/cargo /home/user/.cache/rustup /home/user/.cache/oicio-rs-target
RUN chown -R user:user /home/user/.cache
USER user
RUN cd /home/user/oicio-rs && \
    export CARGO_TARGET_DIR=/home/user/.cache/oicio-rs-target && \
    cargo build --release --bin oicio && \
    cargo build --release --bin oicio_real_rah && \
    cargo build --release --bin oicio_turboquant_real

# Python stage
FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

WORKDIR /home/user

# Copy Python code + Rust binaries
COPY --from=rust-builder /home/user/oicio-rs/ /home/user/oicio-rs/
COPY --from=rust-builder /home/user/.cache/oicio-rs-target/release/oicio /home/user/oicio-rs/oicio
COPY --from=rust-builder /home/user/.cache/oicio-rs-target/release/oicio_real_rah /home/user/oicio-rs/oicio_real_rah
COPY oicio/ /home/user/oicio/
COPY *.md /home/user/
COPY app.py /home/user/
COPY requirements.txt /home/user/ 2>/dev/null || echo "numpy\ntorch --index-url https://download.pytorch.org/whl/cpu\nhuggingface_hub\nsafetensors\nfastapi\nuvicorn\ngradio\npsutil\ntqdm" > /home/user/requirements.txt

# Setup Python venv in .venv (excluded from snapshot logic, but in Docker okay)
RUN python3 -m venv /home/user/.venv/oicio && \
    /home/user/.venv/oicio/bin/pip install --upgrade pip && \
    /home/user/.venv/oicio/bin/pip install -r /home/user/requirements.txt

# Setup swap 10GB, 20GB, 30GB... in .cache (excluded)
RUN mkdir -p /home/user/.cache && \
    fallocate -l 10G /home/user/.cache/swap_10gb && \
    chmod 600 /home/user/.cache/swap_10gb && \
    mkswap /home/user/.cache/swap_10gb || true

# Expose ports for API and Gradio
EXPOSE 8000 7860

# ENV for Rust
ENV CARGO_HOME=/home/user/.cache/cargo
ENV RUSTUP_HOME=/home/user/.cache/rustup
ENV PATH=/home/user/.cache/cargo/bin:/home/user/.venv/oicio/bin:$PATH
ENV CARGO_TARGET_DIR=/home/user/.cache/oicio-rs-target

# Default command: run Gradio app (for HF Spaces) or Jupyter (for MyBinder)
# MyBinder will run jupyter, HF Spaces will run app.py
CMD ["bash", "-c", "source /home/user/.venv/oicio/bin/activate && swapon /home/user/.cache/swap_10gb 2>/dev/null || true && python app.py"]

# Labels for credits
LABEL maintainer="deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh"
LABEL description="OICIO — Optimized Infinite Context Intelligence Orchestration — MatMul-Free CPU-Only — Frontier at 1.58-bit"
LABEL version="0.6.0"
