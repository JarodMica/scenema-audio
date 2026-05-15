# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT
#
# Scenema Audio OSS image
#
# Only models under 1 GB are baked into the image.
# Large checkpoints (audio transformer, pipeline, Gemma, SeedVC)
# are downloaded on first run and cached in a Docker volume.
#
# Build:
#   docker compose build
#
# Run:
#   HF_TOKEN=your_token docker compose up

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

LABEL org.opencontainers.image.title="Scenema Audio"
LABEL org.opencontainers.image.description="Zero-shot expressive voice cloning and speech generation"
LABEL org.opencontainers.image.url="https://scenema.ai"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip \
    git curl wget xz-utils gcc \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ffmpeg
RUN curl -fSL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz \
    | tar -xJ --strip-components=2 -C /usr/local/bin/ \
      ffmpeg-n7.1-latest-linux64-gpl-7.1/bin/ffmpeg \
      ffmpeg-n7.1-latest-linux64-gpl-7.1/bin/ffprobe

WORKDIR /app

# =============================================================================
# PyTorch + LTX Pipeline
# =============================================================================

RUN pip install --no-cache-dir \
    "torch==2.7.1" "torchaudio==2.7.1" \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
    "numpy==2.2.6" \
    "transformers==4.57.6" \
    "accelerate==1.13.0" \
    "safetensors==0.7.0" \
    "sentencepiece==0.2.1" \
    "ltx-core @ git+https://github.com/Lightricks/LTX-2.git@41d924371612b692c0fd1e4d9d94c3dfb3c02cb3#subdirectory=packages/ltx-core" \
    "ltx-pipelines @ git+https://github.com/Lightricks/LTX-2.git@41d924371612b692c0fd1e4d9d94c3dfb3c02cb3#subdirectory=packages/ltx-pipelines"

# SageAttention 2.2.0 (optional, speeds up attention)
ARG SAGE_WHEEL_URL=https://huggingface.co/ScenemaAI/scenema-audio/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
RUN pip install --no-cache-dir "${SAGE_WHEEL_URL}" 2>/dev/null || true

# =============================================================================
# SeedVC + MelBandRoFormer (architecture code only, weights downloaded at runtime)
# =============================================================================

RUN git clone --depth 1 https://github.com/Plachtaa/seed-vc.git /app/seed-vc \
    && cd /app/seed-vc && pip install --no-cache-dir \
    "scipy==1.13.1" "librosa==0.10.2" \
    "huggingface-hub==0.36.2" "munch==4.0.0" "einops==0.8.0" \
    "descript-audio-codec==1.0.0" "pydub==0.25.1" \
    "soundfile==0.12.1" \
    "hydra-core==1.3.2" "pyyaml==6.0.3" "python-dotenv==1.2.2" "diffusers==0.37.1" \
    "onnxruntime==1.25.0" "funasr==1.3.1"

RUN git clone --depth 1 https://github.com/kijai/ComfyUI-MelBandRoFormer /app/melband_roformer_node
RUN pip install --no-cache-dir "rotary-embedding-torch==0.8.9" "beartype==0.22.9"

# =============================================================================
# Server + service dependencies
# =============================================================================

RUN pip install --no-cache-dir \
    "fastapi==0.112.2" \
    "starlette==0.38.2" \
    "uvicorn[standard]==0.46.0" \
    "gradio==4.44.1" \
    "httpx==0.28.1" \
    "psutil==7.2.2" \
    "bitsandbytes==0.49.2"

# Kokoro TTS (82 MB, CPU-only, baked)
RUN pip install --no-cache-dir "kokoro==0.9.4" \
    && python3 -c "from kokoro import KPipeline; KPipeline(lang_code='a')" \
    && echo "Kokoro model cached"

# faster-whisper (speech validation)
RUN pip install --no-cache-dir "faster-whisper==1.2.1" "ctranslate2==4.7.1"

# =============================================================================
# Bake small models (<1 GB)
# =============================================================================

# VAE encoder (42.7 MB)
RUN pip install --no-cache-dir huggingface_hub \
    && python3 -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('ScenemaAI/scenema-audio', 'scenema-audio-vae-encoder.safetensors', local_dir='/app/models')"

# MelBandRoFormer (436 MB)
RUN wget -q -O /app/models/MelBandRoformer_fp16.safetensors \
    https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors

# =============================================================================
# Copy service code
# =============================================================================

COPY src/ /app/src/
COPY tests/ /app/tests/

# Verify key imports
RUN PYTHONPATH=/app/src python3 -c "\
import torch; print(f'torch {torch.__version__} cuda={torch.version.cuda}'); \
from common.handlers.base import ProcessJob; print('common shim OK'); \
from audio_core.compiler import compile_prompt; print('audio_core OK')"

# =============================================================================
# Environment
# =============================================================================

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src
ENV HF_HUB_CACHE=/app/models/hf_cache

# Default model paths (all large models downloaded at runtime into /app/models)
ENV AUDIO_CKPT=/app/models/scenema-audio-transformer-int8.safetensors
ENV VAE_ENCODER_CKPT=/app/models/scenema-audio-vae-encoder.safetensors
ENV PIPELINE_CKPT=/app/models/scenema-audio-pipeline.safetensors
ENV GEMMA_ROOT=/app/models/gemma-3-12b-it
ENV MELBAND_MODEL_PATH=/app/models/MelBandRoformer_fp16.safetensors
ENV MELBAND_NODE_PATH=/app/melband_roformer_node
ENV SEEDVC_PATH=/app/seed-vc
ENV UPLOAD_DIR=/app/uploads

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV LIBRARY_PATH=/usr/local/cuda-12.8/compat
ENV PORT=8000

RUN mkdir -p /app/uploads

EXPOSE 8000

CMD ["python3", "-m", "server"]
