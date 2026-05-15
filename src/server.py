# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio standalone server.

Thin FastAPI wrapper around the production AudioProcessor.
"""

import asyncio
import base64
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download, snapshot_download
import uvicorn

try:
    import gradio as gr

    from ui import UPLOAD_DIR, create_ui
except ImportError:
    gr = None
    UPLOAD_DIR = None
    create_ui = None

logger = logging.getLogger("scenema-audio")

# Must be set before any torch import
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

from audio_core.processor import AudioProcessor  # noqa: E402
from common.handlers.base import ProcessJob  # noqa: E402

# ── Model download ──────────────────────────────────────────────

HF_REPO = "ScenemaAI/scenema-audio"
GEMMA_REPO = "google/gemma-3-12b-it"
SEEDVC_REPO = "Plachta/Seed-VC"
BIGVGAN_REPO = "nvidia/bigvgan_v2_22khz_80band_256x"
WHISPER_REPO = "openai/whisper-small"

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))


def _download_models():
    """Download missing model checkpoints from HuggingFace."""

    token = os.environ.get("HF_TOKEN")

    # Audio transformer (INT8 by default, bf16 when AUDIO_CKPT points at it)
    audio_ckpt = Path(os.environ.get(
        "AUDIO_CKPT",
        str(MODEL_DIR / "scenema-audio-transformer-int8.safetensors"),
    ))
    if not audio_ckpt.exists():
        audio_filename = audio_ckpt.name
        if audio_filename not in {
            "scenema-audio-transformer-int8.safetensors",
            "scenema-audio-transformer.safetensors",
        }:
            audio_filename = "scenema-audio-transformer-int8.safetensors"

        if audio_filename.endswith("-int8.safetensors"):
            logger.info("Downloading audio transformer (INT8, ~4.9 GB)...")
        else:
            logger.info("Downloading audio transformer (bf16, ~9.8 GB)...")

        hf_hub_download(
            HF_REPO,
            audio_filename,
            local_dir=str(audio_ckpt.parent),
            token=token,
        )

    # Pipeline checkpoint
    pipeline_ckpt = Path(os.environ.get(
        "PIPELINE_CKPT",
        str(MODEL_DIR / "scenema-audio-pipeline.safetensors"),
    ))
    if not pipeline_ckpt.exists():
        logger.info("Downloading pipeline checkpoint (~7.1 GB)...")
        hf_hub_download(
            HF_REPO,
            "scenema-audio-pipeline.safetensors",
            local_dir=str(pipeline_ckpt.parent),
            token=token,
        )

    # VAE encoder (small, may already be baked)
    vae_ckpt = Path(os.environ.get(
        "VAE_ENCODER_CKPT",
        str(MODEL_DIR / "scenema-audio-vae-encoder.safetensors"),
    ))
    if not vae_ckpt.exists():
        logger.info("Downloading VAE encoder (~42 MB)...")
        hf_hub_download(
            HF_REPO,
            "scenema-audio-vae-encoder.safetensors",
            local_dir=str(vae_ckpt.parent),
            token=token,
        )

    # Gemma 3 12B IT
    gemma_root = Path(os.environ.get("GEMMA_ROOT", str(MODEL_DIR / "gemma-3-12b-it")))
    if not gemma_root.exists() or not any(gemma_root.glob("*.safetensors")):
        logger.info("Downloading Gemma 3 12B IT (~24 GB, gated model)...")
        snapshot_download(
            GEMMA_REPO,
            local_dir=str(gemma_root),
            ignore_patterns=["*.gguf"],
            token=token,
        )

    # SeedVC
    seedvc_path = Path(os.environ.get("SEEDVC_PATH", "/app/seed-vc"))
    seedvc_cache = seedvc_path / "checkpoints"
    if not seedvc_cache.exists() or not any(seedvc_cache.glob("*.pth")):
        logger.info("Downloading SeedVC checkpoints (~1.6 GB)...")
        hf_cache = seedvc_cache / "hf_cache"
        hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HUB_CACHE"] = str(hf_cache)
        hf_hub_download(
            SEEDVC_REPO,
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
            local_dir=str(seedvc_cache),
            token=token,
        )
        hf_hub_download(
            SEEDVC_REPO,
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
            local_dir=str(seedvc_cache),
            token=token,
        )
        snapshot_download(BIGVGAN_REPO, local_dir=str(hf_cache / "bigvgan"))
        snapshot_download(WHISPER_REPO, local_dir=str(hf_cache / "whisper-small"))


# ── FastAPI app ─────────────────────────────────────────────────

processor = AudioProcessor()
_semaphore = asyncio.Semaphore(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _download_models()
    processor.startup()
    logger.info("Scenema Audio ready on port %s", os.environ.get("PORT", "8000"))
    yield
    processor.shutdown()


app = FastAPI(title="Scenema Audio", lifespan=lifespan)

if gr is not None and create_ui is not None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app = gr.mount_gradio_app(
        app,
        create_ui(processor, _semaphore),
        path="/ui",
        allowed_paths=[str(UPLOAD_DIR)],
    )
else:
    logger.warning("Gradio is not installed; UI will not be available")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: Request):
    body = await request.json()

    job = ProcessJob(
        job_id=str(uuid.uuid4()),
        input=body,
    )

    async with _semaphore:
        result = await processor.process(job)

    if not result.success:
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": result.error or "Generation failed",
            },
        )

    output = result.output
    audio_b64 = base64.b64encode(output.data).decode() if output.data else None

    return {
        "status": "succeeded",
        "audio": audio_b64,
        "content_type": output.content_type or "audio/wav",
        "metadata": output.metadata or {},
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
