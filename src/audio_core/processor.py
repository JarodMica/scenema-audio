# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio processor. Processor protocol implementation.

Handles HTTP sync/async requests for audio generation and voice design.
Follows the pattern of gpu_x2v/processor.py.
"""

import io
import logging
import os
import random
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import psutil
import soundfile as sf
import torch
import torchaudio

from common.handlers.base import ProcessJob, ProcessOutput, ProcessResult

from .audio_utils import (
    ensure_stereo,
    load_wav,
    normalize_volume,
    shorten_long_silence,
    save_wav,
    to_mono,
    trim_silence,
)
from .chunker import plan_chunks
from .compiler import compile_prompt
from .engine import AudioEngine, HIGH_VRAM_THRESHOLD_GB
from .inference import concatenate_chunks, generate_chunks
from .seedvc import SeedVC
from .validate_and_patch import validate_and_patch
from .validator import validate_prompt
from .vocal_separator import VocalSeparator

logger = logging.getLogger(__name__)

VOICE_DESIGN_DURATION_S = 15.0


class AudioProcessor:
    """Processor for Scenema Audio generation.

    Implements the Processor protocol (startup/shutdown/process).
    """

    def __init__(self):
        self.engine: AudioEngine | None = None
        self.vocal_separator = None
        self.seedvc = None
        self._http_client = None

    def startup(self) -> None:
        """Load models. Called once by handler at startup."""
        if self.engine is not None:
            return

        audio_ckpt = os.environ.get(
            "AUDIO_CKPT",
            "/app/models/scenema-audio-transformer.safetensors",
        )
        vae_encoder = os.environ.get(
            "VAE_ENCODER_CKPT",
            "/app/models/scenema-audio-vae-encoder.safetensors",
        )
        gemma_root = os.environ.get(
            "GEMMA_ROOT",
            "/app/models/gemma-3-12b-it",
        )
        pipeline_ckpt = os.environ.get(
            "PIPELINE_CKPT",
            "/app/models/ltx-2.3-22b-distilled.safetensors",
        )

        self.engine = AudioEngine(
            audio_ckpt_path=audio_ckpt,
            vae_encoder_path=vae_encoder,
            gemma_root=gemma_root,
            pipeline_ckpt_path=pipeline_ckpt,
        )
        self.engine.load()

        self.vocal_separator = VocalSeparator()
        self.seedvc = SeedVC()

        # Preload all models on high-VRAM cards (>= 40GB), keep resident
        vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available()
            else 0
        )
        preload_postprocessors = os.environ.get("SCENEMA_PRELOAD_POSTPROCESSORS", "1") == "1"
        self._keep_resident = vram_gb >= HIGH_VRAM_THRESHOLD_GB and preload_postprocessors
        if self._keep_resident:
            self.vocal_separator.load()
            self.seedvc.load()
            logger.info("All models preloaded and resident (%.0fGB VRAM)", vram_gb)
        else:
            logger.info("Low VRAM (%.0fGB), models loaded on-demand", vram_gb)

        logger.info("AudioProcessor ready")

    def shutdown(self) -> None:
        """Unload all models."""
        if self.engine:
            self.engine.unload()
            self.engine = None
        if self.vocal_separator:
            self.vocal_separator.unload()
            self.vocal_separator = None
        if self.seedvc and self.seedvc._loaded:
            self.seedvc.unload()
        logger.info("AudioProcessor shutdown")

    async def process(self, job: ProcessJob) -> ProcessResult:
        """Process an audio generation job."""
        start_time = time.time()
        started_at = datetime.now(timezone.utc).isoformat()
        torch.cuda.reset_peak_memory_stats()

        try:
            if self.engine is None:
                self.startup()

            config = self._parse_input(job)

            if config["mode"] == "voice_design":
                wav_np, sr = await self._voice_design(config)
            else:
                wav_np, sr = await self._generate(config)

            wav_bytes = self._encode_wav(wav_np, sr)
            processing_ms = int((time.time() - start_time) * 1000)

            return ProcessResult(
                job_id=job.job_id,
                success=True,
                output=ProcessOutput(
                    success=True,
                    data=wav_bytes,
                    content_type="audio/wav",
                    metadata=self._build_metadata(
                        config, wav_np, sr, processing_ms, started_at
                    ),
                ),
                processing_ms=processing_ms,
            )
        except Exception as e:
            logger.error("Processing failed: %s", e, exc_info=True)
            processing_ms = int((time.time() - start_time) * 1000)
            return ProcessResult(
                job_id=job.job_id,
                success=False,
                output=ProcessOutput(success=False, error=str(e)),
                error=str(e),
                processing_ms=processing_ms,
            )

    def _parse_input(self, job: ProcessJob) -> dict:
        """Parse and validate job input.

        Input schema:
            prompt: str           - Required. <speak> XML string.
            mode: str             - "generate" (default) or "voice_design".
            reference_voice_url: str | None - URL to reference audio for voice cloning.
            reference_voice_path: str | None - Internal local upload path for UI use.
            background_sfx: bool  - Keep background SFX (default: false, strips via MelBandRoFormer).
            validate: bool        - Enable Whisper speech validation (default: false).
                                    When true, each generated chunk is transcribed by faster-whisper
                                    (GPU, float16, ~1GB VRAM) and compared against the expected text.
                                    If word match ratio falls below 60%, the chunk is regenerated with
                                    extended duration and a new seed (up to 3 retries), keeping the
                                    best result. Adds <1s per chunk on GPU. When false, each chunk is
                                    generated once with no quality gate, which is faster and sufficient
                                    for most prompts.
            seed: int             - Base seed (-1 for random).
        """
        inp = job.input

        prompt = inp.get("prompt")
        if not prompt:
            raise ValueError("Missing required 'prompt' field")

        mode = inp.get("mode", "generate")
        if mode not in ("generate", "voice_design"):
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'generate' or 'voice_design'"
            )

        result = validate_prompt(prompt)
        if not result.valid:
            raise ValueError(f"Invalid prompt XML: {'; '.join(result.errors)}")

        seed = inp.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 999999)

        return {
            "prompt": prompt,
            "mode": mode,
            "reference_voice_url": inp.get("reference_voice_url"),
            "reference_voice_path": self._normalize_reference_voice_path(
                inp.get("reference_voice_path")
            ),
            "background_sfx": inp.get("background_sfx", False),
            "validate": inp.get("validate", True),
            "seed": seed,
            "pace": inp.get("pace", 1.5),
            "min_match_ratio": inp.get("min_match_ratio", 0.90),
            "vc_cfg_rate": inp.get("vc_cfg_rate", 0.5),
            "vc_steps": inp.get("vc_steps", 25),
            "skip_vc": inp.get("skip_vc", False),
        }

    async def _voice_design(self, config: dict) -> tuple[np.ndarray, int]:
        """Generate a 15s voice sample for voice design."""
        compiled = compile_prompt(config["prompt"])
        vc, ac = self.engine.encode_text(compiled.prompt)
        result = self.engine.generate(vc, ac, VOICE_DESIGN_DURATION_S, config["seed"])

        wav = result.waveform_np
        sr = result.sample_rate

        if not config["background_sfx"]:
            wav = self._strip_background(wav, sr)

        wav = trim_silence(wav, sr)
        wav = shorten_long_silence(wav, sr)
        wav = normalize_volume(wav, sr)

        return wav, sr

    async def _generate(self, config: dict) -> tuple[np.ndarray, int]:
        """Full generation pipeline with chunking and post-processing."""
        chunks = plan_chunks(
            config["prompt"], base_seed=config["seed"], pace=config["pace"]
        )
        logger.info("Planned %d chunk(s)", len(chunks))

        ref_wav_path = None
        cleanup_ref_wav = False
        if config["reference_voice_path"]:
            ref_wav_path = config["reference_voice_path"]
        elif config["reference_voice_url"]:
            ref_wav_path = await self._download_reference(config["reference_voice_url"])
            cleanup_ref_wav = True

        # skip_vc: seed every chunk with the reference audio's tail latent,
        # identical to how inter-chunk chaining works. The model sees the
        # reference as "what I just generated" and continues in that voice.
        # Disables the normal chaining (each chunk chains from the ref, not
        # from the previous chunk) to keep the voice anchored to the reference.
        anchor_latent = None
        if config["skip_vc"] and ref_wav_path:
            ref_wav, ref_sr = load_wav(ref_wav_path)
            ref_mono = to_mono(ref_wav)
            tail_seconds = 3.0
            tail_samples = int(tail_seconds * ref_sr)
            if ref_mono.shape[0] > tail_samples:
                ref_tail = ref_mono[-tail_samples:]
            else:
                ref_tail = ref_mono
            anchor_latent = self.engine.encode_reference(ref_tail, ref_sr)
            logger.info(
                "Anchor mode: every chunk seeded from %.1fs reference tail",
                ref_tail.shape[0] / ref_sr,
            )

        with torch.inference_mode():
            results = generate_chunks(
                self.engine,
                chunks,
                ref_latent=anchor_latent,
                anchor_ref=anchor_latent is not None,
                validate=config["validate"],
                min_match_ratio=config["min_match_ratio"],
            )

        wav, sr = concatenate_chunks(results)

        # Strip background music/SFX from the concatenated audio (single pass)
        if not config["background_sfx"]:
            wav = self._strip_background(wav, sr)

        # Cap silence — scale with pace
        max_silence = min(0.5 * config["pace"], 1.5)
        wav = shorten_long_silence(
            wav, sr, max_duration=max_silence, target_duration=max_silence * 0.6
        )

        # Apply SeedVC when: reference voice provided, or multiple chunks (voice consistency).
        # Skip for single-chunk generations without reference (preserves SFX).
        needs_vc = ref_wav_path or len(results) > 1
        if not config["skip_vc"] and needs_vc:
            wav = self._apply_seedvc(
                wav,
                sr,
                results,
                ref_wav_path,
                vc_steps=config["vc_steps"],
                vc_cfg_rate=config["vc_cfg_rate"],
            )

        # Post-SeedVC alignment trimming (disabled by default, needs refinement)
        if config.get("patch", False):
            expected_text = " ".join(c.expected_text for c in chunks)
            wav = validate_and_patch(wav, sr, expected_text)

        # Ensure stereo final output
        wav = ensure_stereo(wav)

        if cleanup_ref_wav and ref_wav_path and os.path.exists(ref_wav_path):
            os.unlink(ref_wav_path)

        return wav, sr

    def _normalize_reference_voice_path(self, path: str | None) -> str | None:
        """Validate an internal UI upload path and return an absolute path."""
        if not path:
            return None

        upload_dir = Path(os.environ.get("UPLOAD_DIR", "/app/uploads")).resolve()
        ref_path = Path(path).resolve()

        try:
            ref_path.relative_to(upload_dir)
        except ValueError as exc:
            raise ValueError("reference_voice_path must be inside UPLOAD_DIR") from exc

        if not ref_path.is_file():
            raise ValueError(f"reference_voice_path does not exist: {ref_path}")

        return str(ref_path)

    def _strip_background(self, wav_np: np.ndarray, sr: int) -> np.ndarray:
        """Strip background music/SFX using MelBandRoFormer.

        Loads the model on-demand and unloads after to free VRAM.
        """
        if self.vocal_separator is None:
            return wav_np

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            input_path = f.name
        vocals_path = input_path.replace(".wav", "_vocals.wav")

        try:
            if not self._keep_resident:
                self.vocal_separator.load()
            stereo = ensure_stereo(wav_np)
            save_wav(stereo, sr, input_path)
            self.vocal_separator.separate(input_path, vocals_path, None)
            vocals, _ = load_wav(vocals_path)
            return vocals
        except Exception as e:
            logger.warning("Vocal separation failed: %s", e)
            return wav_np
        finally:
            if not self._keep_resident:
                self.vocal_separator.unload()
            for p in [input_path, vocals_path]:
                if os.path.exists(p):
                    os.unlink(p)

    def _apply_seedvc(
        self,
        wav: np.ndarray,
        sr: int,
        chunk_results: list,
        ref_wav_path: str | None,
        vc_steps: int = 20,
        vc_cfg_rate: float = 0.5,
    ) -> np.ndarray:
        """Apply SeedVC voice cloning.

        If reference_voice_url provided: convert against reference.
        If no reference: convert all against chunk 0 (first chunk sets identity).
        """
        if self.seedvc is None:
            logger.info("SeedVC not available, skipping voice cloning")
            return wav

        try:
            if not self._keep_resident:
                self.seedvc.load()
            with tempfile.TemporaryDirectory() as tmp:
                source_path = os.path.join(tmp, "source_22k.wav")
                target_path = os.path.join(tmp, "target_22k.wav")

                source_mono = to_mono(wav)
                source_t = torch.from_numpy(source_mono).float().unsqueeze(0)
                source_22k = torchaudio.functional.resample(source_t, sr, 22050)
                save_wav(source_22k.squeeze(0).numpy(), 22050, source_path)

                if ref_wav_path:
                    target_wav, target_sr = load_wav(ref_wav_path)
                    target_mono = to_mono(target_wav)
                    target_t = torch.from_numpy(target_mono).float().unsqueeze(0)
                    target_22k = torchaudio.functional.resample(
                        target_t, target_sr, 22050
                    )
                    save_wav(target_22k.squeeze(0).numpy(), 22050, target_path)
                else:
                    chunk0 = chunk_results[0].waveform_np
                    chunk0_mono = to_mono(chunk0)
                    chunk0_t = torch.from_numpy(chunk0_mono).float().unsqueeze(0)
                    chunk0_22k = torchaudio.functional.resample(
                        chunk0_t, chunk_results[0].sample_rate, 22050
                    )
                    save_wav(chunk0_22k.squeeze(0).numpy(), 22050, target_path)

                converted = self.seedvc.convert(
                    source_path,
                    target_path,
                    diffusion_steps=vc_steps,
                    cfg_rate=vc_cfg_rate,
                )

                conv_t = torch.from_numpy(converted).float().unsqueeze(0)
                result = torchaudio.functional.resample(conv_t, 22050, sr)
                wav = result.squeeze(0).numpy()
                wav = ensure_stereo(wav)

        except Exception as e:
            logger.error("SeedVC failed: %s", e, exc_info=True)
        finally:
            if not self._keep_resident:
                try:
                    self.seedvc.unload()
                except Exception:
                    pass

        return wav

    async def _download_reference(self, url: str) -> str:
        """Download reference audio from URL to temp file."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)

        response = await self._http_client.get(url)
        response.raise_for_status()

        suffix = ".wav"
        if "mp3" in url.lower() or "mpeg" in response.headers.get("content-type", ""):
            suffix = ".mp3"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(response.content)
            logger.info(
                "Downloaded reference: %d bytes to %s", len(response.content), f.name
            )
            return f.name

    def _encode_wav(self, wav_np: np.ndarray, sr: int) -> bytes:
        """Encode numpy array to WAV bytes."""
        buf = io.BytesIO()
        sf.write(buf, wav_np, sr, format="WAV")
        return buf.getvalue()

    def _build_metadata(
        self,
        config: dict,
        wav_np: np.ndarray,
        sr: int,
        processing_ms: int,
        started_at: str = "",
    ) -> dict:
        """Build comprehensive metadata matching x2v pattern."""
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        vram_total_mb = 0
        vram_peak_mb = 0
        if torch.cuda.is_available():
            vram_total_mb = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2
            )
            vram_peak_mb = round(torch.cuda.max_memory_allocated() / 1024**2)

        cpu_cores_total = os.cpu_count() or 0
        system_ram_gb = round(psutil.virtual_memory().total / 1024**3)
        disk = shutil.disk_usage("/")

        return {
            "duration_s": round(wav_np.shape[0] / sr, 2),
            "sample_rate": sr,
            "mode": config["mode"],
            "seed": config["seed"],
            "background_sfx": config["background_sfx"],
            "has_reference_voice": (
                config["reference_voice_url"] is not None
                or config["reference_voice_path"] is not None
            ),
            "validate": config["validate"],
            "processing_ms": processing_ms,
            "vram_peak_mb": vram_peak_mb,
            "vram_total_mb": vram_total_mb,
            "gpu": gpu_name,
            "cpu_cores_total": cpu_cores_total,
            "system_ram_gb": system_ram_gb,
            "disk_total_gb": round(disk.total / 1024**3, 1),
            "disk_free_gb": round(disk.free / 1024**3, 1),
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
