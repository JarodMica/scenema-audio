# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""SeedVC voice conversion for Scenema Audio.

Converts the voice identity of generated audio to match a reference speaker
while preserving prosody, rhythm, and emotion. Uses the Seed-VC model with
DiT backbone, CAMPPlus speaker encoder, and BigVGAN vocoder.

Expects 22050Hz mono WAV input for both source and target.
"""

import inspect
import logging
import os
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_SEEDVC_PATH = Path(os.environ.get("SEEDVC_PATH", "/app/seed-vc"))
DEFAULT_DIFFUSION_STEPS = 25
DEFAULT_CFG_RATE = 0.5


class SeedVC:
    """Voice conversion engine using Seed-VC.

    Converts source audio voice identity to match a target speaker
    while preserving the source's delivery, emotion, and pacing.
    """

    def __init__(self, seedvc_path: Path = DEFAULT_SEEDVC_PATH):
        self.seedvc_path = seedvc_path
        self._loaded = False
        self._original_cwd: str | None = None
        self._app_vc = None

    def load(self) -> None:
        """Load SeedVC models to GPU.

        Changes working directory to seedvc_path (required by SeedVC internals),
        stubs gradio, and loads all models via app_vc.load_models().
        """
        if self._loaded:
            return

        logger.info("Loading SeedVC from %s", self.seedvc_path)

        self._original_cwd = os.getcwd()
        os.chdir(self.seedvc_path)

        if "gradio" not in sys.modules:
            sys.modules["gradio"] = types.ModuleType("gradio")

        seedvc_str = str(self.seedvc_path)
        if seedvc_str not in sys.path:
            sys.path.insert(0, seedvc_str)

        os.environ.setdefault(
            "HF_HUB_CACHE",
            str(self.seedvc_path / "checkpoints" / "hf_cache"),
        )

        # Patch BigVGAN for huggingface_hub compat (same as gpu_vc)
        import modules.bigvgan.bigvgan as _bigvgan_mod

        _orig = _bigvgan_mod.BigVGAN._from_pretrained

        @classmethod
        def _patched(cls, **kwargs):
            kwargs.setdefault("proxies", None)
            kwargs.setdefault("resume_download", False)
            return _orig.__func__(cls, **kwargs)

        _bigvgan_mod.BigVGAN._from_pretrained = _patched

        # Load models (exact pattern from gpu_vc/seedvc_engine.py)
        import app_vc

        self._app_vc = app_vc
        app_vc.device = torch.device("cuda")

        args = Namespace(checkpoint=None, config=None, fp16=True, gpu=0)
        (
            app_vc.model,
            app_vc.semantic_fn,
            app_vc.vocoder_fn,
            app_vc.campplus_model,
            app_vc.to_mel,
            app_vc.mel_fn_args,
        ) = app_vc.load_models(args)

        app_vc.max_context_window = app_vc.sr // app_vc.hop_length * 30
        app_vc.overlap_wave_len = app_vc.overlap_frame_len * app_vc.hop_length

        self._loaded = True
        logger.info("SeedVC loaded: sr=%d, device=%s", app_vc.sr, app_vc.device)

    def unload(self) -> None:
        """Free SeedVC models from GPU."""
        if not self._loaded:
            return

        if self._app_vc is not None:
            for attr in [
                "model",
                "semantic_fn",
                "vocoder_fn",
                "campplus_model",
                "to_mel",
            ]:
                if hasattr(self._app_vc, attr):
                    delattr(self._app_vc, attr)
            self._app_vc = None

        torch.cuda.empty_cache()

        if self._original_cwd:
            if os.path.isdir(self._original_cwd):
                os.chdir(self._original_cwd)
            self._original_cwd = None

        self._loaded = False
        logger.info("SeedVC unloaded")

    def convert(
        self,
        source_wav_path: str,
        target_wav_path: str,
        diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
        cfg_rate: float = DEFAULT_CFG_RATE,
    ) -> np.ndarray:
        """Convert voice identity of source to match target.

        Both files must be 22050Hz mono WAV.

        Args:
            source_wav_path: Path to source audio (generated speech)
            target_wav_path: Path to target audio (reference voice)
            diffusion_steps: Number of diffusion steps (quality vs speed)
            cfg_rate: Classifier-free guidance rate

        Returns:
            Converted audio as float32 numpy array at 22050Hz mono
        """
        if not self._loaded:
            raise RuntimeError("SeedVC not loaded. Call load() first.")

        logger.info(
            "Converting voice: %s -> %s (%d steps, cfg_rate=%.2f)",
            source_wav_path,
            target_wav_path,
            diffusion_steps,
            cfg_rate,
        )

        audio_tuple = None
        vc_kwargs = {
            "source": source_wav_path,
            "target": target_wav_path,
            "diffusion_steps": diffusion_steps,
            "length_adjust": 1.0,
            "inference_cfg_rate": cfg_rate,
        }
        # n_quantizers removed in newer SeedVC versions
        sig = inspect.signature(self._app_vc.voice_conversion)
        if "n_quantizers" in sig.parameters:
            vc_kwargs["n_quantizers"] = 3
        for result in self._app_vc.voice_conversion(**vc_kwargs):
            if isinstance(result, tuple) and len(result) == 2:
                _, audio_tuple = result

        if audio_tuple is None:
            raise RuntimeError("SeedVC produced no output")

        sample_rate, samples = audio_tuple

        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        peak = np.abs(samples).max()
        if peak > 1.0:
            samples = samples / peak

        logger.info("Converted: %.1fs at %dHz", len(samples) / sample_rate, sample_rate)
        return samples
