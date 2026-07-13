# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""MelBandRoFormer vocal separation for Scenema Audio.

Separates vocals from background music/SFX in audio. Used to clean
generated audio that may contain unwanted background sounds from the
diffusion model (which was trained on video with ambient audio).

Expects stereo 44100Hz input. Processes in overlapping chunks for
smooth transitions.
"""

import importlib
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(
    os.environ.get("MELBAND_MODEL_PATH", "/app/models/MelBandRoformer_fp16.safetensors")
)
DEFAULT_NODE_PATH = Path(
    os.environ.get("MELBAND_NODE_PATH", "/app/melband_roformer_node")
)

MODEL_CONFIG = {
    "dim": 384,
    "depth": 6,
    "stereo": True,
    "num_stems": 1,
    "time_transformer_depth": 1,
    "freq_transformer_depth": 1,
    "num_bands": 60,
    "dim_head": 64,
    "heads": 8,
    "attn_dropout": 0,
    "ff_dropout": 0,
    "flash_attn": True,
    "dim_freqs_in": 1025,
    "sample_rate": 44100,
    "stft_n_fft": 2048,
    "stft_hop_length": 441,
    "stft_win_length": 2048,
    "stft_normalized": False,
    "mask_estimator_depth": 2,
    "multi_stft_resolution_loss_weight": 1.0,
    "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
    "multi_stft_hop_size": 147,
    "multi_stft_normalized": False,
}

CHUNK_SIZE = 352800  # ~8 seconds at 44100Hz
OVERLAP_FACTOR = 2


class VocalSeparator:
    """Separates vocals from background audio using MelBandRoFormer.

    Processes audio in overlapping chunks with fade windows for
    smooth transitions. Keeps model loaded on GPU for repeated use.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        node_path: Path = DEFAULT_NODE_PATH,
    ):
        self.model_path = model_path
        self.node_path = node_path
        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Load MelBandRoFormer model to GPU."""
        if self._loaded:
            return

        # The pinned vendor's ``model`` directory is a namespace package and
        # collides with common embedding-application modules such as model.py.
        # Mount it under a private package name so its relative imports remain
        # valid without mutating or copying the pinned submodule.
        package_name = "_scenema_melband_model"
        package_path = self.node_path / "model"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__package__ = package_name
            package.__path__ = [str(package_path)]
            sys.modules[package_name] = package
        melband_module = importlib.import_module(f"{package_name}.mel_band_roformer")
        MelBandRoformer = melband_module.MelBandRoformer

        logger.info("Loading MelBandRoFormer from %s", self.model_path)

        model = MelBandRoformer(**MODEL_CONFIG)
        sd = load_file(str(self.model_path))
        model.load_state_dict(sd)
        del sd

        self._model = model.cuda().eval().float()
        self._loaded = True

        param_count = sum(p.numel() for p in self._model.parameters())
        logger.info("MelBandRoFormer loaded: %.1fM params", param_count / 1e6)

    def unload(self) -> None:
        """Free model from GPU."""
        if not self._loaded:
            return

        self._model = None
        torch.cuda.empty_cache()
        self._loaded = False
        logger.info("MelBandRoFormer unloaded")

    def separate(
        self,
        input_path: str,
        vocals_path: str,
        sfx_path: str | None = None,
    ) -> dict:
        """Separate vocals from background audio.

        Args:
            input_path: Path to input audio file (any format ffmpeg supports)
            vocals_path: Output path for isolated vocals
            sfx_path: Output path for isolated SFX/background (optional)

        Returns:
            Dict with metadata: input_duration, sample_rate
        """
        if not self._loaded:
            raise RuntimeError("VocalSeparator not loaded. Call load() first.")

        sr = MODEL_CONFIG["sample_rate"]

        audio = self._load_audio_ffmpeg(input_path, sr)
        input_duration = audio.shape[1] / sr

        logger.info("Separating: %.1fs audio", input_duration)

        with torch.inference_mode():
            vocals = self._chunked_inference(audio, sr)

        self._save_audio_ffmpeg(vocals, sr, vocals_path)

        if sfx_path:
            sfx = audio - vocals
            self._save_audio_ffmpeg(sfx, sr, sfx_path)

        return {
            "input_duration": input_duration,
            "sample_rate": sr,
        }

    def _chunked_inference(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Run model inference in overlapping chunks with fade windows."""
        total_samples = audio.shape[1]
        chunk_size = CHUNK_SIZE
        overlap = chunk_size // OVERLAP_FACTOR
        step = chunk_size - overlap

        fade_in = np.linspace(0, 1, overlap, dtype=np.float32)
        fade_out = np.linspace(1, 0, overlap, dtype=np.float32)

        result = np.zeros_like(audio)
        weight = np.zeros(total_samples, dtype=np.float32)

        pos = 0
        while pos < total_samples:
            end = min(pos + chunk_size, total_samples)
            chunk = audio[:, pos:end]

            if chunk.shape[1] < chunk_size:
                pad_width = chunk_size - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, pad_width)))

            chunk_t = torch.from_numpy(chunk.copy()).unsqueeze(0).cuda().float()
            out = self._model(chunk_t)
            out_np = out.squeeze(0).cpu().float().numpy()[:, : end - pos]

            chunk_len = end - pos
            w = np.ones(chunk_len, dtype=np.float32)
            if pos > 0:
                fade_len = min(overlap, chunk_len)
                w[:fade_len] *= fade_in[:fade_len]
            if end < total_samples:
                fade_len = min(overlap, chunk_len)
                w[-fade_len:] *= fade_out[:fade_len]

            result[:, pos:end] += out_np * w[np.newaxis, :]
            weight[pos:end] += w

            pos += step

        weight = np.maximum(weight, 1e-8)
        result /= weight[np.newaxis, :]

        return result

    def _load_audio_ffmpeg(self, path: str, target_sr: int) -> np.ndarray:
        """Load audio to stereo float32 numpy via ffmpeg."""
        cmd = [
            "ffmpeg",
            "-i",
            path,
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "2",
            "-ar",
            str(target_sr),
            "-v",
            "quiet",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        audio = np.frombuffer(proc.stdout, dtype=np.float32)
        return audio.reshape(-1, 2).T  # (2, samples)

    def _save_audio_ffmpeg(self, audio: np.ndarray, sr: int, path: str) -> None:
        """Save stereo float32 numpy to WAV via ffmpeg."""
        interleaved = audio.T.astype(np.float32).tobytes()
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "2",
            "-ar",
            str(sr),
            "-i",
            "pipe:0",
            "-acodec",
            "pcm_s16le",
            path,
            "-v",
            "quiet",
        ]
        subprocess.run(cmd, input=interleaved, check=True)
