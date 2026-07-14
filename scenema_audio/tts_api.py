"""Stable class-based Scenema Audio API for external applications."""

from __future__ import annotations

import asyncio
import gc
import html
import os
import shutil
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any


class ScenemaAudioTTSEngine:
    """CUDA-only adapter around Scenema Audio's production ``AudioProcessor``."""

    def __init__(self) -> None:
        self._load_options: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._processor: Any = None
        self._process_job_class: Any = None
        self._reference_audio_path: Path | None = None
        self._upload_temp: tempfile.TemporaryDirectory[str] | None = None

    def close(self) -> None:
        """Unload all backend models and remove temporary reference files."""
        with self._lock:
            if self._processor is not None:
                try:
                    self._processor.shutdown()
                finally:
                    self._processor = None
            if self._upload_temp is not None:
                self._upload_temp.cleanup()
                self._upload_temp = None
            self._load_options = {}
            self._reference_audio_path = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def tts_inference(
        self,
        *,
        text: str,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        reference_audio_path: str | Path | None = None,
        voice_description: str = "A clear, expressive audiobook narrator",
        gender: str = "female",
        scene: str = "",
        language: str = "en",
        shot: str = "closeup",
        action: str = "",
        mode: str = "generate",
        background_sfx: bool = False,
        validate: bool = False,
        seed: int = 42,
        pace: float = 1.5,
        min_match_ratio: float = 0.90,
        skip_vc: bool = False,
        vc_steps: int = 25,
        vc_cfg_rate: float = 0.5,
        raw_prompt: bool = False,
        **load_overrides: Any,
    ) -> Path:
        """Generate one verified 48 kHz WAV file."""
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("SceneMa Audio inference text must not be empty.")

        requested_model_path = Path(model_path).expanduser().resolve() if model_path else None
        if self._processor is None:
            if requested_model_path is None and not load_overrides:
                raise RuntimeError("SceneMa Audio is not loaded. Call tts_load(...) first.")
            self.tts_load(model_path=requested_model_path, **load_overrides)
        elif requested_model_path is not None and self._load_options.get("model_path") != requested_model_path:
            self.tts_load(model_path=requested_model_path, **load_overrides)

        reference = self._resolve_optional_file(
            reference_audio_path,
            "reference audio",
        ) if reference_audio_path else self._reference_audio_path
        runtime_mode, runtime_reference, runtime_skip_vc = self._resolve_generation_contract(
            mode,
            reference,
            skip_vc,
        )
        staged_reference = self._stage_reference(runtime_reference) if runtime_reference else None
        destination = self._resolve_output_path(output_path, output_dir)
        prompt = normalized_text if raw_prompt else self._build_prompt(
            normalized_text,
            voice_description=voice_description,
            gender=gender,
            scene=scene,
            language=language,
            shot=shot,
            action=action,
        )
        payload = {
            "prompt": prompt,
            "mode": runtime_mode,
            "reference_voice_path": str(staged_reference) if staged_reference else None,
            "background_sfx": bool(background_sfx),
            "validate": bool(validate),
            "seed": int(seed),
            "pace": float(pace),
            "min_match_ratio": float(min_match_ratio),
            "skip_vc": runtime_skip_vc,
            "vc_steps": int(vc_steps),
            "vc_cfg_rate": float(vc_cfg_rate),
        }

        with self._lock:
            job = self._process_job_class(job_id=uuid.uuid4().hex, input=payload)
            result = asyncio.run(self._processor.process(job))
        if not result.success or result.output is None or not result.output.success:
            error = result.error or getattr(result.output, "error", None) or "unknown SceneMa Audio failure"
            raise RuntimeError(f"SceneMa Audio inference failed: {error}")
        audio_bytes = result.output.data
        if not audio_bytes:
            raise RuntimeError("SceneMa Audio returned no WAV bytes.")
        destination.write_bytes(audio_bytes)
        if not destination.is_file() or destination.stat().st_size <= 0:
            raise RuntimeError(f"SceneMa Audio did not produce a non-empty WAV file: {destination}")
        return destination

    def tts_load(
        self,
        *,
        model_path: str | Path | None = None,
        reference_audio_path: str | Path | None = None,
        audio_checkpoint_path: str | Path | None = None,
        pipeline_checkpoint_path: str | Path | None = None,
        vae_encoder_path: str | Path | None = None,
        gemma_root: str | Path | None = None,
        melband_model_path: str | Path | None = None,
        melband_source_path: str | Path | None = None,
        seedvc_source_path: str | Path | None = None,
        seedvc_asset_path: str | Path | None = None,
        device: str = "cuda",
        gemma_quantize: str = "nf4",
        transformer_quantize: str = "",
        preload_postprocessors: bool = True,
    ) -> None:
        """Load the complete SceneMa Audio CUDA pipeline once."""
        self._require_cuda(device)
        root = Path(model_path).expanduser().resolve() if model_path else None
        paths = self._resolve_model_paths(
            root,
            audio_checkpoint_path=audio_checkpoint_path,
            pipeline_checkpoint_path=pipeline_checkpoint_path,
            vae_encoder_path=vae_encoder_path,
            gemma_root=gemma_root,
            melband_model_path=melband_model_path,
            melband_source_path=melband_source_path,
            seedvc_source_path=seedvc_source_path,
            seedvc_asset_path=seedvc_asset_path,
        )
        reference = self._resolve_optional_file(reference_audio_path, "reference audio")
        options = {
            "model_path": root,
            **paths,
            "device": str(device),
            "gemma_quantize": str(gemma_quantize),
            "transformer_quantize": str(transformer_quantize),
            "preload_postprocessors": bool(preload_postprocessors),
        }

        with self._lock:
            if self._processor is not None and options == self._load_options:
                self._reference_audio_path = reference
                return
            self.close()
            self._upload_temp = tempfile.TemporaryDirectory(prefix="scenema_audio_refs_")
            self._configure_environment(
                paths,
                gemma_quantize,
                transformer_quantize,
                preload_postprocessors,
            )
            processor_class, process_job_class = self._get_backend_classes()
            processor = processor_class()
            try:
                processor.startup()
            except Exception as exc:
                try:
                    processor.shutdown()
                except Exception:
                    pass
                self._upload_temp.cleanup()
                self._upload_temp = None
                raise RuntimeError(f"Failed to load SceneMa Audio: {exc}") from exc
            self._processor = processor
            self._process_job_class = process_job_class
            self._load_options = options
            self._reference_audio_path = reference

    @staticmethod
    def _build_prompt(
        text: str,
        *,
        voice_description: str,
        gender: str,
        scene: str,
        language: str,
        shot: str,
        action: str,
    ) -> str:
        description = str(voice_description or "").strip()
        normalized_gender = str(gender or "").strip().lower()
        if not description:
            raise ValueError("SceneMa Audio voice_description must not be empty.")
        if normalized_gender not in {"male", "female"}:
            raise ValueError("SceneMa Audio gender must be 'male' or 'female'.")
        attributes = [
            f'voice="{html.escape(description, quote=True)}"',
            f'gender="{normalized_gender}"',
            f'language="{html.escape(str(language or "en"), quote=True)}"',
            f'shot="{html.escape(str(shot or "closeup"), quote=True)}"',
        ]
        if str(scene or "").strip():
            attributes.append(f'scene="{html.escape(str(scene).strip(), quote=True)}"')
        action_xml = f"<action>{html.escape(str(action).strip())}</action>" if str(action or "").strip() else ""
        return f"<speak {' '.join(attributes)}>{action_xml}{html.escape(text)}</speak>"

    def _configure_environment(
        self,
        paths: dict[str, Path],
        gemma_quantize: str,
        transformer_quantize: str,
        preload_postprocessors: bool,
    ) -> None:
        if self._upload_temp is None:
            raise RuntimeError("SceneMa Audio upload staging was not initialized.")
        values = {
            "AUDIO_CKPT": paths["audio_checkpoint_path"],
            "PIPELINE_CKPT": paths["pipeline_checkpoint_path"],
            "VAE_ENCODER_CKPT": paths["vae_encoder_path"],
            "GEMMA_ROOT": paths["gemma_root"],
            "MELBAND_MODEL_PATH": paths["melband_model_path"],
            "MELBAND_NODE_PATH": paths["melband_source_path"],
            "SEEDVC_PATH": paths["seedvc_source_path"],
            "SEEDVC_ASSET_PATH": paths["seedvc_asset_path"],
            "UPLOAD_DIR": Path(self._upload_temp.name),
            "GEMMA_QUANTIZE": str(gemma_quantize),
            "TRANSFORMER_QUANTIZE": str(transformer_quantize),
            "SCENEMA_PRELOAD_POSTPROCESSORS": "1" if preload_postprocessors else "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        for key, value in values.items():
            os.environ[key] = str(value)

    @staticmethod
    def _get_backend_classes():
        repo_root = Path(__file__).resolve().parents[1]
        source_root = repo_root / "src"
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        try:
            from audio_core.processor import AudioProcessor
            from common.handlers.base import ProcessJob
        except Exception as exc:
            raise ImportError(
                "SceneMa Audio backend imports failed. Install the locked CUDA dependencies "
                "and initialize the vendor submodules."
            ) from exc
        return AudioProcessor, ProcessJob

    @staticmethod
    def _require_cuda(device: str) -> None:
        if not str(device).lower().startswith("cuda"):
            raise ValueError("SceneMa Audio 4.5 integration is CUDA-only; device must be 'cuda'.")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("SceneMa Audio requires a CUDA-enabled PyTorch installation.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("SceneMa Audio requires CUDA, but torch.cuda.is_available() is false.")

    @staticmethod
    def _resolve_generation_contract(
        mode: str,
        reference: Path | None,
        skip_vc: bool,
    ) -> tuple[str, Path | None, bool]:
        normalized_mode = str(mode or "generate").strip().lower()
        if normalized_mode == "voice_design":
            return "generate", None, True
        if normalized_mode == "generate":
            return "generate", reference, bool(skip_vc)
        raise ValueError("SceneMa Audio mode must be 'generate' or 'voice_design'.")

    @staticmethod
    def _resolve_model_paths(
        model_root: Path | None,
        **explicit_paths: str | Path | None,
    ) -> dict[str, Path]:
        repo_root = Path(__file__).resolve().parents[1]
        defaults = {
            "audio_checkpoint_path": ("scenema-audio-transformer-int8.safetensors",),
            "pipeline_checkpoint_path": ("scenema-audio-pipeline.safetensors",),
            "vae_encoder_path": ("scenema-audio-vae-encoder.safetensors",),
            "gemma_root": ("gemma-3-12b-it",),
            "melband_model_path": ("MelBandRoformer_fp16.safetensors",),
            "melband_source_path": ("melband_roformer_node",),
            "seedvc_source_path": ("seed-vc",),
            "seedvc_asset_path": ("seedvc",),
        }
        source_fallbacks = {
            "melband_source_path": repo_root / "vendor" / "ComfyUI-MelBandRoFormer",
            "seedvc_source_path": repo_root / "vendor" / "seed-vc",
        }
        resolved: dict[str, Path] = {}
        if model_root is None and not all(explicit_paths.get(key) for key in defaults if key not in source_fallbacks):
            raise ValueError(
                "Provide model_path containing SceneMa Audio assets, or provide all checkpoint/Gemma paths explicitly."
            )
        for key, names in defaults.items():
            candidates: list[Path] = []
            explicit = explicit_paths.get(key)
            if explicit:
                candidates.append(Path(explicit).expanduser())
            if model_root:
                for name in names:
                    candidates.extend((model_root / name, model_root / "base" / name))
            if key in source_fallbacks:
                candidates.append(source_fallbacks[key])
            match = next((candidate.resolve() for candidate in candidates if candidate.resolve().exists()), None)
            if match is None:
                searched = ", ".join(str(candidate.resolve()) for candidate in candidates)
                raise FileNotFoundError(f"SceneMa Audio {key} was not found. Checked: {searched}")
            resolved[key] = match

        directory_keys = {
            "gemma_root",
            "melband_source_path",
            "seedvc_source_path",
            "seedvc_asset_path",
        }
        for key, path in resolved.items():
            expected = path.is_dir() if key in directory_keys else path.is_file()
            if not expected:
                raise ValueError(f"SceneMa Audio {key} has the wrong file type: {path}")
        return resolved

    @staticmethod
    def _resolve_optional_file(path_value: str | Path | None, label: str) -> Path | None:
        if path_value in (None, ""):
            return None
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SceneMa Audio {label} does not exist: {path}")
        return path

    @staticmethod
    def _resolve_output_path(
        output_path: str | Path | None,
        output_dir: str | Path | None,
    ) -> Path:
        if output_path:
            destination = Path(output_path).expanduser()
        elif output_dir:
            destination = Path(output_dir).expanduser() / "scenema_audio_output.wav"
        else:
            raise ValueError("Provide output_path or output_dir for SceneMa Audio inference.")
        destination = destination.resolve()
        if destination.suffix.lower() != ".wav":
            raise ValueError("SceneMa Audio output_path must end in .wav.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def _stage_reference(self, reference: Path) -> Path:
        if self._upload_temp is None:
            raise RuntimeError("SceneMa Audio upload staging is not initialized.")
        suffix = reference.suffix.lower() or ".wav"
        destination = Path(self._upload_temp.name) / f"reference{suffix}"
        shutil.copy2(reference, destination)
        return destination.resolve()
