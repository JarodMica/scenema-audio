# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for AudioProcessor. Integration tests with mocked engine."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from common.handlers.base import ProcessJob
from audio_core.engine import AudioResult
from audio_core.processor import AudioProcessor


@pytest.fixture
def mock_engine():
    """Mock AudioEngine that returns fake audio."""
    engine = MagicMock()
    engine.load = MagicMock()
    engine.unload = MagicMock()
    engine._loaded = True
    engine.vae_sample_rate = 16000

    engine.encode_text = MagicMock(return_value=("mock_vc", "mock_ac"))
    engine.encode_reference = MagicMock(return_value="mock_ref_latent")

    fake_wav = np.sin(np.linspace(0, 100 * np.pi, 48000)).astype(np.float32)
    fake_result = AudioResult(
        waveform_np=fake_wav,
        sample_rate=48000,
        duration_s=1.0,
    )
    engine.generate = MagicMock(return_value=fake_result)
    return engine


@pytest.fixture
def processor(mock_engine):
    """AudioProcessor with mocked dependencies."""
    proc = AudioProcessor()
    proc.engine = mock_engine
    proc.vocal_separator = None
    proc.seedvc = None
    return proc


class TestParseInput:
    def test_valid_generate_input(self, processor):
        job = ProcessJob(
            job_id="test-1",
            input={
                "prompt": '<speak voice="V" gender="male">Hello world.</speak>',
                "seed": 42,
            },
        )
        config = processor._parse_input(job)
        assert config["mode"] == "generate"
        assert config["seed"] == 42
        assert config["background_sfx"] is False

    def test_voice_design_mode(self, processor):
        job = ProcessJob(
            job_id="test-2",
            input={
                "prompt": '<speak voice="V" gender="male">Hello.</speak>',
                "mode": "voice_design",
            },
        )
        config = processor._parse_input(job)
        assert config["mode"] == "voice_design"

    def test_missing_prompt_raises(self, processor):
        job = ProcessJob(job_id="test-3", input={})
        with pytest.raises(ValueError, match="prompt"):
            processor._parse_input(job)

    def test_invalid_xml_raises(self, processor):
        job = ProcessJob(
            job_id="test-4",
            input={"prompt": "<broken>"},
        )
        with pytest.raises(ValueError, match="Invalid prompt"):
            processor._parse_input(job)

    def test_invalid_mode_raises(self, processor):
        job = ProcessJob(
            job_id="test-5",
            input={
                "prompt": '<speak voice="V" gender="male">Hello.</speak>',
                "mode": "invalid",
            },
        )
        with pytest.raises(ValueError, match="Invalid mode"):
            processor._parse_input(job)

    def test_background_sfx_flag(self, processor):
        job = ProcessJob(
            job_id="test-6",
            input={
                "prompt": '<speak voice="V" gender="male">Hello.</speak>',
                "background_sfx": True,
            },
        )
        config = processor._parse_input(job)
        assert config["background_sfx"] is True

    def test_reference_voice_url(self, processor):
        job = ProcessJob(
            job_id="test-7",
            input={
                "prompt": '<speak voice="V" gender="male">Hello.</speak>',
                "reference_voice_url": "https://example.com/voice.wav",
            },
        )
        config = processor._parse_input(job)
        assert config["reference_voice_url"] == "https://example.com/voice.wav"

    def test_reference_voice_path_inside_upload_dir(
        self, processor, tmp_path, monkeypatch
    ):
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        reference = upload_dir / "voice.wav"
        reference.write_bytes(b"fake wav")
        monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))

        job = ProcessJob(
            job_id="test-8",
            input={
                "prompt": '<speak voice="V" gender="male">Hello.</speak>',
                "reference_voice_path": str(reference),
            },
        )

        config = processor._parse_input(job)
        assert config["reference_voice_path"] == str(reference.resolve())

    def test_reference_voice_path_outside_upload_dir_raises(
        self, processor, tmp_path, monkeypatch
    ):
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        reference = tmp_path / "voice.wav"
        reference.write_bytes(b"fake wav")
        monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))

        job = ProcessJob(
            job_id="test-9",
            input={
                "prompt": '<speak voice="V" gender="male">Hello.</speak>',
                "reference_voice_path": str(reference),
            },
        )

        with pytest.raises(ValueError, match="inside UPLOAD_DIR"):
            processor._parse_input(job)

    def test_random_seed(self, processor):
        job = ProcessJob(
            job_id="test-10",
            input={
                "prompt": '<speak voice="V" gender="male">Hello.</speak>',
                "seed": -1,
            },
        )
        config = processor._parse_input(job)
        assert config["seed"] >= 0


class TestProcess:
    @pytest.mark.asyncio
    async def test_voice_design_calls_generate_once(self, processor, mock_engine):
        """Voice design generates a single 15s sample (no chunking)."""
        job = ProcessJob(
            job_id="vd-1",
            input={
                "prompt": '<speak voice="Warm female voice" gender="female">Hello.</speak>',
                "mode": "voice_design",
                "seed": 42,
            },
        )
        result = await processor.process(job)
        assert result.success
        assert result.output.content_type == "audio/wav"
        assert result.output.metadata["mode"] == "voice_design"
        # Voice design: single encode + single generate (no chunking)
        mock_engine.encode_text.assert_called_once()
        mock_engine.generate.assert_called_once()
        # Verify the 15s duration was passed
        call_args = mock_engine.generate.call_args
        assert call_args[0][2] == 15.0  # duration arg

    @pytest.mark.asyncio
    async def test_generate_uses_chunking_pipeline(self, processor, mock_engine):
        """Standard generate goes through plan_chunks + generate_chunks."""
        job = ProcessJob(
            job_id="gen-1",
            input={
                "prompt": '<speak voice="V" gender="male">Hello world. Short text.</speak>',
                "seed": 42,
            },
        )
        result = await processor.process(job)
        assert result.success
        assert result.output.content_type == "audio/wav"
        assert result.output.metadata["mode"] == "generate"
        assert result.output.metadata["seed"] == 42
        # Generate mode: encode_text called (at least once for chunking)
        assert mock_engine.encode_text.called
        assert mock_engine.generate.called

    @pytest.mark.asyncio
    async def test_generate_error_returns_failure(self, processor, mock_engine):
        mock_engine.encode_text.side_effect = RuntimeError("GPU error")
        job = ProcessJob(
            job_id="err-1",
            input={
                "prompt": '<speak voice="V" gender="male">Hello.</speak>',
                "seed": 42,
            },
        )
        result = await processor.process(job)
        assert not result.success
        assert "GPU error" in result.error

    @pytest.mark.asyncio
    async def test_metadata_includes_duration(self, processor, mock_engine):
        job = ProcessJob(
            job_id="meta-1",
            input={
                "prompt": '<speak voice="V" gender="male">Hello.</speak>',
                "mode": "voice_design",
                "seed": 42,
            },
        )
        result = await processor.process(job)
        assert result.success
        assert "duration_s" in result.output.metadata
        assert "processing_ms" in result.output.metadata
        assert result.output.metadata["mode"] == "voice_design"


class TestBuildMetadata:
    def test_metadata_fields(self, processor):
        config = {
            "mode": "generate",
            "background_sfx": False,
            "validate": False,
            "seed": 42,
            "reference_voice_url": None,
            "reference_voice_path": None,
        }
        wav = np.zeros(48000, dtype=np.float32)
        metadata = processor._build_metadata(config, wav, 48000, 1234)

        assert metadata["duration_s"] == 1.0
        assert metadata["sample_rate"] == 48000
        assert metadata["processing_ms"] == 1234
        assert metadata["mode"] == "generate"
        assert metadata["seed"] == 42
        assert metadata["has_reference_voice"] is False
