"""Executable CUDA smoke test for the public Scenema Audio TTS API."""

from __future__ import annotations

import argparse
from pathlib import Path

from scenema_audio import ScenemaAudioTTSEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="Load SceneMa Audio and generate two verified WAV files.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("tts_test_outputs"))
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = ScenemaAudioTTSEngine()
    try:
        engine.tts_load(
            model_path=args.model_path,
            reference_audio_path=args.reference_audio,
            device="cuda",
            gemma_quantize="",
            preload_postprocessors=False,
        )
        first = engine.tts_inference(
            text="The lantern still burned beside the window.",
            output_path=output_dir / "scenema_audio_voice_design.wav",
            voice_description="A calm audiobook narrator with measured warmth",
            gender="female",
            mode="voice_design",
            background_sfx=True,
            seed=42,
        )
        second = engine.tts_inference(
            text="Then the storm arrived, sudden and fierce.",
            output_path=output_dir / "scenema_audio_nondefault.wav",
            reference_audio_path=args.reference_audio,
            voice_description="An urgent storyteller with rising tension",
            gender="female",
            action="Her delivery becomes tense and immediate.",
            background_sfx=True,
            validate=False,
            skip_vc=True,
            pace=1.2,
            seed=43,
        )
        for path in (first, second):
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"SceneMa Audio smoke output is missing or empty: {path}")
            print(path)
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
