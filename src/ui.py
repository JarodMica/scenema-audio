# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Gradio UI for local Scenema Audio Docker deployments."""

import json
import os
import shutil
import uuid
from pathlib import Path

import gradio as gr

from common.handlers.base import ProcessJob

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/app/uploads")).resolve()
OUTPUT_DIR = UPLOAD_DIR / "outputs"
ALLOWED_AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}

DEFAULT_PROMPT = (
    '<speak voice="A warm male voice with a slight British accent. '
    'Measured, thoughtful pacing." gender="male">'
    "The old lighthouse had stood on the cliff for over a century, "
    "its beam cutting through the fog like a blade of light."
    "</speak>"
)


def _persist_upload(upload_path: str | None) -> str | None:
    """Copy a Gradio temp upload into the shared /app/uploads volume."""
    if not upload_path:
        return None

    source = Path(upload_path)
    if not source.is_file():
        raise gr.Error("Reference audio upload was not found.")

    suffix = source.suffix.lower() or ".wav"
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        raise gr.Error(f"Unsupported reference audio format. Use: {allowed}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / f"reference-{uuid.uuid4().hex}{suffix}"
    shutil.copyfile(source, destination)
    return str(destination)


def _write_output_audio(audio_bytes: bytes) -> str:
    """Persist generated audio so Gradio can serve it back to the browser."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"scenema-{uuid.uuid4().hex}.wav"
    output_path.write_bytes(audio_bytes)
    return str(output_path)


def _metadata_json(metadata: dict) -> str:
    if not metadata:
        return "{}"
    return json.dumps(metadata, indent=2, sort_keys=True)


def create_ui(processor, semaphore):
    """Create the Gradio Blocks app mounted by server.py."""

    async def run_generation(
        prompt,
        mode,
        reference_audio,
        reference_voice_url,
        background_sfx,
        validate,
        seed,
        pace,
        min_match_ratio,
        skip_vc,
        vc_steps,
        vc_cfg_rate,
    ):
        prompt = (prompt or "").strip()
        if not prompt:
            raise gr.Error("Prompt is required.")

        reference_path = _persist_upload(reference_audio)
        reference_url = (reference_voice_url or "").strip()

        request = {
            "prompt": prompt,
            "mode": mode,
            "background_sfx": background_sfx,
            "validate": validate,
            "seed": int(seed),
            "pace": float(pace),
            "min_match_ratio": float(min_match_ratio),
            "skip_vc": skip_vc,
            "vc_steps": int(vc_steps),
            "vc_cfg_rate": float(vc_cfg_rate),
        }
        if reference_path:
            request["reference_voice_path"] = reference_path
        elif reference_url:
            request["reference_voice_url"] = reference_url

        job = ProcessJob(job_id=str(uuid.uuid4()), input=request)
        async with semaphore:
            result = await processor.process(job)

        if not result.success:
            raise gr.Error(result.error or "Generation failed.")

        output = result.output
        if not output or not output.data:
            raise gr.Error("Generation did not return audio.")

        return _write_output_audio(output.data), _metadata_json(output.metadata or {})

    with gr.Blocks(
        title="Scenema Audio",
        theme=gr.themes.Soft(),
        fill_height=True,
    ) as demo:
        gr.Markdown("# Scenema Audio")
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    value=DEFAULT_PROMPT,
                    lines=12,
                )
                with gr.Row():
                    mode = gr.Radio(
                        ["generate", "voice_design"],
                        value="generate",
                        label="Mode",
                    )
                    seed = gr.Number(value=42, precision=0, label="Seed")

                with gr.Accordion("Reference voice", open=False):
                    reference_audio = gr.Audio(
                        label="Upload reference audio",
                        sources=["upload"],
                        type="filepath",
                    )
                    reference_voice_url = gr.Textbox(
                        label="Reference voice URL",
                        placeholder="https://example.com/reference.wav",
                    )

                with gr.Accordion("Generation settings", open=False):
                    with gr.Row():
                        background_sfx = gr.Checkbox(
                            value=False,
                            label="Keep background SFX",
                        )
                        validate = gr.Checkbox(value=True, label="Validate speech")
                        skip_vc = gr.Checkbox(value=False, label="Skip SeedVC")
                    pace = gr.Slider(
                        minimum=0.5,
                        maximum=3.0,
                        value=1.5,
                        step=0.05,
                        label="Pace",
                    )
                    min_match_ratio = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.01,
                        label="Minimum match ratio",
                    )
                    vc_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=25,
                        step=1,
                        label="SeedVC steps",
                    )
                    vc_cfg_rate = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="SeedVC CFG rate",
                    )

                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Generated audio", type="filepath")
                metadata_output = gr.Code(label="Metadata", language="json")

        generate_btn.click(
            run_generation,
            inputs=[
                prompt,
                mode,
                reference_audio,
                reference_voice_url,
                background_sfx,
                validate,
                seed,
                pace,
                min_match_ratio,
                skip_vc,
                vc_steps,
                vc_cfg_rate,
            ],
            outputs=[audio_output, metadata_output],
            show_api=False,
        )

    return demo.queue()
