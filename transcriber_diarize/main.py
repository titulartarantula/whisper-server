"""faster-whisper-server with a speaker diarization endpoint added, without modifying upstream code."""

from __future__ import annotations

import os

from fastapi import FastAPI
from faster_whisper_server.main import create_app as create_whisper_app

from transcriber_diarize.router import preload_in_background, router

import faster_whisper

# The OpenAI-style API takes one temperature, which switches off Whisper's fallback re-decode: the step that
# breaks repetition loops. Treat 0 as "start greedy, fall back as usual".
_FALLBACK_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
_transcribe = faster_whisper.WhisperModel.transcribe


def _robust_transcribe(self, audio, *args, **kwargs):  # type: ignore[no-untyped-def]
    if kwargs.get("temperature") in (None, 0, 0.0):
        kwargs["temperature"] = _FALLBACK_TEMPERATURES
    # Feeding each window the previous window's text lets one chant ("Happy birthday, Tess") take over the
    # rest of a long recording. Off unless WHISPER_CONDITION_ON_PREVIOUS_TEXT=1.
    kwargs.setdefault(
        "condition_on_previous_text", os.environ.get("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "0") == "1"
    )
    return _transcribe(self, audio, *args, **kwargs)


faster_whisper.WhisperModel.transcribe = _robust_transcribe


def create_app() -> FastAPI:
    app = create_whisper_app()
    app.include_router(router)

    # The Gradio UI is mounted at "/", which would shadow any route registered after it.
    ours = [r for r in app.router.routes if getattr(r, "path", "").startswith("/v1/audio/diarization")]
    app.router.routes[:] = ours + [r for r in app.router.routes if r not in ours]

    if os.environ.get("DIARIZATION_PRELOAD", "0") == "1":
        preload_in_background()
    return app
