"""faster-whisper-server with a speaker diarization endpoint added, without modifying upstream code."""

from __future__ import annotations

import os

from fastapi import FastAPI
from faster_whisper_server.main import create_app as create_whisper_app

from transcriber_diarize.router import preload_in_background, router


def create_app() -> FastAPI:
    app = create_whisper_app()
    app.include_router(router)

    # The Gradio UI is mounted at "/", which would shadow any route registered after it.
    ours = [r for r in app.router.routes if getattr(r, "path", "").startswith("/v1/audio/diarization")]
    app.router.routes[:] = ours + [r for r in app.router.routes if r not in ours]

    if os.environ.get("DIARIZATION_PRELOAD", "0") == "1":
        preload_in_background()
    return app
