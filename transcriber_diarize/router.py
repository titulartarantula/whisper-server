"""Speaker diarization endpoints, served next to faster-whisper-server's transcription API."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
import time
from typing import Annotated, Any

import numpy as np
import soundfile
from fastapi import APIRouter, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")

logger = logging.getLogger(__name__)
router = APIRouter(tags=["diarization"])

_pipeline: Any = None
_device = "cpu"
_load_lock = threading.Lock()
# One GPU: running two diarizations at once only makes both slower.
_run_lock = asyncio.Lock()


def load_pipeline() -> Any:
    global _pipeline, _device
    with _load_lock:
        if _pipeline is not None:
            return _pipeline

        import torch
        from pyannote.audio import Pipeline

        started = time.perf_counter()
        pipeline = Pipeline.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN") or None)
        if pipeline is None:
            raise RuntimeError(f"{MODEL} could not be loaded. Is HF_TOKEN set, and were its terms accepted on Hugging Face?")
        if torch.cuda.is_available():
            _device = "cuda"
            pipeline.to(torch.device("cuda"))
        _pipeline = pipeline
        logger.info("Loaded %s on %s in %.1fs", MODEL, _device, time.perf_counter() - started)
        return _pipeline


def preload_in_background() -> None:
    def run() -> None:
        try:
            load_pipeline()
        except Exception:
            logger.exception("Preloading %s failed; it will be retried on the first request", MODEL)

    threading.Thread(target=run, name="diarization-preload", daemon=True).start()


def _read_audio(data: bytes) -> tuple[np.ndarray, int]:
    samples, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return samples.mean(axis=1), rate


def _diarize(data: bytes, num: int | None, low: int | None, high: int | None) -> dict[str, Any]:
    import torch

    pipeline = load_pipeline()
    mono, rate = _read_audio(data)
    started = time.perf_counter()

    # Passing the waveform directly skips pyannote's ffmpeg-based file decoding.
    kwargs = {k: v for k, v in {"num_speakers": num, "min_speakers": low, "max_speakers": high}.items() if v}
    output = pipeline({"waveform": torch.from_numpy(mono).unsqueeze(0), "sample_rate": rate}, **kwargs)
    # Exclusive mode gives one speaker at a time, which is what word-level attribution needs.
    annotation = getattr(output, "exclusive_speaker_diarization", output)

    ids: dict[str, int] = {}
    segments = []
    for turn, _, label in sorted(annotation.itertracks(yield_label=True), key=lambda t: t[0].start):
        speaker = ids.setdefault(label, len(ids))
        segments.append({"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": speaker})

    return {
        "duration": round(len(mono) / rate, 3),
        "num_speakers": len(ids),
        "segments": segments,
        "model": MODEL,
        "device": _device,
        "elapsed": round(time.perf_counter() - started, 2),
    }


@router.get("/v1/audio/diarization", summary="Report whether speaker diarization is available.")
def status() -> dict[str, Any]:
    return {"available": True, "model": MODEL, "loaded": _pipeline is not None, "device": _device}


@router.post("/v1/audio/diarization", summary="Split audio into anonymous speaker turns.")
async def diarize(
    file: UploadFile,
    num_speakers: Annotated[int | None, Form()] = None,
    min_speakers: Annotated[int | None, Form()] = None,
    max_speakers: Annotated[int | None, Form()] = None,
) -> dict[str, Any]:
    data = await file.read()
    try:
        async with _run_lock:
            return await run_in_threadpool(_diarize, data, num_speakers, min_speakers, max_speakers)
    except soundfile.LibsndfileError as e:
        raise HTTPException(status_code=400, detail=f"Could not read the audio: {e}") from e
    except Exception as e:
        logger.exception("Diarization failed")
        raise HTTPException(status_code=503, detail=f"Diarization failed: {e}") from e
