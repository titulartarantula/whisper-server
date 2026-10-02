"""Speaker diarization endpoints, served next to faster-whisper-server's transcription API."""

from __future__ import annotations

import asyncio
import gc
import io
import logging
import os
import threading
import time
from typing import Annotated, Any

import av
import numpy as np
from fastapi import APIRouter, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
# Seconds of idleness before the model leaves the GPU, like faster-whisper-server's WHISPER__TTL. -1 keeps it.
TTL = int(os.environ.get("DIARIZATION_TTL", "300"))

logger = logging.getLogger(__name__)
router = APIRouter(tags=["diarization"])

_pipeline: Any = None
_device = "cpu"
_load_lock = threading.Lock()
_unload_timer: threading.Timer | None = None
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


def _schedule_unload() -> None:
    """Restarts the idle countdown; the GPU is shared with other work, so an idle model shouldn't sit on it."""
    global _unload_timer
    if TTL < 0:
        return
    with _load_lock:
        if _unload_timer is not None:
            _unload_timer.cancel()
        _unload_timer = threading.Timer(TTL, _unload)
        _unload_timer.daemon = True
        _unload_timer.start()


def _unload() -> None:
    global _pipeline
    if _run_lock.locked():
        _schedule_unload()  # still busy; try again after another idle period
        return
    with _load_lock:
        if _pipeline is None:
            return
        _pipeline = None

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Unloaded %s after %ss idle", MODEL, TTL)


def preload_in_background() -> None:
    def run() -> None:
        try:
            load_pipeline()
            _schedule_unload()
        except Exception:
            logger.exception("Preloading %s failed; it will be retried on the first request", MODEL)

    threading.Thread(target=run, name="diarization-preload", daemon=True).start()


SAMPLE_RATE = 16000


class NoAudioError(ValueError):
    pass


def _read_audio(data: bytes) -> tuple[np.ndarray, int]:
    """16 kHz mono, decoded the way the transcription endpoint does it, so every format it takes
    (WAV, FLAC, MP3, M4A, AAC, OGG, a video's soundtrack, ...) works here too."""
    from faster_whisper.audio import decode_audio

    samples = decode_audio(io.BytesIO(data), sampling_rate=SAMPLE_RATE)
    if samples.size == 0:
        raise NoAudioError("The file has no audio.")
    return samples, SAMPLE_RATE


def _diarize(data: bytes, num: int | None, low: int | None, high: int | None) -> dict[str, Any]:
    import torch

    pipeline = load_pipeline()
    mono, rate = _read_audio(data)
    started = time.perf_counter()

    # Passing the waveform directly skips pyannote's ffmpeg-based file decoding.
    kwargs = {k: v for k, v in {"num_speakers": num, "min_speakers": low, "max_speakers": high}.items() if v}
    output = pipeline({"waveform": torch.from_numpy(mono).unsqueeze(0), "sample_rate": rate}, **kwargs)
    # Exclusive mode gives one speaker at a time, which is what word-level attribution needs. The full
    # diarization, where turns overlap, shows who kept talking while someone else said "mm-hmm".
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    full = getattr(output, "speaker_diarization", output)

    ids: dict[str, int] = {}

    def turns(annotation: Any) -> list[dict[str, Any]]:
        return [
            {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": ids.setdefault(label, len(ids))}
            for turn, _, label in sorted(annotation.itertracks(yield_label=True), key=lambda t: t[0].start)
        ]

    result: dict[str, Any] = {"segments": turns(exclusive if exclusive is not None else full)}
    if exclusive is not None:
        result["overlap_segments"] = turns(full)

    return {
        "duration": round(len(mono) / rate, 3),
        "num_speakers": len(ids),
        **result,
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
            try:
                return await run_in_threadpool(_diarize, data, num_speakers, min_speakers, max_speakers)
            finally:
                _schedule_unload()
    except (av.error.FFmpegError, NoAudioError, IndexError) as e:
        raise HTTPException(status_code=400, detail=f"Could not read the audio: {e}") from e
    except Exception as e:
        logger.exception("Diarization failed")
        raise HTTPException(status_code=503, detail=f"Diarization failed: {e}") from e
