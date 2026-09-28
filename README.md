# whisper-server

The GPU speech server Transcriber talks to. It's [faster-whisper-server](https://github.com/fedirz/faster-whisper-server)
(OpenAI-compatible `/v1/audio/transcriptions`) with one addition: a speaker diarization endpoint backed by
[pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1). Both run in one
container on port 8000. Upstream code is not modified. `transcriber_diarize/main.py` wraps upstream's app
factory and adds a router.

## Setup

1. Accept the model's terms at https://huggingface.co/pyannote/speaker-diarization-community-1.
2. Create a read token at https://huggingface.co/settings/tokens.
3. Copy `.env.example` to `.env` and set `HF_TOKEN`. The model downloads once into the
   `faster-whisper-cache` volume and is loaded onto the GPU at startup (`DIARIZATION_PRELOAD=1`).
4. `docker compose up -d --build`

The compose file uses the existing external volume `faster-whisper-cache`. If you're starting fresh,
create it first with `docker volume create faster-whisper-cache`.

## API

`GET /v1/audio/diarization` reports whether diarization is available:

```json
{"available": true, "model": "pyannote/speaker-diarization-community-1", "loaded": true, "device": "cuda"}
```

A stock Whisper server returns 404 here. Transcriber treats that as "separate voices on the device".

`POST /v1/audio/diarization` takes multipart form data:

| Field | |
| --- | --- |
| `file` | audio (WAV/FLAC; Transcriber sends 16 kHz mono PCM) |
| `num_speakers` | optional, exact count |
| `min_speakers`, `max_speakers` | optional bounds |

It returns exclusive turns, meaning one speaker at a time, with speakers numbered by first appearance:

```json
{"duration": 48.2, "num_speakers": 2, "model": "…", "device": "cuda", "elapsed": 1.9,
 "segments": [{"start": 0.03, "end": 7.9, "speaker": 0}, {"start": 8.2, "end": 15.1, "speaker": 1}]}
```

Errors return 503 with a `detail` message. Requests run one at a time on the GPU.

```sh
curl -F file=@meeting.wav http://localhost:8000/v1/audio/diarization
```

## Notes

- The base image is pinned by digest. RTX 50xx (Blackwell) cards need the CUDA 12.8 PyTorch wheels the
  Dockerfile installs.
- `huggingface-hub` is held below 1.0 and `h11` below 0.15 so faster-whisper's own dependencies stay intact.
  The build runs `uv pip check` and fails if anything conflicts.
- Whisper is loaded at startup and kept on the GPU (`WHISPER_TTL=-1`). Upstream's default unloads it after
  5 idle minutes, which costs ~15 s on the next request. Set `WHISPER_TTL=300` in `.env` to free the VRAM
  between meetings.
- CUDA compiles kernels for this GPU on first use. The `cuda-cache` volume keeps them, so the first
  request after a restart isn't slow.
- pyannote logs a warning that torchcodec can't load. That's expected: audio is decoded with soundfile and
  passed to pyannote in memory.
