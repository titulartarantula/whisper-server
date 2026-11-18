# whisper-server

The GPU speech server that [Transcriber](https://github.com/titulartarantula/transcriber) (Windows and Android) talks to. It's [faster-whisper-server](https://github.com/fedirz/faster-whisper-server)
(OpenAI-compatible `/v1/audio/transcriptions`) with one addition: a speaker diarization endpoint backed by
[pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1). Both run in one
container on port 8000. Upstream code is not modified. `transcriber_diarize/main.py` wraps upstream's app
factory and adds a router.

## Setup

1. Accept the model's terms at https://huggingface.co/pyannote/speaker-diarization-community-1.
2. Create a read token at https://huggingface.co/settings/tokens.
3. Copy `.env.example` to `.env` and set `HF_TOKEN`. The model downloads once into the
   `faster-whisper-cache` volume.
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
| `file` | audio in any format the transcription endpoint accepts (WAV, FLAC, MP3, M4A, AAC, OGG, …); Transcriber sends 16 kHz mono WAV |
| `num_speakers` | optional, exact count |
| `min_speakers`, `max_speakers` | optional bounds |

`segments` are exclusive turns, meaning one speaker at a time, with speakers numbered by first appearance.
`overlap_segments` are the same speakers' turns where they're allowed to overlap: when someone says
"mm-hmm" over a long turn, the long turn carries on underneath it. Transcriber uses them to keep the
talker's words with the talker.

```json
{"duration": 48.2, "num_speakers": 2, "model": "…", "device": "cuda", "elapsed": 1.9,
 "segments": [{"start": 0.03, "end": 7.9, "speaker": 0}, {"start": 7.9, "end": 8.3, "speaker": 1}, {"start": 8.3, "end": 15.1, "speaker": 0}],
 "overlap_segments": [{"start": 0.03, "end": 15.1, "speaker": 0}, {"start": 7.9, "end": 8.3, "speaker": 1}]}
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
- The GPU is shared with other work, so Whisper and pyannote each unload after 5 idle minutes
  (`WHISPER_TTL`, `DIARIZATION_TTL`; `-1` keeps a model loaded) and reload on the next request, which adds
  a few seconds. The process keeps a small CUDA context (a few hundred MB) while the container runs.
- CUDA compiles kernels for this GPU on first use. The `cuda-cache` volume keeps them, so the first
  request after a restart isn't slow.
- pyannote logs a warning that torchcodec can't load. That's expected: audio is decoded with faster-whisper's
  own decoder (PyAV) and passed to pyannote in memory.
