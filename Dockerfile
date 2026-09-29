# faster-whisper-server plus a pyannote speaker diarization endpoint, in one container on one port.
# The upstream image is pinned so a rebuild never silently changes the transcription server.
FROM fedirz/faster-whisper-server:latest-cuda@sha256:0b64050ad0b9244745746b652473ee42a8d5454d501877a252c3e65f631ffc99

WORKDIR /root/faster-whisper-server

# Blackwell GPUs (RTX 50xx, sm_120) need PyTorch built for CUDA 12.8 or later. Torch brings its own
# CUDA and cuDNN 9 libraries; CTranslate2 keeps using the image's cuDNN 8, so the two don't collide.
# faster-whisper's tokenizers needs huggingface-hub < 1.0, and httpcore needs h11 < 0.15.
# pyannote warns that torchcodec can't load; that's fine, audio is decoded with PyAV and passed in memory.
RUN uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 \
        torch torchaudio \
 && uv pip install --python .venv/bin/python "pyannote.audio>=4.0" "huggingface-hub<1.0" "h11<0.15" \
 && uv pip check --python .venv/bin/python

COPY transcriber_diarize ./transcriber_diarize

# Run the venv's uvicorn directly: "uv run" would re-sync the venv to upstream's lockfile.
CMD ["/root/faster-whisper-server/.venv/bin/uvicorn", "--factory", "transcriber_diarize.main:create_app"]
