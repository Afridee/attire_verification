# syntax=docker/dockerfile:1
#
# Layer order: system deps -> Python deps -> model weights -> app code.
# A push that only touches Python invalidates the final COPY, so the model
# download stays cached.
#
# This image is CPU-only. torch's default Linux wheel pulls CUDA (~2GB of
# nvidia packages) that this service never uses. Those packages are stripped
# from the export, and uv's torch backend installs the CPU wheels instead.
#
# Build:
#   DOCKER_BUILDKIT=1 docker build -t attire-verification .
#
# Run (set SERVICE_API_KEY to the same value Nest sends as X-API-Key):
#   docker run -p 8000:8000 -e SERVICE_API_KEY=... attire-verification

FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

WORKDIR /app

# Deps only. App code is copied after the model layer.
COPY pyproject.toml uv.lock README.md ./

RUN uv export --frozen --no-dev --no-emit-project --no-hashes -o /tmp/reqs.txt \
    && python -c "\
from pathlib import Path; \
src = Path('/tmp/reqs.txt').read_text().splitlines(); \
skip = ('nvidia-', 'cuda-', 'triton'); \
kept = [line for line in src if not line.startswith(skip)]; \
Path('/tmp/reqs-cpu.txt').write_text('\n'.join(kept) + '\n')" \
    && uv pip install --system --no-cache --torch-backend=cpu -r /tmp/reqs-cpu.txt

# Weights are baked in so a container start does not hit the network.
ENV HF_HOME=/root/.cache/huggingface
RUN python -c "\
import open_clip; \
print('downloading FashionSigLIP...'); \
open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP'); \
import mediapipe as mp; \
pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1); \
pose.close(); \
print('models cached.')"

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV ATTIRE_WARMUP=1

COPY src ./src
RUN uv pip install --system --no-cache --no-deps .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "attire_verification.api:app", "--host", "0.0.0.0", "--port", "8000"]
