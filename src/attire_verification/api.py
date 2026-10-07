"""HTTP API so a Nest backend can call attire verification.

Same pipeline as ``attire_verification verify``. One route:

    python -m uvicorn attire_verification.api:app --host 0.0.0.0 --port 8000
    POST /verify
      multipart field ``image``  — JPEG or PNG full-body photo
      multipart field ``role``   — staff role (default ``BR``)

Authenticate with header ``X-API-Key: <SERVICE_API_KEY>``. The check runs
only when that env var is set, so local calls work without it. In any
environment Nest can reach, set ``SERVICE_API_KEY`` and send that same value
from Nest as ``X-API-Key``.

The pose model and FashionSigLIP are loaded once per process. Inference is
serialized: one photo is in the model at a time.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi

from attire_verification.cli import _get_pose_cropper, _get_scorer, run_verify
from attire_verification.roles import Role, parse_role

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("attire_verification.api")

MAX_IMAGE_BYTES = 25 * 1024 * 1024
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
_SUFFIX_BY_CONTENT_TYPE = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
}

# MediaPipe Pose and FashionSigLIP are not safe to call concurrently.
_infer_lock = threading.Lock()


def _warmup_enabled() -> bool:
    return os.getenv("ATTIRE_WARMUP", "1") != "0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _warmup_enabled():
        logger.info("loading pose and FashionSigLIP models")
        _get_pose_cropper()
        _get_scorer()
        logger.info("models ready")
    else:
        logger.info("ATTIRE_WARMUP=0 — models load on the first request")
    if not os.getenv("SERVICE_API_KEY", "").strip():
        logger.warning("SERVICE_API_KEY is unset — POST /verify is unauthenticated")
    yield


app = FastAPI(title="Attire Verification Service", lifespan=lifespan)


def require_auth(x_api_key: str | None = Header(default=None)) -> None:
    """Accept ``X-API-Key`` when ``SERVICE_API_KEY`` is set.

    Unset means local/dev: the route stays open. A configured key that does
    not match is rejected, same as the transcription service's API-key path.
    """
    expected = os.getenv("SERVICE_API_KEY", "").strip()
    if not expected or x_api_key == expected:
        return
    logger.warning("rejected request: invalid X-API-Key")
    raise HTTPException(status_code=401, detail="Missing or invalid credentials")


def custom_openapi():
    """Declare the API key so Swagger UI shows an Authorize dialog."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema.setdefault("components", {})["securitySchemes"] = {
        "ApiKey": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "SERVICE_API_KEY — for the Nest backend and other trusted callers",
        },
    }
    schema["security"] = [{"ApiKey": []}]
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


def _image_suffix(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix in _ALLOWED_SUFFIXES:
        return ".jpg" if suffix == ".jpeg" else suffix
    content_type = (upload.content_type or "").split(";", 1)[0].strip().lower()
    mapped = _SUFFIX_BY_CONTENT_TYPE.get(content_type)
    if mapped:
        return mapped
    raise HTTPException(
        status_code=422,
        detail="image must be a JPEG or PNG",
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/verify", dependencies=[Depends(require_auth)])
def verify_image(
    image: UploadFile = File(..., description="Full-body JPEG or PNG"),
    role: str = Form(Role.BR.value, description="Staff role. BR and BR_SUP require an ID badge"),
) -> dict:
    """Score one full-body photo. Response matches the ``verify`` CLI JSON."""
    try:
        canonical_role = parse_role(role).value
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    suffix = _image_suffix(image)
    raw = image.file.read(MAX_IMAGE_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=422, detail="image is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds 25MB")

    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
        logger.info(
            "verifying image=%s role=%s bytes=%s",
            image.filename,
            canonical_role,
            len(raw),
        )
        with _infer_lock:
            result = run_verify(Path(tmp_name), role=canonical_role)
    except HTTPException:
        raise
    except Exception:
        logger.exception("verification failed for image=%s", image.filename)
        raise HTTPException(status_code=500, detail="internal error verifying attire")
    finally:
        Path(tmp_name).unlink(missing_ok=True)

    result.imagePath = Path(image.filename).name if image.filename else None
    payload = result.public_dict()
    logger.info(
        "verified image=%s role=%s score=%s failReasons=%s",
        result.imagePath,
        canonical_role,
        result.score,
        result.failReasons,
    )
    return payload
