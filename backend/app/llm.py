"""Gemini client: video upload, structured JSON calls, retry + model fallback.

A 503 "model overloaded" during judging would kill a live demo, so every call retries with
backoff and then falls back to the secondary model before giving up.
"""

import json
import logging
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from .config import get_settings

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger("llm")

RETRYABLE = {429, 500, 502, 503, 504}
MAX_ATTEMPTS_PER_MODEL = 4
FILE_TTL_S = 44 * 3600  # Gemini keeps uploads 48 h


@lru_cache
def client() -> genai.Client:
    key = get_settings().gemini_api_key
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=key)


def models() -> list[str]:
    s = get_settings()
    return [m for m in (s.gemini_model, s.gemini_fallback_model) if m]


def upload_video(path: Path, cache_file: Path) -> types.File:
    """Upload once per video; reuse the remote file while it is still alive."""
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if time.time() - cached["uploaded_at"] < FILE_TTL_S:
            try:
                remote = client().files.get(name=cached["name"])
                if remote.state == types.FileState.ACTIVE:
                    return remote
            except errors.APIError:
                pass

    remote = client().files.upload(file=path, config=types.UploadFileConfig(mime_type="video/mp4"))
    while remote.state == types.FileState.PROCESSING:
        time.sleep(3)
        remote = client().files.get(name=remote.name)
    if remote.state != types.FileState.ACTIVE:
        raise RuntimeError(f"Gemini could not process the video: {remote.error}")
    cache_file.write_text(json.dumps({"name": remote.name, "uri": remote.uri, "uploaded_at": time.time()}))
    return remote


def video_part(remote: types.File, start: float, end: float, fps: float = 1.0) -> types.Part:
    return types.Part(
        file_data=types.FileData(file_uri=remote.uri, mime_type="video/mp4"),
        video_metadata=types.VideoMetadata(start_offset=f"{start:.1f}s", end_offset=f"{end:.1f}s", fps=fps),
    )


def generate_json(
    contents: list,
    schema: type[T],
    *,
    system: str | None = None,
    temperature: float = 0.2,
    low_res_media: bool = True,
) -> tuple[T, str]:
    """Call Gemini with a response schema. Returns (parsed result, model that answered)."""
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW if low_res_media else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    last_error: Exception | None = None
    for model in models():
        for attempt in range(MAX_ATTEMPTS_PER_MODEL):
            try:
                response = client().models.generate_content(model=model, contents=contents, config=config)
                return schema.model_validate_json(response.text), model
            except errors.APIError as e:
                last_error = e
                log.warning("%s attempt %d failed: %s %s", model, attempt + 1, e.code, e.status)
                if e.code == 429 and "PerDay" in json.dumps(getattr(e, "details", None) or str(e)):
                    break  # daily quota: retrying this model is pointless today, go to the fallback
                if e.code not in RETRYABLE:
                    if e.code == 404:
                        break  # model name retired: go straight to the fallback
                    raise
            except ValueError as e:  # malformed / truncated JSON: retry
                last_error = e
                log.warning("%s attempt %d returned invalid JSON: %s", model, attempt + 1, str(e)[:200])
            time.sleep(min(30, 2 ** (attempt + 1)) + random.random())
    raise RuntimeError(f"Gemini failed on all models {models()}: {last_error}")