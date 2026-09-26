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
        if _exhausted_today(model):
            continue
        for attempt in range(MAX_ATTEMPTS_PER_MODEL):
            try:
                response = client().models.generate_content(model=model, contents=contents, config=config)
                return schema.model_validate_json(response.text), model
            except errors.APIError as e:
                last_error = e
                log.warning("%s attempt %d failed: %s %s", model, attempt + 1, e.code, e.status)
                if e.code == 429 and "PerDay" in json.dumps(getattr(e, "details", None) or str(e)):
                    _mark_exhausted(model)
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


# ---------- daily-quota memory ----------
# Free tiers have per-day request caps. Once a model reports its daily cap, skip it for the rest of
# the (Pacific) quota day instead of spending a request to rediscover that.

_exhausted: dict[str, float] = {}
QUOTA_DAY_S = 6 * 3600  # re-probe after a few hours; the reset time is not reported


def _exhausted_today(model: str) -> bool:
    return time.time() - _exhausted.get(model, 0.0) < QUOTA_DAY_S


def _mark_exhausted(model: str) -> None:
    log.warning("%s hit its daily quota; using fallbacks for now", model)
    _exhausted[model] = time.time()


# ---------- Groq (text reasoning + ASR) ----------

@lru_cache
def groq_client():
    from groq import Groq

    key = get_settings().groq_api_key
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=key, max_retries=0, timeout=30.0)


def _groq_reasoning(model: str) -> dict | None:
    # Free tier allows ~8k tokens/min; hidden reasoning tokens count too, so keep them minimal.
    if "qwen" in model:
        return {"reasoning_effort": "none"}
    if "gpt-oss" in model:
        return {"reasoning_effort": "low"}
    return None


def _parse_duration(value: str | None) -> float:
    """Groq reset headers look like '1m26.4s', '17.09s' or '750ms'."""
    if not value:
        return 0.0
    total, number = 0.0, ""
    units = {"h": 3600.0, "m": 60.0, "s": 1.0}
    i = 0
    while i < len(value):
        ch = value[i]
        if ch.isdigit() or ch == ".":
            number += ch
        elif value.startswith("ms", i):
            total += float(number or 0) / 1000
            number, i = "", i + 1
        elif ch in units:
            total += float(number or 0) * units[ch]
            number = ""
        i += 1
    return total


def _groq_wait(headers) -> float:
    return max(float(headers.get("retry-after", 0) or 0),
               _parse_duration(headers.get("x-ratelimit-reset-tokens")),
               min(_parse_duration(headers.get("x-ratelimit-reset-requests")), 61.0) if
               headers.get("x-ratelimit-remaining-requests") == "0" else 0.0)


def text_json(
    prompt: str,
    schema: type[T],
    *,
    system: str,
    prefer: str = "text",
    temperature: float = 0.1,
) -> tuple[T, str]:
    """Text-only structured call. Groq models first (free, fast), Gemini as the last resort.

    prefer="audit" puts the audit model first so the auditor is a different model from the matcher.
    """
    import groq

    s = get_settings()
    chain = [s.groq_text_model, s.groq_audit_model]
    if prefer == "audit":
        chain.reverse()
    instructions = (
        f"{system}\n\nReply with ONE JSON object only, matching this JSON schema:\n"
        f"{json.dumps(schema.model_json_schema())}"
    )
    last_error: Exception | None = None
    for model in chain:
        if _exhausted_today(model):
            continue
        for attempt in range(3):
            try:
                response = groq_client().chat.completions.create(
                    model=model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": instructions}, {"role": "user", "content": prompt}],
                    extra_body=_groq_reasoning(model),
                )
                return schema.model_validate_json(response.choices[0].message.content), model
            except groq.RateLimitError as e:
                last_error = e
                wait = _groq_wait(e.response.headers)
                log.warning("%s rate limited, waiting %.1fs", model, wait)
                if wait > 60:
                    _mark_exhausted(model)
                    break
                time.sleep(wait + 0.5)
            except groq.APIStatusError as e:
                last_error = e
                log.warning("%s attempt %d failed: %s", model, attempt + 1, e.status_code)
                if e.status_code not in RETRYABLE:
                    break
                time.sleep(2 ** (attempt + 1))
            except (groq.APIConnectionError, ValueError) as e:
                last_error = e
                log.warning("%s attempt %d error: %s", model, attempt + 1, str(e)[:200])
                time.sleep(2 ** attempt)
    log.warning("Groq chain failed (%s); falling back to Gemini text", last_error)
    return generate_json([prompt], schema, system=system, temperature=temperature, low_res_media=False)


def transcribe(wav_bytes: bytes, language: str = "bn") -> dict | None:
    """Word-timestamped ASR of a short clip (Groq Whisper). None when unavailable."""
    import groq

    try:
        result = groq_client().audio.transcriptions.create(
            model=get_settings().groq_asr_model,
            file=("clip.wav", wav_bytes),
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            temperature=0.0,
        )
    except groq.APIError as e:
        log.warning("ASR unavailable: %s", str(e)[:200])
        return None
    return result.model_dump() if hasattr(result, "model_dump") else dict(result)