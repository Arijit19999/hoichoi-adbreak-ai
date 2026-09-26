"""Check that the API keys in .env work.

Run from the repo root:
    uv run --directory backend python ../scripts/check_keys.py
"""
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def check_ffmpeg():
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} not on PATH (reopen terminal after installing)")
    return "ffmpeg + ffprobe found"


def check_gemini():
    from google import genai

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is empty in .env")
    client = genai.Client()
    flash = [m.name for m in client.models.list() if "flash" in m.name]
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    reply = client.models.generate_content(model=model, contents="Reply with just: OK")
    return f"{model} replied {reply.text.strip()!r} | flash models: {flash[:6]}"


def check_groq():
    from groq import Groq

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is empty in .env")
    whisper = [m.id for m in Groq().models.list().data if "whisper" in m.id]
    if not whisper:
        raise RuntimeError("key works but no whisper models listed")
    return f"whisper models: {whisper}"


if __name__ == "__main__":
    failed = False
    for name, check in [("ffmpeg", check_ffmpeg), ("Gemini", check_gemini), ("Groq", check_groq)]:
        try:
            print(f"[PASS] {name}: {check()}")
        except Exception as e:
            failed = True
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print("\nAll good - setup Step 7 done." if not failed else "\nFix the FAIL lines above.")