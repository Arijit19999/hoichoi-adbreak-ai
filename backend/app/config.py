from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_fallback_model: str = "gemini-flash-lite-latest"
    groq_api_key: str = ""
    groq_text_model: str = "openai/gpt-oss-120b"
    groq_audit_model: str = "qwen/qwen3.8-27b"
    groq_asr_model: str = "whisper-large-v3"
    anthropic_api_key: str = ""
    output_dir: Path = Path("outputs")
    public_base_url: str = "http://localhost:8000/"
    # >0: upload a small proxy (this height, 5 fps) to Gemini instead of the original.
    # Useful on slow uplinks; leave 0 on a small server (transcoding is CPU-heavy).
    gemini_proxy_height: int = 0
    # Comma-separated video links processed in the background at startup when not already processed.
    # Demo convenience for hosts with an ephemeral disk; results still come from the full pipeline.
    seed_video_urls: str = ""
    brands_path: Path = Path("data/brands/brands.json")

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def outputs(self) -> Path:
        path = self.resolve(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
