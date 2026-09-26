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
