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
    anthropic_api_key: str = ""
    output_dir: Path = Path("outputs")
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
