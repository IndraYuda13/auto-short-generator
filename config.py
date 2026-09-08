"""Configuration module for Auto Short Generator."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Explicitly load .env file if it exists
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class Settings(BaseSettings):
    # 9router local LLM endpoint
    ROUTER_BASE_URL: str = os.getenv("ROUTER_BASE_URL", "http://127.0.0.1:20128/v1")
    ROUTER_API_KEY: str = os.getenv("ROUTER_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini/gemini-3.8-flash")

    # YouTube API
    YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")

    # Discovery queries
    SEARCH_QUERIES: str = os.getenv(
        "SEARCH_QUERIES",
        "podcast viral indonesia,deddy corbuzier podcast,raditya dika podcast,stand up comedy indonesia",
    )
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

    # Platform target: youtube, tiktok, both, or local_only
    TARGET_PLATFORMS: str = os.getenv("TARGET_PLATFORMS", "youtube,tiktok")

    # Paths
    PROJECT_ROOT: Path = Path(os.getenv("PROJECT_ROOT", "/root/projects/auto-short-generator"))
    COOKIES_FILE: Path = Path(
        os.getenv("COOKIES_FILE", "/root/projects/auto-short-generator/cookies.txt")
    )
    DB_PATH: Path = Path(os.getenv("DB_PATH", "/root/projects/auto-short-generator/data/app.db"))
    DOWNLOAD_DIR: Path = Path(
        os.getenv("DOWNLOAD_DIR", "/root/projects/auto-short-generator/downloads")
    )
    OUTPUT_DIR: Path = Path(
        os.getenv("OUTPUT_DIR", "/root/projects/auto-short-generator/output")
    )

    # Editor V2 feature toggles
    EDITOR_V2_ENABLED: bool = os.getenv("EDITOR_V2_ENABLED", "true").lower() in ("true", "1", "yes")
    FACE_TRACKING_ENABLED: bool = os.getenv("FACE_TRACKING_ENABLED", "true").lower() in ("true", "1", "yes")
    PACING_ENABLED: bool = os.getenv("PACING_ENABLED", "false").lower() in ("true", "1", "yes")
    AUDIO_MASTERING_ENABLED: bool = os.getenv("AUDIO_MASTERING_ENABLED", "true").lower() in ("true", "1", "yes")
    LEGACY_RANDOM_HOOKS_ENABLED: bool = os.getenv("LEGACY_RANDOM_HOOKS_ENABLED", "false").lower() in ("true", "1", "yes")

    # Loop Daemon Interval
    LOOP_INTERVAL_SECONDS: int = int(os.getenv("LOOP_INTERVAL_SECONDS", "300"))

    # Clip constraints
    MIN_CLIP_DURATION_SEC: int = int(os.getenv("MIN_CLIP_DURATION_SEC", "30"))
    MAX_CLIP_DURATION_SEC: int = int(os.getenv("MAX_CLIP_DURATION_SEC", "55"))

    # Whisper audio transcription slice limit (transcribe first N seconds of long podcast, 0 = unlimited)
    MAX_AUDIO_TRANSCRIBE_SEC: int = int(os.getenv("MAX_AUDIO_TRANSCRIBE_SEC", "600"))

    # Whisper
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    # YouTube OAuth
    YOUTUBE_CLIENT_SECRETS_FILE: Path = Path(
        os.getenv(
            "YOUTUBE_CLIENT_SECRETS_FILE",
            "/root/projects/auto-short-generator/client_secrets.json",
        )
    )
    YOUTUBE_CREDENTIALS_FILE: Path = Path(
        os.getenv(
            "YOUTUBE_CREDENTIALS_FILE",
            "/root/projects/auto-short-generator/data/youtube_token.json",
        )
    )

    # TikTok API v2
    TIKTOK_CLIENT_KEY: str = os.getenv("TIKTOK_CLIENT_KEY", "")
    TIKTOK_CLIENT_SECRET: str = os.getenv("TIKTOK_CLIENT_SECRET", "")
    TIKTOK_ACCESS_TOKEN: str = os.getenv("TIKTOK_ACCESS_TOKEN", "")

    # Local Proxy Pool (Surfshark nodes 31001-31015)
    PROXIES: list[str] = [
        f"http://127.0.0.1:{port}" for port in range(31001, 31016)
    ]
    DENO_PATH: str = os.getenv("DENO_PATH", "/usr/local/bin/deno")

    def get_random_proxy(self) -> str:
        import random
        return random.choice(self.PROXIES) if self.PROXIES else ""

    class Config:
        extra = "ignore"


settings = Settings()

# Ensure directories exist
settings.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
