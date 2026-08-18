import os
from dataclasses import dataclass


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_text_model: str = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.5-flash-lite")
    gemini_tts_model: str = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
    gemini_tts_voice: str = os.getenv("GEMINI_TTS_VOICE", "Kore")
    pexels_api_key: str = os.getenv("PEXELS_API_KEY", "")

    youtube_client_id: str = os.getenv("YOUTUBE_CLIENT_ID", "")
    youtube_client_secret: str = os.getenv("YOUTUBE_CLIENT_SECRET", "")
    youtube_refresh_token: str = os.getenv("YOUTUBE_REFRESH_TOKEN", "")
    youtube_privacy_status: str = os.getenv("YOUTUBE_PRIVACY_STATUS", "private")
    youtube_category_id: str = os.getenv("YOUTUBE_CATEGORY_ID", "24")
    youtube_made_for_kids: bool = _bool("YOUTUBE_MADE_FOR_KIDS", False)

    max_monthly_spend_eur: float = float(os.getenv("MAX_MONTHLY_SPEND_EUR", "0"))
    allow_paid_apis: bool = _bool("ALLOW_PAID_APIS", False)
    allow_paid_ads: bool = _bool("ALLOW_PAID_ADS", False)

    channel_language: str = os.getenv("CHANNEL_LANGUAGE", "en")
    channel_niche: str = os.getenv("CHANNEL_NICHE", "sci-fi mystery what-if storytelling")
    channel_brand: str = os.getenv("CHANNEL_BRAND", "Project Echo")

    def validate_zero_budget(self) -> None:
        if self.max_monthly_spend_eur != 0:
            raise RuntimeError("Safety stop: MAX_MONTHLY_SPEND_EUR must be 0 for the zero-budget build.")
        if self.allow_paid_apis or self.allow_paid_ads:
            raise RuntimeError("Safety stop: paid APIs/ads are disabled in the zero-budget build.")

    def validate_generation(self) -> None:
        self.validate_zero_budget()
        if not self.gemini_api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        if not self.pexels_api_key:
            raise RuntimeError("Missing PEXELS_API_KEY")

    def validate_youtube(self) -> None:
        missing = [name for name, value in {
            "YOUTUBE_CLIENT_ID": self.youtube_client_id,
            "YOUTUBE_CLIENT_SECRET": self.youtube_client_secret,
            "YOUTUBE_REFRESH_TOKEN": self.youtube_refresh_token,
        }.items() if not value]
        if missing:
            raise RuntimeError("Missing YouTube secrets: " + ", ".join(missing))
