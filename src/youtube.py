from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from .config import Config


SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]


def service(cfg: Config):
    cfg.validate_youtube()
    creds = Credentials(
        token=None,
        refresh_token=cfg.youtube_refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=cfg.youtube_client_id,
        client_secret=cfg.youtube_client_secret,
        scopes=SCOPES,
    )
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def upload(cfg: Config, video: Path, thumbnail: Path, plan: dict, description: str) -> str:
    yt = service(cfg)
    body = {
        "snippet": {
            "title": plan["title"][:100],
            "description": description[:5000],
            "tags": plan.get("tags", [])[:20],
            "categoryId": cfg.youtube_category_id,
            "defaultLanguage": cfg.channel_language,
        },
        "status": {
            "privacyStatus": cfg.youtube_privacy_status,
            "selfDeclaredMadeForKids": cfg.youtube_made_for_kids,
            "containsSyntheticMedia": bool(plan.get("contains_realistic_synthetic_media", False)),
        },
    }
    media = MediaFileUpload(str(video), mimetype="video/mp4", resumable=True, chunksize=8 * 1024 * 1024)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        _, response = req.next_chunk()
    video_id = response["id"]
    if thumbnail.exists():
        yt.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(str(thumbnail), mimetype="image/jpeg")).execute()
    return video_id
