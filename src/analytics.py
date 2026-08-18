import json
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .memory import load_history
from .youtube import service


def main():
    cfg = Config()
    yt = service(cfg)
    history = load_history()
    ids = [x.get("youtube_video_id") for x in history.get("published", []) if x.get("youtube_video_id")]
    ids = ids[-50:]
    if not ids:
        print("No uploaded video IDs yet.")
        return
    rows = []
    for i in range(0, len(ids), 50):
        response = yt.videos().list(part="snippet,statistics,status", id=",".join(ids[i:i+50])).execute()
        for item in response.get("items", []):
            s = item.get("statistics", {})
            rows.append({
                "video_id": item["id"],
                "title": item.get("snippet", {}).get("title"),
                "views": int(s.get("viewCount", 0)),
                "likes": int(s.get("likeCount", 0)),
                "comments": int(s.get("commentCount", 0)),
                "privacy": item.get("status", {}).get("privacyStatus"),
            })
    Path("data/analytics.json").write_text(json.dumps({
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "videos": rows,
    }, indent=2), encoding="utf-8")
    print(json.dumps(rows[-10:], indent=2))


if __name__ == "__main__":
    main()
