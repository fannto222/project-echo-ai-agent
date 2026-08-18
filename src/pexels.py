import hashlib
import json
from pathlib import Path

import requests


class PexelsClient:
    BASE = "https://api.pexels.com/v1/videos/search"

    def __init__(self, api_key: str):
        self.headers = {"Authorization": api_key}

    def search(self, query: str, orientation: str, per_page: int = 8) -> list[dict]:
        r = requests.get(
            self.BASE,
            headers=self.headers,
            params={"query": query, "orientation": orientation, "per_page": per_page},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("videos", [])

    @staticmethod
    def _choose_file(video: dict, portrait: bool) -> dict | None:
        files = [f for f in video.get("video_files", []) if f.get("file_type") == "video/mp4" and f.get("link")]
        if not files:
            return None
        def score(f):
            w, h = f.get("width") or 0, f.get("height") or 0
            orient_bonus = 1_000_000 if ((h >= w) == portrait) else 0
            target = 720 * 1280 if portrait else 1280 * 720
            pixels = w * h
            size_penalty = abs(pixels - target)
            return orient_bonus - size_penalty
        return max(files, key=score)

    def collect(self, search_terms: list[str], out_dir: Path, portrait: bool, wanted: int) -> tuple[list[Path], list[dict]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        orientation = "portrait" if portrait else "landscape"
        clips, credits, seen = [], [], set()

        for term in search_terms:
            if len(clips) >= wanted:
                break
            for video in self.search(term, orientation, per_page=10):
                if len(clips) >= wanted:
                    break
                vid = str(video.get("id"))
                if not vid or vid in seen:
                    continue
                chosen = self._choose_file(video, portrait)
                if not chosen:
                    continue
                seen.add(vid)
                ext = ".mp4"
                name = hashlib.sha1((vid + chosen["link"]).encode()).hexdigest()[:12] + ext
                path = out_dir / name
                with requests.get(chosen["link"], stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(path, "wb") as f:
                        for chunk in r.iter_content(1024 * 1024):
                            if chunk:
                                f.write(chunk)
                clips.append(path)
                user = video.get("user") or {}
                credits.append({
                    "pexels_video_id": video.get("id"),
                    "creator": user.get("name", "Unknown creator"),
                    "creator_url": user.get("url", ""),
                    "pexels_url": video.get("url", ""),
                    "search_term": term,
                })
        if not clips:
            raise RuntimeError("Pexels returned no usable clips. Try broader search terms.")
        (out_dir / "credits.json").write_text(json.dumps(credits, indent=2), encoding="utf-8")
        return clips, credits
