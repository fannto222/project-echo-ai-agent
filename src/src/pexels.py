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
        files = [
            f for f in video.get("video_files", [])
            if f.get("file_type") == "video/mp4" and f.get("link")
        ]
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

    @staticmethod
    def _safe_term(term: str) -> str:
        term = " ".join(term.strip().split())
        return term[:100]

    def _download_video(
        self,
        video: dict,
        term: str,
        out_dir: Path,
        portrait: bool,
        seen: set[str],
    ) -> tuple[Path, dict] | None:
        vid = str(video.get("id") or "")
        if not vid or vid in seen:
            return None
        chosen = self._choose_file(video, portrait)
        if not chosen:
            return None

        seen.add(vid)
        name = hashlib.sha1((vid + chosen["link"]).encode()).hexdigest()[:12] + ".mp4"
        path = out_dir / name
        with requests.get(chosen["link"], stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

        user = video.get("user") or {}
        credit = {
            "pexels_video_id": video.get("id"),
            "creator": user.get("name", "Unknown creator"),
            "creator_url": user.get("url", ""),
            "pexels_url": video.get("url", ""),
            "search_term": term,
        }
        return path, credit

    def collect(
        self,
        search_terms: list[str],
        out_dir: Path,
        portrait: bool,
        wanted: int,
    ) -> tuple[list[Path], list[dict]]:
        """
        Story-first collection.

        Pass 1: take at most ONE unique clip from each search phrase, preserving story order.
        Pass 2: only if more clips are needed, take additional unique results from those searches.
        This prevents the old behavior where the first search phrase could fill the whole video.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        orientation = "portrait" if portrait else "landscape"
        terms = [self._safe_term(t) for t in search_terms if self._safe_term(t)]
        clips: list[Path] = []
        credits: list[dict] = []
        seen: set[str] = set()
        cached: dict[str, list[dict]] = {}

        # Pass 1: one clip per ordered beat/search term.
        for term in terms:
            if len(clips) >= wanted:
                break
            results = self.search(term, orientation, per_page=8)
            cached[term] = results
            for video in results:
                item = self._download_video(video, term, out_dir, portrait, seen)
                if item:
                    path, credit = item
                    clips.append(path)
                    credits.append(credit)
                    break

        # Pass 2: fill remaining slots while still rotating across terms.
        if len(clips) < wanted:
            for result_index in range(1, 8):
                for term in terms:
                    if len(clips) >= wanted:
                        break
                    results = cached.get(term, [])
                    if result_index >= len(results):
                        continue
                    item = self._download_video(
                        results[result_index], term, out_dir, portrait, seen
                    )
                    if item:
                        path, credit = item
                        clips.append(path)
                        credits.append(credit)
                if len(clips) >= wanted:
                    break

        if not clips:
            raise RuntimeError("Pexels returned no usable clips. Try broader search terms.")

        (out_dir / "credits.json").write_text(
            json.dumps(credits, indent=2), encoding="utf-8"
        )
        return clips, credits
