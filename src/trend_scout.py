import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

import requests
from google import genai
from google.genai import types

from .config import Config


YOUTUBE_API = "https://www.googleapis.com/youtube/v3"

# Categories with a reasonable chance of inspiring copyright-safe original formats.
# We deliberately exclude Music, Film, Gaming, Sports and News from the raw candidate pool.
SAFE_CATEGORY_IDS = {"15", "22", "23", "24", "26", "27", "28"}


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _safe_text(value: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", value or "").strip()[:limit]


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"Trend model did not return JSON: {text[:300]}")
    return json.loads(text[start:end + 1])


class YouTubeTrendScout:
    """
    Uses only the official YouTube Data API for automated public trend research.

    TikTok Creative Center is intentionally NOT scraped in this zero-budget build.
    Cross-platform TikTok confirmation can be added later only through an approved,
    stable interface or partner integration.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gemini = genai.Client(api_key=cfg.gemini_api_key)
        self.session = requests.Session()

    def _get(self, endpoint: str, params: dict) -> dict:
        params = {**params, "key": self.cfg.youtube_data_api_key}
        r = self.session.get(f"{YOUTUBE_API}/{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def _popular_region(self, region: str) -> list[dict]:
        data = self._get(
            "videos",
            {
                "part": "snippet,statistics,contentDetails",
                "chart": "mostPopular",
                "regionCode": region,
                "maxResults": 50,
            },
        )
        now = datetime.now(timezone.utc)
        out = []
        for item in data.get("items", []):
            sn = item.get("snippet") or {}
            category = str(sn.get("categoryId") or "")
            if category not in SAFE_CATEGORY_IDS:
                continue
            published = sn.get("publishedAt")
            try:
                age_h = max(1.0, (now - _parse_dt(published)).total_seconds() / 3600)
            except Exception:
                age_h = 9999.0
            stats = item.get("statistics") or {}
            views = int(stats.get("viewCount") or 0)
            out.append(
                {
                    "video_id": item.get("id"),
                    "region": region,
                    "title": _safe_text(sn.get("title"), 140),
                    "channel_title": _safe_text(sn.get("channelTitle"), 80),
                    "category_id": category,
                    "published_at": published,
                    "age_hours": round(age_h, 2),
                    "views": views,
                    "views_per_hour": round(views / age_h, 2),
                }
            )
        return out

    def _derive_signals(self, popular: list[dict]) -> list[dict]:
        compact = [
            {
                "id": x["video_id"],
                "region": x["region"],
                "title": x["title"],
                "age_h": x["age_hours"],
                "views": x["views"],
                "vph": x["views_per_hour"],
            }
            for x in sorted(popular, key=lambda z: z["views_per_hour"], reverse=True)[:120]
        ]

        prompt = f"""
You are a trend analyst for a zero-budget faceless English-language YouTube channel.
Analyze the CURRENT YouTube most-popular evidence below.

Your job is NOT to copy titles, creators, celebrities, copyrighted characters, songs,
movies, games, sports footage, news events, or another creator's story.
Extract repeatable FORMAT/TOPIC PATTERNS that can inspire wholly original content
using stock footage, original narration and editing.

Reject trends that depend on:
- celebrity identity or gossip
- copyrighted franchises/characters
- music or movie clips
- sports footage
- current political/news reporting
- dangerous stunts
- medical, legal or financial claims
- direct reuse of another creator's video

Return ONLY JSON:
{{
  "signals": [
    {{
      "name": "short trend label",
      "format_pattern": "repeatable format, not a copied title",
      "why_it_is_moving": "brief evidence-based reason",
      "search_query": "2-5 word YouTube query for validating this pattern",
      "safe_original_angle": "one original copyright-safe angle",
      "evidence_video_ids": ["id1","id2"],
      "initial_score": 0
    }}
  ]
}}

Return 4-8 signals. initial_score must be 0-100.
Favor patterns appearing across multiple regions and videos with strong views/hour.

EVIDENCE:
{json.dumps(compact, ensure_ascii=False)}
"""
        response = self.gemini.models.generate_content(
            model=self.cfg.gemini_text_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.25),
        )
        data = _extract_json(response.text)
        signals = data.get("signals") or []
        cleaned = []
        seen = set()
        for s in signals:
            name = _safe_text(str(s.get("name") or ""), 80)
            q = _safe_text(str(s.get("search_query") or ""), 80)
            if not name or not q or name.lower() in seen:
                continue
            seen.add(name.lower())
            cleaned.append(
                {
                    "name": name,
                    "format_pattern": _safe_text(str(s.get("format_pattern") or ""), 220),
                    "why_it_is_moving": _safe_text(str(s.get("why_it_is_moving") or ""), 220),
                    "search_query": q,
                    "safe_original_angle": _safe_text(str(s.get("safe_original_angle") or ""), 220),
                    "evidence_video_ids": [str(x) for x in (s.get("evidence_video_ids") or [])[:5]],
                    "initial_score": max(0, min(100, int(s.get("initial_score") or 0))),
                }
            )
        return cleaned[: self.cfg.trend_max_signals]

    def _validate_signal(self, signal: dict, region: str = "US") -> dict:
        after = datetime.now(timezone.utc) - timedelta(hours=self.cfg.trend_lookback_hours)
        search = self._get(
            "search",
            {
                "part": "snippet",
                "type": "video",
                "q": signal["search_query"],
                "order": "viewCount",
                "publishedAfter": after.isoformat().replace("+00:00", "Z"),
                "regionCode": region,
                "relevanceLanguage": self.cfg.trend_language,
                "safeSearch": "strict",
                "videoDuration": "short",
                "maxResults": 10,
            },
        )
        ids = [x.get("id", {}).get("videoId") for x in search.get("items", [])]
        ids = [x for x in ids if x]
        if not ids:
            return {**signal, "validation": {"videos": 0, "median_vph": 0, "peak_vph": 0}}

        details = self._get(
            "videos",
            {"part": "snippet,statistics", "id": ",".join(ids[:10])},
        )
        now = datetime.now(timezone.utc)
        velocities = []
        titles = []
        for item in details.get("items", []):
            sn = item.get("snippet") or {}
            try:
                age_h = max(1.0, (now - _parse_dt(sn.get("publishedAt"))).total_seconds() / 3600)
            except Exception:
                continue
            views = int((item.get("statistics") or {}).get("viewCount") or 0)
            velocities.append(views / age_h)
            titles.append(_safe_text(sn.get("title"), 120))

        if not velocities:
            med = peak = 0.0
        else:
            med = median(velocities)
            peak = max(velocities)

        # Momentum score is deliberately capped and logarithmic so one huge outlier
        # cannot completely dominate the decision.
        momentum = min(100.0, 18.0 * math.log10(max(1.0, med) + 1.0))
        evidence_bonus = min(15.0, len(velocities) * 1.5)
        validated_score = min(
            100,
            round(signal["initial_score"] * 0.55 + momentum * 0.35 + evidence_bonus, 1),
        )
        return {
            **signal,
            "validated_score": validated_score,
            "validation": {
                "videos": len(velocities),
                "median_views_per_hour": round(med, 2),
                "peak_views_per_hour": round(peak, 2),
                "sample_titles": titles[:5],
            },
        }

    def _build_editorial_plan(self, signals: list[dict]) -> dict:
        prompt = f"""
You are the chief editor of Project Echo, a zero-budget faceless content channel.
Use ONLY the trend signals below.

Create a production slate that follows current patterns WITHOUT copying any existing
title, script, creator, celebrity, brand, copyrighted character, song, movie, game,
sports clip, or news footage.

The content must be feasible with:
- original English narration
- Pexels-style stock footage
- FFmpeg motion/captions
- no paid generation
- no copyrighted audio/video

Return ONLY JSON:
{{
  "short_ideas": [
    {{
      "idea": "original video concept",
      "trend_signal": "matching signal name",
      "hook": "first-line hook",
      "why_now": "why this matches the live signal",
      "copyright_safe": true,
      "estimated_score": 0
    }}
  ],
  "long_idea": {{
    "idea": "one original 6-9 minute concept",
    "trend_signal": "matching signal",
    "hook": "cold open",
    "why_now": "reason",
    "copyright_safe": true,
    "estimated_score": 0
  }}
}}

Give exactly 5 short ideas and 1 long idea.
Only use signals with validated_score >= 45.
If there are not enough strong signals, diversify angles from the best safe signals
instead of inventing unrelated topics.

SIGNALS:
{json.dumps(signals, ensure_ascii=False)}
"""
        response = self.gemini.models.generate_content(
            model=self.cfg.gemini_text_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.65),
        )
        return _extract_json(response.text)

    def run(self, out_path: Path | None = None) -> dict:
        popular = []
        for region in self.cfg.trend_regions:
            popular.extend(self._popular_region(region))

        if not popular:
            raise RuntimeError("YouTube mostPopular returned no safe trend candidates.")

        signals = self._derive_signals(popular)
        validated = [self._validate_signal(s) for s in signals]
        validated.sort(key=lambda x: x.get("validated_score", 0), reverse=True)

        plan = self._build_editorial_plan(validated)
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sources": [
                "YouTube Data API v3 videos.list chart=mostPopular",
                "YouTube Data API v3 search.list recent validation",
            ],
            "regions": self.cfg.trend_regions,
            "lookback_hours": self.cfg.trend_lookback_hours,
            "automation_note": (
                "TikTok Creative Center is not scraped. TikTok confirmation is omitted "
                "until an approved stable interface is available."
            ),
            "signals": validated,
            "editorial_plan": plan,
        }
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report


def main() -> None:
    cfg = Config()
    cfg.validate_trends()
    out = Path("output") / "trend_report.json"
    report = YouTubeTrendScout(cfg).run(out)
    print(json.dumps(
        {
            "ok": True,
            "report": str(out),
            "top_signals": [
                {"name": x["name"], "score": x.get("validated_score", 0)}
                for x in report["signals"][:5]
            ],
            "short_ideas": report.get("editorial_plan", {}).get("short_ideas", [])[:5],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
