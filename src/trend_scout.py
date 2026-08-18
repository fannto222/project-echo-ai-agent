import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

import requests
from google import genai
from google.genai import types

from .config import Config


YOUTUBE_API = "https://www.googleapis.com/youtube/v3"

# Public categories that can contain repeatable, copyright-safe ideas.
# Film/music/gaming/sports/news are not used as direct subject pools, but the model
# may still identify a FORMAT pattern from safe categories without copying subjects.
SAFE_CATEGORY_IDS = {"15", "22", "23", "24", "26", "27", "28"}


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _safe_text(value: str, limit: int = 200) -> str:
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
    V4: trend-first editorial scout.

    Automated sources:
    - YouTube Data API v3 mostPopular
    - YouTube Data API v3 recent search validation

    TikTok Creative Center is deliberately not scraped.
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

            views = int((item.get("statistics") or {}).get("viewCount") or 0)
            out.append(
                {
                    "video_id": item.get("id"),
                    "region": region,
                    "title": _safe_text(sn.get("title"), 150),
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
            for x in sorted(popular, key=lambda z: z["views_per_hour"], reverse=True)[:140]
        ]

        prompt = f"""
You are the trend intelligence desk for a zero-budget faceless YouTube channel.
Analyze CURRENT YouTube most-popular evidence.

Separate two things:
1. SUBJECT TREND = the actual topic/subject itself is attracting attention.
2. FORMAT TREND = a repeatable presentation mechanic is attracting attention.
3. HYBRID = both subject and format are reusable.

We want trends we can exploit almost directly while still making wholly original
content. Do NOT "sanitize" a hot trend so aggressively that the audience appeal
disappears. If the original appeal depends on copyrighted IP, celebrity identity,
sports footage, music/movie/game clips, current news, or another creator's footage,
mark that as high copyright/dependency risk instead of inventing a dead replacement.

Return ONLY JSON:
{{
  "signals": [
    {{
      "name": "short trend label",
      "trend_type": "subject|format|hybrid",
      "format_pattern": "what viewers are responding to",
      "subject_pattern": "topic family, or empty string",
      "why_it_is_moving": "evidence-based reason",
      "validation_queries": [
        "query variant 1",
        "query variant 2",
        "query variant 3"
      ],
      "direct_safe_angle": "a close-to-the-trend original angle that preserves the appeal",
      "replication_fit": 0,
      "copyright_dependency_risk": 0,
      "stock_footage_fit": 0,
      "evidence_video_ids": ["id1","id2"],
      "initial_score": 0
    }}
  ]
}}

Rules:
- Return 5-8 signals.
- validation_queries: 3-5 natural YouTube searches, each 2-7 words.
- replication_fit 0-100 = can Project Echo reproduce the core appeal at €0?
- copyright_dependency_risk 0-100 = how much the trend depends on protected IP/footage/identity.
- stock_footage_fit 0-100 = can original narration + stock footage make the idea convincing?
- initial_score 0-100.
- Prefer signals appearing across multiple regions or multiple independent videos.
- A viral copyrighted trailer breakdown can be a FORMAT signal, but should have high
  copyright dependency if the subject itself is the franchise.
- Do not propose the final video idea yet.

EVIDENCE:
{json.dumps(compact, ensure_ascii=False)}
"""
        response = self.gemini.models.generate_content(
            model=self.cfg.gemini_text_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        data = _extract_json(response.text)

        cleaned = []
        seen = set()
        for s in data.get("signals") or []:
            name = _safe_text(str(s.get("name") or ""), 85)
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())

            queries = []
            for q in s.get("validation_queries") or []:
                q = _safe_text(str(q), 90)
                if q and q.lower() not in {x.lower() for x in queries}:
                    queries.append(q)
            if len(queries) < 3:
                continue

            trend_type = str(s.get("trend_type") or "format").lower()
            if trend_type not in {"subject", "format", "hybrid"}:
                trend_type = "format"

            cleaned.append({
                "name": name,
                "trend_type": trend_type,
                "format_pattern": _safe_text(str(s.get("format_pattern") or ""), 250),
                "subject_pattern": _safe_text(str(s.get("subject_pattern") or ""), 180),
                "why_it_is_moving": _safe_text(str(s.get("why_it_is_moving") or ""), 250),
                "validation_queries": queries[:5],
                "direct_safe_angle": _safe_text(str(s.get("direct_safe_angle") or ""), 260),
                "replication_fit": max(0, min(100, int(s.get("replication_fit") or 0))),
                "copyright_dependency_risk": max(
                    0, min(100, int(s.get("copyright_dependency_risk") or 0))
                ),
                "stock_footage_fit": max(0, min(100, int(s.get("stock_footage_fit") or 0))),
                "evidence_video_ids": [str(x) for x in (s.get("evidence_video_ids") or [])[:6]],
                "initial_score": max(0, min(100, int(s.get("initial_score") or 0))),
            })

        return cleaned[: self.cfg.trend_max_signals]

    def _search_recent(self, query: str, region: str) -> list[dict]:
        after = datetime.now(timezone.utc) - timedelta(hours=self.cfg.trend_lookback_hours)
        search = self._get(
            "search",
            {
                "part": "snippet",
                "type": "video",
                "q": query,
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
            return []

        details = self._get(
            "videos",
            {"part": "snippet,statistics", "id": ",".join(ids[:10])},
        )
        now = datetime.now(timezone.utc)
        rows = []
        for item in details.get("items", []):
            sn = item.get("snippet") or {}
            try:
                age_h = max(1.0, (now - _parse_dt(sn.get("publishedAt"))).total_seconds() / 3600)
            except Exception:
                continue
            views = int((item.get("statistics") or {}).get("viewCount") or 0)
            rows.append({
                "video_id": item.get("id"),
                "title": _safe_text(sn.get("title"), 130),
                "region": region,
                "views": views,
                "age_hours": round(age_h, 2),
                "views_per_hour": round(views / age_h, 2),
            })
        return rows

    def _validate_signal(self, signal: dict) -> dict:
        # Validate the same signal through multiple semantic queries and two markets.
        validation_regions = self.cfg.trend_regions[:2] or ["US"]
        query_results = []
        unique_videos: dict[str, dict] = {}
        successful_queries = 0
        regions_with_evidence = set()

        for query in signal["validation_queries"]:
            this_query_ids = set()
            for region in validation_regions:
                rows = self._search_recent(query, region)
                if rows:
                    regions_with_evidence.add(region)
                for row in rows:
                    vid = row["video_id"]
                    this_query_ids.add(vid)
                    old = unique_videos.get(vid)
                    if old is None or row["views_per_hour"] > old["views_per_hour"]:
                        unique_videos[vid] = row
            if len(this_query_ids) >= 2:
                successful_queries += 1
            query_results.append({
                "query": query,
                "unique_videos": len(this_query_ids),
            })

        rows = list(unique_videos.values())
        velocities = [x["views_per_hour"] for x in rows]
        med = median(velocities) if velocities else 0.0
        peak = max(velocities) if velocities else 0.0

        # Log momentum makes VPH meaningful without letting a single giant video dominate.
        momentum_score = min(100.0, 19.0 * math.log10(max(1.0, med) + 1.0))
        query_breadth_score = min(
            100.0, (successful_queries / max(1, len(signal["validation_queries"]))) * 100.0
        )
        region_score = min(100.0, (len(regions_with_evidence) / max(1, len(validation_regions))) * 100.0)
        safety_score = 100.0 - signal["copyright_dependency_risk"]

        validated_score = round(
            signal["initial_score"] * 0.10
            + momentum_score * 0.30
            + query_breadth_score * 0.15
            + region_score * 0.10
            + signal["replication_fit"] * 0.20
            + safety_score * 0.10
            + signal["stock_footage_fit"] * 0.05,
            1,
        )

        qualified = (
            successful_queries >= 2
            and len(rows) >= 5
            and signal["replication_fit"] >= 55
            and signal["copyright_dependency_risk"] <= 45
            and validated_score >= 55
        )

        return {
            **signal,
            "validated_score": min(100.0, validated_score),
            "qualified": qualified,
            "validation": {
                "successful_query_variants": successful_queries,
                "total_query_variants": len(signal["validation_queries"]),
                "unique_recent_videos": len(rows),
                "regions_with_evidence": sorted(regions_with_evidence),
                "median_views_per_hour": round(med, 2),
                "peak_views_per_hour": round(peak, 2),
                "query_results": query_results,
                "sample_titles": [
                    x["title"]
                    for x in sorted(rows, key=lambda y: y["views_per_hour"], reverse=True)[:6]
                ],
            },
        }

    def _build_editorial_plan(self, signals: list[dict]) -> dict:
        qualified = [s for s in signals if s.get("qualified")]

        # Hard gate: do not fabricate a slate from a single weak trend.
        if len(qualified) < 2:
            return {
                "ready_to_produce": False,
                "reason": (
                    f"Only {len(qualified)} independently validated, reproducible trend signal(s) "
                    "passed the V4 gate. Project Echo should wait/recheck instead of forcing content."
                ),
                "short_ideas": [],
                "long_idea": None,
            }

        prompt = f"""
You are the chief editor of Project Echo.

Use ONLY the QUALIFIED live trend signals below. We need ideas that preserve the
actual curiosity/reward mechanism of the trend. Do not turn a hot trend into an
unrelated educational documentary merely because it is safer.

For each idea:
- stay close to the validated subject/format
- be wholly original
- no copied title/script/story
- no celebrity/gossip dependency
- no protected movie/game/music/sports/news footage
- must be executable at €0 using original narration, stock footage, simple graphics,
  screen-recorded original simulations where appropriate, and FFmpeg editing
- if a trend cannot retain its appeal without copyrighted IP, DO NOT use it
- favor concepts understandable in under one sentence
- favor strong curiosity, challenge, transformation, comparison, POV, reveal, test,
  ranking or visual-proof mechanics

Return ONLY JSON:
{{
  "ready_to_produce": true,
  "reason": "why the slate is strong enough now",
  "short_ideas": [
    {{
      "idea": "specific original concept",
      "trend_signal": "matching qualified signal",
      "trend_type": "subject|format|hybrid",
      "hook": "first spoken line",
      "viral_mechanic": "what keeps the viewer watching",
      "why_now": "live evidence connection",
      "production_method": "how to make it at €0",
      "copyright_safe": true,
      "estimated_score": 0
    }}
  ],
  "long_idea": {{
    "idea": "one specific original 6-9 minute concept",
    "trend_signal": "matching qualified signal",
    "hook": "cold open",
    "viral_mechanic": "retention engine",
    "why_now": "live evidence connection",
    "production_method": "€0 production method",
    "copyright_safe": true,
    "estimated_score": 0
  }}
}}

Give exactly 5 Short ideas, using at least 2 different qualified signals.
Do NOT create five variants of the same concept.

QUALIFIED SIGNALS:
{json.dumps(qualified, ensure_ascii=False)}
"""
        response = self.gemini.models.generate_content(
            model=self.cfg.gemini_text_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.6),
        )
        plan = _extract_json(response.text)
        plan["ready_to_produce"] = bool(plan.get("ready_to_produce", True))
        return plan

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
            "scout_version": "v4-multi-query-hard-gate",
            "sources": [
                "YouTube Data API v3 videos.list chart=mostPopular",
                "YouTube Data API v3 search.list multi-query recent validation",
            ],
            "regions": self.cfg.trend_regions,
            "validation_regions": self.cfg.trend_regions[:2],
            "lookback_hours": self.cfg.trend_lookback_hours,
            "automation_note": (
                "TikTok Creative Center is not scraped. TikTok confirmation remains omitted "
                "until an approved stable interface is available."
            ),
            "qualified_signal_count": sum(1 for x in validated if x.get("qualified")),
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
            "scout_version": report["scout_version"],
            "report": str(out),
            "qualified_signal_count": report["qualified_signal_count"],
            "ready_to_produce": report.get("editorial_plan", {}).get("ready_to_produce", False),
            "top_signals": [
                {
                    "name": x["name"],
                    "type": x["trend_type"],
                    "score": x.get("validated_score", 0),
                    "qualified": x.get("qualified", False),
                }
                for x in report["signals"][:6]
            ],
            "short_ideas": report.get("editorial_plan", {}).get("short_ideas", [])[:5],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
