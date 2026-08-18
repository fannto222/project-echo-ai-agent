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

# Categories with a realistic chance of yielding copyright-safe, reproducible ideas.
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
    V5 trend scout.

    Adds two hard filters on top of V4:
    1. Semantic relevance: search results count only when they genuinely belong to
       the trend being validated, not merely because a loose keyword matched.
    2. Appeal preservation: Project Echo must be able to reproduce the CORE reason
       viewers watch the trend with its zero-budget production stack.

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

Separate:
1. SUBJECT TREND = the topic itself is attracting attention.
2. FORMAT TREND = the repeatable presentation mechanic is attracting attention.
3. HYBRID = both subject and format are reusable.

We want trends Project Echo can exploit almost directly while still creating wholly
original content. Do not sanitize a hot trend until its appeal disappears.

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
      "direct_safe_angle": "close-to-trend original angle preserving the appeal",
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
- validation_queries: 3-5 natural YouTube searches, 2-7 words each.
- replication_fit 0-100 = can Project Echo recreate the core experience at €0?
- copyright_dependency_risk 0-100.
- stock_footage_fit 0-100.
- initial_score 0-100.
- Prefer cross-region or multi-video patterns.
- High copyrighted-IP dependence must be reflected in copyright risk.
- Do NOT propose final video ideas yet.

Project Echo's current production abilities:
- original English script/narration
- Gemini TTS
- Pexels stock video
- FFmpeg editing, captions, motion/crops
- simple original graphics
- simple screen-recorded ORIGINAL simulations only when feasible
- no paid video generation
- no actors/team
- no copyrighted clips/music
- no ability to recreate expensive real-world stunts

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
                age_h = max(
                    1.0,
                    (now - _parse_dt(sn.get("publishedAt"))).total_seconds() / 3600
                )
            except Exception:
                continue

            views = int((item.get("statistics") or {}).get("viewCount") or 0)
            rows.append({
                "video_id": item.get("id"),
                "title": _safe_text(sn.get("title"), 150),
                "region": region,
                "views": views,
                "age_hours": round(age_h, 2),
                "views_per_hour": round(views / age_h, 2),
            })
        return rows

    def _semantic_and_appeal_review(
        self,
        signal: dict,
        candidates: list[dict],
    ) -> dict:
        """
        One Gemini call per signal:
        - assigns semantic relevance to each candidate result
        - estimates whether Project Echo can preserve the original trend's core appeal
        """
        compact = [
            {
                "video_id": x["video_id"],
                "title": x["title"],
                "regions": sorted(x["regions"]),
                "queries": sorted(x["matched_queries"]),
                "views_per_hour": x["views_per_hour"],
            }
            for x in sorted(
                candidates,
                key=lambda y: y["views_per_hour"],
                reverse=True,
            )[:45]
        ]

        prompt = f"""
You are validating ONE YouTube trend for Project Echo.

TREND:
{json.dumps(signal, ensure_ascii=False)}

CANDIDATE RECENT SEARCH RESULTS:
{json.dumps(compact, ensure_ascii=False)}

Task A — SEMANTIC RELEVANCE
A candidate counts ONLY if its actual title strongly supports the SAME subject/format
trend described above. Loose keyword overlap is not enough.

Examples:
- Trend = sandbox-game zero-to-hero progression.
  "I Built a Hidden Real-Life Tiny House" => NOT relevant.
  "Minecraft Poor to Rich Challenge" => relevant.
- Trend = real-life extreme game challenges.
  An unrelated video that merely contains "challenge" => NOT relevant.

Task B — APPEAL PRESERVATION
Score 0-100 how much of the ORIGINAL viewer reward Project Echo can preserve using:
- original English narration
- Gemini TTS
- Pexels stock footage
- FFmpeg captions/motion
- simple original graphics
- simple ORIGINAL screen-recorded simulations only when feasible
- no actors/team
- no expensive real-world stunts
- no copyrighted movie/game/music/sports clips
- no paid video generation

Do not confuse "we can talk ABOUT the trend" with "we can preserve why people watch it".
If viewers watch to SEE Minecraft progression, but Project Echo can only narrate over
generic stock footage, appeal preservation should be LOW.
If the viral mechanic is a reveal/comparison/POV/explanation that stock footage and
original graphics can genuinely deliver, it can be HIGH.

Return ONLY JSON:
{{
  "candidate_reviews": [
    {{
      "video_id": "id",
      "relevance_score": 0,
      "relevant": true
    }}
  ],
  "appeal_preservation_score": 0,
  "appeal_preservation_reason": "brief reason",
  "minimum_viable_execution": "what Project Echo would actually need to show",
  "production_gap": "what crucial viral ingredient Project Echo cannot currently reproduce, or empty string"
}}
"""
        response = self.gemini.models.generate_content(
            model=self.cfg.gemini_text_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        data = _extract_json(response.text)

        reviews = {}
        for item in data.get("candidate_reviews") or []:
            vid = str(item.get("video_id") or "")
            if not vid:
                continue
            score = max(0, min(100, int(item.get("relevance_score") or 0)))
            reviews[vid] = {
                "relevance_score": score,
                "relevant": bool(item.get("relevant")) and score >= 70,
            }

        return {
            "candidate_reviews": reviews,
            "appeal_preservation_score": max(
                0, min(100, int(data.get("appeal_preservation_score") or 0))
            ),
            "appeal_preservation_reason": _safe_text(
                str(data.get("appeal_preservation_reason") or ""), 300
            ),
            "minimum_viable_execution": _safe_text(
                str(data.get("minimum_viable_execution") or ""), 300
            ),
            "production_gap": _safe_text(
                str(data.get("production_gap") or ""), 300
            ),
        }

    def _validate_signal(self, signal: dict) -> dict:
        validation_regions = self.cfg.trend_regions[:2] or ["US"]

        # Gather candidates while remembering WHICH query and region produced them.
        candidate_map: dict[str, dict] = {}

        for query in signal["validation_queries"]:
            for region in validation_regions:
                for row in self._search_recent(query, region):
                    vid = row["video_id"]
                    existing = candidate_map.get(vid)
                    if existing is None:
                        candidate_map[vid] = {
                            **row,
                            "regions": {region},
                            "matched_queries": {query},
                        }
                    else:
                        existing["regions"].add(region)
                        existing["matched_queries"].add(query)
                        if row["views_per_hour"] > existing["views_per_hour"]:
                            existing["views_per_hour"] = row["views_per_hour"]
                            existing["views"] = row["views"]
                            existing["age_hours"] = row["age_hours"]

        candidates = list(candidate_map.values())
        review = self._semantic_and_appeal_review(signal, candidates)

        relevant_rows = []
        for row in candidates:
            r = review["candidate_reviews"].get(row["video_id"])
            if r and r["relevant"]:
                relevant_rows.append({
                    **row,
                    "semantic_relevance_score": r["relevance_score"],
                })

        # Query breadth AFTER semantic filtering.
        query_counts = {}
        for query in signal["validation_queries"]:
            ids = {
                row["video_id"]
                for row in relevant_rows
                if query in row["matched_queries"]
            }
            query_counts[query] = len(ids)

        successful_queries = sum(1 for n in query_counts.values() if n >= 2)

        regions_with_evidence = set()
        for row in relevant_rows:
            regions_with_evidence.update(row["regions"])

        velocities = [x["views_per_hour"] for x in relevant_rows]
        med = median(velocities) if velocities else 0.0
        peak = max(velocities) if velocities else 0.0

        momentum_score = min(100.0, 19.0 * math.log10(max(1.0, med) + 1.0))
        query_breadth_score = min(
            100.0,
            (successful_queries / max(1, len(signal["validation_queries"]))) * 100.0,
        )
        region_score = min(
            100.0,
            (len(regions_with_evidence) / max(1, len(validation_regions))) * 100.0,
        )
        safety_score = 100.0 - signal["copyright_dependency_risk"]
        appeal = review["appeal_preservation_score"]

        # Appeal preservation has substantial weight in V5.
        validated_score = round(
            signal["initial_score"] * 0.07
            + momentum_score * 0.23
            + query_breadth_score * 0.12
            + region_score * 0.08
            + signal["replication_fit"] * 0.12
            + safety_score * 0.08
            + signal["stock_footage_fit"] * 0.05
            + appeal * 0.25,
            1,
        )

        qualified = (
            successful_queries >= 2
            and len(relevant_rows) >= 5
            and signal["replication_fit"] >= 55
            and signal["copyright_dependency_risk"] <= 45
            and appeal >= 75
            and validated_score >= 60
        )

        rejected_semantic = len(candidates) - len(relevant_rows)

        return {
            **signal,
            "validated_score": min(100.0, validated_score),
            "appeal_preservation_score": appeal,
            "appeal_preservation_reason": review["appeal_preservation_reason"],
            "minimum_viable_execution": review["minimum_viable_execution"],
            "production_gap": review["production_gap"],
            "qualified": qualified,
            "validation": {
                "raw_candidate_videos": len(candidates),
                "semantic_relevant_videos": len(relevant_rows),
                "semantic_rejected_videos": rejected_semantic,
                "semantic_precision": round(
                    len(relevant_rows) / max(1, len(candidates)), 3
                ),
                "successful_query_variants": successful_queries,
                "total_query_variants": len(signal["validation_queries"]),
                "regions_with_evidence": sorted(regions_with_evidence),
                "median_views_per_hour": round(med, 2),
                "peak_views_per_hour": round(peak, 2),
                "query_results_after_semantic_filter": [
                    {"query": q, "relevant_unique_videos": query_counts[q]}
                    for q in signal["validation_queries"]
                ],
                "sample_relevant_titles": [
                    {
                        "title": x["title"],
                        "semantic_relevance_score": x["semantic_relevance_score"],
                        "views_per_hour": x["views_per_hour"],
                    }
                    for x in sorted(
                        relevant_rows,
                        key=lambda y: y["views_per_hour"],
                        reverse=True,
                    )[:6]
                ],
            },
        }

    def _build_editorial_plan(self, signals: list[dict]) -> dict:
        qualified = [s for s in signals if s.get("qualified")]

        if len(qualified) < 2:
            return {
                "ready_to_produce": False,
                "reason": (
                    f"Only {len(qualified)} trend signal(s) passed V5 semantic relevance + "
                    "appeal-preservation gates. Project Echo should wait/recheck instead of forcing content."
                ),
                "short_ideas": [],
                "long_idea": None,
            }

        prompt = f"""
You are the chief editor of Project Echo.

Use ONLY the QUALIFIED V5 live trend signals below.

Critical rule:
The final concept must preserve the SAME viewer reward measured by
appeal_preservation_score. Do not turn gameplay/progression/stunts into a generic
commentary video if the viewer originally watches to SEE the action.

Every idea must:
- stay close to the qualified subject/format
- preserve the viral mechanic
- be wholly original
- use no copyrighted clips/music/characters
- be executable at €0 with Project Echo's actual toolset
- be understandable in one sentence
- have a visible payoff/reveal/transformation/test/comparison/POV or proof
- use at least 2 different trend signals across the 5 Shorts

Return ONLY JSON:
{{
  "ready_to_produce": true,
  "reason": "why the slate is genuinely producible now",
  "short_ideas": [
    {{
      "idea": "specific original concept",
      "trend_signal": "matching signal",
      "trend_type": "subject|format|hybrid",
      "hook": "first spoken line",
      "viral_mechanic": "what keeps viewers watching",
      "visible_payoff": "what the viewer will actually SEE",
      "production_method": "how Project Echo makes it at €0",
      "copyright_safe": true,
      "estimated_score": 0
    }}
  ],
  "long_idea": {{
    "idea": "specific 6-9 minute concept",
    "trend_signal": "matching qualified signal",
    "hook": "cold open",
    "viral_mechanic": "retention engine",
    "visible_payoff": "what viewers will see",
    "production_method": "€0 production method",
    "copyright_safe": true,
    "estimated_score": 0
  }}
}}

QUALIFIED SIGNALS:
{json.dumps(qualified, ensure_ascii=False)}
"""
        response = self.gemini.models.generate_content(
            model=self.cfg.gemini_text_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.55),
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
            "scout_version": "v5-semantic-appeal-gate",
            "sources": [
                "YouTube Data API v3 videos.list chart=mostPopular",
                "YouTube Data API v3 search.list multi-query recent validation",
                "Gemini semantic relevance filtering",
                "Gemini appeal-preservation evaluation",
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
            out_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
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
            "ready_to_produce": report.get(
                "editorial_plan", {}
            ).get("ready_to_produce", False),
            "top_signals": [
                {
                    "name": x["name"],
                    "type": x["trend_type"],
                    "score": x.get("validated_score", 0),
                    "appeal_preservation": x.get("appeal_preservation_score", 0),
                    "qualified": x.get("qualified", False),
                }
                for x in report["signals"][:6]
            ],
            "short_ideas": report.get(
                "editorial_plan", {}
            ).get("short_ideas", [])[:5],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
