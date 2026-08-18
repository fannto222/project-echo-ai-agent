import json
import math
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

import requests
from google import genai
from google.genai import types

from .config import Config


YOUTUBE_API = "https://www.googleapis.com/youtube/v3"

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
    V6 trend scout.

    Hard gates:
    1. Semantic relevance
    2. Appeal preservation
    3. Third-party IP / identity dependency

    The IP gate is deliberately stricter than ordinary copyright-risk scoring:
    even if commentary might arguably be fair use, Project Echo rejects a trend when
    the audience demand fundamentally depends on a third-party franchise, song,
    movie/TV property, game IP, celebrity/creator identity, sports property/team,
    or another protected/current-media asset.

    Automated sources:
    - YouTube Data API v3 mostPopular
    - YouTube Data API v3 recent search validation
    - Gemini semantic / appeal / IP-dependency classification

    TikTok Creative Center is deliberately not scraped.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gemini = genai.Client(api_key=cfg.gemini_api_key)
        self.session = requests.Session()

    def _get(self, endpoint: str, params: dict) -> dict:
        params = {**params, "key": self.cfg.youtube_data_api_key}

        last_message = ""
        for attempt in range(4):
            r = self.session.get(f"{YOUTUBE_API}/{endpoint}", params=params, timeout=30)

            if r.status_code != 429:
                r.raise_for_status()
                return r.json()

            try:
                body = r.json()
                err = body.get("error") or {}
                last_message = str(err.get("message") or "")
                reasons = [
                    str(x.get("reason") or "")
                    for x in (err.get("errors") or [])
                    if isinstance(x, dict)
                ]
            except Exception:
                last_message = r.text[:300]
                reasons = []

            # A short transient rate limit may recover. A daily/search quota will not,
            # but a few small retries also make the diagnostic explicit in Actions.
            if attempt < 3:
                time.sleep((2, 5, 10)[attempt])
                continue

            reason_text = ", ".join(reasons) or "unknown"
            raise RuntimeError(
                "YouTube Data API search rate/quota limit reached (HTTP 429). "
                f"reason={reason_text}; message={last_message}. "
                "Project Echo stopped safely; no video was generated. "
                "Check Google Cloud YouTube Data API quotas or retry after the quota resets."
            )

        raise RuntimeError("YouTube Data API request failed after rate-limit retries.")

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
            out.append({
                "video_id": item.get("id"),
                "region": region,
                "title": _safe_text(sn.get("title"), 150),
                "channel_title": _safe_text(sn.get("channelTitle"), 80),
                "category_id": category,
                "published_at": published,
                "age_hours": round(age_h, 2),
                "views": views,
                "views_per_hour": round(views / age_h, 2),
            })

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
You are the trend intelligence desk for Project Echo, a zero-budget faceless channel.

Analyze CURRENT YouTube most-popular evidence and identify reusable trend signals.

Classify:
- subject = the topic itself is hot
- format = a repeatable presentation mechanic is hot
- hybrid = both

IMPORTANT: Project Echo has a strict NO-DEPENDENCY policy.
A trend should NOT be considered safely reproducible merely because commentary or
fair use might be possible. If its audience demand fundamentally comes from:
- a movie/TV/anime franchise
- a specific game/game franchise
- a song/artist/music catalogue
- a celebrity, streamer, YouTuber or public personality
- a sports league/team/athlete/event
- a specific brand/product launch
- copyrighted trailer/clip/lore
- breaking entertainment/news event owned or driven by third-party IP
then mark third_party_dependency high.

A generic mechanic such as "before vs after transformation", "visual comparison",
"POV reveal", "ranked test", "satisfying process", or "original hypothetical challenge"
can be low dependency when it remains appealing without a famous underlying property.

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
      "third_party_dependency": 0,
      "third_party_dependency_type": "none|franchise|game_ip|music|celebrity_creator|sports|brand|news_event|mixed",
      "stock_footage_fit": 0,
      "evidence_video_ids": ["id1","id2"],
      "initial_score": 0
    }}
  ]
}}

Rules:
- 5-8 signals.
- 3-5 validation queries, 2-7 words each.
- scores 0-100.
- third_party_dependency means "would this trend still have roughly the same audience
  appeal if all famous names/IP were removed?"
- If NO, dependency should be 70-100.
- If YES, dependency can be 0-25.
- Do not lower dependency merely because commentary/fair use may be legally arguable.

Project Echo can currently make:
- original English scripts
- Gemini TTS
- Pexels stock footage
- FFmpeg captions/motion/crops
- simple original graphics
- simple ORIGINAL screen-recorded simulations where feasible
- no actors/team
- no copyrighted clips/music
- no expensive real-world stunts
- no paid video generation

EVIDENCE:
{json.dumps(compact, ensure_ascii=False)}
"""
        response = self.gemini.models.generate_content(
            model=self.cfg.gemini_text_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.15),
        )
        data = _extract_json(response.text)

        cleaned = []
        seen = set()
        allowed_dep_types = {
            "none", "franchise", "game_ip", "music", "celebrity_creator",
            "sports", "brand", "news_event", "mixed"
        }

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

            dep_type = str(s.get("third_party_dependency_type") or "none").lower()
            if dep_type not in allowed_dep_types:
                dep_type = "mixed"

            cleaned.append({
                "name": name,
                "trend_type": trend_type,
                "format_pattern": _safe_text(str(s.get("format_pattern") or ""), 250),
                "subject_pattern": _safe_text(str(s.get("subject_pattern") or ""), 180),
                "why_it_is_moving": _safe_text(str(s.get("why_it_is_moving") or ""), 250),
                "validation_queries": queries[:3],
                "direct_safe_angle": _safe_text(str(s.get("direct_safe_angle") or ""), 260),
                "replication_fit": max(0, min(100, int(s.get("replication_fit") or 0))),
                "copyright_dependency_risk": max(
                    0, min(100, int(s.get("copyright_dependency_risk") or 0))
                ),
                "third_party_dependency": max(
                    0, min(100, int(s.get("third_party_dependency") or 0))
                ),
                "third_party_dependency_type": dep_type,
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
                "channel_title": _safe_text(sn.get("channelTitle"), 90),
                "region": region,
                "views": views,
                "age_hours": round(age_h, 2),
                "views_per_hour": round(views / age_h, 2),
            })

        return rows

    def _semantic_appeal_ip_review(self, signal: dict, candidates: list[dict]) -> dict:
        compact = [
            {
                "video_id": x["video_id"],
                "title": x["title"],
                "channel_title": x.get("channel_title", ""),
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
You are performing the FINAL validation of ONE YouTube trend for Project Echo.

TREND:
{json.dumps(signal, ensure_ascii=False)}

RECENT CANDIDATES:
{json.dumps(compact, ensure_ascii=False)}

TASK A — SEMANTIC RELEVANCE
A video counts only when its title clearly supports the same trend. Loose keyword
overlap is insufficient.

TASK B — APPEAL PRESERVATION
Score 0-100 how much of the ORIGINAL viewer reward Project Echo can reproduce with:
original script, Gemini TTS, Pexels, FFmpeg, simple graphics, and simple original
simulations. No actors, no copyrighted clips/music, no expensive stunts.

Talking ABOUT a trend is not the same as reproducing its appeal.

TASK C — THIRD-PARTY DEPENDENCY HARD GATE
This is NOT a fair-use test.

Ask:
"If every famous/protected third-party name and asset vanished, would this exact
trend still have roughly the same audience demand?"

Third-party dependency includes:
- film/TV/anime franchises and characters
- game IP and specific games when the game itself drives clicks
- songs, artists and commercial music
- celebrities, streamers, YouTubers and public personalities
- sports leagues, teams, athletes and events
- major brands/product launches
- copyrighted trailers, clips, lore or current franchise announcements

If the answer is NO, set ip_dependency_detected=true and ip_dependency_score >= 70.
Do this EVEN IF commentary/fair use could theoretically be possible.
Do this EVEN IF stock footage could be substituted.
A transformed "analysis of why this franchise is popular" does not remove the
dependency if the franchise is the reason people click.

Generic reusable mechanics may be false/low dependency:
- original before/after transformation
- generic POV scenario
- original test/comparison
- satisfying process
- original challenge rule
- public-domain/raw factual subject not dependent on a famous identity

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
  "minimum_viable_execution": "what viewers would actually see",
  "production_gap": "missing viral ingredient, or empty string",
  "ip_dependency_detected": true,
  "ip_dependency_score": 0,
  "ip_dependency_type": "none|franchise|game_ip|music|celebrity_creator|sports|brand|news_event|mixed",
  "ip_dependency_reason": "why audience demand does or does not depend on third-party IP/identity"
}}
"""
        response = self.gemini.models.generate_content(
            model=self.cfg.gemini_text_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.05),
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

        allowed_dep_types = {
            "none", "franchise", "game_ip", "music", "celebrity_creator",
            "sports", "brand", "news_event", "mixed"
        }
        dep_type = str(data.get("ip_dependency_type") or "none").lower()
        if dep_type not in allowed_dep_types:
            dep_type = "mixed"

        dep_score = max(0, min(100, int(data.get("ip_dependency_score") or 0)))
        dep_detected = bool(data.get("ip_dependency_detected")) or dep_score >= 50

        return {
            "candidate_reviews": reviews,
            "appeal_preservation_score": max(
                0, min(100, int(data.get("appeal_preservation_score") or 0))
            ),
            "appeal_preservation_reason": _safe_text(
                str(data.get("appeal_preservation_reason") or ""), 320
            ),
            "minimum_viable_execution": _safe_text(
                str(data.get("minimum_viable_execution") or ""), 320
            ),
            "production_gap": _safe_text(
                str(data.get("production_gap") or ""), 320
            ),
            "ip_dependency_detected": dep_detected,
            "ip_dependency_score": dep_score,
            "ip_dependency_type": dep_type,
            "ip_dependency_reason": _safe_text(
                str(data.get("ip_dependency_reason") or ""), 340
            ),
        }

    def _validate_signal(self, signal: dict) -> dict:
        validation_regions = self.cfg.trend_regions[:1] or ["US"]
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
        review = self._semantic_appeal_ip_review(signal, candidates)

        relevant_rows = []
        for row in candidates:
            r = review["candidate_reviews"].get(row["video_id"])
            if r and r["relevant"]:
                relevant_rows.append({
                    **row,
                    "semantic_relevance_score": r["relevance_score"],
                })

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

        appeal = review["appeal_preservation_score"]

        # Final dependency is deliberately conservative: use the HIGHER of the first
        # trend-level estimate and the final evidence-level estimate.
        signal_ip_score = signal["third_party_dependency"]
        review_ip_score = review["ip_dependency_score"]
        final_ip_dependency = max(signal_ip_score, review_ip_score)

        # Pick the type belonging to the stronger dependency assessment.
        if review_ip_score >= signal_ip_score:
            final_ip_type = review["ip_dependency_type"]
        else:
            final_ip_type = signal["third_party_dependency_type"]

        # V6.2 hard-block rule:
        # A mere category label (e.g. "news_event" at score 15) is NOT enough to block.
        # The dependency must be materially high.
        ip_hard_block = (
            final_ip_dependency >= 70
            or (
                review["ip_dependency_detected"]
                and review_ip_score >= 60
            )
        )

        dependency_safety = max(0.0, 100.0 - final_ip_dependency)

        validated_score = round(
            signal["initial_score"] * 0.06
            + momentum_score * 0.22
            + query_breadth_score * 0.11
            + region_score * 0.07
            + signal["replication_fit"] * 0.11
            + signal["stock_footage_fit"] * 0.05
            + appeal * 0.23
            + dependency_safety * 0.15,
            1,
        )

        semantic_precision = len(relevant_rows) / max(1, len(candidates))

        # In quota-efficient one-region mode, a single broad query can still provide
        # strong evidence if it returns many clean, fast-moving, semantically matched videos.
        strong_single_query_evidence = (
            successful_queries >= 1
            and len(relevant_rows) >= 8
            and semantic_precision >= 0.80
            and med >= 50
            and peak >= 500
        )
        validation_pass = successful_queries >= 2 or strong_single_query_evidence

        qualified = (
            not ip_hard_block
            and validation_pass
            and len(relevant_rows) >= 5
            and signal["replication_fit"] >= 55
            and appeal >= 75
            and validated_score >= 60
        )

        if ip_hard_block:
            rejection_reason = (
                "IP_DEPENDENCY_HARD_BLOCK: audience demand materially depends on "
                f"third-party {final_ip_type} (dependency score {final_ip_dependency})."
            )
        elif not validation_pass:
            rejection_reason = (
                "Trend validation was not broad/strong enough after semantic filtering."
            )
        elif len(relevant_rows) < 5:
            rejection_reason = "Not enough semantically relevant recent videos."
        elif signal["replication_fit"] < 55:
            rejection_reason = "Project Echo cannot reproduce the format well enough."
        elif appeal < 75:
            rejection_reason = "Project Echo cannot preserve enough of the original viewer appeal."
        elif validated_score < 60:
            rejection_reason = "Overall validated trend score is below threshold."
        else:
            rejection_reason = ""

        return {
            **signal,
            "validated_score": min(100.0, validated_score),
            "appeal_preservation_score": appeal,
            "appeal_preservation_reason": review["appeal_preservation_reason"],
            "minimum_viable_execution": review["minimum_viable_execution"],
            "production_gap": review["production_gap"],
            "ip_dependency_hard_block": ip_hard_block,
            "ip_dependency_score": final_ip_dependency,
            "ip_dependency_type_final": final_ip_type,
            "ip_dependency_reason": review["ip_dependency_reason"],
            "qualified": qualified,
            "rejection_reason": rejection_reason,
            "validation": {
                "raw_candidate_videos": len(candidates),
                "semantic_relevant_videos": len(relevant_rows),
                "semantic_rejected_videos": len(candidates) - len(relevant_rows),
                "semantic_precision": round(semantic_precision, 3),
                "strong_single_query_evidence": strong_single_query_evidence,
                "validation_pass": validation_pass,
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
                    f"Only {len(qualified)} trend signal(s) passed V6.2 semantic, appeal, "
                    "and zero-third-party-IP-dependency hard gates. Project Echo should "
                    "wait/recheck instead of forcing content."
                ),
                "short_ideas": [],
                "long_idea": None,
            }

        prompt = f"""
You are the chief editor of Project Echo.

Use ONLY QUALIFIED V6 signals below.

These signals already passed a strict third-party-IP-dependency gate. Do not reinsert
brands, franchises, games, songs, celebrities, creators, sports properties, or
copyrighted/current entertainment assets into the final ideas.

Every idea must:
- preserve the exact viewer reward of the qualified trend
- remain wholly original
- be executable at €0
- have a visible payoff/reveal/transformation/test/comparison/POV/proof
- work using Project Echo's actual production stack
- not depend on fair use

Return ONLY JSON:
{{
  "ready_to_produce": true,
  "reason": "why the slate is genuinely producible and independent",
  "short_ideas": [
    {{
      "idea": "specific original concept",
      "trend_signal": "matching qualified signal",
      "trend_type": "subject|format|hybrid",
      "hook": "first spoken line",
      "viral_mechanic": "what keeps viewers watching",
      "visible_payoff": "what viewers actually see",
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
    "visible_payoff": "what viewers see",
    "production_method": "€0 production method",
    "copyright_safe": true,
    "estimated_score": 0
  }}
}}

Give exactly 5 Short ideas using at least 2 different qualified signals.

QUALIFIED SIGNALS:
{json.dumps(qualified, ensure_ascii=False)}
"""
        response = self.gemini.models.generate_content(
            model=self.cfg.gemini_text_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.5),
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
            "scout_version": "v6.2-balanced-ip-gate",
            "sources": [
                "YouTube Data API v3 videos.list chart=mostPopular",
                "YouTube Data API v3 search.list multi-query recent validation",
                "Gemini semantic relevance filtering",
                "Gemini appeal-preservation evaluation",
                "Gemini third-party IP/identity dependency hard gate",
            ],
            "regions": self.cfg.trend_regions,
            "validation_regions": self.cfg.trend_regions[:1],
            "search_budget_strategy": "max 3 search queries per signal, 1 validation region; mostPopular discovery remains US/GB/CA/AU",
            "lookback_hours": self.cfg.trend_lookback_hours,
            "automation_note": (
                "TikTok Creative Center is not scraped. TikTok confirmation remains omitted "
                "until an approved stable interface is available."
            ),
            "qualified_signal_count": sum(1 for x in validated if x.get("qualified")),
            "ip_blocked_signal_count": sum(
                1 for x in validated if x.get("ip_dependency_hard_block")
            ),
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
            "ip_blocked_signal_count": report["ip_blocked_signal_count"],
            "ready_to_produce": report.get(
                "editorial_plan", {}
            ).get("ready_to_produce", False),
            "top_signals": [
                {
                    "name": x["name"],
                    "type": x["trend_type"],
                    "score": x.get("validated_score", 0),
                    "appeal_preservation": x.get("appeal_preservation_score", 0),
                    "ip_dependency_score": x.get("ip_dependency_score", 0),
                    "ip_blocked": x.get("ip_dependency_hard_block", False),
                    "qualified": x.get("qualified", False),
                    "rejection_reason": x.get("rejection_reason", ""),
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
