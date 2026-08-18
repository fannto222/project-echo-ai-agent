import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .gemini_agent import GeminiAgent
from .memory import add_publication
from .pexels import PexelsClient
from .render import concat_audio, duration, make_thumbnail, render_video
from .trend_scout import YouTubeTrendScout
from .youtube import upload


def build_description(plan: dict, credits: list[dict]) -> str:
    lines = [plan["description"].strip()]
    if plan.get("fiction_disclaimer"):
        lines += ["", plan["fiction_disclaimer"].strip()]
    lines += ["", "Visual sources: Photos/videos provided by Pexels."]
    for c in credits[:20]:
        creator = c.get("creator") or "Pexels creator"
        url = c.get("pexels_url") or c.get("creator_url") or "https://www.pexels.com"
        lines.append(f"- {creator}: {url}")
    lines += ["", "Narration and editing are original for this channel."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["short", "long"], default="short")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--skip-trends", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.validate_generation()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + args.kind
    out = Path("output") / run_id
    out.mkdir(parents=True, exist_ok=True)

    trend_report = None
    if not args.skip_trends:
        trend_report = YouTubeTrendScout(cfg).run(out / "trend_report.json")
        if not (trend_report.get("editorial_plan") or {}).get("ready_to_produce"):
            raise RuntimeError(
                "V4 Trend Scout says WAIT: not enough independently validated, reproducible "
                "trends. No video was generated."
            )

    agent = GeminiAgent(cfg)
    plan = agent.create_content_plan(args.kind, trend_report=trend_report)
    (out / "plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    voice_parts = agent.synthesize_voice(plan["narration"], out / "voice_parts")
    voice = out / "voice.wav"
    concat_audio(voice_parts, voice)
    voice_seconds = duration(voice)

    if args.kind == "short" and voice_seconds > 48:
        raise RuntimeError(
            f"Short safety stop: narration audio is {voice_seconds:.1f}s. Expected <= 48s."
        )

    portrait = args.kind == "short"
    wanted = 16 if portrait else 24
    clips, credits = PexelsClient(cfg.pexels_api_key).collect(
        plan["search_terms"], out / "assets", portrait=portrait, wanted=wanted
    )

    video = out / ("short.mp4" if portrait else "video.mp4")
    render_video(clips, voice, plan["narration"], video, portrait=portrait)

    thumb = out / "thumbnail.jpg"
    make_thumbnail(video, plan["title"], thumb)
    description = build_description(plan, credits)
    (out / "description.txt").write_text(description, encoding="utf-8")

    rights = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": {"source": "Gemini API free tier", "original_for_channel": True},
        "trend_research": {
            "source": "YouTube Data API v3 official public endpoints",
            "scout_version": (trend_report or {}).get("scout_version"),
            "trend_signal_used": plan.get("trend_signal", ""),
            "tiktok_scraping_used": False,
        },
        "voice": {"source": cfg.gemini_tts_model, "generated_for_channel": True},
        "music": None,
        "visual_assets": credits,
        "pexels_attribution_in_description": True,
        "contains_realistic_synthetic_media": bool(
            plan.get("contains_realistic_synthetic_media", False)
        ),
        "max_spend_eur": cfg.max_monthly_spend_eur,
        "voice_duration_seconds": round(voice_seconds, 2),
        "editor_version": "v4-trend-hard-gate-fast-shorts",
    }
    (out / "rights_ledger.json").write_text(
        json.dumps(rights, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    video_id = None
    if args.upload:
        video_id = upload(cfg, video, thumb, plan, description)

    add_publication({
        "run_id": run_id,
        "kind": args.kind,
        "topic": plan["topic"],
        "trend_signal": plan.get("trend_signal", ""),
        "title": plan["title"],
        "youtube_video_id": video_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    print(json.dumps({
        "ok": True,
        "run_id": run_id,
        "kind": args.kind,
        "trend_signal": plan.get("trend_signal", ""),
        "title": plan["title"],
        "voice_duration_seconds": round(voice_seconds, 2),
        "video": str(video),
        "youtube_video_id": video_id,
    }, indent=2))


if __name__ == "__main__":
    main()
