import base64
import json
import re
import time
import wave
from pathlib import Path

from google import genai
from google.genai import types

from .config import Config
from .memory import recent_topics


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model did not return JSON: {text[:300]}")
    return json.loads(text[start:end + 1])


class GeminiAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = genai.Client(api_key=cfg.gemini_api_key)

    def create_content_plan(self, kind: str, trend_report: dict | None = None) -> dict:
        used = recent_topics(60)

        if kind == "short":
            length = "STRICTLY 60-82 spoken words, designed for 28-43 seconds. Never exceed 82 narration words."
            structure = (
                "hook in the first sentence, immediate payoff, fast escalation, one clear thread, "
                "final reveal/twist/question in the last sentence"
            )
            search_rule = (
                "Return 14-18 concrete Pexels-friendly shot searches in STORY ORDER. "
                "Each phrase must describe a visible real-world shot."
            )
        else:
            length = "800-1100 spoken words, roughly 6-9 minutes"
            structure = "cold open, 5-8 clear beats, escalating stakes, satisfying ending, short CTA"
            search_rule = (
                "Return 20-30 concrete Pexels-friendly shot searches in STORY ORDER. "
                "Every phrase must describe a filmable shot or location."
            )

        if trend_report:
            editorial = trend_report.get("editorial_plan") or {}
            if not editorial.get("ready_to_produce"):
                raise RuntimeError(
                    "Trend safety stop: V4 Trend Scout did not find enough independently "
                    "validated reproducible signals. Do not generate filler content."
                )

            idea_pool = editorial.get("short_ideas") if kind == "short" else [editorial.get("long_idea")]
            idea_pool = [x for x in (idea_pool or []) if x]
            if not idea_pool:
                raise RuntimeError("Trend safety stop: editorial slate is empty.")

            trend_context = json.dumps(
                {
                    "qualified_signals": [
                        x for x in trend_report.get("signals", []) if x.get("qualified")
                    ][:5],
                    "idea_pool": idea_pool,
                },
                ensure_ascii=False,
            )

            source_instruction = f"""
Choose ONE concept from idea_pool. Preserve its viral_mechanic and its connection to
the live trend. You may sharpen the execution, but do not drift into an unrelated
documentary/explainer. The result must remain original and copyright-safe.

LIVE TREND CONTEXT:
{trend_context}
"""
        else:
            source_instruction = (
                "No live trend report was provided. Create one original concept inside the channel direction."
            )

        prompt = f"""
You are the editorial director of an English-language faceless YouTube channel named {self.cfg.channel_brand}.
Channel direction: {self.cfg.channel_niche}.

{source_instruction}

Hard safety:
- no copied titles/scripts/stories
- no celebrity/gossip dependency
- no copyrighted fictional universes/characters
- no protected music/movie/game/sports footage
- no current political/news reporting
- no medical/legal/financial advice
- label fiction when fictional

Target narration length: {length}
Structure: {structure}
Do not repeat previous channel topics: {json.dumps(used[-40:])}
{search_rule}

Return ONLY JSON:
{{
  "topic": "short internal topic label",
  "trend_signal": "qualified live trend signal used, or empty string",
  "title": "original YouTube-ready English title",
  "description": "2-4 sentence description; no asset credits yet",
  "narration": "complete English narration",
  "search_terms": ["ordered stock-footage searches"],
  "tags": ["5-12 relevant tags"],
  "contains_realistic_synthetic_media": false,
  "fiction_disclaimer": "short sentence if fictional, otherwise empty string"
}}
"""
        for attempt in range(3):
            response = self.client.models.generate_content(
                model=self.cfg.gemini_text_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.72 if attempt == 0 else 0.4),
            )
            plan = _extract_json(response.text)

            if kind != "short":
                return self._validate_plan(plan)

            wc = len(re.findall(r"\b[\w'-]+\b", str(plan.get("narration") or "")))
            if 55 <= wc <= 82:
                return self._validate_plan(plan)

            prompt += f"\nPrevious narration was {wc} words. Rewrite to exactly 60-82 spoken words."

        raise RuntimeError("Gemini could not produce a Short within the 60-82 word safety range.")

    @staticmethod
    def _validate_plan(plan: dict) -> dict:
        required = {
            "topic", "trend_signal", "title", "description", "narration", "search_terms",
            "tags", "contains_realistic_synthetic_media", "fiction_disclaimer"
        }
        missing = required - set(plan)
        if missing:
            raise ValueError(f"Plan missing fields: {sorted(missing)}")
        if not isinstance(plan.get("search_terms"), list) or len(plan["search_terms"]) < 6:
            raise ValueError("Plan needs at least 6 visual search terms.")
        return plan

    @staticmethod
    def _chunks(text: str, max_chars: int = 2800) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks, cur = [], ""
        for s in sentences:
            if cur and len(cur) + len(s) + 1 > max_chars:
                chunks.append(cur.strip())
                cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            chunks.append(cur)
        return chunks

    def synthesize_voice(self, narration: str, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for idx, chunk in enumerate(self._chunks(narration)):
            last_error = None
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=self.cfg.gemini_tts_model,
                        contents=(
                            "Read the following narration in a natural, energetic, modern short-form voice. "
                            "Clear and slightly fast, but not rushed. Do not add or remove words.\n\n" + chunk
                        ),
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=types.SpeechConfig(
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name=self.cfg.gemini_tts_voice
                                    )
                                )
                            ),
                        ),
                    )
                    pcm = response.candidates[0].content.parts[0].inline_data.data
                    if isinstance(pcm, str):
                        pcm = base64.b64decode(pcm)
                    path = out_dir / f"voice_{idx:02d}.wav"
                    with wave.open(str(path), "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        wf.writeframes(pcm)
                    outputs.append(path)
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"TTS failed after retries: {last_error}")
        return outputs
