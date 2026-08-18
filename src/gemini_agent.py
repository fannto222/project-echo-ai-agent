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
                "final twist/question in the last sentence"
            )
            search_rule = (
                "Return 14-18 concrete Pexels-friendly shot searches in STORY ORDER. "
                "Each phrase must describe a visible real-world shot. Avoid abstract concepts, brands, "
                "celebrities, copyrighted characters, movie/game names and impossible CGI-only scenes."
            )
        else:
            length = "800-1100 spoken words, roughly 6-9 minutes"
            structure = "cold open, 5-8 clear beats, escalating stakes, satisfying ending, short CTA"
            search_rule = (
                "Return 20-30 concrete Pexels-friendly shot searches in STORY ORDER. "
                "Every phrase must describe a filmable shot or location."
            )

        if trend_report:
            slate = trend_report.get("editorial_plan") or {}
            idea_pool = slate.get("short_ideas") if kind == "short" else [slate.get("long_idea")]
            idea_pool = [x for x in (idea_pool or []) if x]
            trend_context = json.dumps(
                {
                    "top_signals": trend_report.get("signals", [])[:5],
                    "idea_pool": idea_pool,
                },
                ensure_ascii=False,
            )
            source_instruction = f"""
Choose ONE idea from the live trend-derived idea_pool below. You may improve the angle,
but you MUST stay connected to the selected live signal. Do not invent an unrelated topic.
Do not copy any evidence/sample title. The output must be wholly original and feasible with stock footage.
LIVE TREND CONTEXT:
{trend_context}
"""
        else:
            source_instruction = (
                "No live trend report was provided. Create one original concept inside the channel niche."
            )

        prompt = f"""
You are the editorial director of an English-language faceless YouTube channel named {self.cfg.channel_brand}.
Channel direction: {self.cfg.channel_niche}.

{source_instruction}

Hard safety rules:
- no celebrity/gossip dependency
- no copyrighted fictional universes/characters
- no copied titles/scripts/stories
- no music/movie/game/sports footage dependency
- no current political/news reporting
- no medical/legal/financial advice
- clearly label fiction when fictional

Target narration length: {length}
Structure: {structure}
Do not repeat these previous channel topics: {json.dumps(used[-40:])}
{search_rule}

Return ONLY valid JSON:
{{
  "topic": "short internal topic label",
  "trend_signal": "live trend signal used, or empty string",
  "title": "original YouTube-ready English title",
  "description": "2-4 sentence English description; do not include asset credits yet",
  "narration": "complete English narration",
  "search_terms": ["ordered concrete stock-footage searches"],
  "tags": ["5-12 relevant tags"],
  "contains_realistic_synthetic_media": false,
  "fiction_disclaimer": "short sentence if fictional, otherwise empty string"
}}
"""
        for attempt in range(3):
            response = self.client.models.generate_content(
                model=self.cfg.gemini_text_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.75 if attempt == 0 else 0.45),
            )
            plan = _extract_json(response.text)
            if kind != "short":
                return self._validate_plan(plan)

            wc = len(re.findall(r"\b[\w'-]+\b", str(plan.get("narration") or "")))
            if 55 <= wc <= 82:
                return self._validate_plan(plan)

            prompt += f"\nYour previous narration was {wc} words. Rewrite it to 60-82 spoken words exactly."

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
