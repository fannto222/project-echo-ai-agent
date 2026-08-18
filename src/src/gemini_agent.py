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

    def create_content_plan(self, kind: str) -> dict:
        used = recent_topics(60)
        if kind == "short":
            length = "STRICTLY 65-85 spoken words, designed for 30-45 seconds. Never exceed 85 narration words."
            structure = (
                "hook in the first sentence, fast escalation, one clear mystery/story thread, "
                "final twist or unanswered question in the last sentence"
            )
            search_rule = (
                "Return 14-18 concrete stock-footage search phrases in STORY ORDER. Each phrase must describe "
                "a visible shot that can realistically exist on Pexels. Use specific nouns and settings, e.g. "
                "'dark satellite control room', 'astronaut silhouette stars', 'radio telescope night'. "
                "Avoid abstract words, emotions, generic 'technology', copyrighted brands, fictional franchises, "
                "and impossible CGI-only scenes."
            )
        else:
            length = "800-1100 spoken words, roughly 6-9 minutes"
            structure = "cold open, 5-8 clear story beats, escalating stakes, satisfying ending, short CTA"
            search_rule = (
                "Return 20-30 concrete Pexels-friendly search phrases in STORY ORDER. Every phrase must describe "
                "a real filmable shot or location, not an abstract concept."
            )

        base_prompt = f"""
You are the editorial director of an English-language faceless YouTube channel named {self.cfg.channel_brand}.
Niche: {self.cfg.channel_niche}.
Create ONE genuinely original {kind} concept. Avoid celebrity/news claims, copyrighted fictional universes,
movie/anime/game characters, copied stories, medical/legal/financial advice, or claims presented as real when fictional.
The content must feel like a real short-form story, not a generic facts slideshow.
Target narration length: {length}
Structure: {structure}.
Visual planning rule: {search_rule}
Do not repeat or closely imitate these previous topics: {json.dumps(used[-40:])}.

Return ONLY valid JSON with exactly these keys:
{{
  "topic": "short internal topic label",
  "title": "YouTube-ready English title",
  "description": "2-4 sentence English description; do not include asset credits yet",
  "narration": "complete English narration",
  "search_terms": ["ordered concrete Pexels search phrases"],
  "tags": ["5-12 relevant tags"],
  "contains_realistic_synthetic_media": false,
  "fiction_disclaimer": "short sentence if fictional, otherwise empty string"
}}
"""

        last_problem = ""
        for attempt in range(3):
            prompt = base_prompt
            if last_problem:
                prompt += f"\nPrevious attempt was rejected because: {last_problem}. Fix this strictly.\n"

            response = self.client.models.generate_content(
                model=self.cfg.gemini_text_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=1.0),
            )
            plan = _extract_json(response.text)
            required = {
                "topic", "title", "description", "narration", "search_terms", "tags",
                "contains_realistic_synthetic_media", "fiction_disclaimer"
            }
            missing = required - set(plan)
            if missing:
                last_problem = f"missing JSON keys {sorted(missing)}"
                continue

            narration_words = len(str(plan["narration"]).split())
            search_terms = [str(x).strip() for x in plan.get("search_terms", []) if str(x).strip()]
            plan["search_terms"] = search_terms

            if kind == "short":
                if narration_words > 85:
                    last_problem = f"narration had {narration_words} words; maximum is 85"
                    continue
                if narration_words < 55:
                    last_problem = f"narration had only {narration_words} words; target is 65-85"
                    continue
                if len(search_terms) < 12:
                    last_problem = f"only {len(search_terms)} visual search phrases; need at least 12"
                    continue
            return plan

        raise RuntimeError(f"Gemini could not produce a valid {kind} plan after retries: {last_problem}")

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
                            "Read the following narration in natural cinematic English. "
                            "For Shorts, keep a confident brisk pace with clear diction and no dramatic long pauses. "
                            "Do not add or remove words.\n\n" + chunk
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
