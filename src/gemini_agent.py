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
            length = "75-120 spoken words, 25-45 seconds"
            structure = "one immediate hook, escalating explanation/story, strong final twist or question"
        else:
            length = "800-1100 spoken words, roughly 6-9 minutes"
            structure = "cold open, 5-8 clear story beats, escalating stakes, satisfying ending, short CTA"

        prompt = f"""
You are the editorial director of an English-language faceless YouTube channel named {self.cfg.channel_brand}.
Niche: {self.cfg.channel_niche}.
Create ONE genuinely original {kind} concept. Avoid celebrity/news claims, copyrighted fictional universes, movie/anime/game characters, copied stories, medical/legal/financial advice, or claims presented as real when fictional.
The content should be clearly transformative and story-led, not a generic fact slideshow.
Target narration length: {length}.
Structure: {structure}.
Do not repeat or closely imitate these previous topics: {json.dumps(used[-40:])}.

Return ONLY valid JSON with exactly these keys:
{{
  "topic": "short internal topic label",
  "title": "YouTube-ready English title",
  "description": "2-4 sentence English description; do not include asset credits yet",
  "narration": "complete English narration",
  "search_terms": ["6-12 concrete Pexels search phrases"],
  "tags": ["5-12 relevant tags"],
  "contains_realistic_synthetic_media": false,
  "fiction_disclaimer": "short sentence if fictional, otherwise empty string"
}}
"""
        response = self.client.models.generate_content(
            model=self.cfg.gemini_text_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=1.0),
        )
        plan = _extract_json(response.text)
        required = {"topic", "title", "description", "narration", "search_terms", "tags", "contains_realistic_synthetic_media", "fiction_disclaimer"}
        missing = required - set(plan)
        if missing:
            raise ValueError(f"Plan missing fields: {sorted(missing)}")
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
                        contents=f"Read the following narration clearly in a cinematic, natural, calm English voice. Do not add words.\\n\\n{chunk}",
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
