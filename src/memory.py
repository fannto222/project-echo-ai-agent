import json
from datetime import datetime, timezone
from pathlib import Path

HISTORY = Path("data/history.json")


def load_history() -> dict:
    if not HISTORY.exists():
        return {"published": [], "topics_used": [], "last_updated": None}
    return json.loads(HISTORY.read_text(encoding="utf-8"))


def recent_topics(limit: int = 50) -> list[str]:
    return load_history().get("topics_used", [])[-limit:]


def add_publication(record: dict) -> None:
    data = load_history()
    data.setdefault("published", []).append(record)
    topic = record.get("topic")
    if topic:
        data.setdefault("topics_used", []).append(topic)
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
