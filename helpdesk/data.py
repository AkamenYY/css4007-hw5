"""Load the supplied fictional fixtures from data/."""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def helpdesk() -> dict:
    return _load("helpdesk.json")


def service_status() -> dict:
    return _load("service_status.json")


def ticket_template() -> dict:
    return _load("ticket_template.json")


def scenarios() -> list:
    return _load("scenarios.json")["scenarios"]


def status_profiles() -> list:
    return list(service_status()["profiles"])


def detect_topics(message: str) -> set:
    """Topics whose aliases appear in the message, ignoring case (aliases come from helpdesk.json)."""
    text = (message or "").lower()
    return {topic for topic, aliases in helpdesk()["topic_aliases"].items()
            if any(alias in text for alias in aliases)}
