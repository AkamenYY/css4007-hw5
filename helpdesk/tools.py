"""Part 2: the three reusable tools.

Each one is a plain Python function that works without the agent:

    find_instructions("wifi")
    check_service_status("wifi", status_profile="wifi_outage")
    prepare_ticket(category="wifi", summary="...", steps_tried=["..."], ...)

None of them raises for bad input; they return {"error": ...} so a model can read the problem.
The agent (agent.py) passes the application-chosen status profile, user and thread to them;
the model never chooses those.
"""
import copy
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import data

SERVICES = ("wifi", "account_recovery")
STATUSES = ("operational", "outage", "unknown", "not_checked")
SECRET_PATTERN = re.compile(
    r"(password|passcode|recovery code|api[ _-]?key|token)\s*(is|was|:|=)\s*\S+", re.IGNORECASE)


def _topic(value: str):
    value = (value or "").strip().lower()
    if value in data.helpdesk()["instructions"]:
        return value
    topics = data.detect_topics(value)
    return next(iter(topics)) if len(topics) == 1 else None


# --------------------------------------------------------- find_instructions

def find_instructions(topic: str) -> dict:
    """Return the local Wi-Fi or password guide from data/helpdesk.json (retrieval, not memory)."""
    key = _topic(topic)
    if key is None:
        return {"error": f"no instructions for '{topic}'. Available topics: "
                         f"{', '.join(data.helpdesk()['instructions'])}"}
    guide = data.helpdesk()["instructions"][key]
    return {"topic": key, "instruction_id": guide["instruction_id"], "title": guide["title"],
            "steps": list(guide["steps"]), "source": "data/helpdesk.json"}


# ------------------------------------------------------ check_service_status

def check_service_status(service: str, status_profile: str = "normal") -> dict:
    """Look up a service in the selected snapshot of data/service_status.json.

    'unknown' means the status could not be determined: not operating and not a confirmed outage.
    """
    profiles = data.service_status()["profiles"]
    if status_profile not in profiles:
        return {"error": f"unknown status profile '{status_profile}'. Profiles: {', '.join(profiles)}"}
    name = (service or "").strip().lower()
    if name not in SERVICES:  # accept a topic name too: password -> account_recovery
        topic = _topic(name)
        name = data.helpdesk()["topic_services"].get(topic, name) if topic else name
    if name not in profiles[status_profile]:
        return {"error": f"unknown service '{service}'. Services: {', '.join(SERVICES)}"}
    entry = profiles[status_profile][name]
    return {"service": name, "status": entry["status"], "message": entry["message"],
            "status_profile": status_profile, "source": "data/service_status.json",
            "repair_time": None}


# ------------------------------------------------------------ prepare_ticket

def _clean_steps(steps) -> list:
    if isinstance(steps, str):
        steps = [steps]
    return [str(s).strip() for s in (steps or []) if str(s).strip()]


def prepare_ticket(category: str, summary: str, steps_tried, *, user_id: str, thread_id: str,
                   status_profile: str, service_status: str = "not_checked",
                   reference_ids=None, ticket_dir="outputs/tickets") -> dict:
    """Write a local ticket draft (data/ticket_template.json shape). Nothing is sent anywhere.

    Missing problem description or steps -> no file, an error listing what to ask the user.
    """
    topic = _topic(category)
    summary = (summary or "").strip()
    steps = _clean_steps(steps_tried)
    missing = []
    if topic is None:
        missing.append("category (wifi or password)")
    if not summary:
        missing.append("problem description")
    if not steps:
        missing.append("steps already tried")
    if missing:
        return {"error": "missing_information", "missing": missing,
                "ask_user": "Ask the user for: " + ", ".join(missing) + ". Do not invent them."}
    if any(SECRET_PATTERN.search(t) for t in [summary, *steps]):
        return {"error": "secret_information",
                "ask_user": "Remove passwords, recovery codes or keys. They must never be in a ticket."}
    if service_status not in STATUSES:
        service_status = "not_checked"

    ticket = copy.deepcopy(data.ticket_template()["ticket"])
    ticket_id = "TKT-DEMO-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:4]
    ticket.update({
        "ticket_id": ticket_id, "status": "draft", "user_id": user_id, "thread_id": thread_id,
        "category": topic, "summary": summary, "steps_tried": steps,
        "service_status": service_status, "status_profile": status_profile,
        "reference_ids": list(reference_ids or []),
    })
    folder = Path(ticket_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{ticket_id}.json"
    path.write_text(json.dumps({"ticket": ticket}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"ticket_id": ticket_id, "file": path.as_posix(), "status": "draft",
            "service_status": service_status, "sent": False,
            "note": "Local draft only; nothing was sent to a helpdesk."}
