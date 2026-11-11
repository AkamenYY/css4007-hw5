"""Part 3: where memory lives.

Two kinds of memory, two files under HELPDESK_STORAGE_DIR (default: local-storage/, git-ignored):

  conversations.sqlite  LangGraph SqliteSaver checkpointer. Key: thread_id.
                        Holds one conversation's state: its messages (problem, steps tried, ...).
  user_memory.sqlite    LangGraph SqliteStore. Namespace ("users", user_id), key "preferences".
                        Holds the user's saved reply language, shared by all of that user's threads.

Delete the folder (python -m helpdesk reset) to start again.
"""
import os
import re
import shutil
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from .data import ROOT

LANGUAGES = {
    "kazakh": "Kazakh", "қазақ": "Kazakh", "қазақша": "Kazakh", "қазақ тілінде": "Kazakh",
    "russian": "Russian", "русском": "Russian", "русский": "Russian",
    "english": "English", "английском": "English",
}
_LANG_WORDS = "|".join(sorted((re.escape(k) for k in LANGUAGES), key=len, reverse=True))
PREFERENCE_PATTERN = re.compile(
    r"\b(?:prefer|want|like)\s+(?:my\s+|your\s+)?(?:replies|answers|responses)\s+in\s+(" + _LANG_WORDS + r")\b"
    r"|\b(?:reply|respond|answer|write)\s+(?:to\s+me\s+)?in\s+(" + _LANG_WORDS + r")\b",
    re.IGNORECASE)


def storage_dir() -> Path:
    path = Path(os.getenv("HELPDESK_STORAGE_DIR", "local-storage"))
    return path if path.is_absolute() else ROOT / path


def conversations_path() -> Path:
    return storage_dir() / "conversations.sqlite"


def user_memory_path() -> Path:
    return storage_dir() / "user_memory.sqlite"


def open_checkpointer(path=None) -> SqliteSaver:
    path = Path(path or conversations_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
    saver.setup()
    return saver


def open_store(path=None) -> SqliteStore:
    path = Path(path or user_memory_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStore(sqlite3.connect(str(path), check_same_thread=False, isolation_level=None))
    store.setup()
    return store


def close(*objects) -> None:
    for obj in objects:
        conn = getattr(obj, "conn", None)
        if conn is not None:
            conn.close()


def reset_storage() -> Path:
    """Delete all conversation checkpoints and user preferences."""
    path = storage_dir()
    if path.exists():
        shutil.rmtree(path)
    return path


def parse_language_preference(text: str):
    """'Please remember that I prefer replies in Kazakh' -> 'Kazakh'. None if no preference is stated."""
    match = PREFERENCE_PATTERN.search(text or "")
    if not match:
        return None
    word = (match.group(1) or match.group(2)).lower()
    return LANGUAGES.get(word)


USER_NAMESPACE = "users"


def read_preferences(store, user_id: str) -> dict:
    item = store.get((USER_NAMESPACE, user_id), "preferences")
    return dict(item.value) if item else {}


def save_language(store, user_id: str, language: str) -> dict:
    prefs = read_preferences(store, user_id)
    prefs["language"] = language
    store.put((USER_NAMESPACE, user_id), "preferences", prefs)
    return prefs
