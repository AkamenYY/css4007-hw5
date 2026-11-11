"""Part 3: where memory is saved.

Two kinds of memory, in two files inside local-storage/ (git-ignored):

  conversations.sqlite  CONVERSATION memory. LangGraph SqliteSaver (a checkpointer).
                        One saved state per thread_id: the messages of that conversation.
  user_memory.sqlite    USER memory. LangGraph SqliteStore.
                        One record per user_id: the reply language. Shared by all threads of that user.

`python -m helpdesk reset` deletes the folder.
"""
import os
import re
import shutil
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from .data import ROOT

LANGUAGES = ["Kazakh", "Russian", "English"]
PREFERENCE = re.compile(
    r"\b(?:prefer|want)\s+(?:replies|answers)\s+in\s+(kazakh|russian|english)\b"
    r"|\b(?:reply|answer|respond)\s+(?:to\s+me\s+)?in\s+(kazakh|russian|english)\b", re.IGNORECASE)


def storage_dir() -> Path:
    path = Path(os.getenv("HELPDESK_STORAGE_DIR", "local-storage"))
    return path if path.is_absolute() else ROOT / path


def conversations_path() -> Path:
    return storage_dir() / "conversations.sqlite"


def user_memory_path() -> Path:
    return storage_dir() / "user_memory.sqlite"


def open_checkpointer() -> SqliteSaver:
    conversations_path().parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(str(conversations_path()), check_same_thread=False))
    saver.setup()
    return saver


def open_store() -> SqliteStore:
    user_memory_path().parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStore(sqlite3.connect(str(user_memory_path()), check_same_thread=False, isolation_level=None))
    store.setup()
    return store


def close(*things) -> None:
    for thing in things:
        thing.conn.close()


def reset_storage() -> Path:
    """Delete all saved conversations and user preferences."""
    if storage_dir().exists():
        shutil.rmtree(storage_dir())
    return storage_dir()


def parse_language_preference(text: str):
    """'Please remember that I prefer replies in Kazakh' -> 'Kazakh'. None if no preference is stated."""
    match = PREFERENCE.search(text or "")
    return (match.group(1) or match.group(2)).capitalize() if match else None


def read_preferences(store, user_id: str) -> dict:
    item = store.get(("users", user_id), "preferences")
    return dict(item.value) if item else {}


def save_language(store, user_id: str, language: str) -> dict:
    prefs = read_preferences(store, user_id)
    prefs["language"] = language
    store.put(("users", user_id), "preferences", prefs)
    return prefs
