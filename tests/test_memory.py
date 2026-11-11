"""Part 3 memory: saved after a restart, separate between threads and between users."""
import json
import os
import subprocess
import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from helpdesk import memory
from helpdesk.agent import HelpdeskSession
from helpdesk.models import ScriptedFakeModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def chat(storage, user, thread, text, model=None):
    model = model or ScriptedFakeModel(responses=[AIMessage(content="ok")])
    s = HelpdeskSession(model, user, thread, "normal", ticket_dir=str(storage / "tickets"))
    try:
        return s.send(text), model
    finally:
        s.close()  # closes both SQLite connections, like quitting the program


def test_conversation_survives_reopening_storage(storage):
    # Catches: an in-memory checkpointer, or a thread_id that is not used as the key.
    chat(storage, "demo-user-a", "restart-a", "My issue is Wi-Fi. I enabled it and tried reconnecting.")
    turn, model = chat(storage, "demo-user-a", "restart-a", "Which problem and steps did we discuss?")
    seen = [m.content for m in model.seen[0] if isinstance(m, HumanMessage)]
    assert seen[0].startswith("My issue is Wi-Fi") and len(seen) == 2
    assert turn["memory"]["conversation_messages_before_this_turn"] == 2


def test_conversation_survives_a_real_process_restart(storage):
    # Catches: state that only lives in the running process. Two separate Python processes.
    env = dict(os.environ)
    log = storage / "run.jsonl"
    base = [sys.executable, "-m", "helpdesk", "chat", "--fake-model", "--user", "demo-user-a",
            "--thread", "restart-a", "--log", str(log), "--short"]
    subprocess.run(base + ["-m", "My issue is Wi-Fi. I enabled it and tried reconnecting."],
                   cwd=ROOT, env=env, check=True, capture_output=True)
    subprocess.run(base + ["-m", "Which problem and steps did we discuss?"],
                   cwd=ROOT, env=env, check=True, capture_output=True)
    first, second = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert first["process"]["pid"] != second["process"]["pid"]
    assert second["turns"][0]["memory"]["conversation_messages_before_this_turn"] > 0
    assert "My issue is Wi-Fi" in second["turns"][0]["reply"]


def test_preference_is_shared_by_threads_of_the_same_user_only(storage):
    # Catches: the preference stored per thread (lost in a new thread), or the old transcript replayed.
    chat(storage, "demo-user-a", "preference-a1", "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.")
    turn, model = chat(storage, "demo-user-a", "preference-a2", "Which language did I ask you to use?")
    prompt = model.seen[0][0]
    assert isinstance(prompt, SystemMessage) and "Kazakh" in prompt.content
    assert [m.content for m in model.seen[0][1:]] == ["Which language did I ask you to use?"]
    assert turn["memory"]["user_memory_read"] == {"language": "Kazakh"}


def test_preference_does_not_leak_to_another_user(storage):
    # Catches: a store namespace that ignores the user id.
    chat(storage, "demo-user-a", "isolation-a", "Please remember that I prefer replies in Kazakh.")
    turn, model = chat(storage, "demo-user-b", "isolation-b", "Have I told you a preferred language?")
    assert "Kazakh" not in model.seen[0][0].content
    assert turn["memory"]["user_memory_read"] is None
    assert len(model.seen[0]) == 2  # system prompt + this message: nothing from demo-user-a's thread


def test_threads_have_separate_histories(storage):
    chat(storage, "demo-user-a", "t1", "First thread message")
    turn, model = chat(storage, "demo-user-a", "t2", "Second thread message")
    assert "First thread message" not in json.dumps([m.content for m in model.seen[0]])


def test_preference_parsing():
    assert memory.parse_language_preference("Please remember that I prefer replies in Kazakh.") == "Kazakh"
    assert memory.parse_language_preference("Reply in Russian please") == "Russian"
    assert memory.parse_language_preference("I have a Wi-Fi problem") is None


def test_reset_clears_both_kinds_of_memory(storage):
    chat(storage, "demo-user-a", "t1", "Please remember that I prefer replies in Kazakh.")
    assert memory.conversations_path().exists() and memory.user_memory_path().exists()
    memory.reset_storage()
    assert not memory.storage_dir().exists()
    turn, _ = chat(storage, "demo-user-a", "t1", "Hello")
    assert turn["memory"]["user_memory_read"] is None
    assert turn["memory"]["conversation_messages_before_this_turn"] == 0
