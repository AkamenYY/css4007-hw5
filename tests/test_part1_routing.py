"""Part 1 routing (no model)."""
import pytest

from helpdesk import part1

ORIGINAL = [
    ("I cannot connect to Wi-Fi.", "wifi_help", "wifi"),
    ("I forgot my PASSWORD.", "password_help", "password"),
    ("My laptop screen is broken.", "other_help", "other"),
    ("My Wi-Fi password does not work.", "wifi_help", "wifi"),   # Wi-Fi rule is checked first
]


@pytest.mark.parametrize("message,node,category", ORIGINAL)
def test_original_graph_routes(message, node, category):
    # Catches: a wrong rule order, a missing alias, or a broken conditional edge.
    r = part1.run(message, "original")
    assert r["route"] == ["read_request", node]
    assert r["final_state"]["category"] == category


def test_clarification_graph_only_changes_two_topic_messages():
    # Catches: the new node stealing single-topic messages, or two-topic messages not reaching it.
    assert part1.run("My Wi-Fi password does not work.", "with_clarification")["route"] == [
        "read_request", "clarify_problem"]
    for message, node, _ in ORIGINAL[:3]:
        assert part1.run(message, "with_clarification")["route"] == ["read_request", node]


@pytest.mark.parametrize("message", ["WIFI is down", "my wi-fi", "WiFi", "PaSsWoRd reset"])
def test_routing_ignores_case(message):
    # Catches: comparing without lowercasing.
    assert part1.run(message)["route"][1] in ("wifi_help", "password_help")


def test_trace_shows_when_each_field_is_set():
    # Catches: a node overwriting the original message, or the reply being set before a help node runs.
    r = part1.run("I cannot connect to Wi-Fi.")
    first, second = r["trace"]
    assert first["update"] == {"category": "wifi"}
    assert "reply" not in first["state_after"]
    assert set(second["update"]) == {"reply"}
    assert "GUIDE-WIFI" in second["state_after"]["reply"]
    assert all(s["state_after"]["message"] == "I cannot connect to Wi-Fi." for s in r["trace"])


def test_help_nodes_use_helpdesk_json():
    # Catches: hard-coded instructions drifting from the supplied data file.
    reply = part1.run("I forgot my PASSWORD.")["final_state"]["reply"]
    assert "GUIDE-PASSWORD" in reply and "Forgot password" in reply


def test_other_help_asks_which_topic():
    reply = part1.run("My laptop screen is broken.")["final_state"]["reply"]
    assert "Wi-Fi" in reply and "password" in reply and "?" in reply
