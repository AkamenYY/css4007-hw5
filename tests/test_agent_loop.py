"""Part 2 loop, with a clearly labelled FAKE model (no API key needed)."""
import json

from langchain_core.messages import AIMessage, ToolMessage

from helpdesk.agent import MAX_TOOL_ROUNDS, HelpdeskSession
from helpdesk.models import ScriptedFakeModel, tool_call

REQUEST = ("I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not connect. "
           "Check whether Wi-Fi is down. If it is operating, prepare a ticket about this problem.")


def decide_after_status(messages):
    """FAKE model step that reads the status tool result, like a real model should."""
    result = json.loads(next(m for m in reversed(messages) if isinstance(m, ToolMessage)).content)
    if result["status"] == "operational":
        return AIMessage(content="", tool_calls=[tool_call("prepare_ticket", {
            "category": "wifi", "summary": "Campus-Demo will not connect.",
            "steps_tried": ["Enabled Wi-Fi", "Forgot the saved network"]})])
    return AIMessage(content=f"Status is {result['status']}; no ticket prepared.")


def conditional_script():
    return [AIMessage(content="", tool_calls=[tool_call("check_service_status", {"service": "wifi"})]),
            decide_after_status,
            AIMessage(content="Prepared a local draft.")]


def session(storage, model, profile="normal", thread="t1", user="demo-user-a"):
    return HelpdeskSession(model, user, thread, profile, ticket_dir=str(storage / "tickets"))


def test_tool_result_returns_to_model_and_changes_the_action(storage):
    # Catches: tool results not added to the state, or the status profile being ignored.
    outcomes = {}
    for profile in ("normal", "wifi_outage", "status_unavailable"):
        model = ScriptedFakeModel(responses=conditional_script())
        s = session(storage, model, profile, thread=f"t-{profile}")
        turn = s.send(REQUEST)
        s.close()
        outcomes[profile] = turn
        second_call = model.seen[1]
        assert isinstance(second_call[-1], ToolMessage) and second_call[-1].name == "check_service_status"
    assert outcomes["normal"]["tickets"] and outcomes["normal"]["nodes"] == [
        "load_memory", "agent", "tools", "agent", "tools", "agent"]
    assert not outcomes["wifi_outage"]["tickets"] and "outage" in outcomes["wifi_outage"]["reply"]
    assert not outcomes["status_unavailable"]["tickets"] and "unknown" in outcomes["status_unavailable"]["reply"]


def test_ticket_takes_status_and_ids_from_the_program_not_the_model(storage):
    # Catches: the model choosing the user, thread or service status written into a ticket.
    s = session(storage, ScriptedFakeModel(responses=conditional_script()), "normal", thread="t9")
    turn = s.send(REQUEST)
    s.close()
    ticket = json.loads(open(turn["tickets"][0], encoding="utf-8").read())["ticket"]
    assert ticket["service_status"] == "operational"
    assert (ticket["user_id"], ticket["thread_id"], ticket["status_profile"]) == ("demo-user-a", "t9", "normal")


def test_loop_limit_stops_an_endless_model(storage):
    # Catches: a loop with no limit, or a limit that leaves unanswered tool calls in the history.
    endless = [AIMessage(content="", tool_calls=[tool_call("find_instructions", {"topic": "wifi"})])
               for _ in range(MAX_TOOL_ROUNDS + 5)]
    s = session(storage, ScriptedFakeModel(responses=endless))
    turn = s.send("Help with Wi-Fi")
    history = s.history()
    s.close()
    assert turn["loop_limit_reached"] is True
    assert turn["nodes"].count("tools") == MAX_TOOL_ROUNDS and turn["nodes"][-1] == "loop_limit"
    answered = {m.tool_call_id for m in history if isinstance(m, ToolMessage)}
    requested = {c["id"] for m in history for c in (getattr(m, "tool_calls", None) or [])}
    assert requested <= answered


def test_limit_resets_on_the_next_user_message(storage):
    endless = [AIMessage(content="", tool_calls=[tool_call("find_instructions", {"topic": "wifi"})])
               for _ in range(MAX_TOOL_ROUNDS + 1)] + [AIMessage(content="Fine now.")]
    s = session(storage, ScriptedFakeModel(responses=endless))
    s.send("Help with Wi-Fi")
    turn = s.send("Thanks")
    s.close()
    assert turn["loop_limit_reached"] is False and turn["reply"] == "Fine now."


def test_unknown_tool_is_refused_without_crashing(storage):
    # Catches: the program running any name the model invents, or crashing on a bad request.
    script = [AIMessage(content="", tool_calls=[tool_call("send_email", {"to": "it@example.com"})]),
              AIMessage(content="ok")]
    s = session(storage, ScriptedFakeModel(responses=script))
    turn = s.send("Email the helpdesk for me")
    s.close()
    assert "error" in turn["tool_results"][0]["result"] and turn["reply"] == "ok"
