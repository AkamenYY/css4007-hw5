"""Part 1: a helpdesk graph with fixed routing rules and no model.

original:            START -> read_request -> wifi_help | password_help | other_help -> END
with_clarification:  START -> read_request -> wifi_help | password_help | other_help | clarify_problem -> END
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import data

VARIANTS = ("original", "with_clarification")


class Part1State(TypedDict, total=False):
    message: str    # the user's original message (never changed after START)
    category: str   # set by read_request: wifi | password | other | both
    reply: str      # set by exactly one help node


# ------------------------------------------------------------------ nodes

def read_request_original(state: Part1State) -> dict:
    """Reads: message. Changes: category. Wi-Fi is checked first, so it wins over password."""
    topics = data.detect_topics(state["message"])
    if "wifi" in topics:
        category = "wifi"
    elif "password" in topics:
        category = "password"
    else:
        category = "other"
    return {"category": category}


def read_request_with_clarification(state: Part1State) -> dict:
    """Same rules, but a message naming both topics gets category 'both'."""
    topics = data.detect_topics(state["message"])
    if {"wifi", "password"} <= topics:
        return {"category": "both"}
    return read_request_original(state)


def _instructions_reply(topic: str) -> str:
    guide = data.helpdesk()["instructions"][topic]
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(guide["steps"], 1))
    return f"{guide['title']} ({guide['instruction_id']}):\n{steps}"


def wifi_help(state: Part1State) -> dict:
    """Reads: nothing from state (the route already decided). Changes: reply."""
    return {"reply": _instructions_reply("wifi")}


def password_help(state: Part1State) -> dict:
    return {"reply": _instructions_reply("password")}


def other_help(state: Part1State) -> dict:
    return {"reply": "I can only help with Wi-Fi or password problems here. "
                     "Is your problem about Wi-Fi or about a password?"}


def clarify_problem(state: Part1State) -> dict:
    return {"reply": "Your message mentions both Wi-Fi and a password. Which should we solve first: "
                     "connecting to Wi-Fi, or recovering your account password?"}


# ------------------------------------------------------------- the edge

ROUTES = {"wifi": "wifi_help", "password": "password_help",
          "other": "other_help", "both": "clarify_problem"}


def choose_next_node(state: Part1State) -> str:
    """Conditional edge. Reads: category. Changes nothing; it only returns a node name."""
    return ROUTES[state["category"]]


# ------------------------------------------------------------- the graph

def build_graph(variant: str = "original"):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    g = StateGraph(Part1State)
    reader = read_request_original if variant == "original" else read_request_with_clarification
    g.add_node("read_request", reader)
    g.add_node("wifi_help", wifi_help)
    g.add_node("password_help", password_help)
    g.add_node("other_help", other_help)
    targets = ["wifi_help", "password_help", "other_help"]
    if variant == "with_clarification":
        g.add_node("clarify_problem", clarify_problem)
        targets.append("clarify_problem")
    g.add_edge(START, "read_request")
    g.add_conditional_edges("read_request", choose_next_node, targets)
    for node in targets:
        g.add_edge(node, END)
    return g.compile()


def run(message: str, variant: str = "original") -> dict:
    """Run one message and record the node order and the full state after every node."""
    graph = build_graph(variant)
    trace = []
    state = {"message": message}
    for mode, chunk in graph.stream({"message": message}, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node, update in chunk.items():
                trace.append({"node": node, "update": dict(update or {})})
        elif mode == "values" and trace:
            trace[-1]["state_after"] = dict(chunk)
            state = dict(chunk)
    return {"variant": variant, "message": message,
            "route": [step["node"] for step in trace],
            "trace": trace, "final_state": state}


def mermaid(variant: str = "original") -> str:
    return build_graph(variant).get_graph().draw_mermaid()
