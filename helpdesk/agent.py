"""Parts 2 and 3: the model chooses tools; the graph runs them; memory is kept between turns.

    START -> load_memory -> agent --(tool calls, under the limit)--> tools -> agent -> ...
                              |--(tool calls, limit reached)-----> loop_limit -> END
                              '--(no tool calls: an answer)------> END

load_memory  my code: saves a stated language preference to user memory, reads this user's
             preference, resets the per-turn tool-round counter.
agent        the MODEL: reads the conversation and either requests tools or answers.
tools        my code: validates each requested call and executes the plain tool function.
loop_limit   my code: stops the loop after MAX_TOOL_ROUNDS tool rounds in one user turn.
"""
import json
import operator
import os
from dataclasses import dataclass
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from . import data, memory, tools
from .models import model_label

MAX_TOOL_ROUNDS = 4  # tool rounds allowed per user message; enforced in route_after_agent


@dataclass
class HelpdeskContext:
    """Chosen by the application when a session starts. The model can never change these."""
    user_id: str
    thread_id: str
    status_profile: str = "normal"
    ticket_dir: str = "outputs/tickets"


class HelpdeskState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # conversation memory (checkpointed)
    tool_rounds: int                                     # tool rounds in the current user turn
    loop_limit_reached: bool
    preferred_language: Optional[str]                    # copied from user memory each turn
    memory: dict                                         # what this turn read/wrote, for the trace
    tickets: Annotated[list[str], operator.add]          # local ticket files made in this thread


# ------------------------------------------------------- tool definitions for the model

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "find_instructions",
        "description": "Look up the local helpdesk instructions for Wi-Fi or password problems "
                       "(data/helpdesk.json). Returns the guide id, title and steps.",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "enum": ["wifi", "password"]}},
            "required": ["topic"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "check_service_status",
        "description": "Check whether a demo service is operational, has an outage, or has unknown status. "
                       "Wi-Fi -> 'wifi'. Password recovery -> 'account_recovery'. The application chooses "
                       "which status snapshot is used.",
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string", "enum": ["wifi", "account_recovery"]}},
            "required": ["service"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "prepare_ticket",
        "description": "Save a LOCAL ticket draft (nothing is sent). Needs the problem description and the "
                       "steps the user said they already tried, in the user's own terms. The service status "
                       "is filled in automatically from earlier status checks (or 'not_checked').",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "enum": ["wifi", "password"]},
            "summary": {"type": "string", "description": "One-sentence problem description from the user."},
            "steps_tried": {"type": "array", "items": {"type": "string"},
                            "description": "Only steps the user said they tried. Never add steps."}},
            "required": ["category", "summary", "steps_tried"], "additionalProperties": False}}},
]
TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]


def system_prompt(preferred_language, just_saved: bool) -> str:
    hd = data.helpdesk()
    if preferred_language:
        memory_line = (f"User memory: this user asked for replies in {preferred_language}"
                       + (" (saved just now)." if just_saved else " (saved in an earlier conversation).")
                       + f" Always reply in {preferred_language}.")
    else:
        memory_line = ("User memory: no language preference or other information is recorded for this user. "
                       "You know nothing about other users or other conversations.")
    return "\n".join([
        "You are a practice university IT helpdesk assistant. All data is fictional demo data; nothing is "
        "connected to real university systems.",
        "You can help with: " + "; ".join(hd["scope"]) + ".",
        "Rules:",
        "- Get Wi-Fi or password instructions only with find_instructions. Do not write instructions from memory.",
        "- Before saying whether a service works, call check_service_status (Wi-Fi -> wifi, password -> "
        "account_recovery). 'unknown' means the status could not be determined: it is neither operating nor "
        "a confirmed outage. Never invent a repair time and never claim a real university outage.",
        "- Follow the user's conditions exactly. If they say 'if it is operating, prepare a ticket', check the "
        "status first and call prepare_ticket only if the result is 'operational'. If it is 'outage', say you did "
        "not prepare the ticket. If it is 'unknown', do not prepare it; ask whether they want a draft that says "
        "the status could not be checked.",
        "- prepare_ticket needs the problem and the steps the user said they already tried in this conversation. "
        "Use only the user's stated steps; never add steps. If something is missing, ask the user first.",
        "- Never ask for or store passwords, recovery codes or API keys.",
        "- A ticket is a local draft only: say it was not sent and give its file name.",
        "- Use the earlier messages of this conversation to understand references such as 'that' or "
        "'the steps I tried'. Do not claim to remember anything that is not in this conversation or in user memory.",
        memory_line,
    ])


# -------------------------------------------------------------------------- nodes

def load_memory(state: HelpdeskState, runtime: Runtime[HelpdeskContext]) -> dict:
    """Reads: the newest user message, user memory. Writes: user memory (if a preference is stated)."""
    user_id = runtime.context.user_id
    last = state["messages"][-1]
    stated = memory.parse_language_preference(last.content if isinstance(last, HumanMessage) else "")
    written = memory.save_language(runtime.store, user_id, stated) if stated else None
    prefs = memory.read_preferences(runtime.store, user_id)
    return {
        "tool_rounds": 0, "loop_limit_reached": False,
        "preferred_language": prefs.get("language"),
        "memory": {
            "user_id": user_id,
            "user_memory_read": prefs or None,
            "user_memory_written": written,
            "conversation_messages_before_this_turn": len(state["messages"]) - 1,
        },
    }


def make_agent_node(model):
    bound = model.bind_tools(TOOL_SCHEMAS)

    def agent(state: HelpdeskState) -> dict:
        """The model reads the system prompt + the whole conversation and returns one AI message."""
        mem = state.get("memory") or {}
        prompt = system_prompt(state.get("preferred_language"), bool(mem.get("user_memory_written")))
        reply = bound.invoke([SystemMessage(prompt), *state["messages"]])
        return {"messages": [reply]}

    return agent


def _last_status(messages, service):
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "check_service_status":
            try:
                result = json.loads(msg.content)
            except (TypeError, ValueError):
                continue
            if result.get("service") == service and "status" in result:
                return result["status"]
    return "not_checked"


def _reference_ids(messages):
    ids = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "find_instructions":
            try:
                rid = json.loads(msg.content).get("instruction_id")
            except (TypeError, ValueError):
                rid = None
            if rid and rid not in ids:
                ids.append(rid)
    return ids


def execute_tool_call(call, messages, ctx: HelpdeskContext, batch_names) -> dict:
    """Validate one model request and run the plain Python tool. Never raises."""
    name, args = call.get("name"), call.get("args")
    if name not in TOOL_NAMES:
        return {"error": f"unknown tool '{name}'"}
    if not isinstance(args, dict):
        return {"error": "arguments must be a JSON object"}
    try:
        if name == "find_instructions":
            return tools.find_instructions(args.get("topic", ""))
        if name == "check_service_status":
            return tools.check_service_status(args.get("service", ""), status_profile=ctx.status_profile)
        if "check_service_status" in batch_names:
            return {"error": "wait_for_status",
                    "ask_user": "A status check was requested in the same step. Read its result first, "
                                "then decide whether a ticket is still wanted."}
        category = args.get("category", "")
        service = data.helpdesk()["topic_services"].get(category, "wifi")
        return tools.prepare_ticket(
            category=category, summary=args.get("summary", ""), steps_tried=args.get("steps_tried", []),
            user_id=ctx.user_id, thread_id=ctx.thread_id, status_profile=ctx.status_profile,
            service_status=_last_status(messages, service), reference_ids=_reference_ids(messages),
            ticket_dir=ctx.ticket_dir)
    except Exception as exc:  # keep the loop alive
        return {"error": f"{name} failed: {type(exc).__name__}"}


def run_tools(state: HelpdeskState, runtime: Runtime[HelpdeskContext]) -> dict:
    """My program executes every tool the model requested, then hands the results back."""
    request = state["messages"][-1]
    batch = [c["name"] for c in request.tool_calls]
    out, tickets = [], []
    for call in request.tool_calls:
        result = execute_tool_call(call, state["messages"], runtime.context, batch)
        if "file" in result:
            tickets.append(result["file"])
        out.append(ToolMessage(content=json.dumps(result, ensure_ascii=False),
                               tool_call_id=call["id"], name=call["name"]))
    return {"messages": out, "tool_rounds": state.get("tool_rounds", 0) + 1, "tickets": tickets}


def loop_limit(state: HelpdeskState) -> dict:
    """Answer the pending calls without running them and stop."""
    request = state["messages"][-1]
    skipped = [ToolMessage(content=json.dumps({"error": "not run: tool loop limit reached"}),
                           tool_call_id=c["id"], name=c["name"]) for c in request.tool_calls]
    final = AIMessage(content=f"I stopped after {MAX_TOOL_ROUNDS} rounds of tool calls without finishing. "
                              "Please rephrase or give more detail.")
    return {"messages": [*skipped, final], "loop_limit_reached": True}


def route_after_agent(state: HelpdeskState) -> str:
    """Conditional edge: tool requests -> tools (or loop_limit); a plain answer -> END."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "loop_limit" if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS else "tools"
    return END


def build_graph(model, checkpointer=None, store=None):
    g = StateGraph(HelpdeskState, context_schema=HelpdeskContext)
    g.add_node("load_memory", load_memory)
    g.add_node("agent", make_agent_node(model))
    g.add_node("tools", run_tools)
    g.add_node("loop_limit", loop_limit)
    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "agent")
    g.add_conditional_edges("agent", route_after_agent, ["tools", "loop_limit", END])
    g.add_edge("tools", "agent")
    g.add_edge("loop_limit", END)
    return g.compile(checkpointer=checkpointer, store=store)


# ------------------------------------------------------------------- trace helpers

def message_to_dict(msg) -> dict:
    d = {"type": msg.type, "content": msg.content}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [{"name": c["name"], "args": c["args"], "id": c["id"]} for c in msg.tool_calls]
    if isinstance(msg, ToolMessage):
        d["name"], d["tool_call_id"] = msg.name, msg.tool_call_id
        try:
            d["content"] = json.loads(msg.content)
        except (TypeError, ValueError):
            pass
    return d


def _plain(value):
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if hasattr(value, "type") and hasattr(value, "content"):
        return message_to_dict(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


class HelpdeskSession:
    """One terminal session: a fixed user, thread and status snapshot, chosen by the application."""

    def __init__(self, model, user_id, thread_id, status_profile="normal",
                 ticket_dir=None, checkpoint_path=None, store_path=None):
        if status_profile not in data.status_profiles():
            raise ValueError(f"status profile must be one of {data.status_profiles()}")
        self.model = model
        self.ctx = HelpdeskContext(user_id=user_id, thread_id=thread_id, status_profile=status_profile,
                                   ticket_dir=ticket_dir or os.getenv("HELPDESK_TICKET_DIR", "outputs/tickets"))
        self.checkpointer = memory.open_checkpointer(checkpoint_path)
        self.store = memory.open_store(store_path)
        self.graph = build_graph(model, self.checkpointer, self.store)
        self.config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 4 * MAX_TOOL_ROUNDS + 10}

    def history(self):
        return self.graph.get_state(self.config).values.get("messages", [])

    def send(self, message: str) -> dict:
        before = self.history()
        steps = []
        for chunk in self.graph.stream({"messages": [HumanMessage(message)]}, self.config,
                                       context=self.ctx, stream_mode="updates"):
            for node, update in chunk.items():
                steps.append({"node": node, "update": _plain(update or {})})
        state = self.graph.get_state(self.config).values
        new = state["messages"][len(before):]
        tool_requests, tool_results, sources = [], [], set()
        for m in new:
            if getattr(m, "tool_calls", None):
                tool_requests += [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
            if isinstance(m, ToolMessage):
                res = message_to_dict(m)["content"]
                tool_results.append({"name": m.name, "result": res})
                if isinstance(res, dict) and res.get("source"):
                    sources.add("reference file: " + res["source"])
        mem = state.get("memory", {})
        if len(before):
            sources.add(f"conversation state: {len(before)} earlier messages in thread '{self.ctx.thread_id}'")
        if mem.get("user_memory_written"):
            sources.add(f"user memory WRITE: saved {mem['user_memory_written']} for '{self.ctx.user_id}'")
        if mem.get("user_memory_read"):
            sources.add(f"user memory: {mem['user_memory_read']} for '{self.ctx.user_id}'")
        else:
            sources.add(f"user memory: nothing recorded for '{self.ctx.user_id}'")
        if not before:
            sources.add(f"conversation state: new thread '{self.ctx.thread_id}' (no earlier messages)")
        final = state["messages"][-1]
        return {
            "user_id": self.ctx.user_id, "thread_id": self.ctx.thread_id,
            "status_profile": self.ctx.status_profile, "model": model_label(self.model),
            "user_message": message,
            "nodes": [s["node"] for s in steps], "steps": steps,
            "tool_requests": tool_requests, "tool_results": tool_results,
            "memory": mem, "sources": sorted(sources),
            "reply": final.content if isinstance(final, AIMessage) else "",
            "tickets": [r["result"]["file"] for r in tool_results
                        if isinstance(r["result"], dict) and "file" in r["result"]],
            "loop_limit_reached": bool(state.get("loop_limit_reached")),
            "tool_rounds": state.get("tool_rounds", 0),
        }

    def close(self):
        memory.close(self.checkpointer, self.store)
