"""Parts 2 and 3: the model chooses tools, my program runs them, and memory is kept.

    START -> load_memory -> agent --(asks for a tool, under the limit)--> tools -> back to agent
                              |----(asks for a tool, limit reached)---> loop_limit -> END
                              '----(writes an answer)-----------------> END

load_memory  my code: if the user states a reply language, save it to user memory;
             read this user's saved language; reset the tool-round counter.
agent        the MODEL: reads the whole conversation and either asks for a tool or answers.
tools        my code: checks the request and runs the plain Python tool function.
loop_limit   my code: stops after MAX_TOOL_ROUNDS tool rounds for one user message.
"""
import json
import operator
from dataclasses import dataclass
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from . import data, memory, tools
from .models import model_label

MAX_TOOL_ROUNDS = 4  # tool rounds allowed per user message (checked in route_after_agent)


@dataclass
class HelpdeskContext:
    """Set by the program when a session starts. The model cannot change these."""
    user_id: str
    thread_id: str
    status_profile: str = "normal"
    ticket_dir: str = "outputs/tickets"


class HelpdeskState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # the conversation (saved by the checkpointer)
    tool_rounds: int                                     # tool rounds used for the current message
    loop_limit_reached: bool
    preferred_language: Optional[str]                    # copied from user memory every turn
    memory: dict                                         # what this turn read/wrote (for the log)
    tickets: Annotated[list[str], operator.add]          # ticket files made in this thread


# What the model is told about the tools (the "menu").
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "find_instructions",
        "description": "Get the local helpdesk instructions for 'wifi' or 'password' problems.",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "enum": ["wifi", "password"]}},
            "required": ["topic"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "check_service_status",
        "description": "Check if a demo service is operational, has an outage, or is unknown. "
                       "Wi-Fi -> 'wifi'. Password recovery -> 'account_recovery'.",
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string", "enum": ["wifi", "account_recovery"]}},
            "required": ["service"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "prepare_ticket",
        "description": "Save a LOCAL ticket draft (nothing is sent). Needs the problem and the steps the "
                       "user said they already tried. The service status is added automatically.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "enum": ["wifi", "password"]},
            "summary": {"type": "string", "description": "The problem, in one sentence."},
            "steps_tried": {"type": "array", "items": {"type": "string"},
                            "description": "Only the steps the user said. Never add steps."}},
            "required": ["category", "summary", "steps_tried"], "additionalProperties": False}}},
]
TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]


def system_prompt(language) -> str:
    if language:
        memory_line = f"User memory: this user asked for replies in {language}. Always reply in {language}."
    else:
        memory_line = ("User memory: nothing is saved for this user (no language, no earlier problem). "
                       "You know nothing about other users or other conversations.")
    return "\n".join([
        "You are a practice university IT helpdesk. All data is fictional demo data.",
        "You can help with: " + "; ".join(data.helpdesk()["scope"]) + ".",
        "Rules:",
        "- Get Wi-Fi or password instructions only with find_instructions.",
        "- Before you say whether a service works, call check_service_status. 'unknown' means the status "
        "could not be checked: it is not 'working' and not an 'outage'. Never invent a repair time.",
        "- Follow the user's conditions. 'If it is operating, prepare a ticket' means: check the status first, "
        "and call prepare_ticket only if the status is 'operational'. If it is 'outage', say you did not make "
        "the ticket. If it is 'unknown', do not make it; ask if they want a draft that says the status "
        "could not be checked.",
        "- A ticket needs the problem and the steps the user already tried. Use only the user's own steps. "
        "If something is missing, ask the user.",
        "- Prepare a ticket only when the user asks for one.",
        "- Never ask for or store passwords, recovery codes or keys.",
        "- A ticket is a local draft only. Say it was not sent and give its file name.",
        "- Use earlier messages of this conversation to understand words like 'that'. Do not claim to "
        "remember anything that is not in this conversation or in user memory.",
        memory_line,
    ])


# --------------------------------------------------------------------- nodes

def load_memory(state: HelpdeskState, runtime: Runtime[HelpdeskContext]) -> dict:
    user_id = runtime.context.user_id
    text = state["messages"][-1].content
    language = memory.parse_language_preference(text)
    written = memory.save_language(runtime.store, user_id, language) if language else None
    prefs = memory.read_preferences(runtime.store, user_id)
    return {"tool_rounds": 0, "loop_limit_reached": False,
            "preferred_language": prefs.get("language"),
            "memory": {"user_id": user_id, "user_memory_written": written,
                       "user_memory_read": prefs or None,
                       "conversation_messages_before_this_turn": len(state["messages"]) - 1}}


def make_agent_node(model):
    # One tool per step, so the model always sees a result before it decides the next step.
    model_with_tools = model.bind_tools(TOOL_SCHEMAS, parallel_tool_calls=False)

    def agent(state: HelpdeskState) -> dict:
        prompt = SystemMessage(system_prompt(state.get("preferred_language")))
        return {"messages": [model_with_tools.invoke([prompt, *state["messages"]])]}

    return agent


def _earlier_results(messages, tool_name):
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == tool_name:
            try:
                yield json.loads(m.content)
            except ValueError:
                pass


def run_one_tool(call, messages, ctx: HelpdeskContext) -> dict:
    """Check one request from the model and run the tool. Never crashes."""
    name, args = call.get("name"), call.get("args") or {}
    if name not in TOOL_NAMES:
        return {"error": f"unknown tool '{name}'"}
    try:
        if name == "find_instructions":
            return tools.find_instructions(args.get("topic", ""))
        if name == "check_service_status":
            return tools.check_service_status(args.get("service", ""), status_profile=ctx.status_profile)
        # prepare_ticket: the status and references come from earlier tool results, not from the model
        service = data.helpdesk()["topic_services"].get(args.get("category"), "wifi")
        statuses = [r["status"] for r in _earlier_results(messages, "check_service_status")
                    if r.get("service") == service and "status" in r]
        refs = sorted({r["instruction_id"] for r in _earlier_results(messages, "find_instructions")
                       if "instruction_id" in r})
        return tools.prepare_ticket(
            category=args.get("category", ""), summary=args.get("summary", ""),
            steps_tried=args.get("steps_tried", []), user_id=ctx.user_id, thread_id=ctx.thread_id,
            status_profile=ctx.status_profile, service_status=statuses[-1] if statuses else "not_checked",
            reference_ids=refs, ticket_dir=ctx.ticket_dir)
    except Exception as exc:
        return {"error": f"{name} failed: {type(exc).__name__}"}


def run_tools(state: HelpdeskState, runtime: Runtime[HelpdeskContext]) -> dict:
    """My program runs every tool the model asked for and adds the results to the conversation."""
    results, tickets = [], []
    for call in state["messages"][-1].tool_calls:
        result = run_one_tool(call, state["messages"], runtime.context)
        if "file" in result:
            tickets.append(result["file"])
        results.append(ToolMessage(content=json.dumps(result, ensure_ascii=False),
                                   tool_call_id=call["id"], name=call["name"]))
    return {"messages": results, "tool_rounds": state.get("tool_rounds", 0) + 1, "tickets": tickets}


def loop_limit(state: HelpdeskState) -> dict:
    """Do not run the last requests; answer them with 'not run' and stop."""
    not_run = [ToolMessage(content='{"error": "not run: loop limit reached"}', tool_call_id=c["id"],
                           name=c["name"]) for c in state["messages"][-1].tool_calls]
    final = AIMessage(content=f"I stopped after {MAX_TOOL_ROUNDS} tool rounds without an answer. "
                              "Please give more detail.")
    return {"messages": [*not_run, final], "loop_limit_reached": True}


def route_after_agent(state: HelpdeskState) -> str:
    """The conditional edge: tool request -> tools (or loop_limit); normal answer -> END."""
    last = state["messages"][-1]
    if last.tool_calls:
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


# ------------------------------------------------------------------ session

def to_dict(msg) -> dict:
    d = {"type": msg.type, "content": msg.content}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in msg.tool_calls]
    if isinstance(msg, ToolMessage):
        d["name"] = msg.name
        try:
            d["content"] = json.loads(msg.content)
        except ValueError:
            pass
    return d


def _plain(value):
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if hasattr(value, "content") and hasattr(value, "type"):
        return to_dict(value)
    return value


class HelpdeskSession:
    """One run of the terminal program: one user, one thread, one status snapshot."""

    def __init__(self, model, user_id, thread_id, status_profile="normal", ticket_dir="outputs/tickets"):
        if status_profile not in data.status_profiles():
            raise ValueError(f"status profile must be one of {data.status_profiles()}")
        self.model = model
        self.ctx = HelpdeskContext(user_id, thread_id, status_profile, str(ticket_dir))
        self.checkpointer = memory.open_checkpointer()
        self.store = memory.open_store()
        self.graph = build_graph(model, self.checkpointer, self.store)
        self.config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}

    def history(self):
        return self.graph.get_state(self.config).values.get("messages", [])

    def send(self, text: str) -> dict:
        """Run one user message through the graph and return a trace of what happened."""
        before = self.history()
        steps = []
        for chunk in self.graph.stream({"messages": [HumanMessage(text)]}, self.config,
                                       context=self.ctx, stream_mode="updates"):
            for node, update in chunk.items():
                steps.append({"node": node, "update": _plain(update or {})})
        state = self.graph.get_state(self.config).values
        new = state["messages"][len(before):]
        requests = [{"name": c["name"], "args": c["args"]} for m in new for c in (getattr(m, "tool_calls", None) or [])]
        results = [{"name": m.name, "result": to_dict(m)["content"]} for m in new if isinstance(m, ToolMessage)]

        mem = state.get("memory", {})
        read_from = []
        if before:
            read_from.append(f"conversation memory: {len(before)} earlier messages of thread '{self.ctx.thread_id}'")
        else:
            read_from.append(f"conversation memory: new thread '{self.ctx.thread_id}', nothing earlier")
        if mem.get("user_memory_written"):
            read_from.append(f"user memory: SAVED {mem['user_memory_written']} for '{self.ctx.user_id}'")
        read_from.append(f"user memory: {mem.get('user_memory_read') or 'nothing saved'} for '{self.ctx.user_id}'")
        for r in results:
            if isinstance(r["result"], dict) and r["result"].get("source"):
                read_from.append(f"reference file: {r['result']['source']} (via {r['name']})")

        final = state["messages"][-1]
        return {
            "user_id": self.ctx.user_id, "thread_id": self.ctx.thread_id,
            "status_profile": self.ctx.status_profile, "model": model_label(self.model),
            "user_message": text, "nodes": [s["node"] for s in steps], "steps": steps,
            "tool_requests": requests, "tool_results": results, "memory": mem, "read_from": read_from,
            "reply": final.content if isinstance(final, AIMessage) else "",
            "tickets": [r["result"]["file"] for r in results if isinstance(r["result"], dict) and "file" in r["result"]],
            "tool_rounds": state.get("tool_rounds", 0),
            "loop_limit_reached": bool(state.get("loop_limit_reached")),
        }

    def close(self):
        memory.close(self.checkpointer, self.store)
