"""Chat models used by the agent.

real_model()          the real tool-capable model from .env (OpenAI adapter). Used for the Part 2-3 demos.
ScriptedFakeModel     FAKE. Returns a fixed list of replies. Used by the pytest tests and CI.
RuleBasedFakeModel    FAKE. Picks tools with simple keyword rules. Only for an offline dry run
                      (python run_scenarios.py --fake-model); its logs go to outputs/fake/ and are not results.
"""
import json
import os
import re
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def real_model():
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()
    name = os.getenv("MODEL_NAME", "").strip()
    if not os.getenv("OPENAI_API_KEY") or not name:
        raise SystemExit("Set OPENAI_API_KEY and MODEL_NAME in .env (see README), or use --fake-model.")
    kwargs = {}
    effort = os.getenv("MODEL_REASONING_EFFORT", "none").strip()
    if effort:  # gpt-5.6-luna only allows tools in Chat Completions with reasoning_effort="none"
        kwargs["reasoning_effort"] = effort
    return ChatOpenAI(
        model=name, use_responses_api=False, timeout=90, max_retries=2,
        # Avoids a brotli decoding error seen with this proxy/client in earlier homeworks.
        default_headers={"Accept-Encoding": "gzip, deflate"}, **kwargs)


def model_label(model) -> str:
    if isinstance(model, (ScriptedFakeModel, RuleBasedFakeModel)):
        return f"FAKE:{type(model).__name__}"
    name = getattr(model, "model_name", None) or getattr(model, "model", None)
    return f"openai:{name}"


def tool_call(name, args):
    return {"name": name, "args": args, "id": "call_" + uuid.uuid4().hex[:8], "type": "tool_call"}


class _FakeBase(BaseChatModel):
    bound_tools: list = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    @property
    def _llm_type(self) -> str:
        return "fake"


class ScriptedFakeModel(_FakeBase):
    """FAKE model for tests: returns `responses` in order and records what it was shown."""
    responses: list = []
    seen: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        if not self.responses:
            msg = AIMessage(content="(scripted fake model has no more replies)")
        else:
            nxt = self.responses.pop(0)
            msg = nxt(messages) if callable(nxt) else nxt
        return ChatResult(generations=[ChatGeneration(message=msg)])


class RuleBasedFakeModel(_FakeBase):
    """FAKE model with keyword rules, so the whole program can be dry-run without an API key."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._decide(messages))])

    @staticmethod
    def _decide(messages):
        system = messages[0].content if messages else ""
        lang = re.search(r"reply in (\w+)\.", system)
        prefix = f"[{lang.group(1)}] " if lang else ""
        idx = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        human = messages[idx].content.lower()
        results = [m for m in messages[idx + 1:] if isinstance(m, ToolMessage)]
        history = [m.content for m in messages[1:idx] if isinstance(m, HumanMessage)]
        wants_ticket = "ticket" in human or any("ticket" in h.lower() for h in history[-1:])
        topic = "wifi" if re.search(r"wi-?fi", human) else ("password" if "password" in human else None)
        if topic is None:
            for h in reversed(history):
                if re.search(r"wi-?fi", h.lower()):
                    topic = "wifi"
                    break
        steps = [s.strip() for s in re.split(r"[.,]| but ", messages[idx].content)
                 if re.search(r"\b(enabled|tried|forgot|reconnect)", s, re.I)]

        def say(text):
            return AIMessage(content=prefix + text)

        if results:
            last = json.loads(results[-1].content)
            if results[-1].name == "find_instructions" and "steps" in last:
                return say("Steps from " + last["instruction_id"] + ": " + " ".join(last["steps"]))
            if results[-1].name == "check_service_status" and "status" in last:
                if last["status"] == "operational" and wants_ticket and steps:
                    return AIMessage(content="", tool_calls=[tool_call("prepare_ticket", {
                        "category": topic or "wifi", "summary": messages[idx].content[:160],
                        "steps_tried": steps})])
                if last["status"] == "outage":
                    return say(f"The demo status source reports an outage: {last['message']} "
                               "I did not prepare a ticket. No repair time is given.")
                if last["status"] == "unknown":
                    return say("I could not determine the status, so I cannot confirm it is operating. "
                               "Would you like a ticket draft saying the status could not be checked?")
                return say(last["message"])
            if results[-1].name == "prepare_ticket":
                if "file" in last:
                    return say(f"I prepared a local draft {last['file']}. Nothing has been sent.")
                return say("What problem are you having, and what have you already tried?")
            return say("Done.")

        if "language" in human or "preference" in human or "previous problem" in human:
            pref = f"You asked me to reply in {lang.group(1)}." if lang else "No preference or previous problem is recorded for you."
            return say(pref + " I can help with Wi-Fi, passwords, service status and ticket drafts.")
        if "which" in human and ("steps" in human or "problem" in human):
            return say("Earlier you said: " + " | ".join(history) if history else "We have not discussed a problem yet.")
        if any(w in human for w in ("status", "down", "known problem")):
            service = "account_recovery" if topic == "password" else "wifi"
            return AIMessage(content="", tool_calls=[tool_call("check_service_status", {"service": service})])
        if wants_ticket and topic and steps:
            return AIMessage(content="", tool_calls=[tool_call("prepare_ticket", {
                "category": topic, "summary": messages[idx].content[:160], "steps_tried": steps})])
        if "ticket" in human:
            return say("What problem are you having, and what have you already tried?")
        if topic:
            return AIMessage(content="", tool_calls=[tool_call("find_instructions", {"topic": topic})])
        return say("Is your problem about Wi-Fi or a password?")
