"""The chat models.

real_model()        The real model from .env (OpenAI, used for the Part 2-3 demos).
ScriptedFakeModel   FAKE model for tests: it returns replies we wrote in advance.
EchoFakeModel       FAKE model for tests: it never calls tools; it just repeats what it can see.
                    That is enough to prove memory works after a restart (no API key needed).
"""
import os
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def real_model():
    from langchain_openai import ChatOpenAI

    name = os.getenv("MODEL_NAME", "").strip()
    if not os.getenv("OPENAI_API_KEY") or not name:
        raise SystemExit("Set OPENAI_API_KEY and MODEL_NAME in .env (see README).")
    extra = {}
    effort = os.getenv("MODEL_REASONING_EFFORT", "none").strip()
    if effort:  # gpt-5.6-luna allows tools in Chat Completions only with reasoning effort "none"
        extra["reasoning_effort"] = effort
    return ChatOpenAI(model=name, use_responses_api=False, timeout=90, max_retries=2,
                      default_headers={"Accept-Encoding": "gzip, deflate"},  # avoids a brotli error we saw in HW3
                      **extra)


def model_label(model) -> str:
    if isinstance(model, (ScriptedFakeModel, EchoFakeModel)):
        return f"FAKE:{type(model).__name__}"
    return f"openai:{getattr(model, 'model_name', '?')}"


def tool_call(name, args):
    """Build a tool request the way a real model sends it."""
    return {"name": name, "args": args, "id": "call_" + uuid.uuid4().hex[:8], "type": "tool_call"}


class _Fake(BaseChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "fake"


class ScriptedFakeModel(_Fake):
    """Returns `responses` in order. A response can also be a function that looks at the messages."""
    responses: list = []
    seen: list = []   # every list of messages the model was given

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        reply = self.responses.pop(0) if self.responses else AIMessage(content="(no more replies)")
        if callable(reply):
            reply = reply(messages)
        return ChatResult(generations=[ChatGeneration(message=reply)])


class EchoFakeModel(_Fake):
    """Answers with the user messages it was given, so a test can see what memory provided."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        said = [m.content for m in messages if isinstance(m, HumanMessage)]
        text = "FAKE MODEL. I can see these user messages: " + " | ".join(said)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
