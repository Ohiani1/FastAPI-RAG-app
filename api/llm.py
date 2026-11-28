"""
Thin, swappable LLM wrapper. Everything else in the app calls `get_llm().complete(...)`
and never imports a provider SDK directly, so switching providers is one env var:

    LLM_PROVIDER=anthropic   (default)   or   LLM_PROVIDER=openai
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from common.config import LLM_MODEL, LLM_PROVIDER

# Model defaults per provider; override with LLM_MODEL in .env.
DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-5-mini"}


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict], max_tokens: int = 1024) -> LLMResponse: ...


class AnthropicLLM:
    def __init__(self, model: str):
        from anthropic import Anthropic
        self.client, self.model = Anthropic(), model   # reads ANTHROPIC_API_KEY

    def complete(self, system, messages, max_tokens=1024):
        extra = {"system": system} if system else {}   # empty system prompts are rejected
        resp = self.client.messages.create(
            model=self.model, messages=messages, max_tokens=max_tokens, **extra,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResponse(text, self.model, resp.usage.input_tokens, resp.usage.output_tokens)


class OpenAILLM:
    def __init__(self, model: str):
        from openai import OpenAI
        self.client, self.model = OpenAI(), model      # reads OPENAI_API_KEY

    def complete(self, system, messages, max_tokens=1024):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=([{"role": "system", "content": system}] if system else []) + messages,
            max_completion_tokens=max_tokens,
        )
        return LLMResponse(resp.choices[0].message.content or "", self.model,
                           resp.usage.prompt_tokens, resp.usage.completion_tokens)


def make_llm(provider: str, model: str = "") -> LLM:
    """Build a client for any provider/model (used for the answering LLM and the eval judge)."""
    cls = {"anthropic": AnthropicLLM, "openai": OpenAILLM}[provider]
    return cls(model or DEFAULT_MODELS[provider])


_llm: LLM | None = None


def get_llm() -> LLM:
    """The LLM that answers user questions, configured by LLM_PROVIDER / LLM_MODEL."""
    global _llm
    if _llm is None:
        _llm = make_llm(LLM_PROVIDER, LLM_MODEL)
    return _llm
