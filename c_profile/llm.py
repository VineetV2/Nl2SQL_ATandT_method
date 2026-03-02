"""
llm.py
------
LLM backend abstraction for OpenAI GPT-5.2.

Factory:
  make_backend("openai", model_id="gpt-5.2")

Backends are cached so repeated calls in the same process do not reload weights.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


class LLMBackend:
    def generate(self, messages: List[Dict[str, str]],
                 max_new_tokens: int = 128, **gen_kwargs) -> str:
        raise NotImplementedError

    def generate_with_meta(self, messages: List[Dict[str, str]],
                           max_new_tokens: int = 128, **gen_kwargs) -> Dict[str, Any]:
        txt = self.generate(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return {"text": txt, "raw": txt, "thoughts": ""}


class OpenAIBackend(LLMBackend):
    def __init__(self, model_id: str = "gpt-5.2", api_key: Optional[str] = None):
        import openai
        self.model_id = model_id
        self.kind = "openai"
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._openai = openai

    def generate(self, messages: List[Dict[str, str]],
                 max_new_tokens: int = 128, **gen_kwargs) -> str:
        token_param = (
            "max_completion_tokens" if self.model_id in ["gpt-5-mini", "gpt-5.2"]
            else "max_tokens"
        )
        call_kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": gen_kwargs.get("temperature", 0),
            token_param: max_new_tokens,
        }
        if "seed" in gen_kwargs and gen_kwargs["seed"] is not None:
            call_kwargs["seed"] = gen_kwargs["seed"]
        response = self._openai.ChatCompletion.create(**call_kwargs)
        return (response["choices"][0]["message"]["content"] or "").strip()

    def generate_with_meta(self, messages: List[Dict[str, str]],
                           max_new_tokens: int = 128, **gen_kwargs) -> Dict[str, Any]:
        text = self.generate(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return {"text": text, "raw": text, "thoughts": ""}


_BACKEND_CACHE: Dict[Tuple, LLMBackend] = {}


def clear_backend_cache() -> None:
    """Clear cached backends (use when switching models)."""
    _BACKEND_CACHE.clear()


def make_backend(kind: str, model_id: Optional[str] = None,
                 cache: bool = True, **kwargs) -> LLMBackend:
    """Create (or retrieve cached) OpenAI LLM backend."""
    k = (kind or "").lower().strip()

    if k in ("openai", "gpt"):
        mid = model_id or "gpt-5.2"
        key = ("openai", mid)
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        backend: LLMBackend = OpenAIBackend(mid)
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    raise ValueError(f"Unknown backend kind: {kind!r}. Use 'openai'.")
