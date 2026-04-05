"""
llm.py
------
LLM backend abstraction for HuggingFace local and OpenRouter API inference.

Factory:
  make_backend("huggingface", model_id="openai/gpt-oss-120b")
  make_backend("openrouter", model_id="openai/gpt-oss-120b")

Backends are cached so repeated calls in the same process do not reload weights.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple


class LLMBackend:
    def generate(self, messages: List[Dict[str, str]],
                 max_new_tokens: int = 128, **gen_kwargs) -> str:
        raise NotImplementedError

    def generate_with_meta(self, messages: List[Dict[str, str]],
                           max_new_tokens: int = 128, **gen_kwargs) -> Dict[str, Any]:
        txt = self.generate(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return {"text": txt, "raw": txt, "thoughts": ""}


# ─────────────────────────────────────────────────────────────
# Marker-based extraction for reasoning models
# ─────────────────────────────────────────────────────────────
FINAL_BEGIN = "###FINAL_BEGIN###"
FINAL_END = "###FINAL_END###"

_MARKER_INSTRUCTION = (
    "\n\n"
    "IMPORTANT OUTPUT FORMAT (MANDATORY):\n"
    "1) Put your FINAL answer between these exact markers:\n"
    f"{FINAL_BEGIN}\n"
    "<final answer>\n"
    f"{FINAL_END}\n"
    "2) Do NOT write anything inside the markers except the final answer.\n"
    "3) You may write reasoning OUTSIDE the markers.\n"
)


def _inject_marker_instruction(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Inject marker instructions into system message and last user message."""
    out: List[Dict[str, str]] = []
    sys_injected = False

    for msg in messages:
        if msg.get("role") == "system" and not sys_injected:
            out.append({"role": "system", "content": (msg.get("content") or "") + _MARKER_INSTRUCTION})
            sys_injected = True
        else:
            out.append(dict(msg))

    if not sys_injected:
        out.insert(0, {"role": "system", "content": _MARKER_INSTRUCTION.strip()})

    # Also inject into last user message
    last_user_idx = None
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is not None:
        out[last_user_idx] = {"role": "user", "content": (out[last_user_idx].get("content") or "") + _MARKER_INSTRUCTION}
    else:
        out.append({"role": "user", "content": _MARKER_INSTRUCTION.strip()})

    return out


def _extract_final_block(raw: str) -> str:
    """Extract content between ###FINAL_BEGIN### and ###FINAL_END### markers."""
    raw = raw or ""
    i = raw.rfind(FINAL_BEGIN)
    if i == -1:
        return ""
    j = raw.rfind(FINAL_END)
    if j == -1 or j < i:
        return ""
    return (raw[i + len(FINAL_BEGIN): j] or "").strip()


def _extract_final_and_thoughts(raw: str) -> Tuple[str, str]:
    """Extract final answer and reasoning from raw model output.
    Returns (final_text, thoughts_text)."""
    raw = (raw or "").strip()
    if not raw:
        return "", ""

    final = _extract_final_block(raw)
    if final:
        i = raw.rfind(FINAL_BEGIN)
        j = raw.rfind(FINAL_END)
        before = raw[:i].strip()
        after = raw[j + len(FINAL_END):].strip()
        thoughts = (before + ("\n" if before and after else "") + after).strip()
        return final, thoughts

    # Fallback: return raw as-is (caller will use regex extraction)
    return raw, ""


# ─────────────────────────────────────────────────────────────
# OpenRouter Backend
# ─────────────────────────────────────────────────────────────

class OpenRouterBackend(LLMBackend):
    """OpenRouter API backend. Works with openai>=1.0 (new client API)."""

    def __init__(self, model_id: str = "openai/gpt-oss-120b",
                 inject_final_markers: bool = True):
        from openai import OpenAI

        self.model_id = model_id
        self.kind = "openrouter"
        self.inject_final_markers = inject_final_markers

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get your key at https://openrouter.ai/keys"
            )
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        print(f"OpenRouter backend ready: {model_id}")

    def generate(self, messages: List[Dict[str, str]],
                 max_new_tokens: int = 1900, **gen_kwargs) -> str:
        meta = self.generate_with_meta(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return (meta.get("text") or "").strip()

    def generate_with_meta(self, messages: List[Dict[str, str]],
                           max_new_tokens: int = 1900, **gen_kwargs) -> Dict[str, Any]:
        temperature = gen_kwargs.get("temperature", 0)

        use_messages = (_inject_marker_instruction(messages)
                        if self.inject_final_markers else messages)

        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=use_messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        raw = (response.choices[0].message.content or "").strip()

        final, thoughts = _extract_final_and_thoughts(raw)
        return {"text": final, "raw": raw, "thoughts": thoughts}


# ─────────────────────────────────────────────────────────────
# HuggingFace Backend (local inference)
# ─────────────────────────────────────────────────────────────

class HuggingFaceBackend(LLMBackend):
    def __init__(self, model_id: str = "openai/gpt-oss-120b"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.model_id = model_id
        self.kind = "huggingface"
        print(f"Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print("Model ready for inference.")

    def generate(self, messages: List[Dict[str, str]],
                 max_new_tokens: int = 128, **gen_kwargs) -> str:
        import torch
        temperature = gen_kwargs.get("temperature", 0)

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"<|{role}|>\n{content}")
            parts.append("<|assistant|>")
            prompt = "\n".join(parts)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        gen_config: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if self.tokenizer.eos_token_id is not None:
            gen_config["eos_token_id"] = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is not None:
            gen_config["pad_token_id"] = self.tokenizer.pad_token_id
        if temperature == 0:
            gen_config["do_sample"] = False
        else:
            gen_config["do_sample"] = True
            gen_config["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_config)

        new_tokens = output_ids[0][input_len:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Free GPU memory after each call to avoid OOM on long prompts
        del inputs, output_ids, new_tokens
        torch.cuda.empty_cache()

        return result

    def generate_with_meta(self, messages: List[Dict[str, str]],
                           max_new_tokens: int = 128, **gen_kwargs) -> Dict[str, Any]:
        text = self.generate(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return {"text": text, "raw": text, "thoughts": ""}


# ─────────────────────────────────────────────────────────────
# Backend Cache + Factory
# ─────────────────────────────────────────────────────────────

_BACKEND_CACHE: Dict[Tuple, LLMBackend] = {}


def clear_backend_cache() -> None:
    """Clear cached backends (use when switching models)."""
    _BACKEND_CACHE.clear()


def make_backend(kind: str, model_id: Optional[str] = None,
                 cache: bool = True, **kwargs) -> LLMBackend:
    """Create (or retrieve cached) LLM backend.

    Supported kinds:
      - 'huggingface' / 'hf': Local HuggingFace inference
      - 'openrouter': OpenRouter API (requires OPENROUTER_API_KEY env var)
    """
    k = (kind or "").lower().strip()

    if k in ("huggingface", "hf"):
        mid = model_id or "openai/gpt-oss-120b"
        key = ("huggingface", mid)
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        backend: LLMBackend = HuggingFaceBackend(mid)
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    if k in ("openrouter",):
        mid = model_id or "openai/gpt-oss-120b"
        inject = kwargs.get("inject_final_markers", True)
        key = ("openrouter", mid, inject)
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        backend = OpenRouterBackend(mid, inject_final_markers=inject)
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    raise ValueError(f"Unknown backend kind: {kind!r}. Use 'huggingface' or 'openrouter'.")
