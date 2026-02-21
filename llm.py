"""
llm.py
------
Reusable LLM backend abstraction.

Consolidates: llm_backends_local.py

Backends:
  - OpenAIBackend          GPT-5.2 via OpenAI API (primary)
  - HFTransformersBackend  Qwen/GPT-OSS local models via HuggingFace
  - OSSHFPBackend          GPT-OSS via transformers.pipeline

Factory:
  make_backend("openai", model_id="gpt-5.2")
  make_backend("qwen",   model_id="Qwen/...")
  make_backend("gptoss", model_id="openai/...")

Backends are cached so repeated calls in the same process do not reload weights.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

FINAL_BEGIN = "###FINAL_BEGIN###"
FINAL_END   = "###FINAL_END###"


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
        response = self._openai.ChatCompletion.create(
            model=self.model_id, messages=messages,
            temperature=gen_kwargs.get("temperature", 0),
            **{token_param: max_new_tokens}
        )
        return (response["choices"][0]["message"]["content"] or "").strip()

    def generate_with_meta(self, messages: List[Dict[str, str]],
                           max_new_tokens: int = 128, **gen_kwargs) -> Dict[str, Any]:
        text = self.generate(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return {"text": text, "raw": text, "thoughts": ""}


class HFTransformersBackend(LLMBackend):
    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct",
                 device_map: str = "auto", dtype: Optional[str] = None,
                 trust_remote_code: bool = True):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        torch_dtype = None
        if dtype:
            dmap = {"float16": torch.float16, "fp16": torch.float16,
                    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                    "float32": torch.float32, "fp32": torch.float32}
            torch_dtype = dmap.get(dtype.lower())
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device_map, torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code)
        self.model.eval()

    def generate(self, messages: List[Dict[str, str]],
                 max_new_tokens: int = 128, **gen_kwargs) -> str:
        import torch
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
        use_args = {k: gen_kwargs.pop(k, None)
                    for k in ("do_sample","temperature","top_p","num_beams")}
        use_args.update(gen_kwargs)
        use_args = {k: v for k, v in use_args.items() if v is not None}
        with torch.inference_mode():
            out_ids = self.model.generate(**enc, max_new_tokens=max_new_tokens, **use_args)
        gen_ids = out_ids[0][enc["input_ids"].shape[1]:]
        return (self.tok.decode(gen_ids, skip_special_tokens=True) or "").strip()


def _pull_raw_from_hf(out_obj) -> str:
    if not out_obj:
        return ""
    if isinstance(out_obj, list) and out_obj:
        out_obj = out_obj[0]
    if isinstance(out_obj, dict):
        for k in ("generated_text", "text"):
            if k in out_obj and isinstance(out_obj[k], str):
                return out_obj[k]
        if isinstance(out_obj.get("generated_text"), list):
            msgs = out_obj["generated_text"]
            if msgs and isinstance(msgs[-1], dict) and "content" in msgs[-1]:
                return str(msgs[-1]["content"])
    return str(out_obj)


def _extract_final_block(raw: str) -> str:
    i = raw.rfind(FINAL_BEGIN)
    if i == -1:
        return ""
    j = raw.rfind(FINAL_END)
    if j == -1 or j < i:
        return ""
    return (raw[i + len(FINAL_BEGIN): j] or "").strip()


def _inject_marker_instruction(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    extra = (f"\n\nIMPORTANT: Put your FINAL answer between these exact markers:\n"
             f"{FINAL_BEGIN}\n<final answer>\n{FINAL_END}\n"
             "Write reasoning OUTSIDE the markers.")
    out: List[Dict[str, str]] = []
    sys_injected = False
    for msg in messages:
        if msg.get("role") == "system" and not sys_injected:
            out.append({"role": "system", "content": (msg.get("content") or "") + extra})
            sys_injected = True
        else:
            out.append(msg)
    if not sys_injected:
        out.insert(0, {"role": "system", "content": extra.strip()})
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i] = {"role": "user", "content": (out[i].get("content") or "") + extra}
            break
    return out


class OSSHFPBackend(LLMBackend):
    def __init__(self, model_id: str = "openai/gpt-oss-20b",
                 device_map: str = "auto", dtype: Optional[str] = None,
                 trust_remote_code: bool = True, inject_final_markers: bool = True):
        import torch
        from transformers import pipeline
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        self.model_id = model_id
        self.inject_final_markers = inject_final_markers
        torch_dtype = None
        if dtype:
            dmap = {"float16": torch.float16, "fp16": torch.float16,
                    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
            torch_dtype = dmap.get(dtype.lower())
        self.pipe = pipeline("text-generation", model=model_id, device_map=device_map,
                             torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
        try:
            self._pad_id = getattr(self.pipe, "tokenizer", None).eos_token_id
        except Exception:
            self._pad_id = None

    def generate(self, messages: List[Dict[str, str]],
                 max_new_tokens: int = 128, **gen_kwargs) -> str:
        return (self.generate_with_meta(messages, max_new_tokens=max_new_tokens,
                                        **gen_kwargs).get("text") or "").strip()

    def generate_with_meta(self, messages: List[Dict[str, str]],
                           max_new_tokens: int = 128, **gen_kwargs) -> Dict[str, Any]:
        use_messages = _inject_marker_instruction(messages) if self.inject_final_markers else messages
        out = self.pipe(use_messages, max_new_tokens=max_new_tokens,
                        do_sample=gen_kwargs.pop("do_sample", False),
                        temperature=gen_kwargs.pop("temperature", None),
                        pad_token_id=gen_kwargs.pop("pad_token_id", self._pad_id),
                        **gen_kwargs)
        raw = _pull_raw_from_hf(out).strip()
        final = _extract_final_block(raw)
        if not final:
            paras = [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]
            final = paras[-1].strip() if paras else raw
        thoughts = raw[:raw.rfind(FINAL_BEGIN)].strip() if FINAL_BEGIN in raw else ""
        return {"text": final, "raw": raw, "thoughts": thoughts}


_BACKEND_CACHE: Dict[Tuple, LLMBackend] = {}


def clear_backend_cache() -> None:
    """Clear cached backends (use when switching models)."""
    _BACKEND_CACHE.clear()


def make_backend(kind: str, model_id: Optional[str] = None,
                 device_map: str = "auto", dtype: Optional[str] = None,
                 trust_remote_code: bool = True, cache: bool = True) -> LLMBackend:
    """
    Create (or retrieve cached) LLM backend.

    kind:
      "openai" / "gpt"   OpenAIBackend (GPT-5.2)
      "qwen"  / "hf"     HFTransformersBackend (local Qwen)
      "gptoss"/ "oss"    OSSHFPBackend (HF pipeline)
    """
    k = (kind or "").lower().strip()

    if k in ("openai", "gpt"):
        mid = model_id or "gpt-5.2"
        key = (k, mid, "", "", False)
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        backend: LLMBackend = OpenAIBackend(mid)
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    if k in ("qwen", "hf"):
        mid = model_id or "Qwen/Qwen2.5-7B-Instruct"
        if dtype is None and device_map != "cpu":
            dtype = "bfloat16"
        key = (k, mid, str(device_map), str(dtype), bool(trust_remote_code))
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        backend = HFTransformersBackend(mid, device_map=device_map, dtype=dtype,
                                        trust_remote_code=trust_remote_code)
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    if k in ("gptoss", "oss"):
        mid = model_id or "openai/gpt-oss-20b"
        if dtype is None and device_map != "cpu":
            dtype = "bfloat16"
        key = (k, mid, str(device_map), str(dtype), bool(trust_remote_code))
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        backend = OSSHFPBackend(mid, device_map=device_map, dtype=dtype,
                                trust_remote_code=trust_remote_code)
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    raise ValueError(f"Unknown backend kind: {kind!r}. Use 'openai', 'qwen', or 'gptoss'.")
