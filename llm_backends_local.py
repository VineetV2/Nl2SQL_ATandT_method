"""
llm_backends_local.py
---------------------
Reusable local LLM backends (Qwen via HF generate; GPT-OSS via HF pipeline).

Update (Jan 2026):
- GPT-OSS may emit "analysis"/reasoning in the same stream.
- We extract and return ONLY the final answer by default using explicit
  begin/end markers (multi-line safe).
- Marker instructions are injected into BOTH the system message and the
  LAST user message (more reliable).
- IMPORTANT: backends are now CACHED so a notebook rerun will NOT reload
  model weights and kill your kernel.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any

import os
import re

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# -----------------------------
# Marker-based final extraction
# -----------------------------
FINAL_BEGIN = "###FINAL_BEGIN###"
FINAL_END = "###FINAL_END###"


class LLMBackend:
    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs) -> str:
        raise NotImplementedError

    def generate_with_meta(
        self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs
    ) -> Dict[str, Any]:
        txt = self.generate(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return {"text": txt, "raw": txt, "thoughts": ""}


class HFTransformersBackend(LLMBackend):
    """
    Chat-style HF backend using AutoTokenizer/AutoModelForCausalLM and the model's chat_template.
    Good for Qwen2.5-* Instruct.
    """
    def __init__(self,
                 model_id: str = "Qwen/Qwen2.5-7B-Instruct",
                 device_map: str = "auto",
                 dtype: Optional[str] = None,
                 trust_remote_code: bool = True):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype

        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

        torch_dtype = None
        if dtype:
            dmap = {
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float32": torch.float32, "fp32": torch.float32,
            }
            torch_dtype = dmap.get(dtype.lower())

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs) -> str:
        import torch

        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt")

        # device placement
        if hasattr(self.model, "device"):
            enc = {k: v.to(self.model.device) for k, v in enc.items()}

        # Safe defaults
        use_args = {
            "do_sample": gen_kwargs.pop("do_sample", False),
            "temperature": gen_kwargs.pop("temperature", None),
            "top_p": gen_kwargs.pop("top_p", None),
            "num_beams": gen_kwargs.pop("num_beams", None),
        }
        use_args.update(gen_kwargs)
        use_args = {k: v for k, v in use_args.items() if v is not None}

        with torch.inference_mode():
            out_ids = self.model.generate(**enc, max_new_tokens=max_new_tokens, **use_args)

        gen_ids = out_ids[0][enc["input_ids"].shape[1]:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        return (text or "").strip()


def _pull_raw_from_hf_chat_output(out_obj) -> str:
    if not out_obj:
        return ""
    if isinstance(out_obj, list) and out_obj:
        out_obj = out_obj[0]
    if isinstance(out_obj, dict):
        for k in ("generated_text", "text"):
            if k in out_obj and isinstance(out_obj[k], str):
                return out_obj[k]
        if "generated_text" in out_obj and isinstance(out_obj["generated_text"], list):
            msgs = out_obj["generated_text"]
            if msgs and isinstance(msgs[-1], dict) and "content" in msgs[-1]:
                return str(msgs[-1]["content"])
    return str(out_obj)


def _extract_final_block(raw: str) -> str:
    raw = raw or ""
    i = raw.rfind(FINAL_BEGIN)
    if i == -1:
        return ""
    j = raw.rfind(FINAL_END)
    if j == -1 or j < i:
        return ""
    return (raw[i + len(FINAL_BEGIN): j] or "").strip()


def _fallback_extract_answer(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    quotes = re.findall(r'"([^"]{1,2000})"', raw)
    if quotes:
        return quotes[-1].strip()
    paras = [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]
    if paras:
        tail = re.sub(r"^\s*analysis\s*", "", paras[-1], flags=re.IGNORECASE)
        return tail.strip()
    return raw


def _extract_final_and_thoughts(raw: str) -> Tuple[str, str]:
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

    # fallback
    final_guess = _fallback_extract_answer(raw)
    return final_guess, raw


def _inject_marker_instruction(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    extra = (
        "\n\n"
        "IMPORTANT OUTPUT FORMAT (MANDATORY):\n"
        f"1) Put your FINAL answer between these exact markers:\n"
        f"{FINAL_BEGIN}\n"
        f"<final answer>\n"
        f"{FINAL_END}\n"
        "2) Do NOT write anything inside the markers except the final answer.\n"
        "3) You may write reasoning OUTSIDE the markers.\n"
    )

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

    last_user_idx = None
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is not None:
        out[last_user_idx] = {"role": "user", "content": (out[last_user_idx].get("content") or "") + extra}
    else:
        out.append({"role": "user", "content": extra.strip()})

    return out


class OSSHFPBackend(LLMBackend):
    """
    OSS backend using transformers.pipeline text-generation. Works for openai/gpt-oss-20b.
    Returns final answer by default (marker extraction).
    """
    def __init__(self,
                 model_id: str = "openai/gpt-oss-20b",
                 device_map: str = "auto",
                 dtype: Optional[str] = None,
                 trust_remote_code: bool = True,
                 inject_final_markers: bool = True):
        import torch
        from transformers import pipeline
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        self.inject_final_markers = inject_final_markers

        torch_dtype = None
        if dtype:
            dmap = {
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float32": torch.float32, "fp32": torch.float32,
            }
            torch_dtype = dmap.get(dtype.lower())

        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

        self._default_pad_token_id = None
        try:
            self._default_pad_token_id = getattr(self.pipe, "tokenizer", None).eos_token_id
        except Exception:
            self._default_pad_token_id = None

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs) -> str:
        meta = self.generate_with_meta(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return (meta.get("text") or "").strip()

    def generate_with_meta(
        self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs
    ) -> Dict[str, Any]:
        do_sample = gen_kwargs.pop("do_sample", False)
        temperature = gen_kwargs.pop("temperature", None)

        pad_token_id = gen_kwargs.pop("pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self._default_pad_token_id

        use_messages = _inject_marker_instruction(messages) if self.inject_final_markers else messages

        pipe_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=pad_token_id,
        )
        pipe_kwargs.update(gen_kwargs)

        out = self.pipe(use_messages, **pipe_kwargs)
        raw = _pull_raw_from_hf_chat_output(out).strip()

        final, thoughts = _extract_final_and_thoughts(raw)
        return {"text": final, "raw": raw, "thoughts": thoughts}


# -----------------------------
# OpenAI API Backend
# -----------------------------

class OpenAIBackend(LLMBackend):
    """
    OpenAI API backend using GPT-5.2 (or any OpenAI model).
    Drop-in replacement for Qwen/GPT-OSS backends.
    """
    def __init__(self, model_id: str = "gpt-5.2", api_key: Optional[str] = None):
        import openai, os
        self.model_id = model_id
        self.kind = "openai"
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._openai = openai

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs) -> str:
        token_param = "max_completion_tokens" if self.model_id in ["gpt-5-mini", "gpt-5.2"] else "max_tokens"
        response = self._openai.ChatCompletion.create(
            model=self.model_id,
            messages=messages,
            temperature=gen_kwargs.get("temperature", 0),
            **{token_param: max_new_tokens}
        )
        return (response["choices"][0]["message"]["content"] or "").strip()

    def generate_with_meta(self, messages: List[Dict[str, str]], max_new_tokens: int = 128, **gen_kwargs) -> Dict[str, Any]:
        text = self.generate(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return {"text": text, "raw": text, "thoughts": ""}


# -----------------------------
# Backend cache (CRITICAL)
# -----------------------------
_BACKEND_CACHE: Dict[Tuple[str, str, str, str, bool], LLMBackend] = {}


def clear_backend_cache() -> None:
    """
    Clear cached backends. Use when switching models or if GPU memory is stuck.
    After calling, restart kernel is still the cleanest option.
    """
    _BACKEND_CACHE.clear()


def make_backend(kind: str,
                 model_id: Optional[str] = None,
                 device_map: str = "auto",
                 dtype: Optional[str] = None,
                 trust_remote_code: bool = True,
                 cache: bool = True) -> LLMBackend:
    """
    kind:
      - "qwen"  -> HFTransformersBackend
      - "gptoss"-> OSSHFPBackend

    NOTE: cache=True means: repeated calls in the same kernel reuse the model instance.
    """
    k = (kind or "").lower().strip()

    if k in ("qwen", "hf"):
        mid = model_id or "Qwen/Qwen2.5-7B-Instruct"
        # strongly recommended default on A100 if user didn't set dtype
        if dtype is None and device_map != "cpu":
            dtype = "bfloat16"
        key = (k, mid, str(device_map), str(dtype), bool(trust_remote_code))
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        backend = HFTransformersBackend(mid, device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code)
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
        backend = OSSHFPBackend(mid, device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code)
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    if k in ("openai", "gpt"):
        mid = model_id or "gpt-5.2"
        key = (k, mid, "", "", False)
        if cache and key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        backend = OpenAIBackend(mid)
        if cache:
            _BACKEND_CACHE[key] = backend
        return backend

    raise ValueError(f"Unknown backend kind: {kind!r}. Use 'qwen', 'gptoss', or 'openai'.")
