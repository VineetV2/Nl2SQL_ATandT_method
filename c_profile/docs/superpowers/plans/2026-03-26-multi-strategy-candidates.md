# Multi-Strategy Candidate Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Beat Anik's 61% BIRD mini-dev accuracy by adding OpenRouter backend, multi-strategy prompting, enhanced feedback loop, and LLM tiebreaker voting.

**Architecture:** Add OpenRouterBackend to llm.py with marker-based extraction. Refactor candidates.py to generate 3 structurally different candidates (full-schema, pruned+full-profiles, pruned+long-profiles), enhance feedback loop to retry on empty generation and use cross-candidate hints, and replace random fallback with LLM tiebreaker.

**Tech Stack:** Python, openai==0.28.1 (OpenRouter-compatible), SQLite, FAISS, sqlglot

---

### Task 1: Add OpenRouter Backend to llm.py

**Files:**
- Modify: `/Users/vora/Documents/PyCharm/DAIL-SQL/c_profile/llm.py`

This task adds the OpenRouterBackend class with marker injection/extraction and updates make_backend().

- [ ] **Step 1: Add marker constants and injection helper after LLMBackend class (line 26)**

Add after line 26 (after the `LLMBackend` class):

```python
# ─────────────────────────────────────────────────────────────
# Marker-based extraction for reasoning models
# ─────────────────────────────────────────────────────────────
FINAL_BEGIN = "###FINAL_BEGIN###"
FINAL_END = "###FINAL_END###"

_MARKER_INSTRUCTION = (
    "\n\n"
    "IMPORTANT OUTPUT FORMAT (MANDATORY):\n"
    f"1) Put your FINAL answer between these exact markers:\n"
    f"{FINAL_BEGIN}\n"
    f"<final answer>\n"
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
```

- [ ] **Step 2: Add OpenRouterBackend class after the marker helpers**

```python
class OpenRouterBackend(LLMBackend):
    """OpenRouter API backend. Uses openai==0.28.x with custom api_base."""

    def __init__(self, model_id: str = "openai/gpt-oss-120b",
                 inject_final_markers: bool = True):
        import os
        self.model_id = model_id
        self.kind = "openrouter"
        self.inject_final_markers = inject_final_markers

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get your key at https://openrouter.ai/keys"
            )
        self._api_key = api_key
        print(f"OpenRouter backend ready: {model_id}")

    def generate(self, messages: List[Dict[str, str]],
                 max_new_tokens: int = 1900, **gen_kwargs) -> str:
        meta = self.generate_with_meta(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        return (meta.get("text") or "").strip()

    def generate_with_meta(self, messages: List[Dict[str, str]],
                           max_new_tokens: int = 1900, **gen_kwargs) -> Dict[str, Any]:
        import openai

        temperature = gen_kwargs.get("temperature", 0)

        use_messages = (_inject_marker_instruction(messages)
                        if self.inject_final_markers else messages)

        # OpenRouter uses OpenAI-compatible API
        prev_key = openai.api_key
        prev_base = openai.api_base
        try:
            openai.api_key = self._api_key
            openai.api_base = "https://openrouter.ai/api/v1"

            response = openai.ChatCompletion.create(
                model=self.model_id,
                messages=use_messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            raw = (response["choices"][0]["message"]["content"] or "").strip()
        finally:
            openai.api_key = prev_key
            openai.api_base = prev_base

        final, thoughts = _extract_final_and_thoughts(raw)
        return {"text": final, "raw": raw, "thoughts": thoughts}
```

- [ ] **Step 3: Update make_backend() to support "openrouter"**

Replace the existing `make_backend` function:

```python
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
```

- [ ] **Step 4: Update module docstring**

Replace lines 1-10:

```python
"""
llm.py
------
LLM backend abstraction for HuggingFace local and OpenRouter API inference.

Factory:
  make_backend("huggingface", model_id="openai/gpt-oss-120b")
  make_backend("openrouter", model_id="openai/gpt-oss-120b")

Backends are cached so repeated calls in the same process do not reload weights.
"""
```

- [ ] **Step 5: Add import for re at top of file**

Add `import re` and `import os` to the imports section (after `from __future__ import annotations`):

```python
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple
```

- [ ] **Step 6: Verify the module loads without errors**

Run: `cd /Users/vora/Documents/PyCharm/DAIL-SQL/c_profile && python -c "from llm import make_backend; print('OK')"`

Expected: `OK` (no import errors; OpenRouter backend won't instantiate without key, but import works)

- [ ] **Step 7: Commit**

```bash
git add llm.py
git commit -m "feat: add OpenRouter backend with marker extraction to llm.py"
```

---

### Task 2: Add Full Schema Renderer and New Prompt Builders to candidates.py

**Files:**
- Modify: `/Users/vora/Documents/PyCharm/DAIL-SQL/c_profile/candidates.py`

This task adds the rendering and prompt building functions needed for the 3 candidate strategies.

- [ ] **Step 1: Add render_full_schema_highlighted() after render_column_profiles() (after line 103)**

```python
def render_full_schema_highlighted(
    schema: Dict[str, List[Dict[str, Any]]],
    linked_keys: Set[str],
    short_profiles: Dict[str, str],
) -> str:
    """
    Full-schema rendering: all tables, all columns.
    Linked columns get short profile inline comment + /* LINKED */ marker.
    Primary keys get -- primary key label.
    Non-linked columns: bare name + type.
    """
    blocks = []
    for table in schema:
        rendered = []
        for col in schema.get(table, []):
            key = f"{table}.{col['column']}"
            line = f"  {col['column']} {col['type']}"
            if col.get("pk"):
                line += "  -- primary key"
            if key in linked_keys:
                profile = short_profiles.get(key, "")
                if profile:
                    line += f"  -- {profile}"
                line += "  /* LINKED */"
            rendered.append(line)
        if rendered:
            blocks.append(f"CREATE TABLE {table} (\n" + ",\n".join(rendered) + "\n);")
    return "\n\n".join(blocks)


def render_full_profiles_block(
    linked_keys: Set[str],
    full_profiles: Dict[str, str],
    long_profiles: Dict[str, str],
) -> str:
    """
    Render detailed profile block for Candidate B (pruned + full profiles).
    Uses full_profiles (stats + dev doc), falls back to long_profiles.
    """
    sections = []
    for key in sorted(linked_keys):
        profile_text = full_profiles.get(key, "")
        if not profile_text:
            profile_text = long_profiles.get(key, "")
        if profile_text:
            sections.append(f"**{key}**\n{profile_text}")
    return "\n\n".join(sections)


def render_long_profiles_block(
    linked_keys: Set[str],
    long_profiles: Dict[str, str],
) -> str:
    """
    Render long profile descriptions for Candidate C (pruned + long profiles).
    Uses 'Field X means:' format with LLM-enhanced statistical descriptions.
    """
    sections = []
    for key in sorted(linked_keys):
        profile_text = long_profiles.get(key, "")
        if profile_text:
            sections.append(f"Field {key} means: {profile_text}")
    return "\n".join(sections)
```

- [ ] **Step 2: Add build_full_schema_prompt() and build_long_profile_prompt() after build_pruned_prompt() (after line 151)**

```python
def build_full_schema_prompt(
    question: str,
    evidence: str,
    schema_text: str,
    few_shots: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build prompt for Candidate A: full schema + short profiles inline + few-shot."""
    parts = [
        "You are a SQLite expert. Given the database schema and column profiles below, "
        "write a SQL query to answer the question.\n"
    ]
    if few_shots:
        parts.append(f"\n### Examples\n{_render_few_shots(few_shots)}")
    parts.append(f"\n### Database Schema\n{schema_text}\n")
    if evidence:
        parts.append(f"\n### Hint\n{evidence}")
    parts.append(
        f"\n### Question\n{question}\n\n"
        "### SQL\nWrite only the SQL query with no explanation.\n\n"
        "SQL:"
    )
    return "\n".join(parts)


def build_deep_focus_prompt(
    question: str,
    evidence: str,
    schema_text: str,
    profiles_block: str,
    few_shots: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build prompt for Candidate B: pruned schema + full profiles block + few-shot."""
    parts = [
        "You are a SQLite expert. Given the database schema and column profiles below, "
        "write a SQL query to answer the question.\n"
    ]
    if few_shots:
        parts.append(f"\n### Examples\n{_render_few_shots(few_shots)}")
    parts.append(f"\n### Database Schema\n{schema_text}\n")
    if profiles_block:
        parts.append(f"\n### Column Profiles\n{profiles_block}\n")
    if evidence:
        parts.append(f"\n### Hint\n{evidence}")
    parts.append(
        f"\n### Question\n{question}\n\n"
        "### SQL\nWrite only the SQL query with no explanation.\n\n"
        "SQL:"
    )
    return "\n".join(parts)


def build_independent_prompt(
    question: str,
    evidence: str,
    schema_text: str,
    profiles_text: str,
) -> str:
    """Build prompt for Candidate C: pruned schema + long profiles + NO few-shot."""
    parts = [
        "You are a SQLite expert. Given the database schema and column profiles below, "
        "write a SQL query to answer the question.\n"
    ]
    parts.append(f"\n### Database Schema\n{schema_text}\n")
    if profiles_text:
        parts.append(f"\n### Column Descriptions\n{profiles_text}")
    if evidence:
        parts.append(f"\n### Hint\n{evidence}")
    parts.append(
        f"\n### Question\n{question}\n\n"
        "### SQL\nWrite only the SQL query with no explanation.\n\n"
        "SQL:"
    )
    return "\n".join(parts)
```

- [ ] **Step 3: Verify the new functions are importable**

Run: `cd /Users/vora/Documents/PyCharm/DAIL-SQL/c_profile && python -c "from candidates import render_full_schema_highlighted, build_full_schema_prompt, build_deep_focus_prompt, build_independent_prompt; print('OK')"`

Expected: `OK`

```
```
---

### Task 3: Update generate_sql() to Use max_new_tokens=1900

**Files:**
- Modify: `/Users/vora/Documents/PyCharm/DAIL-SQL/c_profile/candidates.py`

- [ ] **Step 1: Change max_new_tokens from 512 to 1900 in generate_sql()**

At line 176, change:
```python
        response = backend.generate(messages, max_new_tokens=512, **gen_kwargs)
```
to:
```python
        response = backend.generate(messages, max_new_tokens=1900, **gen_kwargs)
```


---

### Task 4: Enhance Feedback Loop — Retry on Empty + Cross-Candidate Hints

**Files:**
- Modify: `/Users/vora/Documents/PyCharm/DAIL-SQL/c_profile/candidates.py`

This task modifies `generate_with_correction()` to retry empty generations and accept a `hint_sql` parameter for cross-candidate references.

- [ ] **Step 1: Replace the generate_with_correction() function (lines 226-306)**

Replace the entire function with:

```python
def generate_with_correction(
    prompt: str,
    backend,
    db_path: str,
    schema_text: str,
    question: str,
    evidence: str = "",
    profiles_text: str = "",
    temperature: float = 0,
    seed: Optional[int] = None,
    max_retry: int = 3,
    hint_sql: str = "",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate SQL, execute it, and if it fails or is empty, re-prompt the LLM
    with the error message for up to max_retry attempts (paper Section 4).

    hint_sql: optional working SQL from another candidate to reference in corrections.

    Returns:
        (final_sql, correction_log)  where correction_log tracks each attempt.
    """
    correction_log: List[Dict[str, Any]] = []

    # Initial generation
    sql = generate_sql(prompt, backend, temperature=temperature, seed=seed)

    # Handle empty generation — retry instead of giving up
    if not sql or not sql.strip():
        correction_log.append({"attempt": 0, "sql": "", "error": "empty_generation", "action": "will_retry"})

        for attempt in range(1, max_retry + 1):
            print(f"      [retry {attempt}/{max_retry}] empty generation, retrying...")
            retry_prompt = (
                "The previous attempt produced no valid SQL output.\n"
                "Generate a valid SQLite SQL query for the following question.\n\n"
                f"### Database Schema\n{schema_text}\n\n"
            )
            if profiles_text:
                retry_prompt += f"### Column Descriptions\n{profiles_text}\n\n"
            if evidence:
                retry_prompt += f"### Hint\n{evidence}\n\n"
            if hint_sql:
                retry_prompt += f"### Reference SQL from another approach\n{hint_sql}\n\n"
            retry_prompt += (
                f"### Question\n{question}\n\n"
                "Write only the SQL query.\n\nSQL:"
            )
            sql = generate_sql(retry_prompt, backend, temperature=temperature, seed=seed)
            if sql and sql.strip():
                correction_log.append({"attempt": attempt, "sql": sql, "error": None, "action": "recovered"})
                break
            correction_log.append({"attempt": attempt, "sql": "", "error": "empty_generation",
                                   "action": "will_retry" if attempt < max_retry else "exhausted"})

        if not sql or not sql.strip():
            return "", correction_log

    # Try executing
    exec_result = _try_execute(sql, db_path)
    if exec_result["success"]:
        correction_log.append({"attempt": 0, "sql": sql, "error": None, "action": "success"})
        return sql, correction_log

    # Initial execution failed — start correction loop
    correction_log.append({"attempt": 0, "sql": sql, "error": exec_result["error"], "action": "will_retry"})

    for attempt in range(1, max_retry + 1):
        error_msg = exec_result["error"]
        print(f"      [correction {attempt}/{max_retry}] error: {error_msg}")

        # Build correction prompt with the error message
        correction_prompt = (
            "You are a SQLite expert. The SQL query below has an error. "
            "Fix the SQL query and return ONLY the corrected SQL with no explanation.\n\n"
            f"### Database Schema\n{schema_text}\n\n"
        )
        if profiles_text:
            correction_prompt += f"### Column Descriptions\n{profiles_text}\n\n"
        if evidence:
            correction_prompt += f"### Hint\n{evidence}\n\n"
        if hint_sql:
            correction_prompt += (
                f"### Working SQL from another approach\n{hint_sql}\n\n"
            )
        correction_prompt += (
            f"### Question\n{question}\n\n"
            f"### Previous SQL (with error)\n{sql}\n\n"
            f"### Error Message\n{error_msg}\n\n"
            "### Corrected SQL\nWrite only the corrected SQL query.\n\n"
            "SQL:"
        )

        new_sql = generate_sql(correction_prompt, backend, temperature=temperature, seed=seed)

        if not new_sql or not new_sql.strip():
            correction_log.append({"attempt": attempt, "sql": "", "error": "empty_generation", "action": "stop"})
            break

        if new_sql.strip() == sql.strip():
            # LLM returned the same SQL — no point retrying
            correction_log.append({"attempt": attempt, "sql": new_sql, "error": "same_as_previous", "action": "stop"})
            break

        sql = new_sql
        exec_result = _try_execute(sql, db_path)

        if exec_result["success"]:
            correction_log.append({"attempt": attempt, "sql": sql, "error": None, "action": "fixed"})
            print(f"      [correction {attempt}/{max_retry}] fixed!")
            return sql, correction_log

        correction_log.append({"attempt": attempt, "sql": sql, "error": exec_result["error"],
                               "action": "will_retry" if attempt < max_retry else "exhausted"})

    # Exhausted retries — return the last SQL (may still be broken)
    return sql, correction_log
```

- [ ] **Step 2: Verify syntax**

Run: `cd /Users/vora/Documents/PyCharm/DAIL-SQL/c_profile && python -c "from candidates import generate_with_correction; print('OK')"`

Expected: `OK`



---

### Task 5: Add LLM Tiebreaker Voting

**Files:**
- Modify: `/Users/vora/Documents/PyCharm/DAIL-SQL/c_profile/candidates.py`

This task adds the LLM tiebreaker function and updates majority_vote() to use it instead of random fallback.

- [ ] **Step 1: Add llm_tiebreaker() function after majority_vote() (after line 402)**

```python
def llm_tiebreaker(
    candidate_sqls: List[str],
    candidate_names: List[str],
    db_path: str,
    question: str,
    evidence: str,
    backend,
) -> Dict[str, Any]:
    """
    LLM-based tiebreaker for when majority vote has no agreement.
    Asks the LLM to pick the best SQL from valid candidates.
    """
    # Filter to valid (executable) candidates
    valid = []
    for i, sql in enumerate(candidate_sqls):
        if sql and sql.strip():
            result = execute_sql(sql, db_path)
            if result is not None:
                valid.append((i, sql))

    # If only 1 valid, pick it
    if len(valid) == 1:
        idx, sql = valid[0]
        return {
            "winner_idx": idx,
            "winner": candidate_names[idx],
            "method": "tiebreaker_single",
            "agreement": 1,
            "sql": sql,
        }

    # If 0 valid, pick first non-empty
    if len(valid) == 0:
        for i, sql in enumerate(candidate_sqls):
            if sql and sql.strip():
                return {
                    "winner_idx": i,
                    "winner": candidate_names[i],
                    "method": "tiebreaker_fallback",
                    "agreement": 0,
                    "sql": sql,
                }
        return {
            "winner_idx": 0,
            "winner": candidate_names[0] if candidate_names else "",
            "method": "tiebreaker_fallback",
            "agreement": 0,
            "sql": candidate_sqls[0] if candidate_sqls else "",
        }

    # 2+ valid but disagree — ask LLM
    letters = "ABCDEFGHIJ"
    options = []
    option_map = {}  # letter -> candidate index
    for li, (idx, sql) in enumerate(valid):
        letter = letters[li]
        option_map[letter] = idx
        options.append(f"{letter}) {sql}")

    options_text = "\n\n".join(options)

    tiebreaker_prompt = (
        "You are a SQLite expert. Given a question and multiple SQL queries that produce "
        "different results, select the most correct one.\n\n"
        f"### Question\n{question}\n"
    )
    if evidence:
        tiebreaker_prompt += f"\n### Hint\n{evidence}\n"
    tiebreaker_prompt += (
        f"\n### Candidates\n{options_text}\n\n"
        "Which candidate is most likely correct? Reply with ONLY the letter.\n\n"
        "Answer:"
    )

    try:
        messages = [{"role": "user", "content": tiebreaker_prompt}]
        response = backend.generate(messages, max_new_tokens=50, temperature=0)
        # Parse letter from response
        chosen_letter = None
        for letter in option_map:
            if letter in response[:10]:
                chosen_letter = letter
                break
        if chosen_letter and chosen_letter in option_map:
            winner_idx = option_map[chosen_letter]
        else:
            # Fallback to first valid
            winner_idx = valid[0][0]
    except Exception as e:
        print(f"    [TIEBREAKER ERROR] {e}")
        winner_idx = valid[0][0]

    return {
        "winner_idx": winner_idx,
        "winner": candidate_names[winner_idx],
        "method": "tiebreaker_llm",
        "agreement": 1,
        "sql": candidate_sqls[winner_idx],
    }
```

- [ ] **Step 2: Update majority_vote() to call llm_tiebreaker instead of random fallback**

Replace the fallback section of majority_vote() (lines 381-402) with:

```python
    # No agreement — return None to signal tiebreaker needed
    return None
```

This changes majority_vote() to return `None` when there's no agreement, so the caller can invoke llm_tiebreaker.

- [ ] **Step 3: Verify syntax**

Run: `cd /Users/vora/Documents/PyCharm/DAIL-SQL/c_profile && python -c "from candidates import llm_tiebreaker, majority_vote; print('OK')"`

Expected: `OK`



---

### Task 6: Refactor process_question() for Multi-Strategy Candidates

**Files:**
- Modify: `/Users/vora/Documents/PyCharm/DAIL-SQL/c_profile/candidates.py`

This is the core task — replaces the 3-same-prompt loop with 3 structurally different candidate strategies.

- [ ] **Step 1: Replace the _CANDIDATES config and process_question() function (lines 573-696)**

Replace the `_CANDIDATES` list and the entire `process_question()` function:

```python
# ─────────────────────────────────────────────────────────────
# Multi-Strategy Candidate Configuration
# ─────────────────────────────────────────────────────────────

_CANDIDATE_STRATEGIES = [
    {"name": "wide_net",           "strategy": "full_schema",   "temperature": 0,   "seed": None, "use_fewshot": True},
    {"name": "deep_focus",         "strategy": "pruned_full",   "temperature": 0,   "seed": None, "use_fewshot": True},
    {"name": "independent_thinker","strategy": "pruned_long",   "temperature": 0.7, "seed": 42,   "use_fewshot": False},
]


def process_question(
    q: Dict[str, Any],
    db: DBLoader,
    backend,
    use_mcs: bool = False,
    few_shot_retriever: Optional[FewShotRetriever] = None,
    feedback: bool = False,
    max_retry: int = 3,
) -> Dict[str, Any]:
    """
    Generate 3 SQL candidates using structurally different prompts and select the best.

    Candidate A (wide_net):           Full schema + short profiles inline + few-shot, temp=0
    Candidate B (deep_focus):         Pruned schema + full profiles block + few-shot, temp=0
    Candidate C (independent_thinker): Pruned schema + long profiles + NO few-shot, temp=0.7

    Selection: majority vote, with LLM tiebreaker when no agreement.
    """
    question = q["question"]
    evidence = q.get("evidence", "")
    schema_links = q.get("schema_links", [])

    # Build set of linked column keys
    linked_keys: Set[str] = set()
    for sl in schema_links:
        linked_keys.add(f"{sl['table']}.{sl['column']}")

    print(f"  Linked columns: {len(linked_keys)}")

    # Retrieve 8 few-shot examples (paper Section 4)
    few_shots: Optional[List[Dict[str, Any]]] = None
    if few_shot_retriever is not None:
        few_shots = few_shot_retriever.retrieve(question, k=8, exclude_question=question)
        print(f"  Few-shot examples: {len(few_shots)}")

    # Pre-render schema variants
    schema_pruned = render_pruned_schema(db.schema, linked_keys)
    schema_full = render_full_schema_highlighted(db.schema, linked_keys, db.short_profiles)
    profiles_full_block = render_full_profiles_block(linked_keys, db.full_profiles, db.long_profiles)
    profiles_long_text = render_long_profiles_block(linked_keys, db.long_profiles)

    # Generate candidates — 3 structurally different prompts
    candidate_sqls: List[str] = []
    candidate_names: List[str] = []
    candidates_detail: List[Dict[str, Any]] = []

    for i, cfg in enumerate(_CANDIDATE_STRATEGIES):
        name = cfg["name"]
        strategy = cfg["strategy"]
        temp = cfg["temperature"]
        seed = cfg["seed"]
        use_fewshot = cfg["use_fewshot"]

        print(f"    [{i+1}/{len(_CANDIDATE_STRATEGIES)}] {name} (strategy={strategy}, temp={temp})...")

        # Build strategy-specific prompt
        if strategy == "full_schema":
            prompt = build_full_schema_prompt(
                question, evidence, schema_full,
                few_shots=few_shots if use_fewshot else None,
            )
            schema_for_correction = schema_full
            profiles_for_correction = ""
        elif strategy == "pruned_full":
            prompt = build_deep_focus_prompt(
                question, evidence, schema_pruned, profiles_full_block,
                few_shots=few_shots if use_fewshot else None,
            )
            schema_for_correction = schema_pruned
            profiles_for_correction = profiles_full_block
        elif strategy == "pruned_long":
            prompt = build_independent_prompt(
                question, evidence, schema_pruned, profiles_long_text,
            )
            schema_for_correction = schema_pruned
            profiles_for_correction = profiles_long_text
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Cross-candidate hint: use first successful candidate's SQL
        hint_sql = ""
        if feedback and candidate_sqls:
            for prev_sql in candidate_sqls:
                if prev_sql and prev_sql.strip():
                    prev_result = _try_execute(prev_sql, db.db_path)
                    if prev_result["success"]:
                        hint_sql = prev_sql
                        break

        if feedback:
            sql, corr_log = generate_with_correction(
                prompt, backend, db.db_path, schema_for_correction, question, evidence,
                profiles_text=profiles_for_correction,
                temperature=temp, seed=seed, max_retry=max_retry,
                hint_sql=hint_sql,
            )
            n_corrections = len([c for c in corr_log if c.get("action") in ("fixed", "recovered")])
            if n_corrections > 0:
                print(f"    -> {sql[:80]}..." if len(sql) > 80 else f"    -> {sql}")
                print(f"    fixed after {len(corr_log)-1} correction(s)")
            else:
                print(f"    -> {sql[:80]}..." if len(sql) > 80 else f"    -> {sql}")
        else:
            sql = generate_sql(prompt, backend, temperature=temp, seed=seed)
            corr_log = []
            print(f"    -> {sql[:80]}..." if len(sql) > 80 else f"    -> {sql}")

        candidate_sqls.append(sql)
        candidate_names.append(name)
        detail = {
            "sql": sql,
            "strategy": strategy,
            "temperature": temp,
            "seed": seed,
            "use_fewshot": use_fewshot,
        }
        if corr_log:
            detail["correction_log"] = corr_log
        candidates_detail.append(detail)

    # ── Selection: Majority Vote + LLM Tiebreaker ──
    print("    Selecting via majority vote...")
    vote = majority_vote(candidate_sqls, candidate_names, db.db_path)

    if vote is None:
        # No majority agreement — use LLM tiebreaker
        print("    No majority — using LLM tiebreaker...")
        vote = llm_tiebreaker(
            candidate_sqls, candidate_names, db.db_path,
            question, evidence, backend,
        )

    print(f"    Winner: {vote['winner']} ({vote['method']}, agreement={vote['agreement']})")

    return {
        "question_id": q.get("question_id"),
        "db_id": q.get("db_id", db.db_id),
        "question": question,
        "evidence": evidence,
        "gold_sql": q.get("gold_sql", ""),
        "difficulty": q.get("difficulty", ""),
        "schema_links": schema_links,
        "candidates": candidates_detail,
        "voted_sql": vote["sql"],
        "vote_details": {
            "method": vote["method"],
            "winner": vote["winner"],
            "agreement": vote["agreement"],
        },
    }
```

- [ ] **Step 2: Verify syntax**

Run: `cd /Users/vora/Documents/PyCharm/DAIL-SQL/c_profile && python -c "from candidates import process_question; print('OK')"`

Expected: `OK`



---

### Task 7: Update CLI to Support OpenRouter Backend

**Files:**
- Modify: `/Users/vora/Documents/PyCharm/DAIL-SQL/c_profile/candidates.py`

- [ ] **Step 1: Update the argparse --backend argument (line 893-895)**

Replace:
```python
    parser.add_argument(
        "--backend", type=str, default="huggingface",
        help="LLM backend: huggingface (default: huggingface)",
    )
```
with:
```python
    parser.add_argument(
        "--backend", type=str, default="openrouter",
        choices=["openrouter", "huggingface"],
        help="LLM backend (default: openrouter). Requires OPENROUTER_API_KEY env var.",
    )
```

- [ ] **Step 2: Update the --model help text (lines 896-899)**

Replace:
```python
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model ID (default: openai/gpt-oss-120b)",
    )
```
with:
```python
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model ID override (default: openai/gpt-oss-120b for openrouter)",
    )
```

- [ ] **Step 3: Remove the --mcs argument (lines 900-903)**

Delete these lines since we're not using MCS:
```python
    parser.add_argument(
        "--mcs", action="store_true",
        help="Use MCS-SQL multiple-choice selection instead of majority voting",
    )
```

- [ ] **Step 4: Update the run section to remove mcs references**

Replace lines 943-946:
```python
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Selection: {'MCS-SQL' if args.mcs else 'Majority Vote'}")
    if args.feedback:
```
with:
```python
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Selection: Majority Vote + LLM Tiebreaker")
    if args.feedback:
```

- [ ] **Step 5: Update the run_candidates call to remove use_mcs**

Replace:
```python
    results = run_candidates(
        input_path=input_path,
        backend=backend,
        use_mcs=args.mcs,
        output_path=output_path,
        few_shot_retriever=few_shot_retriever,
        feedback=args.feedback,
        max_retry=args.max_retry,
    )
```
with:
```python
    results = run_candidates(
        input_path=input_path,
        backend=backend,
        use_mcs=False,
        output_path=output_path,
        few_shot_retriever=few_shot_retriever,
        feedback=args.feedback,
        max_retry=args.max_retry,
    )
```

- [ ] **Step 6: Verify CLI help works**

Run: `cd /Users/vora/Documents/PyCharm/DAIL-SQL/c_profile && python candidates.py --help`

Expected: Shows updated help with `--backend` choices including `openrouter` as default.



---

### Task 8: End-to-End Smoke Test

**Files:**
- No files created or modified — this is a verification task

- [ ] **Step 1: Set OPENROUTER_API_KEY and test single-question generation**

Run a quick test with 1 question to verify the full pipeline works:

```bash
cd /Users/vora/Documents/PyCharm/DAIL-SQL/c_profile

# Test that OpenRouter backend connects and generates SQL
python -c "
import os
os.environ.setdefault('OPENROUTER_API_KEY', 'YOUR_KEY_HERE')
from llm import make_backend
backend = make_backend('openrouter')

messages = [
    {'role': 'system', 'content': 'You are a Text-to-SQL assistant for SQLite.\nReturn ONLY a single valid SQLite SQL query.\nNo explanations.'},
    {'role': 'user', 'content': '### Database Schema\nCREATE TABLE users (id INTEGER, name TEXT, age INTEGER);\n\n### Question\nHow many users are there?\n\n### SQL\nWrite only the SQL query with no explanation.\n\nSQL:'},
]
result = backend.generate_with_meta(messages, max_new_tokens=200)
print('Text:', result['text'])
print('Raw:', result['raw'][:200])
print('Thoughts:', result['thoughts'][:200] if result['thoughts'] else '(none)')
"
```

Expected: Clean SQL output like `SELECT COUNT(*) FROM users` in the `text` field, with reasoning (if any) in `thoughts`.

- [ ] **Step 2: Run candidates.py on a small subset**

Create a test with just the first 2 questions from the schema links file:

```bash
cd /Users/vora/Documents/PyCharm/DAIL-SQL/c_profile

python -c "
import json
with open('/Users/vora/Downloads/Results_metadata_paper/schema_links_all_dbs_Noneper_20260314_221301.json') as f:
    data = json.load(f)
data['results'] = data['results'][:2]
with open('results/test_2q_schema_links.json', 'w') as f:
    json.dump(data, f, indent=2)
print('Wrote 2-question test file')
"
```

Then run:
```bash
python candidates.py \
    --input results/test_2q_schema_links.json \
    --backend openrouter \
    --feedback \
    --out results/test_2q_candidates.json
```

Expected: Generates 3 candidates per question (wide_net, deep_focus, independent_thinker), runs feedback loop, votes, saves results. Check output file for correct structure.

- [ ] **Step 3: Verify output structure**

```bash
python -c "
import json
with open('results/test_2q_candidates.json') as f:
    data = json.load(f)
r = data['results'][0]
print('Candidates:', len(r['candidates']))
for c in r['candidates']:
    print(f\"  {c.get('strategy', '?')}: {c['sql'][:60]}...\")
print(f\"Voted SQL: {r['voted_sql'][:60]}...\")
print(f\"Vote method: {r['vote_details']['method']}\")
print(f\"Winner: {r['vote_details']['winner']}\")
"
```

Expected: 3 candidates with strategies `full_schema`, `pruned_full`, `pruned_long`. A voted_sql selected by majority or tiebreaker.

- [ ] **Step 4: Commit (no code changes — just verify)**

No commit needed. Pipeline is verified and ready for full 500-question run.

---

### Task 9: Full 500-Question Run

**Files:**
- No code changes — execution task

- [ ] **Step 1: Run the full pipeline with feedback**

```bash
cd /Users/vora/Documents/PyCharm/DAIL-SQL/c_profile

export OPENROUTER_API_KEY=your_key_here

python candidates.py \
    --input /Users/vora/Downloads/Results_metadata_paper/schema_links_all_dbs_Noneper_20260314_221301.json \
    --backend openrouter \
    --feedback \
    --max_retry 3 \
    --out results/candidates_openrouter_multistrategy.json
```

Expected: ~2000-2500 API calls, ~$0.50 cost, generates results for all 500 questions. Summary at end shows accuracy breakdown.

- [ ] **Step 2: Check accuracy summary**

The `print_summary()` function at the end will show:
- Oracle accuracy (any candidate correct)
- Voted accuracy (selected candidate correct)
- Vote method distribution (majority vs tiebreaker_llm vs tiebreaker_fallback)

Target: voted accuracy > 61% (Anik's baseline).

- [ ] **Step 3: If accuracy is below target, run without feedback for comparison**

```bash
python candidates.py \
    --input /Users/vora/Downloads/Results_metadata_paper/schema_links_all_dbs_Noneper_20260314_221301.json \
    --backend openrouter \
    --out results/candidates_openrouter_nofeedback.json
```

Compare the two results to measure feedback loop impact.
