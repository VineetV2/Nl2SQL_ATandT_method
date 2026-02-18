#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mention_extract_local.py
------------------------
Rule-first mention extraction from a SpaCy dependency parse.

Goal
----
Given:
  - question text
  - SpaCy dependency parse (DependencyParse from dependency_tree_local.py)
  - already-extracted literals (from literal_extract_only.py)

Return:
  - high-recall "mention candidates" (mostly noun-phrase spans) that likely
    correspond to schema concepts/attributes (e.g., "customer name",
    "transaction id", "annual average consumption").

Design choices (fits your pipeline)
-----------------------------------
- Deterministic + fast (no LLM required).
- High recall: emit multiple overlapping candidates (head-only + expanded NP).
- Literal tokens are *excluded* from NP expansion by default (you already handle literals).
- Optional: attach mention↔literal links using dependency heads (useful later for grounding).

This module is meant to be used *before* concept/segment grounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set

import re


# -------------------------
# Utilities
# -------------------------

_WS = re.compile(r"\s+")


def _norm_space(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())


def _span_chars(dpt, start_i: int, end_i: int) -> Tuple[int, int]:
    """
    Convert token span [start_i, end_i] (inclusive) into char span [c0, c1).
    Uses dpt.token_char_spans from dependency_tree_local.DependencyParse.
    """
    spans = getattr(dpt, "token_char_spans", None)
    if not spans:
        # fallback: no char spans available
        return (0, 0)
    c0 = spans[start_i][0]
    c1 = spans[end_i][1]
    return int(c0), int(c1)


def _span_text(dpt, start_i: int, end_i: int) -> str:
    """
    Exact surface text for a token span, sliced from the original question.
    """
    text = getattr(dpt, "text", "") or ""
    c0, c1 = _span_chars(dpt, start_i, end_i)
    if c1 > c0 and c1 <= len(text):
        return _norm_space(text[c0:c1])
    # fallback join
    toks = getattr(dpt, "tokens", [])
    return _norm_space(" ".join(toks[start_i:end_i + 1]))


def _is_content_pos(pos: str) -> bool:
    # spaCy POS tags: NOUN, PROPN, ADJ, VERB, NUM, etc.
    return pos in ("NOUN", "PROPN", "ADJ")  # keep ADJ for things like "average" used as noun-ish


def _is_mention_head(pos: str) -> bool:
    return pos in ("NOUN", "PROPN")


def _children_of(dpt, head_i: int) -> List[int]:
    """
    Children indices of head_i.

    IMPORTANT:
    - The canonical source of head/child relations is `dpt.heads` (same indexing as tokens).
    - `dependency_tree_local.DependencyParse.arcs` is kept for debugging and is a list of dicts
      like {child_i, head_i, dep, ...}. Older code sometimes used (head, child, dep) triples.
      This helper is robust to either representation.
    """
    # Prefer heads[] (fast + stable)
    heads = getattr(dpt, "heads", None)
    if heads is not None:
        out = [i for i, h in enumerate(heads) if int(h) == int(head_i)]
        out.sort()
        return out

    # Fallback: derive from arcs, supporting both dict and tuple formats.
    arcs = getattr(dpt, "arcs", []) or []
    out: List[int] = []
    for a in arcs:
        try:
            if isinstance(a, dict):
                h = int(a.get("head_i"))
                c = int(a.get("child_i"))
            else:
                # tuple/list format: (head, child, dep)
                h = int(a[0])
                c = int(a[1])
        except Exception:
            continue
        if h == int(head_i):
            out.append(c)
    out.sort()
    return out


def _dep_of(dpt, i: int) -> str:
    deps = getattr(dpt, "deps", []) or []
    return deps[i] if 0 <= i < len(deps) else ""


def _pos_of(dpt, i: int) -> str:
    pos = getattr(dpt, "pos", []) or []
    return pos[i] if 0 <= i < len(pos) else ""


def _lemma_of(dpt, i: int) -> str:
    lemmas = getattr(dpt, "lemmas", []) or []
    if lemmas and 0 <= i < len(lemmas):
        return str(lemmas[i])
    # fallback: lowercase token
    toks = getattr(dpt, "tokens", []) or []
    return str(toks[i]).lower() if 0 <= i < len(toks) else ""


def _token(dpt, i: int) -> str:
    toks = getattr(dpt, "tokens", []) or []
    return str(toks[i]) if 0 <= i < len(toks) else ""


def _token_is_punct(dpt, i: int) -> bool:
    return _pos_of(dpt, i) == "PUNCT" or _token(dpt, i) in {",", ".", "?", "!", ";", ":"}


# -------------------------
# Output structures
# -------------------------

@dataclass(frozen=True)
class MentionCandidate:
    mention_id: str
    start_token: int
    end_token: int  # inclusive
    head_token: int
    text: str
    head_text: str
    lemma: str
    pos: str
    kind: str  # "head" | "np" | "np_of"
    source: str  # short rule label
    contains_literal: bool


@dataclass(frozen=True)
class MentionLiteralLink:
    mention_id: str
    literal_id: str
    rel: str  # "arg" | "value" | ...
    # keep a light copy of the literal
    literal_kind: str
    literal_text: str


# -------------------------
# Mention extraction
# -------------------------

# conservative modifier deps for NP expansion
_LEFT_MOD_DEPS = {
    "compound", "amod", "nummod", "poss", "npadvmod", "appos", "quantmod", "advmod"
}

# prepositions that often attach constraints
_PREP_DEPS = {"prep", "agent"}

# If we include "of" phrase, allow these deps inside object NP.
_OBJ_NP_DEPS = {"compound", "amod", "nummod", "poss", "npadvmod", "appos"}



def _coerce_literals(literals: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """
    Accept literals as either:
      - list[dict] (already serialized), OR
      - list[Literal] dataclass objects from literal_extract_only.py

    Returns list[dict] with inclusive token spans:
      { literal_id, kind, text, start_token, end_token }
    """
    out: List[Dict[str, Any]] = []
    if not literals:
        return out
    for i, lit in enumerate(literals):
        if isinstance(lit, dict):
            # allow either inclusive or exclusive; prefer inclusive if provided
            if "start_token" in lit and "end_token" in lit:
                out.append(dict(lit))
                continue
            if "token_span" in lit and isinstance(lit["token_span"], (list, tuple)) and len(lit["token_span"]) == 2:
                s, e_excl = int(lit["token_span"][0]), int(lit["token_span"][1])
                out.append({
                    "literal_id": lit.get("literal_id") or f"L:{i}",
                    "kind": lit.get("kind") or lit.get("typ") or "",
                    "text": lit.get("text") or "",
                    "start_token": s,
                    "end_token": max(s, e_excl - 1),
                })
                continue
            out.append(dict(lit))
            continue

        # dataclass-like
        kind = getattr(lit, "kind", None) or getattr(lit, "typ", "")
        text = getattr(lit, "text", "")
        token_span = getattr(lit, "token_span", None)
        if isinstance(token_span, (list, tuple)) and len(token_span) == 2:
            s, e_excl = int(token_span[0]), int(token_span[1])
            out.append({
                "literal_id": getattr(lit, "literal_id", None) or f"L:{i}",
                "kind": str(kind),
                "text": str(text),
                "start_token": s,
                "end_token": max(s, e_excl - 1),
            })
        else:
            # unknown structure; ignore
            continue
    return out


def _literal_token_mask(literals: List[Dict[str, Any]], n_tokens: int) -> List[bool]:
    mask = [False] * n_tokens
    for lit in (literals or []):
        try:
            s = int(lit.get("start_token"))
            e = int(lit.get("end_token"))
        except Exception:
            continue
        if s < 0 or e < 0:
            continue
        s = max(0, min(n_tokens - 1, s))
        e = max(0, min(n_tokens - 1, e))
        for i in range(s, e + 1):
            mask[i] = True
    return mask


def _expand_np_indices(
    dpt,
    head_i: int,
    *,
    literal_mask: Optional[List[bool]] = None,
    allow_of_phrase: bool = True,
) -> Tuple[Set[int], str]:
    """
    Expand a noun phrase around head_i using local dependency modifiers.
    Returns: (indices_set, kind_tag)
    """
    n = len(getattr(dpt, "tokens", []) or [])
    literal_mask = literal_mask or ([False] * n)

    idxs: Set[int] = {head_i}
    source_kind = "np"

    # include left-side modifiers that are direct children of head
    for ch in _children_of(dpt, head_i):
        if ch < 0 or ch >= n:
            continue
        if literal_mask[ch]:
            continue
        dep = _dep_of(dpt, ch)
        if dep in _LEFT_MOD_DEPS:
            idxs.add(ch)
            # include chained compounds on that modifier
            for ch2 in _children_of(dpt, ch):
                if literal_mask[ch2]:
                    continue
                if _dep_of(dpt, ch2) == "compound":
                    idxs.add(ch2)

    # optionally include "of" phrase: head -> prep(of) -> pobj(noun)
    if allow_of_phrase:
        for ch in _children_of(dpt, head_i):
            if _dep_of(dpt, ch) in _PREP_DEPS and _token(dpt, ch).lower() == "of":
                # find pobj / pcomp of 'of'
                pobj = None
                for ch2 in _children_of(dpt, ch):
                    if _dep_of(dpt, ch2) in ("pobj", "pcomp"):
                        pobj = ch2
                        break
                if pobj is None:
                    continue
                if literal_mask[pobj]:
                    continue
                if not _is_mention_head(_pos_of(dpt, pobj)):
                    continue

                idxs.add(ch)  # include 'of'
                idxs.add(pobj)
                source_kind = "np_of"

                # include object NP modifiers
                for ch3 in _children_of(dpt, pobj):
                    if literal_mask[ch3]:
                        continue
                    if _dep_of(dpt, ch3) in _OBJ_NP_DEPS:
                        idxs.add(ch3)
                        for ch4 in _children_of(dpt, ch3):
                            if literal_mask[ch4]:
                                continue
                            if _dep_of(dpt, ch4) == "compound":
                                idxs.add(ch4)

    return idxs, source_kind


def _indices_to_span(idxs: Set[int]) -> Optional[Tuple[int, int]]:
    if not idxs:
        return None
    s = min(idxs)
    e = max(idxs)
    return (s, e)


def _span_ok(dpt, s: int, e: int, *, max_len: int = 8) -> bool:
    if s > e:
        return False
    if (e - s + 1) > max_len:
        return False
    # Avoid spans that are mostly punctuation
    toks = getattr(dpt, "tokens", []) or []
    if not toks:
        return False
    non_punct = [i for i in range(s, e + 1) if not _token_is_punct(dpt, i)]
    return len(non_punct) > 0


def _dedupe_mentions(mentions: List[MentionCandidate]) -> List[MentionCandidate]:
    seen = set()
    out: List[MentionCandidate] = []
    for m in mentions:
        key = (m.start_token, m.end_token, m.head_token, m.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def extract_mentions(
    dpt,
    literals: Optional[List[Dict[str, Any]]] = None,
    *,
    max_np_len: int = 8,
    include_head_only: bool = True,
    include_np: bool = True,
    allow_of_phrase: bool = True,
    attach_literal_links: bool = True,
) -> Dict[str, Any]:
    """
    Extract mention candidates from a DependencyParse.

    Parameters
    ----------
    dpt:
      DependencyParse from dependency_tree_local.dependency_parse
    literals:
      list of literal dicts; expected keys include:
        - literal_id (optional)
        - kind
        - text
        - start_token, end_token   (inclusive)
    Returns
    -------
    dict with:
      - "mentions": [ ... ]
      - "mention_literal_links": [ ... ]  (optional)
    """
    toks = getattr(dpt, "tokens", []) or []
    n = len(toks)
    pos = getattr(dpt, "pos", []) or [""] * n

    literals = _coerce_literals(literals)
    literal_mask = _literal_token_mask(literals, n)

    mentions: List[MentionCandidate] = []

    # 1) Head-only mentions (one per NOUN/PROPN token not in literal span)
    if include_head_only:
        for i in range(n):
            if literal_mask[i]:
                continue
            if not _is_mention_head(pos[i]):
                continue
            txt = _span_text(dpt, i, i)
            if not txt:
                continue
            mentions.append(MentionCandidate(
                mention_id=f"M:{len(mentions)}",
                start_token=i,
                end_token=i,
                head_token=i,
                text=txt,
                head_text=_token(dpt, i),
                lemma=_lemma_of(dpt, i),
                pos=pos[i],
                kind="head",
                source="head_noun",
                contains_literal=False,
            ))

    # 2) Expanded noun-phrase mentions around each head noun
    if include_np:
        for i in range(n):
            if literal_mask[i]:
                continue
            if not _is_mention_head(pos[i]):
                continue

            idxs, kind_tag = _expand_np_indices(
                dpt, i, literal_mask=literal_mask, allow_of_phrase=allow_of_phrase
            )
            span = _indices_to_span(idxs)
            if not span:
                continue
            s, e = span
            if (s == i and e == i):
                continue  # already have head-only
            if not _span_ok(dpt, s, e, max_len=max_np_len):
                continue
            txt = _span_text(dpt, s, e)
            if not txt:
                continue
            # quick stop: spans like "the" are impossible here, but keep a guard
            if len(txt) <= 1:
                continue

            mentions.append(MentionCandidate(
                mention_id=f"M:{len(mentions)}",
                start_token=s,
                end_token=e,
                head_token=i,
                text=txt,
                head_text=_token(dpt, i),
                lemma=_lemma_of(dpt, i),
                pos=pos[i],
                kind=kind_tag,
                source="np_expand",
                contains_literal=False,
            ))

    # Dedupe, then reassign stable mention IDs in order
    mentions = _dedupe_mentions(mentions)
    mentions_sorted = sorted(
        mentions,
        key=lambda m: (m.start_token, m.end_token, (0 if m.kind == "np_of" else 1), m.head_token),
    )
    mentions_final: List[MentionCandidate] = []
    for k, m in enumerate(mentions_sorted):
        mentions_final.append(MentionCandidate(
            mention_id=f"M:{k}",
            start_token=m.start_token,
            end_token=m.end_token,
            head_token=m.head_token,
            text=m.text,
            head_text=m.head_text,
            lemma=m.lemma,
            pos=m.pos,
            kind=m.kind,
            source=m.source,
            contains_literal=m.contains_literal,
        ))

    # 3) Optional mention↔literal links (use head of literal span)
    links: List[MentionLiteralLink] = []
    if attach_literal_links and literals:
        # map token index -> best mention that covers it (prefer smallest span)
        cover: Dict[int, MentionCandidate] = {}
        for m in mentions_final:
            for i in range(m.start_token, m.end_token + 1):
                # prefer smaller spans for token coverage
                if i not in cover or (m.end_token - m.start_token) < (cover[i].end_token - cover[i].start_token):
                    cover[i] = m

        heads = getattr(dpt, "heads", []) or []
        for li, lit in enumerate(literals):
            # expected: start_token/end_token inclusive
            try:
                ls = int(lit.get("start_token"))
                le = int(lit.get("end_token"))
            except Exception:
                continue
            if ls < 0 or le < 0 or ls >= n:
                continue
            le = min(le, n - 1)

            # find an attachment token: take the head of the first token in span
            attach_tok = None
            if heads and 0 <= ls < len(heads):
                attach_tok = int(heads[ls])
            # sometimes head points inside the literal span; step once to escape
            if attach_tok is not None and ls <= attach_tok <= le and heads and 0 <= attach_tok < len(heads):
                attach_tok2 = int(heads[attach_tok])
                if attach_tok2 != attach_tok:
                    attach_tok = attach_tok2

            if attach_tok is None or attach_tok < 0 or attach_tok >= n:
                continue
            if literal_mask[attach_tok]:
                continue

            m = cover.get(attach_tok)
            if not m:
                # fallback: choose mention whose head is attach_tok
                cands = [mm for mm in mentions_final if mm.head_token == attach_tok]
                m = cands[0] if cands else None
            if not m:
                continue

            lit_id = lit.get("literal_id") or f"L:{li}"
            links.append(MentionLiteralLink(
                mention_id=m.mention_id,
                literal_id=str(lit_id),
                rel="value",
                literal_kind=str(lit.get("kind") or ""),
                literal_text=str(lit.get("text") or ""),
            ))

    # serialize
    out_mentions = [
        {
            "mention_id": m.mention_id,
            "text": m.text,
            "start_token": m.start_token,
            "end_token": m.end_token,
            "head_token": m.head_token,
            "head_text": m.head_text,
            "lemma": m.lemma,
            "pos": m.pos,
            "kind": m.kind,
            "source": m.source,
            "contains_literal": bool(m.contains_literal),
        }
        for m in mentions_final
    ]
    out_links = [
        {
            "mention_id": lk.mention_id,
            "literal_id": lk.literal_id,
            "rel": lk.rel,
            "literal_kind": lk.literal_kind,
            "literal_text": lk.literal_text,
        }
        for lk in links
    ]

    return {
        "mentions": out_mentions,
        "mention_literal_links": out_links,
    }
