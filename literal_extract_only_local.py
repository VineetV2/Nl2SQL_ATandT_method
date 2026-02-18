"""
literal_extract_only.py - 5. after Dep tree, next 6. 
-----------------------
Extract literals from a question using:
  - NER (spaCy)
  - Rule patterns (quoted, currency, percent, ISO dates, numbers, time, months, weekdays, currency words, uppercase codes)

IMPORTANT CHANGE FOR THIS PROJECT:
- In addition to character spans, every literal is aligned to the dependency-tree token indices.
  That means you can reference literals by the same token indices used in dependency_tree.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable
from functools import lru_cache
import re

# Local project module (keeps token indexing consistent across pipeline)
from dependency_tree_local import DependencyParse, dependency_parse, get_spacy_model


# -----------------------
# Data structure
# -----------------------

@dataclass
class Literal:
    text: str
    typ: str                 # 'date','time','money','percent','number','person','org','location','misc',...
    norm: str
    variants: List[str]
    char_span: Tuple[int, int]               # (start_char, end_char) in original question
    token_span: Tuple[int, int]              # (start_token_i, end_token_i_exclusive) in dependency tokens
    token_indices: List[int]                 # explicit token indices covered by the literal span
    source: str                              # 'ner' or 'rule'


# -----------------------
# Regex / lexicons (kept from your script)
# -----------------------

_MONTHS = {
    "january":  ("january","jan","01","1"),
    "february": ("february","feb","02","2"),
    "march":    ("march","mar","03","3"),
    "april":    ("april","apr","04","4"),
    "may":      ("may","may","05","5"),
    "june":     ("june","jun","06","6"),
    "july":     ("july","jul","07","7"),
    "august":   ("august","aug","08","8"),
    "september":("september","sep","09","9"),
    "october":  ("october","oct","10","10"),
    "november": ("november","nov","11","11"),
    "december": ("december","dec","12","12"),
}

_MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)

_WEEKDAY_RE = re.compile(
    r"\b(mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
    re.IGNORECASE,
)

_CURRENCY_WORDS = {
    "dollar":"usd","dollars":"usd","usd":"usd","$":"usd",
    "euro":"eur","euros":"eur","eur":"eur","€":"eur",
    "pound":"gbp","pounds":"gbp","gbp":"gbp","£":"gbp",
    "yen":"jpy","jpy":"jpy","¥":"jpy",
    "rupee":"inr","rupees":"inr","inr":"inr","₹":"inr",
}

_RE_QUOTED   = re.compile(r'(["\'])(.*?)\1')
_RE_NUMBER   = re.compile(r"\b\d+(?:\.\d+)?\b")
_RE_PERCENT  = re.compile(r"\b\d+(?:\.\d+)?%")
_RE_CURRENCY = re.compile(r"(?:[$€£₹¥]\s?\d[\d,]*(?:\.\d+)?)")
_RE_TIME     = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d(?:\s?(?:AM|PM|am|pm))?\b")
_RE_ISO_DATE = re.compile(r"\b(20\d{2}|19\d{2})[-/.](0?[1-9]|1[0-2])(?:[-/.](0?[1-9]|[12]\d|3[01]))?\b")


# -----------------------
# Utility helpers
# -----------------------

def _dedup(xs: Iterable[str]) -> List[str]:
    seen=set(); out=[]
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _basic_variants(s: str) -> List[str]:
    s1 = s.strip()
    no_punct = re.sub(r"[^\w\s]", "", s1)
    no_spaces = s1.replace(" ", "")
    hy2sp = s1.replace("-", " ")
    sp2hy = s1.replace(" ", "-")
    return _dedup([s1, s1.lower(), s1.title(), s1.upper(), no_punct, no_spaces, hy2sp, sp2hy])

def _month_key(tok: str) -> Optional[str]:
    t = tok.lower()
    m = {"jan":"january","feb":"february","mar":"march","apr":"april","jun":"june","jul":"july",
         "aug":"august","sep":"september","oct":"october","nov":"november","dec":"december"}
    if t in _MONTHS: return t
    if t in m: return m[t]
    return None

def _expand_month_variants(month_token: str) -> List[str]:
    key = _month_key(month_token)
    if not key: return [month_token]
    return list(dict.fromkeys(_MONTHS[key]))


def _char_span_to_token_span(token_char_spans: List[Tuple[int,int]], span: Tuple[int,int]) -> Tuple[int,int,List[int]]:
    """
    Map a character span to a contiguous token span (start_i, end_i_excl) and explicit token indices.

    We include tokens whose character spans overlap the literal span.
    """
    a, b = span
    covered = []
    for i, (ta, tb) in enumerate(token_char_spans):
        # overlap if intervals intersect
        if tb <= a:
            continue
        if ta >= b:
            continue
        covered.append(i)

    if not covered:
        return (-1, -1, [])

    start = min(covered)
    end_excl = max(covered) + 1
    # ensure contiguity
    token_indices = list(range(start, end_excl))
    return (start, end_excl, token_indices)


# -----------------------
# NER (spaCy)
# -----------------------

@lru_cache(maxsize=1)
def _get_spacy_model_cached():
    # reuse dependency_tree's loader to stay consistent
    return get_spacy_model("en_core_web_trf")


def _ner_spacy(q: str, dpt: DependencyParse) -> List[Literal]:
    try:
        import spacy
        try:
            spacy.prefer_gpu()
        except Exception:
            pass
    except Exception:
        pass

    nlp = _get_spacy_model_cached()
    doc = nlp(q)

    map_lbl = {
        "PERSON":"person","ORG":"org","GPE":"location","LOC":"location","FAC":"location",
        "DATE":"date","TIME":"time","MONEY":"money","PERCENT":"percent","QUANTITY":"quantity",
        "NORP":"misc","EVENT":"misc","PRODUCT":"misc","LAW":"misc","LANGUAGE":"misc",
        "CARDINAL":"number","ORDINAL":"number"
    }

    out: List[Literal] = []
    for ent in doc.ents:
        typ = map_lbl.get(ent.label_, "misc")
        text = ent.text
        char_span = (ent.start_char, ent.end_char)

        # Prefer token indices from *dependency parse* via char alignment for consistency
        t0, t1, tids = _char_span_to_token_span(dpt.token_char_spans, char_span)

        out.append(Literal(
            text=text,
            typ=typ,
            norm=text.lower(),
            variants=_basic_variants(text),
            char_span=char_span,
            token_span=(t0, t1),
            token_indices=tids,
            source="ner",
        ))
    return out


# -----------------------
# Rule extraction (your logic, but aligned to token indices)
# -----------------------

def _rules_enrich(q: str, dpt: DependencyParse, existing_char_spans: List[Tuple[int,int]], max_variants_per_literal: int) -> List[Literal]:
    spanset = {(a,b) for (a,b) in existing_char_spans}
    lits: List[Literal] = []

    def add(text: str, typ: str, char_span: Tuple[int,int], variants: List[str]):
        if char_span in spanset:
            return
        t0, t1, tids = _char_span_to_token_span(dpt.token_char_spans, char_span)
        lits.append(Literal(
            text=text,
            typ=typ,
            norm=text.lower(),
            variants=_dedup(variants)[:max_variants_per_literal],
            char_span=char_span,
            token_span=(t0, t1),
            token_indices=tids,
            source="rule",
        ))

    # 1) quoted strings
    for m in _RE_QUOTED.finditer(q):
        s = (m.group(2) or "").strip()
        if s:
            add(s, "misc", (m.start(2), m.end(2)), _basic_variants(s))

    # 2) currency amounts like $100
    for m in _RE_CURRENCY.finditer(q):
        s = m.group(0)
        cur = "usd" if "$" in s else "eur" if "€" in s else "gbp" if "£" in s else "inr" if "₹" in s else "jpy" if "¥" in s else "money"
        add(s, "money", (m.start(), m.end()), [s, cur] + _basic_variants(s))

    # 3) percent
    for m in _RE_PERCENT.finditer(q):
        s = m.group(0)
        add(s, "percent", (m.start(), m.end()), [s])

    # 4) ISO dates
    for m in _RE_ISO_DATE.finditer(q):
        s = m.group(0)
        add(s, "date", (m.start(), m.end()), _basic_variants(s))

    # 5) plain numbers
    for m in _RE_NUMBER.finditer(q):
        s = m.group(0)
        add(s, "number", (m.start(), m.end()), [s])

    # 6) time
    for m in _RE_TIME.finditer(q):
        s = m.group(0)
        add(s, "time", (m.start(), m.end()), [s, s.lower()])

    # 7) month names (+ month variants)
    for m in _MONTH_RE.finditer(q):
        s = m.group(1)
        add(s, "date", (m.start(1), m.end(1)), _expand_month_variants(s) + _basic_variants(s))

    # 8) weekdays
    for m in _WEEKDAY_RE.finditer(q):
        s = m.group(1)
        add(s, "date", (m.start(1), m.end(1)), [s, s.capitalize()])

    # 9) currency words (dollars, eur, etc)
    for m in re.finditer(r"\b([A-Za-z]+)\b", q):
        w = m.group(1).lower()
        if w in _CURRENCY_WORDS:
            cur = _CURRENCY_WORDS[w]
            add(m.group(1), "money", (m.start(1), m.end(1)), [m.group(1), cur] + _basic_variants(m.group(1)))

    # 10) uppercase/codes like SME, CZK, A320
    for m in re.finditer(r"\b[A-Z0-9]{3,}\b", q):
        s = m.group(0)
        if s.lower() in _MONTHS:
            continue
        add(s, "misc", (m.start(), m.end()), _basic_variants(s))

    return lits


# -----------------------
# Main extraction API
# -----------------------

def extract_literals(
    q: str,
    dpt: Optional[DependencyParse] = None,
    ner_backend: str = "spacy",   # "spacy" or "none"
    max_variants_per_literal: int = 6
) -> List[Literal]:
    """
    Returns deduplicated literals with:
      - char_span
      - token_span + token_indices aligned to the dependency tree token indices.
    """
    if dpt is None:
        dpt = dependency_parse(q)

    ner_lits: List[Literal] = []
    if ner_backend == "spacy":
        ner_lits = _ner_spacy(q, dpt)
    elif ner_backend == "none":
        ner_lits = []
    else:
        raise ValueError("ner_backend must be 'spacy' or 'none'")

    ner_char_spans = [L.char_span for L in ner_lits]
    rule_lits = _rules_enrich(q, dpt, ner_char_spans, max_variants_per_literal=max_variants_per_literal)

    # Merge by (text.lower(), typ). Keep the earliest token_span if conflicts.
    merged: Dict[Tuple[str, str], Literal] = {}
    for L in ner_lits + rule_lits:
        key = (L.text.lower(), L.typ)
        if key not in merged:
            merged[key] = L
        else:
            merged[key].variants = _dedup(merged[key].variants + L.variants)[:max_variants_per_literal]
            # prefer a valid token span if previous was invalid
            if merged[key].token_span == (-1, -1) and L.token_span != (-1, -1):
                merged[key].token_span = L.token_span
                merged[key].token_indices = L.token_indices
                merged[key].char_span = L.char_span

    out = list(merged.values())
    out.sort(key=lambda x: (x.token_span[0] if x.token_span[0] >= 0 else 10**9, x.char_span[0], x.char_span[1]))
    return out


# -----------------------
# Optional demo
# -----------------------

if __name__ == "__main__":
    q = "What is the difference in the annual average consumption of the customers with the least amount of consumption paid in CZK for 2013 between SME and LAM?"
    dpt = dependency_parse(q)
    lits = extract_literals(q, dpt=dpt, ner_backend="spacy", max_variants_per_literal=6)
    for L in lits:
        print(f"{L.text!r:>12}  type={L.typ:<8} token_span={L.token_span} token_indices={L.token_indices} char_span={L.char_span} source={L.source}")
