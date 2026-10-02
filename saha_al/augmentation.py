"""
SAHA-AL Data Augmentation Module
═══════════════════════════════════════════════════════════════════════
Expands the gold-standard dataset using three complementary, peer-
reviewed augmentation strategies.  Applied ONLY to verified gold
entries to prevent error propagation.

Strategies implemented
──────────────────────
1. Entity Swap      (Dai & Adel, COLING 2020)          ×4
   → Swap every PII span with a fresh Faker replacement while
     keeping the surrounding context verbatim.  Forces the model
     to learn positional/contextual patterns instead of memorising
     specific entity values.

2. Template Fill    (Anaby-Tavor et al., AAAI 2020)    ×5
   → Build an entity pool from gold data, define diverse sentence
     templates, and fill them with randomly sampled entities.
     Produces novel entity *combinations* in novel *structures*.

3. EDA – Easy Data Augmentation  (Wei & Zou, EMNLP 2019)  ×3
   → Four lightweight word-level ops on NON-PII context:
       a) Synonym replacement   b) Random insertion
       c) Random swap           d) Random deletion
     Entity spans are *never* touched.  Teaches the model that
     the same PII pattern appears in varied context phrasings.

Usage
─────
    python scripts/run_augmentation.py --strategy all
    python scripts/run_augmentation.py --strategy swap --multiplier 6
    python scripts/run_augmentation.py --strategy template --count 10000
    python scripts/run_augmentation.py --strategy eda --multiplier 5
"""

import argparse
import copy
import json
import logging
import os
import random
import re
from collections import Counter
from datetime import datetime

from saha_al.config import (
    GOLD_STANDARD_PATH,
    AUGMENTED_DATA_PATH,
    TRAINING_DATA_PATH,
    ENTITY_TYPES,
    AUG_ENTITY_SWAP_MULTIPLIER,
    AUG_TEMPLATE_FILL_COUNT,
    AUG_EDA_MULTIPLIER,
    AUG_EDA_ALPHA,
    AUG_BACKTRANSL_MULTIPLIER,
    AUG_BACKTRANSL_PIVOT_LANG,
    AUG_CONTEXTUAL_MLM_MULTIPLIER,
    AUG_CONTEXTUAL_MLM_PROB,
)
from saha_al.utils.io_helpers import read_jsonl, write_jsonl
from saha_al.utils.faker_replacements import generate_replacements

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Augmentation] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ─── reproducibility ────────────────────────────────────────────────
random.seed(42)


# ═════════════════════════════════════════════════════════════════════
#  STRATEGY 1 :  Entity Swap   (Dai & Adel, COLING 2020)
# ═════════════════════════════════════════════════════════════════════

def augment_entity_swap(entry: dict, n_variants: int = 4) -> list:
    """
    Generate *n_variants* copies of an entry where every detected
    entity is replaced with a **new** Faker value of the same type.
    The surrounding context text is kept verbatim.

    Why it works
    ────────────
    Given "My SSN is 123-45-6789", the model could memorise the
    specific digit sequence.  After swapping → "My SSN is 847-29-3156"
    the model learns that *any* XXX-XX-XXXX sequence in this position
    is an SSN.
    """
    original_text = entry.get("original_text", "")
    entities = entry.get("entities", [])

    if not entities or not original_text.strip():
        return []

    variants = []
    for vi in range(n_variants):
        # Sort entities right-to-left so replacements don't shift offsets
        sorted_ents = sorted(
            [dict(e) for e in entities],
            key=lambda e: e.get("start", 0),
            reverse=True,
        )

        new_text = original_text
        new_entities = []
        replacement_map = {}

        for ent in sorted_ents:
            etype = ent.get("label", "")
            pii_text = ent.get("value", "")
            start = ent.get("start", 0)
            end = ent.get("end", 0)

            # Generate a single fresh replacement
            try:
                candidates = generate_replacements(etype, pii_text, n=1)
                new_val = candidates[0] if candidates else pii_text
            except Exception:
                new_val = pii_text

            # Perform the swap in-text
            new_text = new_text[:start] + new_val + new_text[end:]

            # Build updated entity record
            new_ent = dict(ent)
            new_ent["value"] = new_val
            new_ent["end"] = start + len(new_val)
            new_entities.append(new_ent)

            replacement_map[pii_text] = new_val

        # Entities were collected in reverse order – flip them
        new_entities.reverse()

        # Re-search to fix cascaded offset drift
        new_entities = _reanchor_entities(new_text, new_entities)

        # Build anonymised version (swap again with fresh Faker values)
        anon_text, anon_map = _build_anon_text(new_text, new_entities)

        variants.append({
            "entry_id": f"{entry.get('entry_id')}_swap_{vi}",
            "original_text": new_text,
            "anonymized_text": anon_text,
            "entities": new_entities,
            "replacements": anon_map,
            "metadata": {
                "augmentation_strategy": "entity_swap",
                "augmentation_ref": "Dai & Adel, COLING 2020",
                "source_entry_id": entry.get("entry_id"),
                "variant_index": vi,
                "swap_map": replacement_map,
                "augmented_at": datetime.now().isoformat(),
            },
        })

    return variants


# ═════════════════════════════════════════════════════════════════════
#  STRATEGY 2 :  Template Fill   (Anaby-Tavor et al., AAAI 2020)
# ═════════════════════════════════════════════════════════════════════

# 40 hand-crafted templates covering immigration, legal, financial,
# medical, administrative, and general PII-bearing contexts.
TEMPLATES = [
    # ── immigration / legal ──
    "{FULLNAME} (ID: {ID_NUMBER}) has an appointment on {DATE} at {ADDRESS}.",
    "Please contact {FULLNAME} at {EMAIL} regarding case {ID_NUMBER}.",
    "Passport {PASSPORT} for {FULLNAME} from {LOCATION} expires on {DATE}.",
    "{FULLNAME} used {PHONE} to schedule a hearing on {DATE}.",
    "To: {EMAIL}. Subject: Document verification for {FULLNAME} (ref {ID_NUMBER}).",
    "A memo has been sent to {FULLNAME} regarding the interview on {DATE} at {ADDRESS}.",
    "{FULLNAME}'s file (ref: {ID_NUMBER}) was last updated on {DATE}.",
    "Emergency contact for {FULLNAME}: {PHONE} or {EMAIL}.",

    # ── financial ──
    "Account {ACCOUNT_NUMBER} belonging to {FULLNAME} was flagged on {DATE}.",
    "{FULLNAME} transferred funds to account {ACCOUNT_NUMBER} on {DATE}.",
    "Credit card {CREDIT_CARD} associated with {FULLNAME} expires {DATE}.",
    "Please verify SSN {SSN} for {FULLNAME} before processing the claim.",
    "{FULLNAME}'s tax return (SSN: {SSN}) was filed on {DATE}.",
    "Transfer from {FULLNAME}'s account {ACCOUNT_NUMBER} to {ORGANIZATION} on {DATE}.",
    "{ORGANIZATION} charged card {CREDIT_CARD} for {FULLNAME} on {DATE}.",

    # ── general administrative ──
    "{FULLNAME} lives at {ADDRESS}. Contact: {PHONE}.",
    "Dear {FULLNAME}, your reservation for {DATE} at {ADDRESS} is confirmed.",
    "{FULLNAME} registered with email {EMAIL} and phone {PHONE}.",
    "The event on {DATE} at {ADDRESS} is organized by {ORGANIZATION}.",
    "{FULLNAME} is a member of {ORGANIZATION} since {DATE}.",
    "{FULLNAME} from {LOCATION} called {PHONE} about {ORGANIZATION}.",

    # ── single/dual entity (simpler patterns) ──
    "The file for {FULLNAME} has been updated.",
    "Please call {PHONE} for more information.",
    "Send all correspondence to {EMAIL}.",
    "Your reference number is {ID_NUMBER}.",
    "The deadline is {DATE}.",
    "{FULLNAME} can be reached at {PHONE}.",
    "{FULLNAME}'s email is {EMAIL}.",
    "On {DATE}, {FULLNAME} submitted the documents.",
    "{FULLNAME} works at {ORGANIZATION}.",
    "{EMAIL} is the contact for {ORGANIZATION}.",

    # ── complex multi-entity ──
    "{FULLNAME} (SSN: {SSN}, Phone: {PHONE}) registered on {DATE}.",
    "Dear {FULLNAME}, your account {ACCOUNT_NUMBER} at {ORGANIZATION} was updated on {DATE}.",
    "{FULLNAME} ({EMAIL}, {PHONE}) submitted application {ID_NUMBER} on {DATE}.",
    "{FULLNAME} at {ADDRESS}, {LOCATION} can be reached at {PHONE} or {EMAIL}.",
    "Meeting between {FULLNAME} and {ORGANIZATION} scheduled for {DATE} at {ADDRESS}.",
    "Invoice for {FULLNAME} (card: {CREDIT_CARD}) from {ORGANIZATION}, dated {DATE}.",
    "Record: {FULLNAME}, DOB {DATE}, ID {ID_NUMBER}, residing at {ADDRESS}, {LOCATION}.",
    "Patient {FULLNAME} (ID: {ID_NUMBER}) visited {ORGANIZATION} on {DATE}.",
    "{ORGANIZATION} sent {FULLNAME} a notice at {ADDRESS} on {DATE} via {EMAIL}.",
]


def _build_entity_pools(gold_data: list) -> dict:
    """
    Collect unique entity values from gold standard grouped by type.
    Returns { entity_type: [value_str, ...], ... }
    """
    pools: dict[str, list[str]] = {et: [] for et in ENTITY_TYPES}
    seen: dict[str, set[str]] = {et: set() for et in ENTITY_TYPES}

    for entry in gold_data:
        for ent in entry.get("entities", []):
            et = ent.get("label", "")
            val = ent.get("value", "").strip()
            if val and et in pools and val not in seen[et]:
                pools[et].append(val)
                seen[et].add(val)

    return pools


def _expand_pools_with_faker(pools: dict, n_extra: int = 30) -> dict:
    """Top-up small pools with Faker-generated values."""
    expandable = [
        "FULLNAME", "FIRST_NAME", "LAST_NAME", "PHONE", "EMAIL",
        "ADDRESS", "DATE", "ORGANIZATION", "LOCATION",
    ]
    for et in expandable:
        if et not in pools:
            pools[et] = []
        sample = pools[et][0] if pools[et] else "placeholder"
        try:
            extras = generate_replacements(et, sample, n=n_extra)
            for v in extras:
                if v and v not in pools[et]:
                    pools[et].append(v)
        except Exception:
            pass
    return pools


def augment_template_fill(
    gold_data: list,
    n_entries: int = 5000,
    entity_pools: dict | None = None,
) -> list:
    """
    Generate brand-new entries by sampling entities from pools and
    plugging them into sentence templates.

    Why it works
    ────────────
    In the gold data, entity *X* only co-occurs with entity *Y*.
    Template fill breaks those spurious correlations by putting X
    next to completely unrelated entities in a novel sentence
    structure the model hasn't seen before.
    """
    if entity_pools is None:
        entity_pools = _build_entity_pools(gold_data)
    entity_pools = _expand_pools_with_faker(entity_pools)

    generated = []
    attempts = 0
    max_attempts = n_entries * 5

    while len(generated) < n_entries and attempts < max_attempts:
        attempts += 1
        template = random.choice(TEMPLATES)
        required = re.findall(r"\{(\w+)\}", template)

        # Check all required types have non-empty pools
        if not all(required_type in entity_pools and entity_pools[required_type]
                    for required_type in required):
            continue

        # Fill placeholders
        text = template
        chosen_values: dict[str, str] = {}
        used: set[str] = set()

        for rtype in required:
            pool = entity_pools[rtype]
            # Pick unique value (avoid duplicates within one entry)
            candidates = [v for v in pool if v not in used]
            if not candidates:
                candidates = pool
            val = random.choice(candidates)
            used.add(val)
            chosen_values[rtype] = val
            text = text.replace("{" + rtype + "}", val, 1)

        # Build entity list with offsets
        entities = []
        for rtype, val in chosen_values.items():
            start = text.find(val)
            if start >= 0:
                entities.append({
                    "value": val,
                    "start": start,
                    "end": start + len(val),
                    "label": rtype,
                })
        entities.sort(key=lambda e: e["start"])

        # Build anonymised version
        anon_text, anon_map = _build_anon_text(text, entities)

        generated.append({
            "entry_id": f"tmpl_{len(generated)}",
            "original_text": text,
            "anonymized_text": anon_text,
            "entities": entities,
            "replacements": anon_map,
            "metadata": {
                "augmentation_strategy": "template_fill",
                "augmentation_ref": "Anaby-Tavor et al., AAAI 2020",
                "template_used": template,
                "augmented_at": datetime.now().isoformat(),
            },
        })

    return generated


# ═════════════════════════════════════════════════════════════════════
#  STRATEGY 3 :  EDA – Easy Data Augmentation  (Wei & Zou, EMNLP 2019)
# ═════════════════════════════════════════════════════════════════════
#
# Four word-level operations applied ONLY to the non-entity context:
#   SR – Synonym Replacement
#   RI – Random Insertion  (insert a synonym of a random word nearby)
#   RS – Random Swap       (swap two context words)
#   RD – Random Deletion   (delete a context word with prob α)
#
# Entity spans are preserved *exactly* (text + offsets recalculated).

# Tiny synonym dict – no external WordNet dependency needed.
_SYNONYMS: dict[str, list[str]] = {
    "send": ["forward", "transmit", "dispatch", "deliver"],
    "contact": ["reach", "call", "notify", "message"],
    "please": ["kindly", "we request you to", "we ask that you"],
    "need": ["require", "must have", "should provide"],
    "bring": ["provide", "present", "carry", "supply"],
    "meet": ["see", "visit", "have a session with"],
    "discuss": ["review", "talk about", "go over", "examine"],
    "schedule": ["arrange", "set up", "plan", "organize"],
    "confirm": ["verify", "validate", "check", "affirm"],
    "submit": ["file", "hand in", "turn in", "deliver"],
    "request": ["ask for", "seek", "demand"],
    "provide": ["supply", "furnish", "give", "offer"],
    "review": ["examine", "check", "inspect", "assess"],
    "attend": ["be present at", "join", "go to", "participate in"],
    "regarding": ["about", "concerning", "with respect to", "relating to"],
    "attached": ["included", "enclosed", "appended", "added"],
    "new": ["updated", "revised", "recent", "latest"],
    "upcoming": ["next", "approaching", "forthcoming", "coming"],
    "important": ["critical", "essential", "vital", "key"],
    "information": ["details", "data", "records", "particulars"],
    "document": ["file", "record", "form", "paper"],
    "application": ["request", "form", "submission", "petition"],
    "required": ["needed", "necessary", "mandatory", "obligatory"],
    "verify": ["confirm", "validate", "authenticate", "check"],
    "update": ["revise", "modify", "change", "amend"],
    "located": ["situated", "found", "based", "positioned"],
    "process": ["handle", "manage", "deal with", "complete"],
    "issue": ["problem", "matter", "concern", "case"],
    "receive": ["get", "obtain", "acquire", "be given"],
    "additional": ["extra", "further", "more", "supplementary"],
    "available": ["accessible", "obtainable", "on hand"],
    "current": ["present", "existing", "ongoing"],
    "previous": ["prior", "earlier", "former", "past"],
    "complete": ["finish", "finalize", "conclude", "wrap up"],
    "immediately": ["at once", "right away", "promptly", "without delay"],
    "appointment": ["meeting", "session", "consultation", "visit"],
    "deadline": ["due date", "cutoff", "time limit"],
    "Dear": ["Hello", "Hi", "Greetings"],
    "registration": ["enrollment", "sign-up"],
    "called": ["phoned", "rang", "contacted", "reached out to"],
    "lives": ["resides", "stays", "is located"],
    "works": ["is employed", "operates", "serves"],
}


def _get_synonym(word: str) -> str | None:
    """Return a random synonym for *word*, or None."""
    key = word.lower().strip(".,;:!?")
    syns = _SYNONYMS.get(key)
    if syns:
        chosen = random.choice(syns)
        # Match original capitalisation
        if word[0].isupper():
            chosen = chosen[0].upper() + chosen[1:]
        return chosen
    return None


def _eda_synonym_replace(tokens: list[str], alpha: float) -> list[str]:
    """SR: Replace ≈ α * n random context words with synonyms."""
    n_changes = max(1, int(alpha * len(tokens)))
    indices = list(range(len(tokens)))
    random.shuffle(indices)
    changed = 0
    new_tokens = list(tokens)
    for i in indices:
        if changed >= n_changes:
            break
        syn = _get_synonym(tokens[i])
        if syn:
            new_tokens[i] = syn
            changed += 1
    return new_tokens


def _eda_random_insertion(tokens: list[str], alpha: float) -> list[str]:
    """RI: Insert ≈ α * n synonyms of random words at random positions."""
    n_inserts = max(1, int(alpha * len(tokens)))
    new_tokens = list(tokens)
    for _ in range(n_inserts):
        # Pick a random word, get its synonym, insert at random pos
        word = random.choice(tokens)
        syn = _get_synonym(word)
        if syn:
            pos = random.randint(0, len(new_tokens))
            new_tokens.insert(pos, syn)
    return new_tokens


def _eda_random_swap(tokens: list[str], alpha: float) -> list[str]:
    """RS: Swap ≈ α * n pairs of random words."""
    n_swaps = max(1, int(alpha * len(tokens)))
    new_tokens = list(tokens)
    for _ in range(n_swaps):
        if len(new_tokens) < 2:
            break
        i, j = random.sample(range(len(new_tokens)), 2)
        new_tokens[i], new_tokens[j] = new_tokens[j], new_tokens[i]
    return new_tokens


def _eda_random_deletion(tokens: list[str], alpha: float) -> list[str]:
    """RD: Delete each word with probability α."""
    if len(tokens) <= 1:
        return tokens
    new_tokens = [t for t in tokens if random.random() > alpha]
    return new_tokens if new_tokens else [random.choice(tokens)]


# EDA operations registry
_EDA_OPS = [
    _eda_synonym_replace,
    _eda_random_insertion,
    _eda_random_swap,
    _eda_random_deletion,
]


def augment_eda(entry: dict, n_variants: int = 3, alpha: float = 0.1) -> list:
    """
    Generate *n_variants* of an entry by applying one random EDA
    operation to each non-entity context region.

    Entity spans are left untouched; only the surrounding words
    change.  After modification the entity character offsets are
    recalculated by exact string search.

    Why it works
    ────────────
    "Please send to user@mail.com for verification"
    → "Kindly forward to user@mail.com to be verified"
    The model learns that EMAIL is signalled by its format, not by
    the specific surrounding words "please send to … for verification".
    """
    original_text = entry.get("original_text", "")
    entities = entry.get("entities", [])

    if not original_text.strip():
        return []

    # Identify entity spans (sorted by start)
    ent_spans = sorted(
        [(e.get("start", 0), e.get("end", 0), e) for e in entities],
        key=lambda x: x[0],
    )

    # Decompose text into (region_text, is_entity, entity_or_None) segments
    segments = _decompose_text(original_text, ent_spans)

    variants = []
    for vi in range(n_variants):
        eda_op = random.choice(_EDA_OPS)
        new_segments = []
        ops_applied = []

        for seg_text, is_entity, ent_ref in segments:
            if is_entity:
                # Keep entity text verbatim
                new_segments.append((seg_text, True, ent_ref))
            else:
                # Tokenise non-entity region and apply EDA
                tokens = seg_text.split()
                if len(tokens) >= 2:
                    new_tokens = eda_op(tokens, alpha)
                    new_seg = " ".join(new_tokens)
                    if new_seg != seg_text:
                        ops_applied.append({
                            "op": eda_op.__name__,
                            "before": seg_text[:60],
                            "after": new_seg[:60],
                        })
                    new_segments.append((new_seg, False, None))
                else:
                    new_segments.append((seg_text, False, None))

        if not ops_applied:
            # No changes were made — skip this variant
            continue

        # Reassemble text and recalculate entity offsets
        new_text = ""
        new_entities = []
        for seg_text, is_entity, ent_ref in new_segments:
            if is_entity and ent_ref is not None:
                new_ent = dict(ent_ref)
                new_ent["start"] = len(new_text)
                new_ent["end"] = len(new_text) + len(seg_text)
                new_entities.append(new_ent)
            new_text += seg_text

        # Build anonymised version
        anon_text, anon_map = _build_anon_text(new_text, new_entities)

        variants.append({
            "entry_id": f"{entry.get('entry_id')}_eda_{vi}",
            "original_text": new_text,
            "anonymized_text": anon_text,
            "entities": new_entities,
            "replacements": anon_map,
            "metadata": {
                "augmentation_strategy": "eda",
                "augmentation_ref": "Wei & Zou, EMNLP 2019",
                "source_entry_id": entry.get("entry_id"),
                "variant_index": vi,
                "eda_operations": ops_applied,
                "alpha": alpha,
                "augmented_at": datetime.now().isoformat(),
            },
        })

    return variants


# ═════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════

def _decompose_text(
    text: str,
    ent_spans: list[tuple[int, int, dict]],
) -> list[tuple[str, bool, dict | None]]:
    """
    Split *text* into alternating (non-entity, entity) segments,
    preserving exact character boundaries.
    """
    segments = []
    prev_end = 0
    for start, end, ent in ent_spans:
        if start > prev_end:
            segments.append((text[prev_end:start], False, None))
        segments.append((text[start:end], True, ent))
        prev_end = end
    if prev_end < len(text):
        segments.append((text[prev_end:], False, None))
    return segments


def _reanchor_entities(text: str, entities: list[dict]) -> list[dict]:
    """
    Re-find each entity's text in *text* to fix cascaded offset drift
    after multi-entity swaps.  Uses sequential scanning to avoid
    matching the same span twice.
    """
    anchored = []
    search_from = 0
    for ent in sorted(entities, key=lambda e: e.get("start", 0)):
        val = ent.get("value", "")
        if not val:
            continue
        pos = text.find(val, search_from)
        if pos >= 0:
            ent["start"] = pos
            ent["end"] = pos + len(val)
            anchored.append(ent)
            search_from = pos + len(val)
    return anchored


def _build_anon_text(
    text: str,
    entities: list[dict],
) -> tuple[str, dict]:
    """
    Build an anonymised version of *text* by replacing every entity
    with a fresh Faker value.  Returns (anonymised_text, replacement_map).
    """
    sorted_ents = sorted(entities, key=lambda e: e.get("start", 0), reverse=True)
    anon = text
    rmap: dict[str, str] = {}
    for ent in sorted_ents:
        pii = ent.get("value", "")
        et = ent.get("label", "")
        start = ent.get("start", 0)
        end = ent.get("end", 0)
        if pii not in rmap:
            try:
                cands = generate_replacements(et, pii, n=1)
                rmap[pii] = cands[0] if cands else f"[{et}]"
            except Exception:
                rmap[pii] = f"[{et}]"
        anon = anon[:start] + rmap[pii] + anon[end:]
    return anon, rmap


# ═════════════════════════════════════════════════════════════════════
#  STRATEGY 4 :  Back-Translation   (Sennrich et al., WMT 2016)
# ═════════════════════════════════════════════════════════════════════
#
# Round-trip translate NON-ENTITY context through a pivot language
# (en→de→en) to obtain a natural paraphrase.  Entity spans are
# replaced with placeholders before translation and restored after,
# so PII values are never altered.

# Lazy-loaded translation models (heavy — only loaded if strategy is used)
_bt_models = {}


def _load_bt_model(direction: str):
    """Lazy-load a MarianMT model for a given direction (e.g. 'en-de')."""
    if direction in _bt_models:
        return _bt_models[direction]
    try:
        from transformers import MarianMTModel, MarianTokenizer
        model_name = f"Helsinki-NLP/opus-mt-{direction}"
        log.info(f"  Loading translation model: {model_name}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _bt_models[direction] = (tokenizer, model)
        return tokenizer, model
    except Exception as e:
        log.warning(f"  Failed to load {direction} model: {e}")
        return None


def _translate(text: str, direction: str) -> str:
    """Translate text using MarianMT."""
    loaded = _load_bt_model(direction)
    if loaded is None:
        return text
    tokenizer, model = loaded
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        translated = model.generate(**inputs, max_length=512, num_beams=4)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def augment_back_translation(
    entry: dict,
    n_variants: int = 2,
    pivot_lang: str = "de",
) -> list:
    """
    Generate *n_variants* paraphrases of non-entity context via
    round-trip translation: en → pivot → en.

    Entity spans are shielded with XML-style placeholders before
    translation, then restored after.  This produces natural
    context rephrasing while preserving all PII exactly.

    Why it works
    ────────────
    "Please contact John Smith at john@email.com regarding case X23"
    → (de) "Bitte kontaktieren Sie <E0> unter <E1> bezüglich Fall <E2>"
    → (en) "Please get in touch with John Smith at john@email.com about case X23"
    The model learns that different phrasings of the same context
    still contain the same PII pattern.
    """
    original_text = entry.get("original_text", "")
    entities = entry.get("entities", [])

    if not original_text.strip():
        return []

    # Sort entities by start position
    sorted_ents = sorted(
        [dict(e) for e in entities],
        key=lambda e: e.get("start", 0),
    )

    variants = []
    for vi in range(n_variants):
        # Step 1: Replace entity spans with placeholders (right-to-left)
        shielded = original_text
        placeholder_map = {}  # placeholder → (original_value, entity_dict)
        for i, ent in enumerate(reversed(sorted_ents)):
            start = ent.get("start", 0)
            end = ent.get("end", 0)
            value = ent.get("value", "")
            placeholder = f"PIIENT{len(sorted_ents) - 1 - i}"
            shielded = shielded[:start] + placeholder + shielded[end:]
            placeholder_map[placeholder] = (value, ent)

        # Step 2: Round-trip translate
        try:
            translated = _translate(shielded, f"en-{pivot_lang}")
            back_translated = _translate(translated, f"{pivot_lang}-en")
        except Exception:
            continue

        # Step 3: Restore entity values from placeholders
        restored = back_translated
        for placeholder, (value, _) in placeholder_map.items():
            restored = restored.replace(placeholder, value)

        # Skip if back-translation didn't change anything meaningful
        if restored.strip() == original_text.strip():
            continue

        # Step 4: Re-anchor entities by searching for their values
        new_entities = []
        search_from = 0
        for ent in sorted_ents:
            val = ent.get("value", "")
            if not val:
                continue
            pos = restored.find(val, search_from)
            if pos >= 0:
                new_ent = dict(ent)
                new_ent["start"] = pos
                new_ent["end"] = pos + len(val)
                new_entities.append(new_ent)
                search_from = pos + len(val)

        # Build anonymised version
        anon_text, anon_map = _build_anon_text(restored, new_entities)

        variants.append({
            "entry_id": f"{entry.get('entry_id')}_bt_{vi}",
            "original_text": restored,
            "anonymized_text": anon_text,
            "entities": new_entities,
            "replacements": anon_map,
            "metadata": {
                "augmentation_strategy": "back_translation",
                "augmentation_ref": "Sennrich et al., WMT 2016",
                "source_entry_id": entry.get("entry_id"),
                "variant_index": vi,
                "pivot_language": pivot_lang,
                "augmented_at": datetime.now().isoformat(),
            },
        })

    return variants


# ═════════════════════════════════════════════════════════════════════
#  STRATEGY 5 :  Contextual MLM   (Kobayashi, ACL 2018)
# ═════════════════════════════════════════════════════════════════════
#
# Use a pretrained masked language model to replace non-entity context
# words with contextually appropriate alternatives.  Unlike EDA's
# static synonym dictionary, this produces *context-sensitive*
# replacements that maintain grammaticality and semantic coherence.

_mlm_pipeline = None


def _get_mlm_pipeline():
    """Lazy-load a fill-mask pipeline (distilroberta-base for speed)."""
    global _mlm_pipeline
    if _mlm_pipeline is not None:
        return _mlm_pipeline
    try:
        from transformers import pipeline
        log.info("  Loading MLM pipeline: distilroberta-base")
        _mlm_pipeline = pipeline("fill-mask", model="distilroberta-base",
                                  top_k=5, device=-1)  # CPU to avoid OOM
        return _mlm_pipeline
    except Exception as e:
        log.warning(f"  Failed to load MLM pipeline: {e}")
        return None


def augment_contextual_mlm(
    entry: dict,
    n_variants: int = 2,
    replace_prob: float = 0.15,
) -> list:
    """
    Generate *n_variants* of an entry by selectively masking non-entity
    context words and filling them with a pretrained MLM's predictions.

    Why it works
    ────────────
    "Please send the documents to John Smith at john@email.com"
    → "Kindly submit the paperwork to John Smith at john@email.com"
    Unlike EDA's static synonym table, the MLM generates replacements
    that are *contextually coherent* — it "understands" the sentence
    and picks words that fit the specific context.
    """
    original_text = entry.get("original_text", "")
    entities = entry.get("entities", [])

    if not original_text.strip():
        return []

    mlm = _get_mlm_pipeline()
    if mlm is None:
        return []

    mask_token = mlm.tokenizer.mask_token

    # Sort entities by start
    sorted_ents = sorted(
        [(e.get("start", 0), e.get("end", 0), e) for e in entities],
        key=lambda x: x[0],
    )

    # Decompose into segments
    segments = _decompose_text(original_text, sorted_ents)

    variants = []
    for vi in range(n_variants):
        new_segments = []
        any_change = False

        for seg_text, is_entity, ent_ref in segments:
            if is_entity:
                new_segments.append((seg_text, True, ent_ref))
                continue

            # Tokenise context and randomly select words to replace
            words = seg_text.split()
            if len(words) < 2:
                new_segments.append((seg_text, False, None))
                continue

            n_replace = max(1, int(replace_prob * len(words)))
            # Only replace content words (skip very short words)
            replaceable = [i for i, w in enumerate(words)
                           if len(w) > 2 and w.isalpha()]
            if not replaceable:
                new_segments.append((seg_text, False, None))
                continue

            to_replace = random.sample(
                replaceable, min(n_replace, len(replaceable))
            )

            new_words = list(words)
            for idx in to_replace:
                original_word = words[idx]
                # Build sentence with mask
                masked_words = list(words)
                masked_words[idx] = mask_token
                masked_sentence = " ".join(masked_words)

                try:
                    predictions = mlm(masked_sentence)
                    if predictions:
                        # Pick a random prediction from top-5 that differs
                        candidates = [
                            p["token_str"].strip()
                            for p in predictions
                            if p["token_str"].strip().lower() != original_word.lower()
                            and p["token_str"].strip().isalpha()
                        ]
                        if candidates:
                            replacement = random.choice(candidates[:3])
                            # Preserve capitalisation
                            if original_word[0].isupper():
                                replacement = replacement[0].upper() + replacement[1:]
                            new_words[idx] = replacement
                            any_change = True
                except Exception:
                    pass  # Keep original word on failure

            new_segments.append((" ".join(new_words), False, None))

        if not any_change:
            continue

        # Reassemble text and recalculate entity offsets
        new_text = ""
        new_entities = []
        for seg_text, is_entity, ent_ref in new_segments:
            if is_entity and ent_ref is not None:
                new_ent = dict(ent_ref)
                new_ent["start"] = len(new_text)
                new_ent["end"] = len(new_text) + len(seg_text)
                new_entities.append(new_ent)
            new_text += seg_text

        # Build anonymised version
        anon_text, anon_map = _build_anon_text(new_text, new_entities)

        variants.append({
            "entry_id": f"{entry.get('entry_id')}_cmlm_{vi}",
            "original_text": new_text,
            "anonymized_text": anon_text,
            "entities": new_entities,
            "replacements": anon_map,
            "metadata": {
                "augmentation_strategy": "contextual_mlm",
                "augmentation_ref": "Kobayashi, ACL 2018",
                "source_entry_id": entry.get("entry_id"),
                "variant_index": vi,
                "replace_prob": replace_prob,
                "augmented_at": datetime.now().isoformat(),
            },
        })

    return variants


# ═════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════

def run_augmentation(
    strategy: str = "all",
    swap_multiplier: int | None = None,
    template_count: int | None = None,
    eda_multiplier: int | None = None,
    eda_alpha: float | None = None,
    bt_multiplier: int | None = None,
    bt_pivot_lang: str | None = None,
    cmlm_multiplier: int | None = None,
    cmlm_prob: float | None = None,
    output_path: str | None = None,
) -> list[dict]:
    """
    Run one or all augmentation strategies on the gold standard.

    Parameters
    ──────────
    strategy       "swap", "template", "eda", "backtransl", "cmlm", or "all"
    swap_multiplier  Overrides config AUG_ENTITY_SWAP_MULTIPLIER
    template_count   Overrides config AUG_TEMPLATE_FILL_COUNT
    eda_multiplier   Overrides config AUG_EDA_MULTIPLIER
    eda_alpha        Overrides config AUG_EDA_ALPHA
    bt_multiplier    Overrides config AUG_BACKTRANSL_MULTIPLIER
    bt_pivot_lang    Overrides config AUG_BACKTRANSL_PIVOT_LANG
    cmlm_multiplier  Overrides config AUG_CONTEXTUAL_MLM_MULTIPLIER
    cmlm_prob        Overrides config AUG_CONTEXTUAL_MLM_PROB
    output_path      Overrides config AUGMENTED_DATA_PATH
    """
    swap_mult = swap_multiplier or AUG_ENTITY_SWAP_MULTIPLIER
    tmpl_cnt = template_count or AUG_TEMPLATE_FILL_COUNT
    eda_mult = eda_multiplier or AUG_EDA_MULTIPLIER
    alpha = eda_alpha or AUG_EDA_ALPHA
    bt_mult = bt_multiplier or AUG_BACKTRANSL_MULTIPLIER
    bt_lang = bt_pivot_lang or AUG_BACKTRANSL_PIVOT_LANG
    cmlm_mult = cmlm_multiplier or AUG_CONTEXTUAL_MLM_MULTIPLIER
    cmlm_p = cmlm_prob or AUG_CONTEXTUAL_MLM_PROB
    out_path = output_path or AUGMENTED_DATA_PATH

    # ── Load gold data ──────────────────────────────────────────────
    log.info(f"Loading gold standard from {GOLD_STANDARD_PATH}")
    gold = read_jsonl(GOLD_STANDARD_PATH)
    if not gold:
        log.error("No gold standard data found — annotate first.")
        return []
    log.info(f"Loaded {len(gold)} verified gold entries")

    all_augmented: list[dict] = []

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    # ── 1. Entity Swap ──────────────────────────────────────────────
    if strategy in ("swap", "all"):
        log.info(f"{'═'*60}")
        log.info(f"Strategy 1: ENTITY SWAP  (×{swap_mult} per entry)")
        log.info(f"  Ref: Dai & Adel, COLING 2020")
        log.info(f"{'═'*60}")
        swap_out: list[dict] = []
        iterator = tqdm(gold, desc="Entity Swap") if tqdm else gold
        for entry in iterator:
            try:
                swap_out.extend(augment_entity_swap(entry, n_variants=swap_mult))
            except Exception as exc:
                log.debug(f"Swap failed for {entry.get('entry_id')}: {exc}")
        log.info(f"  → {len(swap_out)} entries generated")
        all_augmented.extend(swap_out)

    # ── 2. Template Fill ────────────────────────────────────────────
    if strategy in ("template", "all"):
        log.info(f"{'═'*60}")
        log.info(f"Strategy 2: TEMPLATE FILL  ({tmpl_cnt} entries)")
        log.info(f"  Ref: Anaby-Tavor et al., AAAI 2020")
        log.info(f"{'═'*60}")
        pools = _build_entity_pools(gold)
        pool_stats = {k: len(v) for k, v in pools.items() if v}
        log.info(f"  Entity pool sizes: {pool_stats}")
        tmpl_out = augment_template_fill(gold, n_entries=tmpl_cnt, entity_pools=pools)
        log.info(f"  → {len(tmpl_out)} entries generated")
        all_augmented.extend(tmpl_out)

    # ── 3. EDA ──────────────────────────────────────────────────────
    if strategy in ("eda", "all"):
        log.info(f"{'═'*60}")
        log.info(f"Strategy 3: EDA  (×{eda_mult}, α={alpha})")
        log.info(f"  Ref: Wei & Zou, EMNLP 2019")
        log.info(f"{'═'*60}")
        eda_out: list[dict] = []
        iterator = tqdm(gold, desc="EDA") if tqdm else gold
        for entry in iterator:
            try:
                eda_out.extend(augment_eda(entry, n_variants=eda_mult, alpha=alpha))
            except Exception as exc:
                log.debug(f"EDA failed for {entry.get('entry_id')}: {exc}")
        log.info(f"  → {len(eda_out)} entries generated")
        all_augmented.extend(eda_out)

    # ── 4. Back-Translation ─────────────────────────────────────────
    if strategy in ("backtransl", "all"):
        log.info(f"{'═'*60}")
        log.info(f"Strategy 4: BACK-TRANSLATION  (×{bt_mult}, pivot={bt_lang})")
        log.info(f"  Ref: Sennrich et al., WMT 2016")
        log.info(f"{'═'*60}")
        bt_out: list[dict] = []
        iterator = tqdm(gold, desc="Back-Translation") if tqdm else gold
        for entry in iterator:
            try:
                bt_out.extend(augment_back_translation(
                    entry, n_variants=bt_mult, pivot_lang=bt_lang))
            except Exception as exc:
                log.debug(f"BackTransl failed for {entry.get('entry_id')}: {exc}")
        log.info(f"  → {len(bt_out)} entries generated")
        all_augmented.extend(bt_out)

    # ── 5. Contextual MLM ──────────────────────────────────────────
    if strategy in ("cmlm", "all"):
        log.info(f"{'═'*60}")
        log.info(f"Strategy 5: CONTEXTUAL MLM  (×{cmlm_mult}, p={cmlm_p})")
        log.info(f"  Ref: Kobayashi, ACL 2018")
        log.info(f"{'═'*60}")
        cmlm_out: list[dict] = []
        iterator = tqdm(gold, desc="Contextual MLM") if tqdm else gold
        for entry in iterator:
            try:
                cmlm_out.extend(augment_contextual_mlm(
                    entry, n_variants=cmlm_mult, replace_prob=cmlm_p))
            except Exception as exc:
                log.debug(f"CMLM failed for {entry.get('entry_id')}: {exc}")
        log.info(f"  → {len(cmlm_out)} entries generated")
        all_augmented.extend(cmlm_out)

    # ── Write output ────────────────────────────────────────────────
    log.info(f"{'═'*60}")
    log.info(f"AUGMENTATION COMPLETE")
    log.info(f"{'═'*60}")
    log.info(f"  Gold entries:      {len(gold)}")
    log.info(f"  Augmented entries: {len(all_augmented)}")
    factor = len(all_augmented) / max(len(gold), 1)
    log.info(f"  Expansion factor:  {factor:.1f}×")

    write_jsonl(out_path, all_augmented)
    log.info(f"  Augmented data  → {out_path}")

    # ── Combine gold + augmented, shuffle, write training set ──────
    combined = gold + all_augmented
    random.shuffle(combined)
    write_jsonl(TRAINING_DATA_PATH, combined)
    log.info(f"  Training data   → {TRAINING_DATA_PATH}  ({len(combined)} total)")

    # ── Per-strategy breakdown ──────────────────────────────────────
    strat_counts = Counter(
        e.get("metadata", {}).get("augmentation_strategy", "?")
        for e in all_augmented
    )
    log.info("  Breakdown:")
    for s, c in strat_counts.most_common():
        log.info(f"    {s:20s}  {c:>7}")

    return all_augmented


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SAHA-AL Data Augmentation  (Entity Swap + Template Fill + EDA + Back-Translation + Contextual MLM)",
    )
    parser.add_argument(
        "--strategy",
        choices=["swap", "template", "eda", "backtransl", "cmlm", "all"],
        default="all",
        help="Which augmentation strategy to run (default: all)",
    )
    parser.add_argument(
        "--multiplier", type=int, default=None,
        help=f"Entity Swap / EDA / BackTransl / CMLM variants per entry (default: swap={AUG_ENTITY_SWAP_MULTIPLIER}, eda={AUG_EDA_MULTIPLIER})",
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help=f"Template Fill entry count (default: {AUG_TEMPLATE_FILL_COUNT})",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help=f"EDA α parameter — fraction of words changed (default: {AUG_EDA_ALPHA})",
    )
    parser.add_argument(
        "--pivot-lang", type=str, default=None,
        help=f"Back-Translation pivot language (default: {AUG_BACKTRANSL_PIVOT_LANG})",
    )
    parser.add_argument(
        "--cmlm-prob", type=float, default=None,
        help=f"Contextual MLM replacement probability (default: {AUG_CONTEXTUAL_MLM_PROB})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (default: data/augmented_data.jsonl)",
    )
    args = parser.parse_args()

    run_augmentation(
        strategy=args.strategy,
        swap_multiplier=args.multiplier,
        template_count=args.count,
        eda_multiplier=args.multiplier,
        eda_alpha=args.alpha,
        bt_multiplier=args.multiplier,
        bt_pivot_lang=args.pivot_lang,
        cmlm_multiplier=args.multiplier,
        cmlm_prob=args.cmlm_prob,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
