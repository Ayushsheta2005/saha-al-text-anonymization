#!/usr/bin/env python3
"""
Step 1 — Adversarial Dataset Generator
=======================================
Generates 40,000 PII-rich sentences using Gemini 2.5 Flash.
Implements 6 probing strategies for model inversion attack research.

Each generated sentence is enriched with dense metadata BY CODE
(not by Gemini) so every field is reliable and consistent.

Run:
    python3 generate_dataset.py

Resumes automatically if interrupted.
Output: output/adversarial_dataset_raw.jsonl
        → uploaded to HuggingFace: JALAPENO11/model-inversion-adversarial
"""

import os
import sys
import json
import time
import uuid
import random
import logging
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from collections import defaultdict

# ── google-genai SDK (pip install google-genai) ────────────────────────────
from google import genai
from google.genai import types as genai_types

# ── HuggingFace Hub ────────────────────────────────────────────────────────
from huggingface_hub import HfApi, login as hf_login

HF_TOKEN     = os.environ.get("HF_TOKEN", "")
HF_USER      = "JALAPENO11"
HF_REPO_ID   = f"{HF_USER}/model-inversion-adversarial"
HF_REPO_TYPE = "dataset"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMP, GEMINI_MAX_TOKENS,
    RPM_LIMIT, RPD_LIMIT,
    BATCH_SIZE, MAX_PARALLEL_WORKERS, TOTAL_TARGET, EVAL_FRACTION,
    STRATEGY_DISTRIBUTION, PROBE_NAMES, CONTEXT_DOMAINS, PII_COMBO_TYPES,
    OUTPUT_DIR, DATA_DIR, LOGS_DIR,
    GENERATED_DATASET_FILE, GENERATION_STATE_FILE,
)

# ── logging ────────────────────────────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "generate_dataset.log"), mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Silence the google-genai SDK's "AFC is enabled" spam —
# AFC = Automatic Function Calling, an SDK feature we don't use.
# It logs on every client init at INFO level by default.
logging.getLogger("google.genai.client").setLevel(logging.WARNING)
logging.getLogger("google.api_core").setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT  (sent with every call as context)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a synthetic data generator for NLP privacy research.

Your job: Generate realistic English sentences containing PII (Personally
Identifiable Information) that EXACTLY match the style of the ai4privacy dataset.

════════ CRITICAL STYLE RULES — READ CAREFULLY ════════

RULE 1 — NAME STYLE (most important):
Names must be SYNTHETIC MULTICULTURAL MASHUPS — combine syllables/fragments
from different world cultures. They must sound plausible but belong to no
real person.

CORRECT style (copy this pattern exactly):
  "Kukshi Welshet", "Amyna Tharathip", "Svenja-Maria Glayce",
  "Mbangila Migali Bajraj", "Qokaj Donnell", "Ljuljeta Shanon",
  "Betselot Mengsteab", "Yidnekachew Tsolakis", "Joévine Roliana",
  "Valentinetti Stravs Perwein", "Igorche Ramtin Eshekary",
  "Fauzieh Mostaghni", "Onome Akpovire", "Masooma Mirsada"

WRONG (never use these):
  ✗ "John Smith"   — too clean Western
  ✗ "Rahul Kumar"  — too clean Indian
  ✗ Any real celebrity or public figure

RULE 2 — PHONE FORMATS (globally varied, NOT just US):
Use these exact varied international formats:
  "+181 397 128 2418",  "018.4861 7786",  "0052 726359332",
  "+322-33.004-6734",   "0111-284596398", "68689 34395",
  "+4631910-3170",      "9683 174 1488",  "0057.642.765-7789",
  "+40-52372 3157",     "+74 67238.2613", "83264.02802"

RULE 3 — EMAIL HANDLES (short and realistic):
  "L@outlook.com",  "DG@aol.com",  "75HEB@hotmail.com",
  "keshascozzaro@yahoo.com",  "torbjancellotti@outlook.com",
  "vagfreyermut@gmail.com",   "EC@tutanota.com",  "LE@hotmail.com"
  Providers: gmail, yahoo, outlook, hotmail, protonmail, aol, tutanota

RULE 4 — SENTENCE TEMPLATES (use all 7 structures across your output):
  CHAT:    "[HH:MM] [Name]: '[casual message containing PII]'"
  FORMAL:  "Dear [Name], [formal request involving PHONE or EMAIL or ID]"
  TODO:    "Reminder/To-do: [Name] needs to [action] at [TIME] on [DATE]"
  RECORD:  "[Document/form type] for [Name]: [ID_NUMBER] and [DATE] required"
  INTRO:   "Please [welcome/confirm/register] [Name] [at/for] [context]"
  ORDER:   "[Action] for [item/service] for [PHONE or NAME] confirmed."
  MIXED:   "[Name] [verb] [Name2] about [topic] at [LOCATION/ADDRESS]"

RULE 5 — LENGTH: 10-25 words preferred (aim for 70% of output in this range)

RULE 6 — ID DOCUMENT formats (alphanumeric, varied):
  "FB36241EB", "WH08544JB", "0KZ3VPZ9S9", "4CKKGVJZ2L",
  "GW31236PD", "16340AG778", "MARGA7101160M9183", "XE7801373"

RULE 7 — PII TYPE DISTRIBUTION across any batch:
  ~40% sentences: NAME only (or NAME + TIME)
  ~20% sentences: NAME + PHONE
  ~15% sentences: NAME + EMAIL
  ~10% sentences: NAME + DATE
  ~10% sentences: PHONE or EMAIL only
  ~5%  sentences: NAME + ID_DOCUMENT

RULE 8 — DIVERSITY: Every sentence MUST use a unique opening structure.
  Never start two sentences the same way within a batch. Vary subject
  position, verb tense, sentence type (statement/question/imperative).
  Use a mix of active and passive voice. Vary sentence length (8-30 words).

════════ OUTPUT FORMAT ════════
Return ONLY a valid JSON array, no markdown, no explanation:
[
  {
    "sentence": "...",
    "probe_entity": "the PRIMARY PII entity in this sentence",
    "entity_type": "NAME|PHONE|EMAIL|DATE|ID_DOCUMENT|ADDRESS|TIME|ORG"
  },
  ...
]
"""

# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_last_sentences(last: List[str], n: int = 5) -> str:
    if not last:
        return "  (none yet)"
    return "\n".join(f"  • {s}" for s in last[-n:])


def _fmt_used_entities(used: List[str], n: int = 50) -> str:
    if not used:
        return "  (none yet)"
    shown = used[-n:] if len(used) > n else used
    return ", ".join(f'"{e}"' for e in shown)


def build_s1_prompt(
    probe_name: Dict,
    context_domain: str,
    batch_num: int,
    generated_so_far: int,
    last_sentences: List[str],
    used_probe_entities: List[str],
) -> str:
    name = probe_name["name"]
    origin = "+".join(probe_name["origin_mix"])
    rarity = probe_name["rarity"]

    return f"""\
SESSION STATE:
  Batch #{batch_num} | Generated so far: {generated_so_far:,} / 20,000
  Strategy: S1_entity_consistency
  Primary probe name: "{name}"  (origin: {origin}, rarity: {rarity})
  Current context domain: {context_domain}

Other probe names already used (do NOT use these as primary probe):
{_fmt_used_entities(used_probe_entities)}

Last {min(5,len(last_sentences))} sentences generated (DO NOT repeat structure):
{_fmt_last_sentences(last_sentences)}

════ TASK ════
Generate exactly {BATCH_SIZE} sentences.
RULES FOR THIS BATCH:
  • Every sentence MUST contain the name "{name}" as the primary PII entity.
  • Context domain: {context_domain}
    (all sentences should fit this domain's register and style)
  • Vary the sentence TEMPLATE across the batch (CHAT, FORMAL, TODO, RECORD, etc.)
  • Some sentences may contain additional PII (phone, email, date) alongside the name.
  • Do NOT generate the same sentence structure twice in this batch.
  • The name "{name}" must appear verbatim — exact spelling and capitalisation.

Return JSON array with {BATCH_SIZE} objects.
"""


def build_s2_prompt(
    combo_type: str,
    batch_num: int,
    generated_so_far: int,
    last_sentences: List[str],
    used_probe_entities: List[str],
) -> str:
    combo_instructions = {
        "NAME_PHONE":         "Include a synthetic multicultural name AND a phone number. No other PII.",
        "NAME_EMAIL":         "Include a synthetic multicultural name AND an email address. No other PII.",
        "NAME_DATE":          "Include a synthetic multicultural name AND a date (birth/appointment/event). No other PII.",
        "NAME_PHONE_EMAIL":   "Include a synthetic multicultural name, a phone number, AND an email. All three must appear.",
        "NAME_PHONE_DATE":    "Include a synthetic multicultural name, a phone number, AND a date. All three must appear.",
        "NAME_EMAIL_DATE":    "Include a synthetic multicultural name, an email address, AND a date. All three must appear.",
        "NAME_ID_DOCUMENT":   "Include a synthetic multicultural name AND an alphanumeric ID document number.",
        "NAME_ORG_DATE":      "Include a synthetic multicultural name, an organisation name, AND a date.",
        "PHONE_EMAIL":        "Include ONLY a phone number and an email address. NO person name in this sentence.",
        "NAME_ADDRESS_PHONE": "Include a name, a street address (road/street/avenue + number), AND a phone number.",
    }
    instruction = combo_instructions.get(combo_type, "Include multiple PII types.")

    return f"""\
SESSION STATE:
  Batch #{batch_num} | Generated so far: {generated_so_far:,} / 20,000
  Strategy: S2_combinatorial_pii
  PII combo type: {combo_type}

Probe entities already used (avoid reusing same names/phones/emails):
{_fmt_used_entities(used_probe_entities)}

Last {min(5,len(last_sentences))} sentences (DO NOT repeat structure):
{_fmt_last_sentences(last_sentences)}

════ TASK ════
Generate exactly {BATCH_SIZE} sentences.
COMBINATION RULE: {instruction}
  • Use different names, phones, emails, dates across all {BATCH_SIZE} sentences.
  • Vary sentence templates (CHAT, FORMAL, TODO, RECORD, MIXED, etc.)
  • Sentences should come from different context domains (medical, financial, chat, etc.)
  • All names must be multicultural mashup style (see system rules).

Return JSON array with {BATCH_SIZE} objects.
"""


def build_s3_prompt(
    entity_set: Dict,
    group_id: int,
    batch_num: int,
    generated_so_far: int,
    last_sentences: List[str],
) -> str:
    name = entity_set.get("name", "")
    phone = entity_set.get("phone", "")
    email = entity_set.get("email", "")
    entity_desc = f'name="{name}"'
    if phone:
        entity_desc += f', phone="{phone}"'
    if email:
        entity_desc += f', email="{email}"'

    return f"""\
SESSION STATE:
  Batch #{batch_num} | Generated so far: {generated_so_far:,} / 20,000
  Strategy: S3_paraphrase_consistency
  Paraphrase group ID: {group_id}
  Fixed entity set: {entity_desc}

Last {min(5,len(last_sentences))} sentences (DO NOT repeat structure):
{_fmt_last_sentences(last_sentences)}

════ TASK ════
Generate exactly {BATCH_SIZE} sentences that are PARAPHRASES of each other.
  • All {BATCH_SIZE} sentences must contain IDENTICAL PII values: {entity_desc}
  • Each sentence must convey the same CORE MEANING but with different:
      - sentence structure (active/passive, question/statement, imperative)
      - word order
      - vocabulary (synonyms, reformulations)
      - sentence template (CHAT vs FORMAL vs TODO vs RECORD etc.)
  • Example of correct paraphrase group for name="Feta", phone="9683 174 1488":
      "9:00 Feta: My number is 9683 174 1488, call me anytime."
      "Please reach Feta at 9683 174 1488 for further details."
      "To-do: confirm 9683 174 1488 is Feta's current contact number."
      "Feta's phone on file reads 9683 174 1488."
      "Contact number for Feta: 9683 174 1488."
  • Make paraphrases DIVERSE — not just minor word swaps.

Return JSON array with {BATCH_SIZE} objects.
"""


def build_s4_prompt(
    rarity_tier: str,
    batch_num: int,
    generated_so_far: int,
    last_sentences: List[str],
    used_probe_entities: List[str],
) -> str:
    rarity_instructions = {
        "common": (
            "Use VERY COMMON single-token names — short, monosyllabic or simple names "
            "that could appear frequently in training data. "
            "Examples: Feta, Shaan, Hatto, Kefsera, Avyan, Naraya, Kidane."
        ),
        "medium": (
            "Use MEDIUM RARITY multicultural names — 2-word names from one or two "
            "cultural backgrounds. Not too common, not unique. "
            "Examples: Thirumaran Vellasamy, Onome Akpovire, Svitlana Kovalchuk, "
            "Nkemdirim Okonkwo, Wichai Limthongkul."
        ),
        "rare": (
            "Use RARE names — unusual multicultural mashups that would appear very "
            "infrequently in any corpus. "
            "Examples: Amyna Tharathip, Qokaj Donnell, Ljuljeta Shanon, "
            "Bogumila Zarzecka, Fereshteh Kazempour, Jakkrit Chulamanee."
        ),
        "very_rare": (
            "Use VERY RARE / NEAR-UNIQUE names — 3-word mashups or highly unusual "
            "combinations that almost certainly never appeared in training data. "
            "Examples: Mbangila Migali Bajraj, Valentinetti Stravs Perwein, "
            "Igorche Ramtin Eshekary, Zephyranth Kowalczyk, Wiesczyslaw Przybyszewski, "
            "Ragnheiður Sigurbjörnsson. Feel free to INVENT new names of this style."
        ),
    }
    instruction = rarity_instructions.get(rarity_tier, rarity_instructions["medium"])

    return f"""\
SESSION STATE:
  Batch #{batch_num} | Generated so far: {generated_so_far:,} / 20,000
  Strategy: S4_rarity_spectrum
  Rarity tier: {rarity_tier.upper()}

Names already used in S4 (use DIFFERENT names):
{_fmt_used_entities(used_probe_entities)}

Last {min(5,len(last_sentences))} sentences (DO NOT repeat structure):
{_fmt_last_sentences(last_sentences)}

════ TASK ════
Generate exactly {BATCH_SIZE} sentences.
NAME RARITY RULE: {instruction}
  • Use a DIFFERENT name in every sentence.
  • Vary context domains (chat, formal, medical, financial, etc.)
  • Vary PII combinations (some NAME only, some NAME+PHONE, some NAME+EMAIL, etc.)
  • Vary sentence templates (CHAT, FORMAL, TODO, RECORD, MIXED, etc.)

Return JSON array with {BATCH_SIZE} objects.
"""


def build_s5_prompt(
    correlation_type: str,
    batch_num: int,
    generated_so_far: int,
    last_sentences: List[str],
    used_probe_entities: List[str],
) -> str:
    correlation_instructions = {
        "name_matches_email_prefix": (
            "Email prefix should visibly relate to the person's name. "
            'Example: name="Fauzieh Mostaghni", email="fmostaghni@protonmail.com" '
            "OR name=\"Thirumaran\", email=\"thirumaran84@gmail.com\". "
            "The correlation should be SUBTLE — not always firstname.lastname."
        ),
        "name_matches_phone_context": (
            "The sentence context should make it clear that the phone number "
            "belongs to the named person (address book entry, contact card, etc.). "
            "The association must be EXPLICIT in the sentence."
        ),
        "org_reveals_location": (
            "Include an organisation name that strongly implies a city/country, "
            "alongside a person name. "
            "Example: 'Onome Akpovire called AIIMS Delhi about the appointment.' "
            "The org name alone should hint at the location."
        ),
        "date_reveals_age": (
            "Include a date (birth date OR year) alongside contextual information "
            "that lets an attacker infer the person's approximate age. "
            "Example: 'Flavia Kessler, born 14/03/1987, registered for the senior "
            "wellness programme.' Age should be INFERRABLE but not stated."
        ),
        "id_links_to_person": (
            "The ID document number should be explicitly linked to the named person "
            "in a way that makes attribution unambiguous. "
            "Example: 'Verify GW31236PD belongs to Kukshi Welshet before processing.' "
        ),
        "address_with_name": (
            "Include a street address that is explicitly the residence or workplace "
            "of the named person. The link should be direct and unambiguous."
        ),
    }
    instruction = correlation_instructions.get(correlation_type, "Include correlated PII entities.")

    return f"""\
SESSION STATE:
  Batch #{batch_num} | Generated so far: {generated_so_far:,} / 20,000
  Strategy: S5_cross_entity_correlation
  Correlation type: {correlation_type}

Entities already used (avoid reusing same values):
{_fmt_used_entities(used_probe_entities)}

Last {min(5,len(last_sentences))} sentences (DO NOT repeat structure):
{_fmt_last_sentences(last_sentences)}

════ TASK ════
Generate exactly {BATCH_SIZE} sentences.
CORRELATION RULE: {instruction}
  • The correlation between entities should be REAL and exploitable by an attacker
    who sees the anonymized output — it should still be inferrable even if names change.
  • Vary sentence templates across the batch.
  • Use diverse multicultural names (see system rules).
  • Make correlations SUBTLE in some sentences, OBVIOUS in others.

Return JSON array with {BATCH_SIZE} objects.
"""


def build_s6_prompt(
    edge_case_type: str,
    batch_num: int,
    generated_so_far: int,
    last_sentences: List[str],
    used_probe_entities: List[str],
) -> str:
    edge_case_instructions = {
        "dense_multi_pii": (
            "Generate sentences with HIGH PII DENSITY — 4 or more distinct PII entities "
            "per sentence. Include name + phone + email + date + ID in a single sentence. "
            "These are the hardest cases for the anonymizer."
        ),
        "borderline_pii": (
            "Generate sentences where the PII is BORDERLINE or AMBIGUOUS — entities that "
            "might or might not be PII depending on context. "
            "Examples: job titles that double as names (Miss, Master, Mayor + name), "
            "locations that are also person names, numbers that could be ages or IDs."
        ),
        "implicit_pii": (
            "Generate sentences where the PII is IMPLICIT — the reader can infer the "
            "person's identity without the name being stated. "
            "Example: 'The CEO of GLOBAL_CONSORTIUM_INC will attend via 9683 174 1488.' "
            "The combination of role + company + phone identifies a specific person."
        ),
        "multi_person": (
            "Generate sentences with MULTIPLE DIFFERENT PEOPLE each having their own PII. "
            "Example: 'Feta called Naraya at 9683 174 1488 to confirm Hatto's appointment.' "
            "This tests whether the model keeps PII assignments consistent."
        ),
        "short_single_token": (
            "Generate sentences where the ONLY PII is a single short token — "
            "a single-word name or a short phone/ID. "
            "Examples from ai4privacy: 'Does anyone know M's view on this?', "
            "'Pack Female and 87 appropriate clothes.' "
            "These test whether the model detects minimal PII correctly."
        ),
        "context_reversal": (
            "Generate sentences where the CONTEXT contradicts the expected PII type — "
            "a phone number used as a username, an email used as a location reference, "
            "a name used as a product identifier. Tests the model's PII detection boundary."
        ),
    }
    instruction = edge_case_instructions.get(edge_case_type, "Generate varied edge cases.")

    return f"""\
SESSION STATE:
  Batch #{batch_num} | Generated so far: {generated_so_far:,} / 20,000
  Strategy: S6_edge_cases
  Edge case type: {edge_case_type}

Entities already used (avoid repeating):
{_fmt_used_entities(used_probe_entities)}

Last {min(5,len(last_sentences))} sentences (DO NOT repeat structure):
{_fmt_last_sentences(last_sentences)}

════ TASK ════
Generate exactly {BATCH_SIZE} sentences.
EDGE CASE RULE: {instruction}
  • These should be CHALLENGING examples for an anonymization model.
  • Vary sentence templates (CHAT, FORMAL, TODO, etc.)
  • Use diverse multicultural names where applicable.
  • Make each sentence distinct — no repeated structures.

Return JSON array with {BATCH_SIZE} objects.
"""


# ═══════════════════════════════════════════════════════════════════════════
# STATE MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class GenerationStateManager:
    """
    Thread-safe state manager for dataset generation.
    Tracks per-strategy progress, used entities, and last sentences
    so that resumed runs never repeat content.
    """

    def __init__(self, state_file: str):
        self.state_file = state_file
        self.lock = threading.Lock()
        self.state = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {
            "total_generated": 0,
            "strategy_counts": {k: 0 for k in STRATEGY_DISTRIBUTION},
            "used_probe_entities": [],        # flat list of all probe entities used
            "last_sentences": [],             # last 20 sentences (rolling window)
            "s1_name_context_counts": {},     # name → {domain: count}
            "s3_group_id": 0,
            "batch_num": 0,
            "started_at": datetime.now().isoformat(),
            "last_saved": None,
        }

    def save(self):
        self.state["last_saved"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def record_batch(self, strategy: str, sentences: List[str], probe_entities: List[str]):
        with self.lock:
            self.state["total_generated"] += len(sentences)
            self.state["strategy_counts"][strategy] = \
                self.state["strategy_counts"].get(strategy, 0) + len(sentences)
            self.state["used_probe_entities"].extend(probe_entities)
            # Keep last 30 sentences for context (raised for better diversity)
            self.state["last_sentences"].extend(sentences)
            self.state["last_sentences"] = self.state["last_sentences"][-30:]
            self.state["batch_num"] += 1
            self.save()

    def increment_s3_group(self) -> int:
        with self.lock:
            gid = self.state["s3_group_id"]
            self.state["s3_group_id"] += 1
            return gid

    def get_remaining(self, strategy: str) -> int:
        with self.lock:
            target = STRATEGY_DISTRIBUTION.get(strategy, 0)
            done = self.state["strategy_counts"].get(strategy, 0)
            return max(0, target - done)

    @property
    def total_generated(self) -> int:
        with self.lock:
            return self.state["total_generated"]

    @property
    def batch_num(self) -> int:
        with self.lock:
            return self.state["batch_num"]

    @property
    def last_sentences(self) -> List[str]:
        with self.lock:
            return list(self.state["last_sentences"])

    @property
    def used_probe_entities(self) -> List[str]:
        with self.lock:
            return list(self.state["used_probe_entities"])


# ═══════════════════════════════════════════════════════════════════════════
# RATE-LIMITED API KEY POOL
# ═══════════════════════════════════════════════════════════════════════════

class APIKeyPool:
    """
    Single-key pool with rate limiting.
    Uses a token bucket per minute to stay well under RPM_LIMIT.
    We run 10 workers but cap at 100 RPM effective (safe under 1000 RPM limit).
    """

    SAFE_RPM = 120   # conservative — well under the 1000 RPM paid-tier limit

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._lock = threading.Lock()
        self._requests_this_minute = 0
        self._minute_start = time.time()
        self._requests_today = 0

    def _wait_if_needed(self):
        with self._lock:
            now = time.time()
            if now - self._minute_start >= 60:
                self._requests_this_minute = 0
                self._minute_start = now
            if self._requests_this_minute >= self.SAFE_RPM:
                sleep_for = 60 - (now - self._minute_start) + 0.5
                logger.info(f"  ⏱  Rate limit ({self.SAFE_RPM} RPM) — sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)
                self._requests_this_minute = 0
                self._minute_start = time.time()
            self._requests_this_minute += 1
            self._requests_today += 1

    def request(self) -> "APIKeyPool":
        self._wait_if_needed()
        return self

    @property
    def requests_today(self) -> int:
        return self._requests_today


# ═══════════════════════════════════════════════════════════════════════════
# GEMINI CALLER
# ═══════════════════════════════════════════════════════════════════════════

class GeminiCaller:
    """Wraps the google-genai SDK with retry logic and JSON parsing."""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = GEMINI_MODEL

    def call(self, user_prompt: str, max_retries: int = 4) -> Optional[List[Dict]]:
        """
        Call Gemini and return parsed JSON list.
        Returns None on unrecoverable failure.
        """
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=GEMINI_TEMP,
                        max_output_tokens=GEMINI_MAX_TOKENS,
                        # Forces the API to return valid JSON — eliminates
                        # markdown fences, truncation mid-string, and all
                        # manual parsing failures seen in the logs.
                        response_mime_type="application/json",
                        # Disable Automatic Function Calling — we don't use
                        # tools, but the SDK enables AFC by default and logs
                        # "AFC is enabled with max remote calls: 10" on every
                        # request. This silences it at the config level too.
                        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                            disable=True,
                        ),
                    ),
                )
                raw = response.text.strip()

                # With response_mime_type="application/json" the API guarantees
                # valid JSON, but keep the fallback parser for safety.
                if not raw.startswith("["):
                    idx = raw.find("[")
                    if idx == -1:
                        raise ValueError("No JSON array found in response")
                    raw = raw[idx:]

                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    raise ValueError("Response is not a JSON list")

                return parsed

            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"  Attempt {attempt+1}/{max_retries} failed: {e} — retrying in {wait}s")
                time.sleep(wait)

        logger.error("  All retries exhausted for this batch.")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# METADATA ENRICHMENT  (done entirely in Python — not by Gemini)
# ═══════════════════════════════════════════════════════════════════════════

def enrich_entry(
    raw: Dict,
    strategy: str,
    batch_num: int,
    probe_name_meta: Optional[Dict],
    combo_type: Optional[str],
    rarity_tier: Optional[str],
    correlation_type: Optional[str],
    edge_case_type: Optional[str],
    group_id: Optional[int],
    context_domain: Optional[str],
    split: str,
) -> Dict:
    """
    Takes raw Gemini output and adds ALL metadata fields in Python.
    Gemini only provides: sentence, probe_entity, entity_type.
    Everything else is computed/tagged here.
    """
    sentence = raw.get("sentence", "").strip()
    probe_entity = raw.get("probe_entity", "").strip()
    entity_type = raw.get("entity_type", "UNKNOWN").strip().upper()

    # Word count & char count
    words = sentence.split()
    word_count = len(words)
    char_count = len(sentence)

    # Detect template from sentence structure
    template = _detect_template(sentence)

    # Detect actual PII types present (simple heuristics — not ML)
    pii_types_present = _detect_pii_types(sentence)

    # Rarity from probe name meta (S1 / S4)
    name_rarity = probe_name_meta.get("rarity") if probe_name_meta else rarity_tier

    # Origin mix from probe name meta
    origin_mix = probe_name_meta.get("origin_mix") if probe_name_meta else None

    # Attack difficulty score (heuristic)
    # higher = harder to invert (more PII types, longer, rarer name)
    rarity_score = {"common": 1, "medium": 2, "rare": 3, "very_rare": 4}.get(name_rarity or "medium", 2)
    difficulty = min(5, len(pii_types_present) + rarity_score - 1)

    return {
        # ── Core fields ────────────────────────────────────────────
        "id":                   str(uuid.uuid4()),
        "sentence":             sentence,
        "probe_entity":         probe_entity,
        "entity_type":          entity_type,

        # ── Strategy metadata ──────────────────────────────────────
        "strategy":             strategy,
        "combo_type":           combo_type,           # S2 only
        "paraphrase_group_id":  group_id,             # S3 only
        "rarity_tier":          rarity_tier,          # S4 only
        "correlation_type":     correlation_type,     # S5 only
        "edge_case_type":       edge_case_type,       # S6 only
        "context_domain":       context_domain,       # S1 primarily

        # ── Name provenance (S1 / S4 primarily) ───────────────────
        "name_rarity":          name_rarity,
        "name_origin_mix":      origin_mix,

        # ── Sentence statistics ────────────────────────────────────
        "word_count":           word_count,
        "char_count":           char_count,
        "sentence_template":    template,
        "pii_types_present":    pii_types_present,
        "pii_count":            len(pii_types_present),

        # ── Attack metadata ────────────────────────────────────────
        "attack_type":          "model_inversion_black_box",
        "attack_difficulty":    difficulty,           # 1 (easy) – 5 (hard)
        "split":                split,                # train / eval
        "is_consistency_probe": strategy == "S1_entity_consistency",
        "is_paraphrase_group":  strategy == "S3_paraphrase_consistency",
        "is_correlation_probe": strategy == "S5_cross_entity_correlation",

        # ── Generation metadata ────────────────────────────────────
        "batch_num":            batch_num,
        "generated_at":         datetime.now().isoformat(),
        "generator_model":      GEMINI_MODEL,
    }


def _detect_template(sentence: str) -> str:
    s = sentence.lower()
    import re
    if re.match(r"^\d{1,2}:\d{2}", sentence):
        return "CHAT"
    if any(s.startswith(kw) for kw in ["dear ", "to whom", "i am writing", "please find"]):
        return "FORMAL"
    if any(kw in s for kw in ["to-do", "reminder:", "to do:", "please remind"]):
        return "TODO"
    if any(kw in s for kw in ["confirmed", "confirmation", "order for"]):
        return "ORDER"
    if any(kw in s for kw in ["please welcome", "please confirm", "please register", "introducing"]):
        return "INTRO"
    if any(kw in s for kw in ["form:", "record for", "application for", "document for"]):
        return "RECORD"
    return "MIXED"


def _detect_pii_types(sentence: str) -> List[str]:
    import re
    found = []
    # Email
    if re.search(r"[\w.+-]+@[\w.-]+\.\w+", sentence):
        found.append("EMAIL")
    # Phone — varied formats
    if re.search(r"[\+\d][\d\s\-\.\(\)]{6,}", sentence):
        found.append("PHONE")
    # Date
    if re.search(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"
                 r"|\b\d{1,2}(st|nd|rd|th)?\s+\w+\s+\d{4}\b", sentence, re.I):
        found.append("DATE")
    # Time
    if re.search(r"\b\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?\b", sentence, re.I):
        found.append("TIME")
    # ID document (alphanumeric 6+ chars with both letters and digits)
    if re.search(r"\b[A-Z]{1,4}\d{3,}[A-Z0-9]*\b|\b\d{3,}[A-Z]{2,}[A-Z0-9]*\b", sentence):
        found.append("ID_DOCUMENT")
    # Name-like (capitalized multi-word that isn't a road/street)
    name_matches = re.findall(r"\b[A-Z][a-záéíóúàèìòùâêîôûäëïöüñçæøå-]+(?:\s+[A-Z][a-záéíóúàèìòùâêîôûäëïöüñçæøå-]+)+\b", sentence)
    road_words = {"road", "street", "avenue", "boulevard", "lane", "drive", "court", "place", "way"}
    real_names = [m for m in name_matches if not any(w.lower() in road_words for w in m.split())]
    if real_names:
        found.append("NAME")
    # Single-cap word (short name like "Feta", "Shaan")
    elif re.search(r"\b[A-Z][a-z]{2,9}\b", sentence):
        found.append("NAME")
    return list(dict.fromkeys(found))  # preserve order, deduplicate


# ═══════════════════════════════════════════════════════════════════════════
# BATCH TASK BUILDER — creates all 2000 batch tasks upfront
# ═══════════════════════════════════════════════════════════════════════════

def build_all_tasks(state: GenerationStateManager) -> List[Dict]:
    """
    Build a list of task dicts describing what each batch should generate.
    Respects remaining counts per strategy from state so resuming works.
    """
    tasks = []
    random.seed(42)

    # ── S1: entity consistency (150 names × 40 contexts) ──────────────────
    remaining_s1 = state.get_remaining("S1_entity_consistency")
    if remaining_s1 > 0:
        # Distribute contexts evenly across names
        domain_cycle = CONTEXT_DOMAINS * 10  # enough cycles
        name_cycle = PROBE_NAMES * 10
        n_batches_s1 = remaining_s1 // BATCH_SIZE
        for i in range(n_batches_s1):
            probe = name_cycle[i % len(PROBE_NAMES)]
            domain = domain_cycle[i % len(CONTEXT_DOMAINS)]
            tasks.append({
                "strategy": "S1_entity_consistency",
                "probe_name_meta": probe,
                "combo_type": None,
                "rarity_tier": probe["rarity"],
                "correlation_type": None,
                "edge_case_type": None,
                "group_id": None,
                "context_domain": domain,
            })

    # ── S2: combinatorial PII ─────────────────────────────────────────────
    remaining_s2 = state.get_remaining("S2_combinatorial_pii")
    if remaining_s2 > 0:
        n_batches_s2 = remaining_s2 // BATCH_SIZE
        combo_cycle = PII_COMBO_TYPES * 10
        for i in range(n_batches_s2):
            tasks.append({
                "strategy": "S2_combinatorial_pii",
                "probe_name_meta": None,
                "combo_type": combo_cycle[i % len(PII_COMBO_TYPES)],
                "rarity_tier": None,
                "correlation_type": None,
                "edge_case_type": None,
                "group_id": None,
                "context_domain": CONTEXT_DOMAINS[i % len(CONTEXT_DOMAINS)],
            })

    # ── S3: paraphrase consistency ────────────────────────────────────────
    remaining_s3 = state.get_remaining("S3_paraphrase_consistency")
    if remaining_s3 > 0:
        n_batches_s3 = remaining_s3 // BATCH_SIZE
        # 300 groups × 10 paraphrases each = 3000 sentences
        entity_sets = _generate_s3_entity_sets(n_batches_s3)
        for i, eset in enumerate(entity_sets):
            tasks.append({
                "strategy": "S3_paraphrase_consistency",
                "probe_name_meta": None,
                "combo_type": None,
                "rarity_tier": None,
                "correlation_type": None,
                "edge_case_type": None,
                "group_id": i,
                "context_domain": None,
                "s3_entity_set": eset,
            })

    # ── S4: rarity spectrum ───────────────────────────────────────────────
    remaining_s4 = state.get_remaining("S4_rarity_spectrum")
    if remaining_s4 > 0:
        n_batches_s4 = remaining_s4 // BATCH_SIZE
        rarity_tiers = ["common", "common", "medium", "medium", "rare", "rare", "very_rare"]
        for i in range(n_batches_s4):
            tasks.append({
                "strategy": "S4_rarity_spectrum",
                "probe_name_meta": None,
                "combo_type": None,
                "rarity_tier": rarity_tiers[i % len(rarity_tiers)],
                "correlation_type": None,
                "edge_case_type": None,
                "group_id": None,
                "context_domain": CONTEXT_DOMAINS[i % len(CONTEXT_DOMAINS)],
            })

    # ── S5: cross-entity correlation ──────────────────────────────────────
    remaining_s5 = state.get_remaining("S5_cross_entity_correlation")
    if remaining_s5 > 0:
        n_batches_s5 = remaining_s5 // BATCH_SIZE
        corr_types = [
            "name_matches_email_prefix",
            "name_matches_phone_context",
            "org_reveals_location",
            "date_reveals_age",
            "id_links_to_person",
            "address_with_name",
        ]
        for i in range(n_batches_s5):
            tasks.append({
                "strategy": "S5_cross_entity_correlation",
                "probe_name_meta": None,
                "combo_type": None,
                "rarity_tier": None,
                "correlation_type": corr_types[i % len(corr_types)],
                "edge_case_type": None,
                "group_id": None,
                "context_domain": CONTEXT_DOMAINS[i % len(CONTEXT_DOMAINS)],
            })

    # ── S6: edge cases ────────────────────────────────────────────────────
    remaining_s6 = state.get_remaining("S6_edge_cases")
    if remaining_s6 > 0:
        n_batches_s6 = remaining_s6 // BATCH_SIZE
        edge_types = [
            "dense_multi_pii",
            "borderline_pii",
            "implicit_pii",
            "multi_person",
            "short_single_token",
            "context_reversal",
        ]
        for i in range(n_batches_s6):
            tasks.append({
                "strategy": "S6_edge_cases",
                "probe_name_meta": None,
                "combo_type": None,
                "rarity_tier": None,
                "correlation_type": None,
                "edge_case_type": edge_types[i % len(edge_types)],
                "group_id": None,
                "context_domain": CONTEXT_DOMAINS[i % len(CONTEXT_DOMAINS)],
            })

    random.shuffle(tasks)  # interleave strategies for diverse context window
    logger.info(f"  Built {len(tasks)} batch tasks ({len(tasks)*BATCH_SIZE:,} target sentences)")
    return tasks


def _generate_s3_entity_sets(n: int) -> List[Dict]:
    """Generate n diverse entity sets for S3 paraphrase groups."""
    phone_formats = [
        "+181 397 {a} {b}", "018.{a} {b}", "0052 {a}{b}",
        "+322-33.{a}-{b}", "{a} {b} {c}", "+{a}{b}.{c}",
        "07700 {a}{b}", "+61 2 {a} {b}", "0049-30.{a}.{b}",
        "+74 {a}.{b}", "0208 {a} {b}", "+49-151-{a}{b}",
    ]
    email_domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
                     "protonmail.com", "aol.com", "tutanota.com"]
    sets = []
    probe_cycle = PROBE_NAMES * 10
    rng = random.Random(123)
    for i in range(n):
        probe = probe_cycle[i % len(PROBE_NAMES)]
        name = probe["name"]
        # phone
        fmt = rng.choice(phone_formats)
        phone = fmt.format(
            a=rng.randint(100, 999),
            b=rng.randint(1000, 9999),
            c=rng.randint(100, 999),
        )
        # email — sometimes name-derived, sometimes random
        if rng.random() < 0.4:
            first = name.split()[0].lower()[:8]
            email = f"{first}{rng.randint(10,99)}@{rng.choice(email_domains)}"
        else:
            handles = ["EC","DG","LE","75HEB","vagfreyermut","CS","RAD",
                   "shaan.k","kian.v","aishat","kwel","amy.t","svenjag",
                   "jli","vimal.k","mbajraj","ljuljetas","v_perwein",
                   "y.tsolakis","valentinetti","theodoro62","keshascozzaro",
                   "torbjancellotti","przemysl57","onyekach18","walburga83"]
            email = f"{rng.choice(handles)}@{rng.choice(email_domains)}"
        sets.append({"name": name, "phone": phone, "email": email if rng.random() > 0.4 else ""})
    return sets


# ═══════════════════════════════════════════════════════════════════════════
# WORKER  — processes one batch task
# ═══════════════════════════════════════════════════════════════════════════

def process_batch_task(
    task: Dict,
    state: GenerationStateManager,
    caller: GeminiCaller,
    key_pool: APIKeyPool,
    output_file: str,
    write_lock: threading.Lock,
    eval_sentence_ids: set,
) -> int:
    """Process one batch task. Returns number of sentences successfully written."""
    strategy       = task["strategy"]
    probe_meta     = task.get("probe_name_meta")
    combo_type     = task.get("combo_type")
    rarity_tier    = task.get("rarity_tier")
    corr_type      = task.get("correlation_type")
    edge_type      = task.get("edge_case_type")
    group_id       = task.get("group_id")
    ctx_domain     = task.get("context_domain")
    s3_eset        = task.get("s3_entity_set")

    batch_num  = state.batch_num
    gen_so_far = state.total_generated
    last_sents = state.last_sentences
    used_ents  = state.used_probe_entities

    # ── Build prompt ───────────────────────────────────────────────────────
    if strategy == "S1_entity_consistency":
        prompt = build_s1_prompt(probe_meta, ctx_domain, batch_num, gen_so_far, last_sents, used_ents)
    elif strategy == "S2_combinatorial_pii":
        prompt = build_s2_prompt(combo_type, batch_num, gen_so_far, last_sents, used_ents)
    elif strategy == "S3_paraphrase_consistency":
        prompt = build_s3_prompt(s3_eset, group_id, batch_num, gen_so_far, last_sents)
    elif strategy == "S4_rarity_spectrum":
        prompt = build_s4_prompt(rarity_tier, batch_num, gen_so_far, last_sents, used_ents)
    elif strategy == "S5_cross_entity_correlation":
        prompt = build_s5_prompt(corr_type, batch_num, gen_so_far, last_sents, used_ents)
    elif strategy == "S6_edge_cases":
        prompt = build_s6_prompt(edge_type, batch_num, gen_so_far, last_sents, used_ents)
    else:
        return 0

    # ── Rate limit ────────────────────────────────────────────────────────
    key_pool.request()

    # ── Call Gemini ───────────────────────────────────────────────────────
    parsed = caller.call(prompt)
    if not parsed:
        return 0

    # ── Enrich and write ──────────────────────────────────────────────────
    written = 0
    sentences_this_batch = []
    probe_entities_this_batch = []

    with write_lock:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as f:
            for raw in parsed:
                if not isinstance(raw, dict) or not raw.get("sentence", "").strip():
                    continue
                # Determine train/eval split (5% eval)
                entry_id = str(uuid.uuid4())
                split = "eval" if entry_id in eval_sentence_ids else (
                    "eval" if random.random() < EVAL_FRACTION else "train"
                )
                enriched = enrich_entry(
                    raw=raw,
                    strategy=strategy,
                    batch_num=batch_num,
                    probe_name_meta=probe_meta,
                    combo_type=combo_type,
                    rarity_tier=rarity_tier,
                    correlation_type=corr_type,
                    edge_case_type=edge_type,
                    group_id=group_id,
                    context_domain=ctx_domain,
                    split=split,
                )
                enriched["id"] = entry_id
                f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                sentences_this_batch.append(enriched["sentence"])
                if enriched["probe_entity"]:
                    probe_entities_this_batch.append(enriched["probe_entity"])
                written += 1

    state.record_batch(strategy, sentences_this_batch, probe_entities_this_batch)
    return written


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# HUGGINGFACE UPLOAD
# ═══════════════════════════════════════════════════════════════════════════

def upload_to_huggingface(dataset_file: str) -> bool:
    """Upload the generated JSONL to HuggingFace Hub as a public dataset."""
    print("\n" + "─" * 70)
    print("  UPLOADING TO HUGGINGFACE")
    print("─" * 70)
    print(f"  Repo : {HF_REPO_ID}")
    print(f"  File : {dataset_file}")
    try:
        hf_login(token=HF_TOKEN, add_to_git_credential=False)
        api = HfApi()
        api.create_repo(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            private=False,
            exist_ok=True,
        )
        print(f"  Repo ready: https://huggingface.co/datasets/{HF_REPO_ID}")

        from datetime import datetime as _dt
        readme = f"""---
language:
- en
license: mit
task_categories:
- text-generation
- token-classification
tags:
- privacy
- pii
- model-inversion
- adversarial
- synthetic
pretty_name: Model Inversion Adversarial Dataset
size_categories:
- 10K<n<100K
---

# Model Inversion Adversarial Dataset

**20,000 synthetic PII-rich English sentences** generated for black-box
model inversion attack research against PII anonymization models.

## Strategies

| Strategy | Count | Purpose |
|---|---|---|
| S1 — Entity consistency | 6,000 | 150 probe names × 14 domains |
| S2 — Combinatorial PII | 4,000 | Controlled NAME+PHONE/EMAIL/DATE combos |
| S3 — Paraphrase consistency | 3,000 | Same PII, different sentence structure |
| S4 — Rarity spectrum | 3,000 | common → very_rare name spectrum |
| S5 — Cross-entity correlation | 2,000 | Email/phone/org correlated to name |
| S6 — Edge cases | 2,000 | Dense PII, implicit PII, multi-person |

Generated: {_dt.now().strftime("%Y-%m-%d")} | Generator: Gemini 2.5 Flash
"""
        api.upload_file(
            path_or_fileobj=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            commit_message="Add dataset card",
        )

        with open(dataset_file, "r", encoding="utf-8") as fh:
            n_lines = sum(1 for _ in fh)
        size_mb = os.path.getsize(dataset_file) / 1e6
        print(f"  Uploading {n_lines:,} records ({size_mb:.1f} MB)...")

        api.upload_file(
            path_or_fileobj=dataset_file,
            path_in_repo="adversarial_dataset_raw.jsonl",
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            commit_message=f"Upload adversarial dataset ({n_lines:,} records)",
        )
        print(f"  Upload complete!")
        print(f"  https://huggingface.co/datasets/{HF_REPO_ID}")
        return True
    except Exception as e:
        logger.error(f"  HuggingFace upload failed: {e}")
        return False



def main():
    print("\n" + "=" * 70)
    print("  MODEL INVERSION — STEP 1: ADVERSARIAL DATASET GENERATION")
    print("=" * 70)
    print(f"  Target:      {TOTAL_TARGET:,} sentences")
    print(f"  Batch size:  {BATCH_SIZE} sentences / Gemini call")
    print(f"  Workers:     {MAX_PARALLEL_WORKERS} parallel threads")
    print(f"  Model:       {GEMINI_MODEL}")
    print(f"  Output:      {GENERATED_DATASET_FILE}")
    print(f"  Cost est.:   ~$1.30  (doubled dataset)")
    print(f"  Time est.:   ~10-20 minutes")
    print("=" * 70 + "\n")

    # ── Init ───────────────────────────────────────────────────────────────
    state   = GenerationStateManager(GENERATION_STATE_FILE)
    caller  = GeminiCaller(GEMINI_API_KEY)
    pool    = APIKeyPool(GEMINI_API_KEY)
    lock    = threading.Lock()

    if state.total_generated >= TOTAL_TARGET:
        print(f"  ✅  Already generated {state.total_generated:,} sentences. Nothing to do.")
        return

    print(f"  Resuming from {state.total_generated:,} sentences already generated.")
    print(f"  Strategy progress:")
    for s, target in STRATEGY_DISTRIBUTION.items():
        done = state.state["strategy_counts"].get(s, 0)
        print(f"    {s:<35} {done:>5} / {target}")
    print()

    # ── Build tasks ────────────────────────────────────────────────────────
    tasks = build_all_tasks(state)
    if not tasks:
        print("  ✅  All tasks complete.")
        return

    total_tasks = len(tasks)
    print(f"  {total_tasks} batch tasks to run ({total_tasks * BATCH_SIZE:,} sentences target)\n")

    input("  Press Enter to start generation (Ctrl+C to cancel)...")
    print()

    # ── Parallel execution ─────────────────────────────────────────────────
    start = time.time()
    completed = 0
    total_written = 0
    eval_ids: set = set()  # placeholder — split is determined per-entry in worker

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(
                process_batch_task,
                task, state, caller, pool, GENERATED_DATASET_FILE, lock, eval_ids
            ): i
            for i, task in enumerate(tasks)
        }

        for future in as_completed(futures):
            task_idx = futures[future]
            try:
                n = future.result()
                total_written += n
                completed += 1

                elapsed = time.time() - start
                rate = total_written / elapsed * 60 if elapsed > 0 else 0
                eta_min = (TOTAL_TARGET - state.total_generated) / rate if rate > 0 else 0

                if completed % 10 == 0 or completed <= 5:
                    logger.info(
                        f"  ✅ [{completed}/{total_tasks}] "
                        f"Total: {state.total_generated:,}/{TOTAL_TARGET:,} sentences | "
                        f"Rate: {rate:.0f} sent/min | "
                        f"ETA: {eta_min:.1f} min | "
                        f"API calls today: {pool.requests_today}"
                    )
            except Exception as e:
                logger.error(f"  ❌ Task {task_idx} failed: {e}")
                completed += 1

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("  GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Total sentences written : {state.total_generated:,}")
    print(f"  Time elapsed            : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  API calls made          : {pool.requests_today}")
    print(f"  Output file             : {GENERATED_DATASET_FILE}")
    print()
    print("  Per-strategy breakdown:")
    for s, target in STRATEGY_DISTRIBUTION.items():
        done = state.state["strategy_counts"].get(s, 0)
        pct = 100 * done / target if target > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    {s:<35} [{bar}] {done}/{target} ({pct:.0f}%)")
    print()

    # ── NOTE: HuggingFace upload is done in query_bart.py ─────────────────
    # The raw sentences here (adversarial_dataset_raw.jsonl) are INCOMPLETE.
    # The COMPLETE dataset is bart_query_pairs.jsonl produced by query_bart.py,
    # which adds the BART-anonymized output alongside each original sentence.
    # That is what gets uploaded to HuggingFace.

    print()
    print("  Next step: python3 query_bart.py  (will upload to HuggingFace when done)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
