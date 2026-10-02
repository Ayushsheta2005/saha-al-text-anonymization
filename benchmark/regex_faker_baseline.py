"""
SAHA-AL Baseline Generator
===========================
Generates predictions.jsonl for three baselines:
  1. Regex+Faker   (no ML, pattern matching only)
  2. spaCy+Faker   (NER + Faker)
  3. Presidio       (industry standard)

Usage:
  python baselines.py --gold data/test.jsonl --mode regex
  python baselines.py --gold data/test.jsonl --mode spacy
  python baselines.py --gold data/test.jsonl --mode presidio

Then evaluate:
  python benchmark_eval.py --gold data/test.jsonl --pred predictions/regex_predictions.jsonl
"""

import json
import re
import argparse
import os
from collections import OrderedDict

# ── Faker setup ──────────────────────────────────────────────
try:
    from faker import Faker
    fake = Faker()
    Faker.seed(42)  # Reproducible
except ImportError:
    print("Install faker: pip install faker")
    exit(1)


# ═══════════════════════════════════════════════════════════════
# REGEX BASELINE
# ═══════════════════════════════════════════════════════════════
# 23 priority-ranked patterns from SAHA-AL Layer 1 (mid-term §3.2.1)
# Priority: higher priority patterns win on span overlaps.

REGEX_PATTERNS = OrderedDict([
    # --- Critical (structured, high-confidence) ---
    ("EMAIL", re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    )),
    ("SSN", re.compile(
        r'\b\d{3}-\d{2}-\d{4}\b'
    )),
    ("CREDIT_CARD", re.compile(
        r'\b(?:\d[ -]*?){13,19}\b'
    )),
    ("IBAN", re.compile(
        r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})?\b'
    )),
    ("IP_ADDRESS", re.compile(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    )),
    # --- High ---
    ("PHONE", re.compile(
        r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'
    )),
    ("URL", re.compile(
        r'https?://[^\s<>\"\']+|www\.[^\s<>\"\']+\.[^\s<>\"\']+'
    )),
    # --- Medium (date formats) ---
    ("DATE", re.compile(
        r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b'
        r'|\b\d{4}[/.-]\d{1,2}[/.-]\d{1,2}\b'
        r'|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?'
        r'|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?'
        r'|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b',
        re.IGNORECASE
    )),
    # --- Account/ID patterns ---
    ("ACCOUNT", re.compile(
        r'\b(?:account|acct)[#:\s]*\d{6,}\b',
        re.IGNORECASE
    )),
    ("ID_NUMBER", re.compile(
        r'\b[A-Z]{1,3}\d{6,10}\b'
    )),
])

# Faker generators per type
FAKER_GENERATORS = {
    "EMAIL":       lambda: fake.email(),
    "SSN":         lambda: fake.ssn(),
    "CREDIT_CARD": lambda: fake.credit_card_number(),
    "IBAN":        lambda: fake.iban(),
    "IP_ADDRESS":  lambda: fake.ipv4(),
    "PHONE":       lambda: fake.phone_number(),
    "URL":         lambda: fake.url(),
    "DATE":        lambda: fake.date(pattern="%m/%d/%Y"),
    "ACCOUNT":     lambda: f"Acct#{fake.numerify('########')}",
    "ID_NUMBER":   lambda: fake.bothify('??######'),
    # spaCy/Presidio entity types
    "PERSON":      lambda: fake.name(),
    "ORG":         lambda: fake.company(),
    "GPE":         lambda: fake.city(),
    "LOC":         lambda: fake.city(),
    "LOCATION":    lambda: fake.city(),
    "FULLNAME":    lambda: fake.name(),
    "FIRSTNAME":   lambda: fake.first_name(),
    "LASTNAME":    lambda: fake.last_name(),
    "ADDRESS":     lambda: fake.street_address(),
    "USERNAME":    lambda: fake.user_name(),
    "PASSWORD":    lambda: fake.password(),
}


def get_faker_replacement(entity_type, original_text=""):
    """Get a Faker replacement, falling back to a generic one."""
    gen = FAKER_GENERATORS.get(entity_type)
    if gen:
        return gen()
    # Fallback: return a placeholder
    return f"[{entity_type}]"


def resolve_overlapping_spans(spans):
    """
    Given a list of (start, end, type, text) tuples,
    remove overlapping spans keeping the earlier (higher-priority) one.
    Patterns are in priority order in REGEX_PATTERNS (OrderedDict).
    """
    # Sort by start position, then by length descending (prefer longer match)
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    resolved = []
    last_end = -1
    for span in spans:
        if span[0] >= last_end:
            resolved.append(span)
            last_end = span[1]
    return resolved


def regex_anonymize(text):
    """Detect PII via regex, replace with Faker. Returns anonymized text."""
    spans = []
    for ent_type, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), ent_type, match.group()))

    spans = resolve_overlapping_spans(spans)

    if not spans:
        return text

    # Replace right-to-left to preserve offsets
    result = text
    for start, end, ent_type, original in reversed(spans):
        replacement = get_faker_replacement(ent_type, original)
        result = result[:start] + replacement + result[end:]

    return result


# ═══════════════════════════════════════════════════════════════
# SPACY BASELINE
# ═══════════════════════════════════════════════════════════════

def spacy_anonymize(text, nlp):
    """Detect PII via spaCy NER, replace with Faker."""
    doc = nlp(text)

    # Collect spaCy entities
    spacy_spans = []
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "DATE",
                          "FAC", "NORP", "EVENT"):
            spacy_spans.append((ent.start_char, ent.end_char,
                                ent.label_, ent.text))

    # Also run regex for structured types spaCy misses
    for ent_type, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            spacy_spans.append((match.start(), match.end(),
                                ent_type, match.group()))

    spans = resolve_overlapping_spans(spacy_spans)

    if not spans:
        return text

    result = text
    for start, end, ent_type, original in reversed(spans):
        replacement = get_faker_replacement(ent_type, original)
        result = result[:start] + replacement + result[end:]

    return result


# ═══════════════════════════════════════════════════════════════
# PRESIDIO BASELINE
# ═══════════════════════════════════════════════════════════════

def presidio_anonymize(text, analyzer, anonymizer):
    """Detect + anonymize PII via Microsoft Presidio."""
    results = analyzer.analyze(text=text, language='en')
    if not results:
        return text
    anon_result = anonymizer.anonymize(text=text, analyzer_results=results)
    return anon_result.text


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline predictions for SAHA-AL benchmark"
    )
    parser.add_argument("--gold", type=str, default="data/test.jsonl",
                        help="Path to gold test.jsonl")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["regex", "spacy", "presidio"],
                        help="Baseline type")
    parser.add_argument("--output-dir", type=str, default="predictions",
                        help="Output directory for predictions")
    args = parser.parse_args()

    # Load test data
    with open(args.gold, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"Loaded {len(records)} test records.")

    # Initialize models if needed
    anonymize_fn = None

    if args.mode == "regex":
        anonymize_fn = regex_anonymize
        print("Mode: Regex + Faker (no ML)")

    elif args.mode == "spacy":
        try:
            import spacy
            nlp = spacy.load("en_core_web_lg")
            print("Mode: spaCy (en_core_web_lg) + Regex + Faker")
        except (ImportError, OSError):
            print("Install spaCy + model:")
            print("  pip install spacy")
            print("  python -m spacy download en_core_web_lg")
            exit(1)
        anonymize_fn = lambda text: spacy_anonymize(text, nlp)

    elif args.mode == "presidio":
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            analyzer = AnalyzerEngine()
            anonymizer = AnonymizerEngine()
            print("Mode: Microsoft Presidio")
        except ImportError:
            print("Install Presidio:")
            print("  pip install presidio-analyzer presidio-anonymizer")
            print("  python -m spacy download en_core_web_lg")
            exit(1)
        anonymize_fn = lambda text: presidio_anonymize(text, analyzer, anonymizer)

    # Generate predictions
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.mode}_predictions.jsonl")

    predictions = []
    failed = 0
    for i, record in enumerate(records):
        text = record.get("original_text", "")
        try:
            anon = anonymize_fn(text)
        except Exception as e:
            # On failure, return original text (worst case = 100% leakage
            # for this example, which is honest)
            anon = text
            failed += 1
            if failed <= 5:
                print(f"  [WARN] Failed on record {record['id']}: {e}")

        predictions.append({
            "id": record["id"],
            "anonymized_text": anon
        })

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(records)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    print(f"\nDone! {len(predictions)} predictions saved to {output_path}")
    if failed > 0:
        print(f"  ({failed} records failed, fell back to original text)")
    print(f"\nEvaluate with:")
    print(f"  python benchmark_eval.py --gold {args.gold} --pred {output_path}")


if __name__ == "__main__":
    main()