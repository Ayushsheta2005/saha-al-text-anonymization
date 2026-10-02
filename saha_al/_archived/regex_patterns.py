"""
Regex Patterns for Structured PII Detection
Each pattern has: name, regex, entity_type, priority, description.
Higher priority = more specific = wins conflicts when spans overlap.
"""

import re


PATTERNS = [
    # ─── EMAIL (highest specificity) ───
    {
        "name": "email",
        "pattern": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        "entity_type": "EMAIL",
        "priority": 10,
        "description": "Email addresses",
    },
    # ─── SSN (US format) ───
    {
        "name": "ssn_us",
        "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
        "entity_type": "SSN",
        "priority": 10,
        "description": "US Social Security Number (XXX-XX-XXXX)",
    },
    # ─── CREDIT CARD (16 digits) ───
    {
        "name": "credit_card",
        "pattern": r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
        "entity_type": "CREDIT_CARD",
        "priority": 9,
        "description": "16-digit credit card number",
    },
    # ─── PHONE: international with + ───
    {
        "name": "phone_plus",
        "pattern": r"\+\d{1,3}[\s.\-]?\d{1,4}[\s.\-]?\d{2,4}[\s.\-]?\d{2,4}[\s.\-]?\d{0,4}",
        "entity_type": "PHONE",
        "priority": 9,
        "description": "International phone starting with +",
    },
    # ─── PHONE: parenthesized area code ───
    {
        "name": "phone_parens",
        "pattern": r"\(\d{2,4}\)[.\s\-]?\d{3,4}[.\s\-]?\d{3,4}",
        "entity_type": "PHONE",
        "priority": 9,
        "description": "Phone with parenthesized area code like (509).1090420",
    },
    # ─── PHONE: digits with separators (10+ digits total) ───
    {
        "name": "phone_separated",
        "pattern": r"\b\d{2,5}[\s.\-]\d{2,5}[\s.\-]\d{2,5}(?:[\s.\-]\d{2,5})?\b",
        "entity_type": "PHONE",
        "priority": 8,
        "description": "Phone with space/dot/dash separators",
    },
    # ─── DATE: ISO format with T ───
    {
        "name": "date_iso",
        "pattern": r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b",
        "entity_type": "DATE",
        "priority": 10,
        "description": "ISO 8601 date like 1990-05-05T00:00:00",
    },
    # ─── DATE: DD/MM/YYYY or MM/DD/YYYY ───
    {
        "name": "date_slash",
        "pattern": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        "entity_type": "DATE",
        "priority": 7,
        "description": "Date with slashes",
    },
    # ─── DATE: verbal ordinal format ───
    {
        "name": "date_verbal_ordinal",
        "pattern": r"\b\d{1,2}(?:st|nd|rd|th)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        "entity_type": "DATE",
        "priority": 8,
        "description": "Verbal date like 4th August 1942",
    },
    # ─── DATE: Month Day, Year ───
    {
        "name": "date_month_day_year",
        "pattern": r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        "entity_type": "DATE",
        "priority": 8,
        "description": "Date like March 15, 2024 or June 6th, 1978",
    },
    # ─── DATE: Month/YY ───
    {
        "name": "date_month_year",
        "pattern": r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)/\d{2,4}\b",
        "entity_type": "DATE",
        "priority": 6,
        "description": "Date like February/15 or October/03",
    },
    # ─── DATE: Day Month, Year (verbal) ───
    {
        "name": "date_day_month_year",
        "pattern": r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b",
        "entity_type": "DATE",
        "priority": 8,
        "description": "Date like 17th June 2021",
    },
    # ─── TIME: HH:MM:SS AM/PM ───
    {
        "name": "time_hms_ampm",
        "pattern": r"\b\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM|am|pm)\b",
        "entity_type": "TIME",
        "priority": 9,
        "description": "Time like 5:52:48 AM",
    },
    # ─── TIME: HH:MM AM/PM ───
    {
        "name": "time_hm_ampm",
        "pattern": r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b",
        "entity_type": "TIME",
        "priority": 8,
        "description": "Time like 1:52 PM",
    },
    # ─── TIME: HH:MM:SS (24h) ───
    {
        "name": "time_hms_24",
        "pattern": r"\b\d{1,2}:\d{2}:\d{2}\b",
        "entity_type": "TIME",
        "priority": 7,
        "description": "24-hour time like 02:05:58",
    },
    # ─── TIME: HH:MM (24h) — careful, can clash with other patterns ───
    {
        "name": "time_hm_24",
        "pattern": r"\b\d{1,2}:\d{2}\b",
        "entity_type": "TIME",
        "priority": 5,
        "description": "24-hour time like 10:17",
    },
    # ─── ID: alphanumeric mixed (letters + digits, 6+ chars) ───
    {
        "name": "id_alphanumeric_mixed",
        "pattern": r"\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{6,}\b",
        "entity_type": "ID_NUMBER",
        "priority": 7,
        "description": "Alphanumeric ID codes like MARGA7101160M9183, K4VMKGVUXE",
    },
    # ─── ID: with embedded dash (TALJA-870061-T9-197 style) ───
    {
        "name": "id_dashed",
        "pattern": r"\b[A-Z0-9]{2,}-[A-Z0-9]{2,}(?:-[A-Z0-9]{1,}){0,3}\b",
        "entity_type": "ID_NUMBER",
        "priority": 7,
        "description": "Dashed ID codes like TALJA-870061-T9-197",
    },
    # ─── ZIPCODE: US 5+4 ───
    {
        "name": "zipcode_us_plus4",
        "pattern": r"\b\d{5}-\d{4}\b",
        "entity_type": "ZIPCODE",
        "priority": 8,
        "description": "US ZIP+4 code like 43431-9599",
    },
    # ─── ORGANISATION PLACEHOLDER (already in data) ───
    {
        "name": "org_placeholder",
        "pattern": r"\[(?:ORGANISATION|ORGANIZATION)PLACEHOLDER_\d+\]",
        "entity_type": "ORGANIZATION",
        "priority": 10,
        "description": "Pre-existing organisation placeholder tokens",
    },
    # ─── AMOUNT PLACEHOLDER ───
    {
        "name": "amount_placeholder",
        "pattern": r"\[AMOUNTPLACEHOLDER_\d+\]",
        "entity_type": "NUMBER",
        "priority": 10,
        "description": "Pre-existing amount placeholder tokens",
    },
    # ─── LANGUAGE PLACEHOLDER ───
    {
        "name": "lang_placeholder",
        "pattern": r"\[LANGUAGEPLACEHOLDER_\d+\]",
        "entity_type": "OTHER_PII",
        "priority": 10,
        "description": "Pre-existing language placeholder tokens",
    },
    # ─── ACCOUNT NUMBER: long pure digits (10-18 digits) ───
    {
        "name": "account_number_long",
        "pattern": r"\b\d{10,18}\b",
        "entity_type": "ACCOUNT_NUMBER",
        "priority": 3,
        "description": "Long digit sequences (10-18 digits), likely account numbers",
    },
]


def run_all_patterns(text: str) -> list:
    """
    Run every regex pattern on the input text.
    Returns a list of match dicts: {text, start, end, entity_type, source, ...}
    """
    results = []
    for p in PATTERNS:
        try:
            for match in re.finditer(p["pattern"], text):
                results.append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "entity_type": p["entity_type"],
                    "source": "regex",
                    "pattern_name": p["name"],
                    "priority": p["priority"],
                    "confidence": round(0.7 + (p["priority"] / 50), 3),
                })
        except re.error:
            continue  # skip broken patterns gracefully
    return results
