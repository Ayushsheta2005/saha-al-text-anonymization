"""
Faker-based Replacement Candidate Generator
Generates plausible anonymized replacements that preserve format, length, and diversity.
"""

import random
import re
import string

try:
    from faker import Faker
except ImportError:
    Faker = None

# ─── Initialize Faker with multiple locales ───
if Faker:
    _LOCALE_MAP = {
        "en": Faker("en_US"),
        "en_gb": Faker("en_GB"),
        "fr": Faker("fr_FR"),
        "de": Faker("de_DE"),
        "es": Faker("es_MX"),
        "ja": Faker("ja_JP"),
        "ru": Faker("ru_RU"),
    }
    # Seed for reproducibility during batch processing (but varied per call)
    Faker.seed(0)
else:
    _LOCALE_MAP = {}

_DEFAULT_LOCALES = list(_LOCALE_MAP.keys()) if _LOCALE_MAP else ["en"]


def _get_faker(locale=None):
    """Get a Faker instance for the given locale, fallback to en_US."""
    if not _LOCALE_MAP:
        return None
    if locale and locale in _LOCALE_MAP:
        return _LOCALE_MAP[locale]
    return _LOCALE_MAP.get("en")


def generate_replacements(entity_type: str, original_text: str, n: int = 3) -> list:
    """
    Generate n replacement candidates for a given entity type.
    Falls back gracefully if Faker is not installed.
    """
    try:
        if entity_type in ("FULLNAME", "FIRST_NAME", "LAST_NAME"):
            return _generate_name_replacements(entity_type, original_text, n)
        elif entity_type in ("ID_NUMBER", "PASSPORT", "SSN"):
            return _generate_id_replacements(original_text, n)
        elif entity_type == "PHONE":
            return _generate_phone_replacements(original_text, n)
        elif entity_type == "EMAIL":
            return _generate_email_replacements(original_text, n)
        elif entity_type == "ADDRESS":
            return _generate_address_replacements(original_text, n)
        elif entity_type == "DATE":
            return _generate_date_replacements(original_text, n)
        elif entity_type == "TIME":
            return _generate_time_replacements(original_text, n)
        elif entity_type == "LOCATION":
            return _generate_location_replacements(n)
        elif entity_type == "ORGANIZATION":
            return _generate_org_replacements(original_text, n)
        elif entity_type in ("ACCOUNT_NUMBER", "CREDIT_CARD"):
            return _generate_digit_replacements(original_text, n)
        elif entity_type == "ZIPCODE":
            return _generate_id_replacements(original_text, n)
        elif entity_type == "TITLE":
            return _generate_title_replacements(n)
        elif entity_type == "GENDER":
            return _generate_gender_replacements(n)
        elif entity_type == "NUMBER":
            return _generate_number_replacements(original_text, n)
        else:
            return [f"[REDACTED_{entity_type}]" for _ in range(n)]
    except Exception:
        return [f"[REDACTED_{entity_type}]" for _ in range(n)]


# ────────────────────────────────────────────────────────────────
# Private generators per entity type
# ────────────────────────────────────────────────────────────────

def _generate_name_replacements(entity_type, original, n):
    """Generate name replacements matching original token count."""
    results = []
    token_count = len(original.split())
    f = _get_faker()

    for _ in range(n):
        locale = random.choice(_DEFAULT_LOCALES)
        f = _get_faker(locale)
        if f is None:
            results.append("John Doe")
            continue

        if entity_type == "FIRST_NAME" or token_count <= 1:
            results.append(f.first_name())
        elif entity_type == "LAST_NAME":
            results.append(f.last_name())
        else:
            # Full name — match token count
            name_parts = [f.first_name()]
            while len(name_parts) < token_count:
                inner_f = _get_faker(random.choice(_DEFAULT_LOCALES))
                if inner_f:
                    name_parts.append(inner_f.first_name() if len(name_parts) < token_count - 1 else inner_f.last_name())
                else:
                    name_parts.append("Smith")
            results.append(" ".join(name_parts[:token_count]))

    return results


def _generate_id_replacements(original, n):
    """Preserve character-class pattern: letter→letter, digit→digit."""
    results = []
    for _ in range(n):
        replacement = ""
        for char in original:
            if char.isupper():
                replacement += random.choice(string.ascii_uppercase)
            elif char.islower():
                replacement += random.choice(string.ascii_lowercase)
            elif char.isdigit():
                replacement += str(random.randint(0, 9))
            else:
                replacement += char  # preserve separators, dashes, etc.
        results.append(replacement)
    return results


def _generate_phone_replacements(original, n):
    """Preserve phone format: replace digits, keep separators."""
    results = []
    for _ in range(n):
        replacement = ""
        for char in original:
            if char.isdigit():
                replacement += str(random.randint(0, 9))
            else:
                replacement += char
        results.append(replacement)
    return results


def _generate_email_replacements(original, n):
    """Generate plausible fake emails."""
    results = []
    domains = ["email.com", "mail.org", "inbox.net", "post.io", "msg.com",
               "outlook.com", "protonmail.com", "yahoo.com"]
    
    for _ in range(n):
        f = _get_faker(random.choice(_DEFAULT_LOCALES))
        if f:
            username = f.user_name()
        else:
            username = "user" + str(random.randint(100, 999))
        domain = random.choice(domains)
        results.append(f"{username}@{domain}")
    return results


def _generate_address_replacements(original, n):
    """Generate fake addresses, try to preserve structure."""
    results = []
    f = _get_faker("en")
    if f:
        for _ in range(n):
            locale = random.choice(_DEFAULT_LOCALES)
            lf = _get_faker(locale)
            if lf:
                addr = lf.street_address()
                results.append(addr)
            else:
                results.append(f.street_address())
    else:
        for _ in range(n):
            results.append(f"{random.randint(1, 9999)} Oak Street")
    return results


def _generate_date_replacements(original, n):
    """Preserve date format, randomize values."""
    results = []
    f = _get_faker("en")

    for _ in range(n):
        if f:
            date = f.date_between(start_date="-80y", end_date="-5y")
        else:
            import datetime
            date = datetime.date(random.randint(1940, 2020), random.randint(1, 12), random.randint(1, 28))

        # Detect format from original
        if "T" in original and "-" in original:
            # ISO format
            results.append(date.strftime("%Y-%m-%dT00:00:00"))
        elif "/" in original:
            results.append(date.strftime("%d/%m/%Y"))
        elif re.search(r"\d{1,2}(?:st|nd|rd|th)", original):
            day = date.day
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day if day < 20 else day % 10, "th")
            results.append(f"{day}{suffix} {date.strftime('%B %Y')}")
        elif re.search(r"[A-Z][a-z]+ \d", original):
            results.append(date.strftime("%B %d, %Y"))
        else:
            results.append(date.strftime("%d/%m/%Y"))

    return results


def _generate_time_replacements(original, n):
    """Preserve time format, randomize values."""
    results = []
    for _ in range(n):
        h = random.randint(0, 23)
        m = random.randint(0, 59)
        s = random.randint(0, 59)

        if "AM" in original or "PM" in original or "am" in original or "pm" in original:
            period = "AM" if h < 12 else "PM"
            h12 = h % 12 or 12
            if re.search(r"\d+:\d+:\d+", original):
                results.append(f"{h12}:{m:02d}:{s:02d} {period}")
            else:
                results.append(f"{h12}:{m:02d} {period}")
        else:
            if re.search(r"\d+:\d+:\d+", original):
                results.append(f"{h:02d}:{m:02d}:{s:02d}")
            else:
                results.append(f"{h}:{m:02d}")

    return results


def _generate_location_replacements(n):
    """Generate plausible city/place names."""
    cities = [
        "Springfield", "Oakville", "Riverdale", "Greenfield", "Fairview",
        "Lakewood", "Westport", "Bridgewater", "Eastville", "Northgate",
        "Pinehurst", "Cedar Falls", "Stonebridge", "Clearwater", "Brookside",
        "Maplewood", "Sunnydale", "Hillcrest", "Ashford", "Windermere",
    ]
    return random.sample(cities, min(n, len(cities)))


def _generate_org_replacements(original, n):
    """Generate plausible organization names."""
    # If it's a placeholder like [ORGANISATIONPLACEHOLDER_14], generate real-ish names
    orgs = [
        "Meridian Corp", "Atlas Foundation", "Pinnacle Services",
        "Vanguard Institute", "Horizon Group", "Summit Partners",
        "Nexus Solutions", "Crestview Ltd", "Apex Consulting",
        "Sterling Associates", "Catalyst Holdings", "Evergreen Industries",
    ]
    return random.sample(orgs, min(n, len(orgs)))


def _generate_digit_replacements(original, n):
    """Replace digits only, preserve everything else."""
    results = []
    for _ in range(n):
        replacement = ""
        for char in original:
            if char.isdigit():
                replacement += str(random.randint(0, 9))
            else:
                replacement += char
        results.append(replacement)
    return results


def _generate_title_replacements(n):
    """Generate title/honorific replacements."""
    titles = ["Mr", "Mrs", "Ms", "Dr", "Prof", "Miss"]
    return random.sample(titles, min(n, len(titles)))


def _generate_gender_replacements(n):
    """Generate gender replacements."""
    genders = ["Male", "Female", "Non-binary"]
    return random.sample(genders, min(n, len(genders)))


def _generate_number_replacements(original, n):
    """Generate numeric replacements with similar magnitude."""
    results = []
    try:
        val = int(re.sub(r"[^\d]", "", original))
        for _ in range(n):
            offset = random.uniform(0.5, 1.5)
            new_val = max(1, int(val * offset))
            results.append(str(new_val))
    except (ValueError, TypeError):
        for _ in range(n):
            results.append(str(random.randint(10, 999)))
    return results
