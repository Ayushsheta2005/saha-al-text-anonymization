"""
SAHA-AL Configuration
Central source of truth for all paths, thresholds, entity types, and settings.
"""

import os

# ─── BASE DIRECTORY ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── DATA PATHS ───
HF_DATASET_NAME = "ai4privacy/open-pii-masking-500k-ai4privacy"
ANNOTATION_QUEUE_PATH = os.path.join(BASE_DIR, "data", "annotation_queue.jsonl")
GOLD_STANDARD_PATH = os.path.join(BASE_DIR, "data", "gold_standard.jsonl")
SKIPPED_PATH = os.path.join(BASE_DIR, "data", "skipped.jsonl")
FLAGGED_PATH = os.path.join(BASE_DIR, "data", "flagged.jsonl")
BACKUP_DIR = os.path.join(BASE_DIR, "data", "backups")

# ─── LOG PATHS ───
ANNOTATION_LOG_PATH = os.path.join(BASE_DIR, "logs", "annotation_log.jsonl")

# ─── QUALITY CHECK THRESHOLDS ───
MAX_LENGTH_RATIO = 1.5   # anonymized text shouldn't be 1.5x longer than original
MIN_LENGTH_RATIO = 0.5   # or less than half

# ─── AUGMENTATION PATHS ───
AUGMENTED_DATA_PATH = os.path.join(BASE_DIR, "data", "augmented_data.jsonl")
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.jsonl")

# ─── AUGMENTATION DEFAULTS ───
AUG_ENTITY_SWAP_MULTIPLIER = 4     # Dai & Adel, COLING 2020
AUG_TEMPLATE_FILL_COUNT = 5000     # Anaby-Tavor et al., AAAI 2020
AUG_EDA_MULTIPLIER = 3             # Wei & Zou, EMNLP 2019
AUG_EDA_ALPHA = 0.1                # Fraction of words changed per EDA operation
AUG_BACKTRANSL_MULTIPLIER = 2      # Sennrich et al., WMT 2016
AUG_BACKTRANSL_PIVOT_LANG = "de"   # Pivot language for back-translation
AUG_CONTEXTUAL_MLM_MULTIPLIER = 2  # Kobayashi, ACL 2018
AUG_CONTEXTUAL_MLM_PROB = 0.15     # Fraction of non-entity words to replace

# ─── AI4PRIVACY ENTITY TYPES ───
ENTITY_TYPES = [
    "FIRSTNAME",
    "LASTNAME",
    "MIDDLENAME",
    "TELEPHONENUM",
    "EMAILMASK",
    "STREETADDRESS",
    "CITY",
    "STATE",
    "ZIPCODE",
    "COUNTRY",
    "CREDITCARDNUMBER",
    "SSN",
    "COMPANYNAME",
    "USERNAME",
    "PASSWORD",
    "DOB",
    "AGE",
    "IPADDRESS",
    "MACADDRESS",
    "URL",
    "ETHEREUMADDRESS",
    "BITCOINADDRESS",
    "LITECOINADDRESS",
    "DRIVINGLICENSE",
    "PASSPORT",
    "IBAN",
    "SWIFTCODE",
    "VEHICLEVIN",
    "VEHICLEVRM",
    "PREFIX",
    "JOBTITLE",
    "BUILDINGNUMBER",
    "SECONDARYADDRESS",
]
