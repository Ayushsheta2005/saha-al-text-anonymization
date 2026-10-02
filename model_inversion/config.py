"""
Model Inversion Attack — Central Configuration
===============================================
All paths, API keys, dataset strategy parameters, and
probe entity pools are defined here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR     = os.path.dirname(BASE_DIR)

DATA_DIR        = os.path.join(BASE_DIR, "data")
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

# Dataset generator outputs
GENERATED_DATASET_FILE  = os.path.join(OUTPUT_DIR, "adversarial_dataset_raw.jsonl")
GENERATION_STATE_FILE   = os.path.join(DATA_DIR,   "generation_state.json")

# BART querier outputs  (step 2)
BART_PAIRS_FILE         = os.path.join(OUTPUT_DIR, "bart_query_pairs.jsonl")
BART_QUERY_STATE_FILE   = os.path.join(DATA_DIR,   "bart_query_state.json")

# Inverter training splits  (step 3)
INVERTER_TRAIN_FILE     = os.path.join(OUTPUT_DIR, "inverter_train.jsonl")
INVERTER_EVAL_FILE      = os.path.join(OUTPUT_DIR, "inverter_eval.jsonl")

# Inverter model checkpoint
INVERTER_CHECKPOINT_DIR = os.path.join(BASE_DIR, "inverter_checkpoint")

# Attack results (step 4)
ATTACK_RESULTS_FILE     = os.path.join(OUTPUT_DIR, "attack_results.json")
ATTACK_REPORT_FILE      = os.path.join(OUTPUT_DIR, "attack_report.txt")

# Consistency analysis (step 2b)
CONSISTENCY_REPORT_FILE = os.path.join(OUTPUT_DIR, "consistency_report.json")

# ── Gemini API ─────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Export it or put it in .env -- never hardcode it. "
        "An earlier revision of this file had the key inline; it has been revoked."
    )
GEMINI_MODEL    = "gemini-2.5-flash"
GEMINI_TEMP     = 0.92          # high creativity — diverse sentences
GEMINI_MAX_TOKENS = 8192        # raised: 2.5-flash uses tokens for thinking BEFORE output

# Rate limits — Gemini 2.5 Flash Tier-1 paid
RPM_LIMIT       = 1000          # requests/minute per key
RPD_LIMIT       = 10_000        # requests/day per key

# ── Generation parameters ──────────────────────────────────────────────────
BATCH_SIZE          = 15        # sentences per Gemini call (raised for throughput)
MAX_PARALLEL_WORKERS = 12       # concurrent threads
TOTAL_TARGET        = 40_000    # total sentences to generate (doubled)
EVAL_FRACTION       = 0.05      # 5% held out for attack evaluation

# ── BART model (victim) ────────────────────────────────────────────────────
# Points to the CE-loss checkpoint (checkpoints/ folder)
BART_MODEL_NAME     = "facebook/bart-base"
BART_CHECKPOINT     = os.path.join(PROJECT_DIR, "Seq2Seq_model", "checkpoints", "bart-base", "best_model.pt")
BART_MAX_INPUT_LEN  = 128
BART_MAX_OUTPUT_LEN = 128
BART_BATCH_SIZE     = 32        # inference batch size

# ── Inverter model ─────────────────────────────────────────────────────────
INVERTER_MODEL_NAME = "facebook/bart-base"   # same arch as victim
INVERTER_EPOCHS     = 20
INVERTER_BATCH_SIZE = 4         # reduced: 3.7 GB VRAM can't fit 16 with fp32
INVERTER_EVAL_BATCH = 8         # reduced accordingly
INVERTER_LR         = 2e-5
INVERTER_MAX_INPUT  = 128       # anonymized text → input to inverter
INVERTER_MAX_TARGET = 128       # original text   → target for inverter
INVERTER_WARMUP     = 200
INVERTER_GRAD_ACCUM = 8         # increased to keep effective batch = 32
INVERTER_NUM_WORKERS = 2

# ── Strategy distribution (must sum to TOTAL_TARGET) ──────────────────────
STRATEGY_DISTRIBUTION = {
    "S1_entity_consistency":       12_000,   # 150 names x ~80 contexts each
    "S2_combinatorial_pii":         8_000,   # controlled PII combos
    "S3_paraphrase_consistency":    6_000,   # same PII, different structure
    "S4_rarity_spectrum":           6_000,   # common -> rare name spectrum
    "S5_cross_entity_correlation":  4_000,   # email prefix matches name, etc.
    "S6_edge_cases":                4_000,   # borderline PII, dense, ambiguous
}
assert sum(STRATEGY_DISTRIBUTION.values()) == TOTAL_TARGET, "Strategy counts must sum to 40000"

# ── S1 Probe entity pool ───────────────────────────────────────────────────
# 150 synthetic multicultural mashup names covering ALL name style types
# from the ai4privacy dataset.  Annotated with origin_mix for analysis.
#
# Categories:
#   MC  = Multicultural mashup  (primary style of ai4privacy)
#   EEU = Eastern European fragment
#   AFR = African fragment
#   SEA = Southeast Asian fragment
#   MID = Middle Eastern fragment
#   LAT = Latin/Romance fragment
#   WEU = Western European
#   SAS = South Asian fragment
#   HYP = Hyphenated / compound
#   RAR = Rare / near-unique

PROBE_NAMES = [
    # ── Multicultural mashups (primary ai4privacy style) ──
    {"name": "Kukshi Welshet",          "origin_mix": ["SAS", "WEU"],  "rarity": "medium"},
    {"name": "Amyna Tharathip",         "origin_mix": ["MID", "SEA"],  "rarity": "rare"},
    {"name": "Svenja-Maria Glayce",     "origin_mix": ["WEU", "HYP"],  "rarity": "medium"},
    {"name": "Mbangila Migali Bajraj",  "origin_mix": ["AFR", "SAS"],  "rarity": "very_rare"},
    {"name": "Qokaj Donnell",           "origin_mix": ["EEU", "WEU"],  "rarity": "rare"},
    {"name": "Ljuljeta Shanon",         "origin_mix": ["EEU", "WEU"],  "rarity": "rare"},
    {"name": "Betselot Mengsteab",      "origin_mix": ["AFR", "AFR"],  "rarity": "very_rare"},
    {"name": "Yidnekachew Tsolakis",    "origin_mix": ["AFR", "EEU"],  "rarity": "very_rare"},
    {"name": "Joévine Roliana",         "origin_mix": ["LAT", "MC"],   "rarity": "rare"},
    {"name": "Valentinetti Stravs Perwein", "origin_mix": ["LAT","EEU","WEU"], "rarity": "very_rare"},
    {"name": "Igorche Ramtin Eshekary","origin_mix": ["EEU", "MID"],  "rarity": "very_rare"},
    {"name": "Fauzieh Mostaghni",       "origin_mix": ["MID", "MID"],  "rarity": "rare"},
    {"name": "Thirumaran Vellasamy",    "origin_mix": ["SAS", "SAS"],  "rarity": "medium"},
    {"name": "Onome Akpovire",          "origin_mix": ["AFR", "AFR"],  "rarity": "medium"},
    {"name": "Mikailo Hrytsenko",       "origin_mix": ["EEU", "EEU"],  "rarity": "medium"},
    {"name": "Arlett Varunee",          "origin_mix": ["WEU", "SEA"],  "rarity": "rare"},
    {"name": "Malpetti Agrawal",        "origin_mix": ["LAT", "SAS"],  "rarity": "rare"},
    {"name": "Enkegaard Testardi",      "origin_mix": ["WEU", "LAT"],  "rarity": "very_rare"},
    {"name": "Xiaoling Murvanidze",     "origin_mix": ["SEA", "EEU"],  "rarity": "very_rare"},
    {"name": "Carnota Muranovic",       "origin_mix": ["LAT", "EEU"],  "rarity": "very_rare"},
    # ── Eastern European fragments ──
    {"name": "Ljiljana Petrovska",      "origin_mix": ["EEU", "EEU"],  "rarity": "medium"},
    {"name": "Bogumila Zarzecka",       "origin_mix": ["EEU", "EEU"],  "rarity": "rare"},
    {"name": "Svitlana Kovalchuk",      "origin_mix": ["EEU", "EEU"],  "rarity": "medium"},
    {"name": "Rastislav Dobronsky",     "origin_mix": ["EEU", "EEU"],  "rarity": "medium"},
    {"name": "Miroslava Hnilicova",     "origin_mix": ["EEU", "EEU"],  "rarity": "rare"},
    {"name": "Dragoljub Stankovic",     "origin_mix": ["EEU", "EEU"],  "rarity": "medium"},
    {"name": "Zuzanna Wisniewska",      "origin_mix": ["EEU", "EEU"],  "rarity": "medium"},
    {"name": "Kazimiera Wojciechowska", "origin_mix": ["EEU", "EEU"],  "rarity": "medium"},
    {"name": "Vlatko Apostolovski",     "origin_mix": ["EEU", "EEU"],  "rarity": "rare"},
    {"name": "Desislava Hristova",      "origin_mix": ["EEU", "EEU"],  "rarity": "medium"},
    # ── African fragments ──
    {"name": "Nkemdirim Okonkwo",       "origin_mix": ["AFR", "AFR"],  "rarity": "medium"},
    {"name": "Abimbola Adeyemi",        "origin_mix": ["AFR", "AFR"],  "rarity": "medium"},
    {"name": "Fatoumata Coulibaly",     "origin_mix": ["AFR", "AFR"],  "rarity": "medium"},
    {"name": "Onyekachi Eze",           "origin_mix": ["AFR", "AFR"],  "rarity": "medium"},
    {"name": "Sibusiso Dlamini",        "origin_mix": ["AFR", "AFR"],  "rarity": "medium"},
    {"name": "Chimamanda Okafor",       "origin_mix": ["AFR", "AFR"],  "rarity": "medium"},
    {"name": "Oluwaseun Akinwande",     "origin_mix": ["AFR", "AFR"],  "rarity": "medium"},
    {"name": "Kwabena Asante-Mensah",   "origin_mix": ["AFR", "AFR"],  "rarity": "medium"},
    {"name": "Tsegay Haileselassie",    "origin_mix": ["AFR", "AFR"],  "rarity": "rare"},
    {"name": "Zodwa Mthethwa",          "origin_mix": ["AFR", "AFR"],  "rarity": "medium"},
    # ── Southeast Asian fragments ──
    {"name": "Pitchaya Sukhonthamat",   "origin_mix": ["SEA", "SEA"],  "rarity": "medium"},
    {"name": "Noppadon Kanchanawat",    "origin_mix": ["SEA", "SEA"],  "rarity": "rare"},
    {"name": "Thitinan Prasomsap",      "origin_mix": ["SEA", "SEA"],  "rarity": "rare"},
    {"name": "Wichai Limthongkul",      "origin_mix": ["SEA", "SEA"],  "rarity": "medium"},
    {"name": "Ratanasak Phumiwasana",   "origin_mix": ["SEA", "SEA"],  "rarity": "very_rare"},
    {"name": "Nguyen Thi Bich Phuong",  "origin_mix": ["SEA", "SEA"],  "rarity": "medium"},
    {"name": "Pattaraporn Srisombat",   "origin_mix": ["SEA", "SEA"],  "rarity": "rare"},
    {"name": "Malinee Wongsawat",       "origin_mix": ["SEA", "SEA"],  "rarity": "medium"},
    {"name": "Jakkrit Chulamanee",      "origin_mix": ["SEA", "SEA"],  "rarity": "rare"},
    {"name": "Saowaluck Phrommas",      "origin_mix": ["SEA", "SEA"],  "rarity": "rare"},
    # ── Middle Eastern fragments ──
    {"name": "Zulfiqar Rahimzadeh",     "origin_mix": ["MID", "MID"],  "rarity": "medium"},
    {"name": "Shahnaz Mohammadpour",    "origin_mix": ["MID", "MID"],  "rarity": "medium"},
    {"name": "Abdulrahman Alhashimi",   "origin_mix": ["MID", "MID"],  "rarity": "medium"},
    {"name": "Fereshteh Kazempour",     "origin_mix": ["MID", "MID"],  "rarity": "rare"},
    {"name": "Gholamhossein Shirazi",   "origin_mix": ["MID", "MID"],  "rarity": "very_rare"},
    {"name": "Marzieh Ebrahimnejad",    "origin_mix": ["MID", "MID"],  "rarity": "rare"},
    {"name": "Noureddine Bensalem",     "origin_mix": ["MID", "MID"],  "rarity": "medium"},
    {"name": "Qamarunnisa Begum",       "origin_mix": ["MID", "SAS"],  "rarity": "rare"},
    {"name": "Tariq Almansouri",        "origin_mix": ["MID", "MID"],  "rarity": "medium"},
    {"name": "Hosseinali Moradi",       "origin_mix": ["MID", "MID"],  "rarity": "rare"},
    # ── Latin/Romance fragments ──
    {"name": "Bartholomeu Vasconcelos", "origin_mix": ["LAT", "LAT"],  "rarity": "medium"},
    {"name": "Xiomara Bustamante",      "origin_mix": ["LAT", "LAT"],  "rarity": "medium"},
    {"name": "Florencio Garmendia",     "origin_mix": ["LAT", "LAT"],  "rarity": "medium"},
    {"name": "Iolanda Nascimento",      "origin_mix": ["LAT", "LAT"],  "rarity": "medium"},
    {"name": "Desiderio Caballero",     "origin_mix": ["LAT", "LAT"],  "rarity": "medium"},
    {"name": "Rosaura Villanueva",      "origin_mix": ["LAT", "LAT"],  "rarity": "medium"},
    {"name": "Epifanio Castellanos",    "origin_mix": ["LAT", "LAT"],  "rarity": "rare"},
    {"name": "Caetana Magalhaes",       "origin_mix": ["LAT", "LAT"],  "rarity": "medium"},
    {"name": "Waldomiro Cavalcante",    "origin_mix": ["LAT", "LAT"],  "rarity": "rare"},
    {"name": "Inmaculada Echeverria",   "origin_mix": ["LAT", "LAT"],  "rarity": "medium"},
    # ── Western European fragments ──
    {"name": "Lieselotte Vanderberghe", "origin_mix": ["WEU", "WEU"],  "rarity": "medium"},
    {"name": "Sigfrid Holmqvist",       "origin_mix": ["WEU", "WEU"],  "rarity": "medium"},
    {"name": "Gudrun Steinsdottir",     "origin_mix": ["WEU", "WEU"],  "rarity": "medium"},
    {"name": "Anneliese Brinkerhoff",   "origin_mix": ["WEU", "WEU"],  "rarity": "medium"},
    {"name": "Bartolomeus Schoenmaker", "origin_mix": ["WEU", "WEU"],  "rarity": "rare"},
    {"name": "Hildegard Winklehner",    "origin_mix": ["WEU", "WEU"],  "rarity": "medium"},
    {"name": "Knud Thorvaldsen",        "origin_mix": ["WEU", "WEU"],  "rarity": "medium"},
    {"name": "Reinhilde Vandermeersch", "origin_mix": ["WEU", "WEU"],  "rarity": "rare"},
    {"name": "Thorbjorn Lindqvist",     "origin_mix": ["WEU", "WEU"],  "rarity": "medium"},
    {"name": "Walburga Steinmetz",      "origin_mix": ["WEU", "WEU"],  "rarity": "medium"},
    # ── South Asian fragments ──
    {"name": "Vethathiri Pillai",       "origin_mix": ["SAS", "SAS"],  "rarity": "rare"},
    {"name": "Gurpreet Dhaliwal",       "origin_mix": ["SAS", "SAS"],  "rarity": "medium"},
    {"name": "Subramanyam Krishnaswamy",    "origin_mix": ["SAS", "SAS"],  "rarity": "medium"},
    {"name": "Manimekalai Subramaniam", "origin_mix": ["SAS", "SAS"],  "rarity": "rare"},
    {"name": "Balakrishnan Nambiar",    "origin_mix": ["SAS", "SAS"],  "rarity": "medium"},
    {"name": "Gangadhar Kulkarni",      "origin_mix": ["SAS", "SAS"],  "rarity": "medium"},
    {"name": "Padmavathi Venkataraman", "origin_mix": ["SAS", "SAS"],  "rarity": "rare"},
    {"name": "Debashish Bhattacharyya", "origin_mix": ["SAS", "SAS"],  "rarity": "medium"},
    {"name": "Sivananda Saraswati",     "origin_mix": ["SAS", "SAS"],  "rarity": "rare"},
    {"name": "Kalavathi Nagarajan",     "origin_mix": ["SAS", "SAS"],  "rarity": "medium"},
    # ── Hyphenated / compound ──
    {"name": "Jan-André Thilina",       "origin_mix": ["HYP", "SEA"],  "rarity": "rare"},
    {"name": "Marie-Céleste Adjoa",     "origin_mix": ["HYP", "AFR"],  "rarity": "rare"},
    {"name": "Daniel-Florin Stanescu",  "origin_mix": ["HYP", "EEU"],  "rarity": "medium"},
    {"name": "Amalia-Elena Ionescu",    "origin_mix": ["HYP", "EEU"],  "rarity": "medium"},
    {"name": "Jean-Baptiste Kouassi",   "origin_mix": ["HYP", "AFR"],  "rarity": "medium"},
    {"name": "Ana-Maria Obretenova",    "origin_mix": ["HYP", "EEU"],  "rarity": "medium"},
    {"name": "Björn-Erik Strömqvist",   "origin_mix": ["HYP", "WEU"],  "rarity": "rare"},
    {"name": "Ioana-Loredana Petrescu", "origin_mix": ["HYP", "EEU"],  "rarity": "medium"},
    {"name": "Karl-Heinz Wusterhausen", "origin_mix": ["HYP", "WEU"],  "rarity": "medium"},
    {"name": "Nur-Aisyah Binte Rashid", "origin_mix": ["HYP", "SEA"],  "rarity": "rare"},
    # ── Very rare / near-unique ──
    {"name": "Zephyranth Kowalczyk",    "origin_mix": ["RAR", "EEU"],  "rarity": "very_rare"},
    {"name": "Yohann Krishnaswamy",     "origin_mix": ["RAR", "SAS"],  "rarity": "very_rare"},
    {"name": "Oscarine Mwangangi",      "origin_mix": ["RAR", "AFR"],  "rarity": "very_rare"},
    {"name": "Theodoros Charalambous",  "origin_mix": ["EEU", "MID"],  "rarity": "very_rare"},
    {"name": "Przemyslaw Wojciechowicz",    "origin_mix": ["EEU", "EEU"], "rarity": "very_rare"},
    {"name": "Ragnheiður Sigurbjörnsson",   "origin_mix": ["WEU", "WEU"], "rarity": "very_rare"},
    {"name": "Wiesczyslaw Przybyszewski",   "origin_mix": ["EEU", "EEU"], "rarity": "very_rare"},
    {"name": "Tenzin Wangchuk Dorje",   "origin_mix": ["SAS", "SAS"],  "rarity": "very_rare"},
    {"name": "Nnamdi Uzochukwu",        "origin_mix": ["AFR", "AFR"],  "rarity": "very_rare"},
    {"name": "Saowanee Phetcharat",     "origin_mix": ["SEA", "SEA"],  "rarity": "very_rare"},
    # ── Common (classic single-token ai4privacy style) ──
    {"name": "Feta",                    "origin_mix": ["MC"],           "rarity": "common"},
    {"name": "Naraya",                  "origin_mix": ["MC"],           "rarity": "common"},
    {"name": "Hatto",                   "origin_mix": ["MC"],           "rarity": "common"},
    {"name": "Kefsera",                 "origin_mix": ["MC"],           "rarity": "common"},
    {"name": "Manina",                  "origin_mix": ["MC"],           "rarity": "common"},
    {"name": "Avyan",                   "origin_mix": ["MC"],           "rarity": "common"},
    {"name": "Kidane",                  "origin_mix": ["AFR"],          "rarity": "common"},
    {"name": "Semainesh",               "origin_mix": ["AFR"],          "rarity": "common"},
    {"name": "Shaan",                   "origin_mix": ["SAS"],          "rarity": "common"},
    {"name": "Cleane",                  "origin_mix": ["MC"],           "rarity": "common"},
    # ── Additional mixed ──
    {"name": "Pieralli Souzade",        "origin_mix": ["LAT", "MC"],   "rarity": "rare"},
    {"name": "Flavian Niblack",         "origin_mix": ["LAT", "WEU"],  "rarity": "rare"},
    {"name": "Masooma Mirsada",         "origin_mix": ["MID", "EEU"],  "rarity": "rare"},
    {"name": "Bentley Edmilson",        "origin_mix": ["WEU", "LAT"],  "rarity": "medium"},
    {"name": "Schneppat Dreizler",      "origin_mix": ["WEU", "WEU"],  "rarity": "very_rare"},
    {"name": "Shula Clercx Abbani",     "origin_mix": ["MID","WEU","MID"],"rarity": "very_rare"},
    {"name": "Mengsteab Jeevithan",     "origin_mix": ["AFR", "SAS"],  "rarity": "very_rare"},
    {"name": "Lizbeth Vanderhoeven",    "origin_mix": ["WEU", "WEU"],  "rarity": "medium"},
    {"name": "Nacira Berrehail",        "origin_mix": ["MID", "MID"],  "rarity": "medium"},
    {"name": "Rasel Chowdhury",         "origin_mix": ["SAS", "SAS"],  "rarity": "medium"},
    {"name": "Aljit Sandhu",            "origin_mix": ["SAS", "SAS"],  "rarity": "medium"},
    {"name": "Paata Kvaratskhelia",     "origin_mix": ["EEU", "EEU"],  "rarity": "rare"},
    {"name": "Kökden Erdogan",          "origin_mix": ["MID", "MID"],  "rarity": "rare"},
    {"name": "Sovan Socheat",           "origin_mix": ["SEA", "SEA"],  "rarity": "medium"},
    {"name": "Lovis Devlete",           "origin_mix": ["WEU", "EEU"],  "rarity": "rare"},
    # ── Final 15 to reach 150 ──
    {"name": "Zornitsa Belcheva",        "origin_mix": ["EEU", "EEU"],  "rarity": "medium"},
    {"name": "Chiamaka Uchenna",         "origin_mix": ["AFR", "AFR"],  "rarity": "medium"},
    {"name": "Phunsuk Wangdu",           "origin_mix": ["SAS", "SAS"],  "rarity": "rare"},
    {"name": "Genoveva Czerniawska",     "origin_mix": ["LAT", "EEU"],  "rarity": "very_rare"},
    {"name": "Natsagsuren Gantulga",     "origin_mix": ["SEA", "SEA"],  "rarity": "very_rare"},
    {"name": "Roswitha Kaltenbrunner",   "origin_mix": ["WEU", "WEU"],  "rarity": "medium"},
    {"name": "Amadou Diallo",            "origin_mix": ["AFR", "AFR"],  "rarity": "common"},
    {"name": "Tarcsisia Vörösmarty",     "origin_mix": ["EEU", "EEU"],  "rarity": "very_rare"},
    {"name": "Bhavishya Nandakumar",     "origin_mix": ["SAS", "SAS"],  "rarity": "rare"},
    {"name": "Saodat Raximova",          "origin_mix": ["MID", "EEU"],  "rarity": "rare"},
    {"name": "Wilfrido Encarnacion",     "origin_mix": ["LAT", "LAT"],  "rarity": "medium"},
    {"name": "Ingibjorg Sigurdardottir", "origin_mix": ["WEU", "WEU"],  "rarity": "rare"},
    {"name": "Cevdet Yildizoglu",        "origin_mix": ["MID", "MID"],  "rarity": "medium"},
    {"name": "Wiphawan Suradech",        "origin_mix": ["SEA", "SEA"],  "rarity": "rare"},
    {"name": "Saoirse Caoimhe Bhriain",  "origin_mix": ["HYP", "WEU"],  "rarity": "very_rare"},
]

assert len(PROBE_NAMES) == 150, f"Expected 150 probe names, got {len(PROBE_NAMES)}"

# ── Context domains for sentence generation ────────────────────────────────
CONTEXT_DOMAINS = [
    "chat_message",
    "formal_letter",
    "todo_reminder",
    "document_record",
    "introduction",
    "order_confirmation",
    "community_notice",
    "medical_note",
    "financial_reference",
    "workshop_schedule",
    "creative_narrative",
    "legal_memo",
    "travel_booking",
    "educational_record",
]

# ── PII combo types for S2 ─────────────────────────────────────────────────
PII_COMBO_TYPES = [
    "NAME_PHONE",
    "NAME_EMAIL",
    "NAME_DATE",
    "NAME_PHONE_EMAIL",
    "NAME_PHONE_DATE",
    "NAME_EMAIL_DATE",
    "NAME_ID_DOCUMENT",
    "NAME_ORG_DATE",
    "PHONE_EMAIL",           # no name — tests non-name recovery
    "NAME_ADDRESS_PHONE",
]
