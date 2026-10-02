from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF = "Xyren2005"

# ── Load Encoder ──────────────────────────────────────────────────────────────
enc_tok   = AutoTokenizer.from_pretrained(f"{HF}/pii-ner-roberta")
enc_model = AutoModelForTokenClassification.from_pretrained(f"{HF}/pii-ner-roberta").eval().to(DEVICE)
id2label = {int(k): v for k, v in enc_model.config.id2label.items()}

# ── Load Fillers ──────────────────────────────────────────────────────────────
mlm_tok = AutoTokenizer.from_pretrained(f"{HF}/pii-ner-filler_deberta-filler")
mlm_model = AutoModelForMaskedLM.from_pretrained(f"{HF}/pii-ner-filler_deberta-filler").eval().to(DEVICE)

seq_tok = AutoTokenizer.from_pretrained(f"{HF}/pii-ner-filler_bart-base")
seq_model = AutoModelForSeq2SeqLM.from_pretrained(f"{HF}/pii-ner-filler_bart-base").eval().to(DEVICE)

ENTITY_TYPES = [
    "FULLNAME", "FIRST_NAME", "LAST_NAME", "ID_NUMBER", "PASSPORT", "SSN",
    "PHONE", "EMAIL", "ADDRESS", "DATE", "TIME", "LOCATION", "ORGANIZATION",
    "ACCOUNT_NUM", "CREDIT_CARD", "ZIPCODE", "TITLE", "GENDER", "NUMBER", "OTHER_PII",
]

# ── Step 1: Encoder → detect and mask PII ────────────────────────────────────
def encode_and_mask(text: str) -> str:
    words = text.split()
    if not words: return text
    enc = enc_tok(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        preds = torch.argmax(enc_model(**enc).logits, -1)[0].tolist()
    labels, prev = [], None
    for j, wid in enumerate(enc.word_ids()):
        if wid is None or wid == prev: continue
        labels.append(id2label.get(preds[j], "O"))
        prev = wid

    masked, i = [], 0
    while i < len(words):
        lab = labels[i]
        if lab.startswith("B-"):
            etype = lab[2:]
            i += 1
            while i < len(words) and labels[i] == f"I-{etype}": i += 1
            masked.append(f"[{etype}]")
        else:
            masked.append(words[i])
            i += 1
    return " ".join(masked)

# ── Step 2: Fillers ──────────────────────────────────────────────────────────
def fill_with_mlm(masked_text: str) -> str:
    filled = masked_text
    for etype in ENTITY_TYPES:
        if f"[{etype}]" in filled:
            # Reverting back to 2 masks (best balance for MLM mathematically)
            filled = filled.replace(f"[{etype}]", f"{mlm_tok.mask_token} {mlm_tok.mask_token}")
    
    inputs = mlm_tok(filled, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        logits = mlm_model(**inputs).logits[0]
    ids = inputs.input_ids[0].clone()
    for pos in (ids == mlm_tok.mask_token_id).nonzero(as_tuple=True)[0]:
        ids[pos] = logits[pos].argmax()
    return mlm_tok.decode(ids, skip_special_tokens=True)

def fill_with_seq2seq(masked_text: str) -> str:
    prompt = f"Replace PII placeholders with realistic entities: {masked_text}"
    inputs = seq_tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        out_ids = seq_model.generate(**inputs, max_new_tokens=192, num_beams=4, do_sample=False)
    # Ensure indices are safely within vocab limits
    out_ids = torch.clamp(out_ids, 0, len(seq_tok) - 1)
    return seq_tok.decode(out_ids[0], skip_special_tokens=True).replace("Replace PII placeholders with realistic entities: ", "").strip()

# ── Test ─────────────────────────────────────────────────────────────────────
test_sentences = [
    "Hi, I'm Sarah Connor and my SSN is 532-12-3456.",
    "Please email me at john.doe@gmail.com or call +1-555-987-6543.",
    "My passport number is X4829301 and I live at 221B Baker Street, London.",
    "Transfer $500 from account 87623910 to card 4111111111111111.",
]

for sentence in test_sentences:
    masked = encode_and_mask(sentence)
    print(f"Original : {sentence}")
    print(f"Masked   : {masked}")
    
    if "[" in masked:
        mlm_anon = fill_with_mlm(masked)
        seq_anon = fill_with_seq2seq(masked)
        print(f"DeBERTa (MLM)    : {mlm_anon}")
        print(f"BART (Seq2Seq)   : {seq_anon}")
    print("-" * 80)
