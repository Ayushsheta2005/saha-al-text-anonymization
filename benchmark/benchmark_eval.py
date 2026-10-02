import json
import re
import argparse
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


def normalize_prediction_text(pred_text):
    if pred_text is None:
        return "[EMPTY]"
    pred_text = pred_text.strip()
    return pred_text if pred_text else "[EMPTY]"


def exact_entity_match(ent_text, pred_text):
    if not ent_text or not pred_text:
        return False
    pattern = rf"(?<!\w){re.escape(ent_text)}(?!\w)"
    return re.search(pattern, pred_text, flags=re.IGNORECASE) is not None


def entity_leakage_rate(gold_records, predictions):
    leaked = 0
    total = 0
    per_type = Counter()
    leaked_per_type = Counter()

    for g, p in zip(gold_records, predictions):
        pred_text = normalize_prediction_text(p.get("anonymized_text", ""))
        for ent in g.get("entities", []):
            ent_text = ent.get("text", "")
            ent_type = ent.get("type", "UNKNOWN") or "UNKNOWN"
            total += 1
            per_type[ent_type] += 1
            if exact_entity_match(ent_text, pred_text):
                leaked += 1
                leaked_per_type[ent_type] += 1

    elr = leaked / total if total > 0 else 0.0
    return elr * 100, leaked, total, per_type, leaked_per_type

def get_capitalized_ngrams(text, n=3):
    """Extract n-grams where at least one token starts with uppercase."""
    tokens = text.split()
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return [ng for ng in ngrams if any(len(t) > 0 and t[0].isupper() for t in ng)]

def crr3(gold_records, predictions):
    """Capitalized 3-gram survival rate."""
    survived = 0
    total = 0
    
    for g, p in zip(gold_records, predictions):
        orig_3grams = get_capitalized_ngrams(g.get("original_text", ""), n=3)
        pred_text = normalize_prediction_text(p.get("anonymized_text", "")).lower()
        
        for ng in orig_3grams:
            total += 1
            if " ".join(ng).lower() in pred_text:
                survived += 1
                
    crr = survived / total if total > 0 else 0.0
    return crr * 100

def _patch_bert_score_sent_encode_for_new_transformers():
    """
    bert_score.utils.sent_encode calls tokenizer.build_inputs_with_special_tokens([])
    for empty strings; recent transformers versions removed that public API.
    Patch only the empty-string path so encode() is used instead.
    """
    import bert_score.utils as bs_utils

    if getattr(bs_utils, "_sent_encode_orig", None) is None:
        bs_utils._sent_encode_orig = bs_utils.sent_encode

        def sent_encode(tokenizer, sent):
            if not sent.strip() and not hasattr(
                tokenizer, "build_inputs_with_special_tokens"
            ):
                return tokenizer.encode("", add_special_tokens=True)
            return bs_utils._sent_encode_orig(tokenizer, sent)

        bs_utils.sent_encode = sent_encode


def calculate_bertscore(gold_records, predictions, model_type="distilbert-base-uncased"):
    """
    BERTScore F1 computes the semantic similarity of the texts.
    Pinned model for reproducible benchmarking computations.
    """
    try:
        from bert_score import score

        _patch_bert_score_sent_encode_for_new_transformers()
    except ImportError:
        print("`bert_score` not installed. Skipping. Install with: pip install bert_score")
        return None
        
    refs = [g.get("original_text", "") for g in gold_records]
    cands = [normalize_prediction_text(p.get("anonymized_text", "")) for p in predictions]

    paired = [(c, r) for c, r in zip(cands, refs) if (r or "").strip()]
    skipped = len(refs) - len(paired)
    if skipped:
        print(
            f"[NOTE] Skipped {skipped} record(s) with empty gold original_text for BERTScore "
            "(bert_score would otherwise force those rows to 0 and print a warning)."
        )
    if not paired:
        print("[NOTE] No non-empty references left; BERTScore skipped.")
        return None

    cands, refs = zip(*paired)
    cands, refs = list(cands), list(refs)

    P, R, F1 = score(cands, refs, lang="en", verbose=False, model_type=model_type)
    return F1.mean().item() * 100

def main():
    parser = argparse.ArgumentParser(description="SAHA-AL Benchmark Evaluator")
    parser.add_argument("--gold", type=str, default="data/test.jsonl", help="Path to gold dataset (e.g. test.jsonl)")
    parser.add_argument("--pred", type=str, required=True, help="Path to predictions JSONL")
    parser.add_argument("--bert-model", type=str, default="distilbert-base-uncased",
                        help="Model to use for BERTScore (e.g., microsoft/deberta-xlarge-mnli)")
    parser.add_argument("--print-types", action="store_true", help="Print per-entity-type ELR breakdown")
    parser.add_argument("--summary-file", type=str, default=None,
                        help="Optional JSON file to write evaluation results to")
    args = parser.parse_args()
    
    with open(args.gold, "r", encoding="utf-8") as f:
        gold_records = [json.loads(line) for line in f]
        
    with open(args.pred, "r", encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f]
        
    # Every metric below pairs gold with predictions via zip(), which silently
    # truncates to the shorter list. Without this check a submission containing
    # only the easy 10% of the test set would pass the per-record id check and
    # score on that subset -- so refuse anything that is not a complete
    # prediction over the frozen split.
    if len(predictions) != len(gold_records):
        raise ValueError(
            f"prediction count {len(predictions)} does not match gold count "
            f"{len(gold_records)}. The frozen split must be predicted in full; "
            "partial submissions are not comparable."
        )

    unknown_entity_count = 0
    total_entity_count = 0

    for g, p in zip(gold_records, predictions):
        gold_id = g.get("id", g.get("entry_id"))
        pred_id = p.get("id")
        if gold_id != pred_id:
            raise ValueError(f"ID mismatch: {gold_id} vs {pred_id}")

        for ent in g.get("entities", []):
            total_entity_count += 1
            if ent.get("type", "UNKNOWN") == "UNKNOWN":
                unknown_entity_count += 1

    if total_entity_count > 0 and unknown_entity_count > 0:
        pct = (unknown_entity_count / total_entity_count) * 100
        print(
            f"[NOTE] {pct:.1f}% of entities in gold ({args.gold}) have type UNKNOWN. "
            "ELR/CRR/BERTScore still use entity text and full documents; only per-type "
            "breakdowns (--print-types) need real types in the gold file."
        )

    print(f"Evaluating {len(predictions)} records...")

    elr, leaked, total_ents, per_type, leaked_per_type = entity_leakage_rate(gold_records, predictions)
    crr_3 = crr3(gold_records, predictions)
    bert_f1 = calculate_bertscore(gold_records, predictions, model_type=args.bert_model)
    
    print("\n" + "="*40)
    print(" SAHA-AL Benchmark Results")
    print("="*40)
    print(f" Entity Leakage Rate (ELR ↓): {elr:5.2f}% ({leaked}/{total_ents} leaked)")
    print(f" Contextual Re-ID (CRR-3 ↓):  {crr_3:5.2f}%")
    if bert_f1 is not None:
        print(f" BERTScore (F1 ↑):            {bert_f1:5.2f} (Model: {args.bert_model})")
    else:
        print(f" BERTScore (F1 ↑):            N/A")

    if args.print_types:
        print("\nPer-entity-type ELR:")
        for ent_type, count in per_type.most_common():
            leaked_count = leaked_per_type.get(ent_type, 0)
            elr_type = (leaked_count / count * 100) if count > 0 else 0.0
            print(f"  {ent_type:15} {elr_type:5.2f}% ({leaked_count}/{count})")

    print("="*40)

    if args.summary_file:
        summary = {
            "gold": args.gold,
            "predictions": args.pred,
            "records": len(predictions),
            "elr": round(elr, 2),
            "leaked": leaked,
            "total_entities": total_ents,
            "crr_3": round(crr_3, 2),
            "bert_f1": round(bert_f1, 2) if bert_f1 is not None else None,
            "bert_model": args.bert_model,
            "entity_types": {
                ent_type: {
                    "count": count,
                    "leaked": leaked_per_type.get(ent_type, 0),
                    "elr": round((leaked_per_type.get(ent_type, 0) / count * 100) if count > 0 else 0.0, 2),
                }
                for ent_type, count in per_type.items()
            },
        }
        with open(args.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to: {args.summary_file}")

if __name__ == "__main__":
    main()
