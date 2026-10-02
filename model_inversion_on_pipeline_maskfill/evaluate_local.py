"""
evaluate_local.py
─────────────────
Local evaluation of the trained PII model-inversion inverters.

Downloads the model from HuggingFace (cached after first run), runs
inference on the local test split, and writes a detailed report.

Usage
─────
  # Evaluate combo 1  (RoBERTa NER + DeBERTa MLM filler)
  python3 evaluate_local.py --combo 1

  # Evaluate combo 2  (RoBERTa NER + BART seq2seq filler)
  python3 evaluate_local.py --combo 2

  # Override any default
  python3 evaluate_local.py --combo 2 --device cpu --batch_size 16

Full options
  --combo         1 or 2  (sets model_id and test_file automatically)
  --model_id      override the HF model repo
  --test_file     override the local test JSONL path
  --output_dir    where to write reports  (default: output/eval_local)
  --batch_size    inference batch size    (default: 32)
  --num_beams     beam search width       (default: 4)
  --num_examples  samples in report       (default: 30)
  --device        cuda or cpu             (auto-falls back to cpu)
"""

import argparse, json, math, os, random, textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Combo presets ─────────────────────────────────────────────────────────────
_OUTPUT_DIR = Path(__file__).parent / "output"
COMBO_PRESETS = {
    1: {
        "model_id"   : "JALAPENO11/pii-inverter-roberta-deberta",
        "test_file"  : str(_OUTPUT_DIR / "combo1_roberta_deberta_test.jsonl"),
        "combo_label": "Combo 1 — RoBERTa NER + DeBERTa MLM filler",
    },
    2: {
        "model_id"   : "JALAPENO11/pii-inverter-roberta-bart",
        "test_file"  : str(_OUTPUT_DIR / "combo2_roberta_bart_test.jsonl"),
        "combo_label": "Combo 2 — RoBERTa NER + BART seq2seq filler",
    },
}

# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--combo", type=int, choices=[1, 2], default=1,
                   help="Which combo to evaluate (1 or 2). Sets model_id and test_file.")
    p.add_argument("--model_id",  default=None,
                   help="Override the HuggingFace model repo.")
    p.add_argument("--test_file", default=None,
                   help="Override the local test JSONL file path.")
    p.add_argument("--output_dir", default=str(
        Path(__file__).parent / "output" / "eval_local"))
    p.add_argument("--batch_size",     type=int, default=32)
    p.add_argument("--num_beams",      type=int, default=4)
    p.add_argument("--max_input_len",  type=int, default=128)
    p.add_argument("--max_target_len", type=int, default=128)
    p.add_argument("--num_examples",   type=int, default=30,
                   help="Number of sample predictions printed in the report")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # Apply preset, then let explicit overrides win
    preset = COMBO_PRESETS[args.combo]
    if args.model_id is None:
        args.model_id = preset["model_id"]
    if args.test_file is None:
        args.test_file = preset["test_file"]
    args.combo_label = preset["combo_label"]
    return args


# ── Metric helpers (no extra deps) ───────────────────────────────────────────
def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def _sentence_bleu(hyp: str, ref: str, max_n: int = 4) -> float:
    h, r = hyp.lower().split(), ref.lower().split()
    if not h:
        return 0.0
    bp = 1.0 if len(h) >= len(r) else math.exp(1 - len(r) / len(h))
    scores = []
    for n in range(1, max_n + 1):
        hn, rn = _ngrams(h, n), _ngrams(r, n)
        if not hn:
            scores.append(0.0)
            continue
        match = sum(min(hn[k], rn[k]) for k in hn)
        scores.append(match / sum(hn.values()))
    if any(s == 0 for s in scores):
        return 0.0
    return bp * math.exp(sum(math.log(s) for s in scores) / len(scores))

def corpus_bleu(preds, refs, max_n=4):
    return sum(_sentence_bleu(p, r, max_n) for p, r in zip(preds, refs)) / max(len(preds), 1)

def _rouge_lcs(hyp_tokens, ref_tokens):
    m, n = len(hyp_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    # DP LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / m
    recall    = lcs / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def _rouge_n(hyp, ref, n):
    hn, rn = _ngrams(hyp.lower().split(), n), _ngrams(ref.lower().split(), n)
    if not rn:
        return 0.0
    match = sum(min(hn[k], rn[k]) for k in rn)
    return match / sum(rn.values())

def compute_rouge(preds, refs):
    r1 = sum(_rouge_n(p, r, 1) for p, r in zip(preds, refs)) / max(len(preds), 1)
    r2 = sum(_rouge_n(p, r, 2) for p, r in zip(preds, refs)) / max(len(preds), 1)
    rl = sum(_rouge_lcs(p.lower().split(), r.lower().split())
             for p, r in zip(preds, refs)) / max(len(preds), 1)
    return r1, r2, rl

def token_accuracy(preds, refs, tokenizer):
    correct = total = 0
    for p, r in zip(preds, refs):
        pt, rt = tokenizer.tokenize(p), tokenizer.tokenize(r)
        ml = min(len(pt), len(rt))
        if ml == 0:
            continue
        correct += sum(a == b for a, b in zip(pt[:ml], rt[:ml]))
        total   += max(len(pt), len(rt))
    return correct / max(total, 1)

def word_accuracy(preds, refs):
    correct = total = 0
    for p, r in zip(preds, refs):
        pw, rw = p.lower().split(), r.lower().split()
        ml = min(len(pw), len(rw))
        if ml == 0:
            continue
        correct += sum(a == b for a, b in zip(pw[:ml], rw[:ml]))
        total   += max(len(pw), len(rw))
    return correct / max(total, 1)

def entity_recovery_rate(preds, refs):
    """Fraction of reference word-tokens (len>2) that appear in the prediction."""
    exact_count = total = 0
    for pred, ref in zip(preds, refs):
        tokens = [t for t in ref.lower().split() if len(t) > 2]
        total += len(tokens)
        pred_lower = pred.lower()
        for t in tokens:
            if t in pred_lower:
                exact_count += 1
    return (exact_count / max(total, 1), total)

def per_sample_bleus(preds, refs):
    return [round(_sentence_bleu(p, r), 4) for p, r in zip(preds, refs)]


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device : {device}")

    # Load model & tokenizer
    print(f"\nLoading model  : {args.model_id}")
    # The tokenizer is never trained — always load from the base BART model.
    # Loading from the HF repo can give the wrong tokenizer (e.g. DeBERTa's)
    # if Seq2SeqTrainer saved the session's last-loaded tokenizer by mistake.
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    model     = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
    model.to(device).eval()
    # The saved generation_config.json may have max_length=20 from the base
    # model checkpoint.  Setting it to None removes the conflict with our
    # max_new_tokens argument and silences the per-batch warning.
    model.generation_config.max_length = None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters     : {n_params/1e6:.1f} M")

    # Load test data
    print(f"\nLoading test file: {args.test_file}")
    rows = []
    with open(args.test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Test samples   : {len(rows):,}")

    all_anons = [r["anonymized"] for r in rows]
    all_refs  = [r["original"]   for r in rows]

    # Generate predictions
    print(f"\nGenerating predictions (batch={args.batch_size}, beams={args.num_beams}) ...")
    all_preds = []
    n_batches = math.ceil(len(all_anons) / args.batch_size)
    for b_idx in range(n_batches):
        start = b_idx * args.batch_size
        batch = all_anons[start : start + args.batch_size]
        enc = tokenizer(
            batch,
            max_length=args.max_input_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=args.max_target_len,
                num_beams=args.num_beams,
            )
        all_preds.extend(tokenizer.batch_decode(out_ids, skip_special_tokens=True))
        if (b_idx + 1) % 10 == 0 or b_idx == n_batches - 1:
            done = min(start + args.batch_size, len(all_anons))
            # Running stats over predictions collected so far
            cur_preds = all_preds
            cur_refs  = all_refs[:len(cur_preds)]
            r_bleu4   = corpus_bleu(cur_preds, cur_refs, max_n=4) * 100
            r_rouge1, _, r_rougeL = compute_rouge(cur_preds, cur_refs)
            r_err, _  = entity_recovery_rate(cur_preds, cur_refs)
            r_exact   = sum(p.strip() == r.strip()
                            for p, r in zip(cur_preds, cur_refs)) / len(cur_preds)
            print(
                f"  [{done:>5}/{len(all_anons)}]  batch {b_idx+1}/{n_batches}"
                f"  |  BLEU-4: {r_bleu4:5.2f}"
                f"  ROUGE-1: {r_rouge1:.4f}"
                f"  ROUGE-L: {r_rougeL:.4f}"
                f"  ERR: {r_err:.4f}"
                f"  Exact: {r_exact:.4f}"
            )

    print("Inference complete.\n")

    # ── Split: trivial (original==anonymized) vs hard (PII was changed) ───
    # When the NER finds no PII the pipeline leaves the sentence unchanged,
    # so original==anonymized.  The model trivially copies the input → inflates
    # every metric.  We report both overall AND hard-only numbers.
    hard_mask = [a.strip() != r.strip() for a, r in zip(all_anons, all_refs)]
    n_trivial = sum(1 for m in hard_mask if not m)
    n_hard    = sum(hard_mask)

    hard_preds = [p for p, m in zip(all_preds, hard_mask) if m]
    hard_refs  = [r for r, m in zip(all_refs,  hard_mask) if m]
    hard_anons = [a for a, m in zip(all_anons, hard_mask) if m]

    print(f"  Trivial samples (original == anonymized): {n_trivial:,}  ({n_trivial/len(rows)*100:.1f}%)")
    print(f"  Hard samples    (PII was changed)       : {n_hard:,}  ({n_hard/len(rows)*100:.1f}%)")

    # ── Compute metrics ────────────────────────────────────────────────────
    print("\nComputing metrics ...")

    def _compute_metric_set(preds, refs):
        b1 = corpus_bleu(preds, refs, max_n=1) * 100
        b2 = corpus_bleu(preds, refs, max_n=2) * 100
        b4 = corpus_bleu(preds, refs, max_n=4) * 100
        r1, r2, rL = compute_rouge(preds, refs)
        ta  = token_accuracy(preds, refs, tokenizer)
        wa  = word_accuracy(preds, refs)
        ex  = sum(p.strip() == r.strip() for p, r in zip(preds, refs)) / max(len(preds), 1)
        err, err_total = entity_recovery_rate(preds, refs)
        sb  = per_sample_bleus(preds, refs)
        return dict(
            BLEU1=round(b1, 4), BLEU2=round(b2, 4), BLEU4=round(b4, 4),
            ROUGE1=round(r1, 4), ROUGE2=round(r2, 4), ROUGEL=round(rL, 4),
            token_accuracy=round(ta, 4), word_accuracy=round(wa, 4),
            exact=round(ex, 4), err=round(err, 4),
            err_total=err_total,
            mean_sbleu=round(sum(sb) / max(len(sb), 1), 4),
            s_bleus=sb,
        )

    overall = _compute_metric_set(all_preds, all_refs)
    hard    = _compute_metric_set(hard_preds, hard_refs) if hard_preds else None

    metrics = {
        "model_id"              : args.model_id,
        "combo_label"           : args.combo_label,
        "test_file"             : args.test_file,
        "n_test_samples"        : len(all_preds),
        "n_trivial_samples"     : n_trivial,
        "n_hard_samples"        : n_hard,
        "num_beams"             : args.num_beams,
        # ── overall ──
        "BLEU-1"                : overall["BLEU1"],
        "BLEU-2"                : overall["BLEU2"],
        "BLEU-4"                : overall["BLEU4"],
        "ROUGE-1"               : overall["ROUGE1"],
        "ROUGE-2"               : overall["ROUGE2"],
        "ROUGE-L"               : overall["ROUGEL"],
        "token_accuracy"        : overall["token_accuracy"],
        "word_accuracy"         : overall["word_accuracy"],
        "exact_sentence_match"  : overall["exact"],
        "ERR_exact"             : overall["err"],
        "ERR_total_tokens_probed": overall["err_total"],
        "mean_sentence_bleu"    : overall["mean_sbleu"],
        # ── hard only ──
        "hard_BLEU-1"           : hard["BLEU1"]          if hard else None,
        "hard_BLEU-2"           : hard["BLEU2"]          if hard else None,
        "hard_BLEU-4"           : hard["BLEU4"]          if hard else None,
        "hard_ROUGE-1"          : hard["ROUGE1"]         if hard else None,
        "hard_ROUGE-2"          : hard["ROUGE2"]         if hard else None,
        "hard_ROUGE-L"          : hard["ROUGEL"]         if hard else None,
        "hard_token_accuracy"   : hard["token_accuracy"] if hard else None,
        "hard_word_accuracy"    : hard["word_accuracy"]  if hard else None,
        "hard_exact_match"      : hard["exact"]          if hard else None,
        "hard_ERR_exact"        : hard["err"]            if hard else None,
        "hard_mean_sentence_bleu": hard["mean_sbleu"]    if hard else None,
        "reported_at"           : datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    s_bleus = overall["s_bleus"]

    # ── Build report ───────────────────────────────────────────────────────
    sep  = "=" * 80
    thin = "-" * 80

    report  = []
    report += [sep,
               "  PII MODEL-INVERSION ATTACK — LOCAL EVALUATION REPORT",
               f"  Model   : {args.model_id}",
               f"  Dataset : {args.test_file}",
               f"  Combo   : {args.combo_label}",
               f"  Samples : {len(all_preds):,}  |  Beams: {args.num_beams}",
               f"  Date    : {metrics['reported_at']}",
               sep, ""]

    report += [
        f"  NOTE: {n_trivial:,} / {len(all_preds):,} samples ({n_trivial/len(all_preds)*100:.1f}%)"
        f" have original == anonymized (NER found no PII).",
        "  These trivially inflate all metrics — see 'Hard only' column below.",
        ""]

    h = hard or {}
    def _hv(key):
        v = h.get(key)
        return f"{v:>12.4f}" if v is not None else f"{'—':>12}"

    report += ["  METRICS", thin,
               f"  {'Metric':<32} {'Overall (all)':>14} {'Hard only (PII changed)':>24}", thin]
    metric_rows = [
        ("BLEU-1",               f"{overall['BLEU1']:>14.4f}", _hv("BLEU1")),
        ("BLEU-2",               f"{overall['BLEU2']:>14.4f}", _hv("BLEU2")),
        ("BLEU-4",               f"{overall['BLEU4']:>14.4f}", _hv("BLEU4")),
        ("Mean Sentence BLEU",   f"{overall['mean_sbleu']:>14.4f}", _hv("mean_sbleu")),
        ("ROUGE-1",              f"{overall['ROUGE1']:>14.4f}", _hv("ROUGE1")),
        ("ROUGE-2",              f"{overall['ROUGE2']:>14.4f}", _hv("ROUGE2")),
        ("ROUGE-L",              f"{overall['ROUGEL']:>14.4f}", _hv("ROUGEL")),
        ("Token Accuracy",       f"{overall['token_accuracy']:>14.4f}", _hv("token_accuracy")),
        ("Word Accuracy",        f"{overall['word_accuracy']:>14.4f}", _hv("word_accuracy")),
        ("Exact Sentence Match", f"{overall['exact']:>14.4f}", _hv("exact")),
        ("Entity Recovery Rate", f"{overall['err']:>14.4f}", _hv("err")),
        ("Samples",              f"{len(all_preds):>14,}", f"{n_hard:>24,}"),
    ]
    for label, col1, col2 in metric_rows:
        report.append(f"  {label:<32} {col1}  {col2}")
    report += [thin, ""]

    report += [
        "  METRIC DESCRIPTIONS", thin,
        "  BLEU-1/2/4     : n-gram precision vs reference; BLEU-4 is most common.",
        "  ROUGE-1/2/L    : recall-oriented overlap (unigrams, bigrams, LCS).",
        "  Token Accuracy : fraction of sub-word tokens matching ref at same position.",
        "  Word Accuracy  : fraction of words matching ref at same position.",
        "  Exact Match    : fraction of sentences reconstructed verbatim.",
        "  ERR            : fraction of ref word-tokens (>2 chars) found in prediction.",
        "  Hard only      : metrics recomputed excluding trivial (unchanged) samples.",
        "                   This is the honest measure of inversion quality.", ""]

    # ── Score distribution ─────────────────────────────────────────────────
    buckets = [0] * 11   # 0.0-0.09, 0.10-0.19, ..., 1.0
    for s in s_bleus:
        buckets[min(int(s * 10), 10)] += 1
    total_s = len(s_bleus)
    report += ["  SENTENCE BLEU DISTRIBUTION", thin]
    for i, cnt in enumerate(buckets):
        lo = i * 0.1
        hi = lo + 0.0999 if i < 10 else 1.0
        bar = "█" * (cnt * 40 // total_s)
        report.append(f"  [{lo:.1f}–{hi:.2f}]  {bar:<40} {cnt:>5}  ({cnt/total_s*100:4.1f}%)")
    report += [""]

    # ── Examples ──────────────────────────────────────────────────────────
    # Sort and sample from HARD examples only so trivial copies don't dominate
    hard_indices = [i for i, m in enumerate(hard_mask) if m]
    hard_s_bleus = [s_bleus[i] for i in hard_indices]
    sorted_hard  = sorted(hard_indices, key=lambda i: s_bleus[i])

    worst_idx = sorted_hard[:5]
    best_idx  = sorted_hard[-5:]
    random.seed(42)
    pool     = [i for i in hard_indices if i not in set(worst_idx + best_idx)]
    n_rand   = min(args.num_examples - 10, len(pool))
    rand_idx = random.sample(pool, n_rand) if n_rand > 0 else []

    report += ["  RECONSTRUCTION EXAMPLES", sep]
    section_labels = (
        [(i, "WORST") for i in worst_idx] +
        [(i, "RANDOM") for i in rand_idx] +
        [(i, "BEST")  for i in best_idx]
    )
    for rank, (i, label) in enumerate(section_labels, 1):
        anon = all_anons[i]
        pred = all_preds[i]
        ref  = all_refs[i]
        exact_m = "✓" if pred.strip() == ref.strip() else "✗"
        report += [
            f"\n  [{label}] Sample {rank:03d}  —  Sentence BLEU: {s_bleus[i]:.4f}  "
            f"Exact: {exact_m}",
            thin,
            f"  INPUT (anonymized) :",
            f"    {textwrap.fill(anon, 74, subsequent_indent='    ')}",
            f"  PREDICTED (original):",
            f"    {textwrap.fill(pred, 74, subsequent_indent='    ')}",
            f"  REFERENCE (original):",
            f"    {textwrap.fill(ref,  74, subsequent_indent='    ')}",
        ]
    report += ["", sep]

    report_text = "\n".join(report)
    print(report_text)

    # ── Save outputs ───────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path  = os.path.join(args.output_dir, f"eval_report_{ts}.txt")
    metrics_path = os.path.join(args.output_dir, f"metrics_{ts}.json")
    preds_path   = os.path.join(args.output_dir, f"predictions_{ts}.jsonl")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(preds_path, "w", encoding="utf-8") as f:
        for row, pred, bleu_s in zip(rows, all_preds, s_bleus):
            f.write(json.dumps({
                "id"       : row.get("id"),
                "anonymized": row["anonymized"],
                "original"  : row["original"],
                "predicted" : pred,
                "sentence_bleu": bleu_s,
                "exact_match"  : pred.strip() == row["original"].strip(),
            }, ensure_ascii=False) + "\n")

    print(f"\nReport   saved → {report_path}")
    print(f"Metrics  saved → {metrics_path}")
    print(f"Preds    saved → {preds_path}")


if __name__ == "__main__":
    main()
