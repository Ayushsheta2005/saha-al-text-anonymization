# 📋 SAHA-AL Text Anonymization Guidelines

> **Version 3.0**
>
> These guidelines govern the human **text rewriting (pseudonymization)** task within the SAHA-AL pipeline. 
> Every team member **must read this document** before starting.

---

## 1. Goal

Provide a parallel translation of the given sentence: translating from realistic text with true Personally Identifiable Information (PII) to an identically meaning, fluent sentence that contains entirely false/synthetic PII.

Your goal isn't to *detect* PII (that part is already done by the AI4Privacy dataset, and highlighted for you). Your goal is to **generate** the new sentence that protects privacy while keeping the exact same grammar, structure, and semantic meaning.

The downstream task is training Seq2Seq Fill models (e.g. BART, DeBERTa-base-filler) to do exactly what you are doing manually.

---

## 2. Core Rule: No Privacy Leakage

This is the most critical requirement. **ZERO original PII text should exist in your rewritten sentence.**

*   If the original text complains about a flight `UA893` arriving in `Boston`, your rewrite must change BOTH the flight number and the location to something plausible, e.g. `DL340` arriving in `Seattle`.
*   A strict character matching leakage checker is applied when you hit *Accept*. If it trips, you will be forced to modify your sentence.

---

## 3. Core Rule: Format/Plurality Preservation

The new fake identifiers you choose must match the general *format* and *contextual plurality* of the original text.

| Original Context | ✅ Good Replacement | ❌ Bad Replacement |
|------------------|---------------------|--------------------|
| `Hello John,` | `Hello Ahmed,` | `Hello Mr. Smith,` |
| `My SSN is 111-22-3333` | `My SSN is 948-28-1039` | `My SSN is 111223333` |
| `Visiting NYC and Boston` | `Visiting Paris and London` | `Visiting NYC and Paris` |

---

## 4. UI Actions

*   ✅ **Accept & Save**: Use when confident in your grammatical/semantic rewrite and NO original PII has bypassed you.
*   🚩 **Flag (Ambiguous)**: Use if you don't know how to rewrite a sentence or there are multiple overlapping or conflicting PII spans that make writing a logical fake sentence impossible.
*   ⏭️ **Skip**: Use if the original data is heavily garbled, not English, or contains no PII and rewriting it provides no value to training.
