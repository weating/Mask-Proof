---
name: extract-proof-formulas
description: Extract maximal formula-only spans from proof LaTeX while preserving exact source text and offsets. Use when Codex needs to separate mathematical expressions from natural-language proof prose, enumerate candidate mask spans in theorem-proof curation, or preprocess `problem`, `reference_solution`, or `mask_text` fields for JSONL pipelines.
---

# Extract Proof Formulas

Use this skill to recover exact math-only spans from a proof without rewriting the proof text. In the masked-proof workflow from `AGENTS.md`, treat the extracted spans as candidate regions only; still apply the downstream self-containedness and mask-quality checks before using any span as `[MASK]`.

## Workflow

1. Run `scripts/extract_formula_blocks.py` on raw proof text, a `.tex` snippet, or a JSONL field such as `reference_solution`.
2. Prefer the returned maximal contiguous math-only spans. Do not normalize macros, remove whitespace, or rewrite nearby text.
3. Reject extracted spans that are mathematically low-value even if they are formula-only; this skill identifies syntactic candidates, not final mask decisions.

## What Counts As A Formula Block

- Display math delimited by `\[...\]` or `$$...$$`.
- Math environments such as `equation`, `align`, `gather`, `multline`, `split`, `aligned`, and `cases`.
- Standalone inline math only when the containing line has no natural-language residue outside the math span.

Reject spans that contain explicit natural-language text commands such as `\text{...}`, `\mbox{...}`, `\intertext{...}`, or `\shortintertext{...}` with textual content.

## Script

Use `scripts/extract_formula_blocks.py`.

Plain text input:

```bash
python3 scripts/extract_formula_blocks.py --input-file /path/to/proof.txt
```

Standard input:

```bash
cat /path/to/proof.txt | python3 scripts/extract_formula_blocks.py
```

JSONL augmentation:

```bash
python3 scripts/extract_formula_blocks.py \
  --input-file /path/to/proofs.jsonl \
  --jsonl \
  --field reference_solution
```

The script returns exact extracted text with:

- `start` and `end` character offsets in the source field, with `end` exclusive
- `line_start` and `line_end` as 1-based line numbers
- `kind` describing the extractor branch

## Output Discipline

- Preserve the source text exactly in `text`.
- Use the extracted spans to inspect candidate mask regions, not to rewrite proofs.
- If a proof mixes prose and formulas too tightly, prefer extracting fewer spans rather than manufacturing boundaries.
