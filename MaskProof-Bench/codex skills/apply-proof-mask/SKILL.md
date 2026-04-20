---
name: apply-proof-mask
description: Apply a single exact `[MASK]` replacement to a proof span while preserving all surrounding text byte-faithfully. Use when mask selection is already complete and Codex needs to convert `reference_solution` into `mask_text` and `mask_content` for JSONL output without paraphrasing, cleanup, or accidental extra edits.
---

# Apply Proof Mask

Use this skill only after mask selection is finished. Its job is narrow: replace one already-chosen contiguous proof span with `[MASK]`, verify that the replacement matches the intended source span, and emit pipeline-ready fields without changing any other character.

## Workflow

1. Prefer offset-based replacement using `start` and `end` from a previously selected candidate span.
2. Use text-based replacement only when offsets are unavailable and the target span is unique or an explicit match index is supplied.
3. Reject ambiguous matches, mismatched expected text, pre-masked source text, or attempts to write multiple masks.

## Recommended Input Shape

Prefer a JSONL record with:

```json
{
  "reference_solution": "...full proof...",
  "selected_mask_candidate": {
    "start": 120,
    "end": 168,
    "text": "\\begin{align*} ... \\end{align*}"
  }
}
```

This pairs naturally with the output of `$extract-proof-formulas`.

## Script

Use `scripts/apply_mask.py`.

Offset mode on plain text:

```bash
python3 scripts/apply_mask.py \
  --input-file /path/to/proof.txt \
  --start 120 \
  --end 168 \
  --expected-text "\\begin{align*} ... \\end{align*}"
```

JSONL mode from a candidate object:

```bash
python3 scripts/apply_mask.py \
  --input-file /path/to/proofs.jsonl \
  --jsonl \
  --candidate-field selected_mask_candidate
```

JSONL mode from explicit fields:

```bash
python3 scripts/apply_mask.py \
  --input-file /path/to/proofs.jsonl \
  --jsonl \
  --start-field mask_start \
  --end-field mask_end \
  --expected-field mask_content
```

## Output Guarantees

- Write exactly one `[MASK]`.
- Preserve all text before and after the selected span exactly.
- Emit the original replaced text as `mask_content`.
- Fail rather than guessing when the selector is ambiguous.

## Notes

- Prefer this skill over ad hoc string editing whenever a masked proof must stay faithful to the original proof.
- Use `$extract-proof-formulas` first if you still need candidate span discovery.
