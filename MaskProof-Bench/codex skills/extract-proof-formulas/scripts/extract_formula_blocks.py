#!/usr/bin/env python3
"""Extract maximal formula-only spans from proof text or JSONL fields."""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


DISPLAY_ENVIRONMENTS = {
    "align",
    "align*",
    "aligned",
    "alignedat",
    "alignedat*",
    "array",
    "bmatrix",
    "Bmatrix",
    "cases",
    "dcases",
    "displaymath",
    "eqnarray",
    "eqnarray*",
    "equation",
    "equation*",
    "flalign",
    "flalign*",
    "gather",
    "gather*",
    "gathered",
    "math",
    "matrix",
    "multline",
    "multline*",
    "pmatrix",
    "smallmatrix",
    "split",
    "Vmatrix",
    "vmatrix",
}

TEXTUAL_COMMANDS = (
    "fbox",
    "framebox",
    "hbox",
    "intertext",
    "mbox",
    "parbox",
    "shortintertext",
    "text",
    "textbf",
    "textit",
    "textmd",
    "textnormal",
    "textrm",
    "textsc",
    "textsf",
    "textsl",
    "texttt",
    "textup",
)

IGNORED_INLINE_RESIDUE = re.compile(
    r"""
    (
        \s+
      | [%.,;:!?()[\]{}]
      | \\[,;:!]
      | \\(?:quad|qquad|enspace|enskip|hfill|hspace\*?\{[^{}]*\}|vspace\*?\{[^{}]*\}|noindent)
    )+
    """,
    re.VERBOSE,
)

START_PATTERN = re.compile(
    r"""
    \\\[
    | \\\(
    | \\begin\{[A-Za-z*]+\}
    | \$\$
    | \$
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    kind: str
    environment: str | None = None
    delimiter: str | None = None


def is_escaped(text: str, index: int) -> bool:
    slash_count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        slash_count += 1
        cursor -= 1
    return slash_count % 2 == 1


def strip_comments(text: str) -> str:
    parts: list[str] = []
    for line in text.splitlines(keepends=True):
        cut = None
        for idx, char in enumerate(line):
            if char == "%" and not is_escaped(line, idx):
                cut = idx
                break
        if cut is None:
            parts.append(line)
        else:
            parts.append(line[:cut])
            if line.endswith("\n"):
                parts.append("\n")
    return "".join(parts)


def extract_braced_content(text: str, brace_index: int) -> tuple[str, int] | None:
    if brace_index >= len(text) or text[brace_index] != "{":
        return None
    depth = 0
    cursor = brace_index
    while cursor < len(text):
        char = text[cursor]
        if char == "{" and not is_escaped(text, cursor):
            depth += 1
        elif char == "}" and not is_escaped(text, cursor):
            depth -= 1
            if depth == 0:
                return text[brace_index + 1 : cursor], cursor + 1
        cursor += 1
    return None


def contains_textual_macro_content(text: str) -> bool:
    for command in TEXTUAL_COMMANDS:
        pattern = re.compile(rf"\\{command}\s*\{{")
        for match in pattern.finditer(text):
            content = extract_braced_content(text, match.end() - 1)
            if content is None:
                continue
            inner, _ = content
            if re.search(r"[A-Za-z\u00C0-\u024F\u4E00-\u9FFF]", inner):
                return True
    return False


def find_unescaped(text: str, token: str, start: int) -> int:
    cursor = start
    while True:
        index = text.find(token, cursor)
        if index == -1:
            return -1
        if not is_escaped(text, index):
            return index
        cursor = index + 1


def find_matching_environment(text: str, env_name: str, start: int) -> int:
    pattern = re.compile(rf"\\(begin|end)\{{{re.escape(env_name)}\}}")
    depth = 1
    for match in pattern.finditer(text, start):
        if match.group(1) == "begin":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return match.end()
    return -1


def iter_candidates(text: str) -> Iterator[Candidate]:
    cursor = 0
    while cursor < len(text):
        match = START_PATTERN.search(text, cursor)
        if match is None:
            return
        token = match.group(0)
        start = match.start()

        if token == "$" and is_escaped(text, start):
            cursor = start + 1
            continue
        if token == "$$" and is_escaped(text, start):
            cursor = start + 1
            continue

        if token == r"\[":
            end_start = find_unescaped(text, r"\]", match.end())
            if end_start == -1:
                cursor = match.end()
                continue
            end = end_start + 2
            yield Candidate(start, end, "display_math_delim", delimiter=r"\[...\]")
            cursor = end
            continue

        if token == r"\(":
            end_start = find_unescaped(text, r"\)", match.end())
            if end_start == -1:
                cursor = match.end()
                continue
            end = end_start + 2
            yield Candidate(start, end, "inline_math", delimiter=r"\(...\)")
            cursor = end
            continue

        if token == "$$":
            end_start = find_unescaped(text, "$$", match.end())
            if end_start == -1:
                cursor = match.end()
                continue
            end = end_start + 2
            yield Candidate(start, end, "display_math_delim", delimiter="$$...$$")
            cursor = end
            continue

        if token == "$":
            if start + 1 < len(text) and text[start + 1] == "$":
                cursor = start + 2
                continue
            end_start = find_unescaped(text, "$", match.end())
            if end_start == -1:
                cursor = match.end()
                continue
            end = end_start + 1
            yield Candidate(start, end, "inline_math", delimiter="$...$")
            cursor = end
            continue

        env_match = re.fullmatch(r"\\begin\{([A-Za-z*]+)\}", token)
        if env_match is None:
            cursor = match.end()
            continue
        env_name = env_match.group(1)
        if env_name not in DISPLAY_ENVIRONMENTS:
            cursor = match.end()
            continue
        end = find_matching_environment(text, env_name, match.end())
        if end == -1:
            cursor = match.end()
            continue
        yield Candidate(start, end, "display_math_env", environment=env_name)
        cursor = end


def line_starts(text: str) -> list[int]:
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def offset_to_line(starts: list[int], offset: int) -> int:
    return bisect.bisect_right(starts, offset)


def inline_residue_is_formula_only(full_text: str, candidate: Candidate) -> bool:
    line_start = full_text.rfind("\n", 0, candidate.start) + 1
    line_end = full_text.find("\n", candidate.end)
    if line_end == -1:
        line_end = len(full_text)
    before = full_text[line_start:candidate.start]
    after = full_text[candidate.end:line_end]
    residue = before + after
    cleaned = IGNORED_INLINE_RESIDUE.sub("", residue)
    return cleaned == ""


def candidate_is_formula_only(full_text: str, candidate: Candidate) -> bool:
    span = full_text[candidate.start:candidate.end]
    stripped = strip_comments(span)
    if contains_textual_macro_content(stripped):
        return False
    if candidate.kind == "inline_math":
        return inline_residue_is_formula_only(full_text, candidate)
    return True


def extract_formula_blocks(text: str) -> list[dict[str, object]]:
    starts = line_starts(text)
    blocks: list[dict[str, object]] = []
    for index, candidate in enumerate(iter_candidates(text)):
        if not candidate_is_formula_only(text, candidate):
            continue
        blocks.append(
            {
                "index": len(blocks),
                "kind": candidate.kind,
                "environment": candidate.environment,
                "delimiter": candidate.delimiter,
                "start": candidate.start,
                "end": candidate.end,
                "line_start": offset_to_line(starts, candidate.start),
                "line_end": offset_to_line(starts, max(candidate.end - 1, candidate.start)),
                "text": text[candidate.start:candidate.end],
            }
        )
    return blocks


def load_text(args: argparse.Namespace) -> str:
    if args.input_file:
        return Path(args.input_file).read_text(encoding="utf-8")
    return sys.stdin.read()


def process_text_mode(args: argparse.Namespace) -> int:
    text = load_text(args)
    output = {
        "formula_blocks": extract_formula_blocks(text),
    }
    json.dump(output, sys.stdout, ensure_ascii=False, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    return 0


def iter_jsonl_records(text: str) -> Iterator[tuple[int, dict[str, object]]]:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at line {line_number}")
        yield line_number, payload


def process_jsonl_mode(args: argparse.Namespace) -> int:
    if not args.field:
        raise ValueError("--field is required with --jsonl")
    raw = load_text(args)
    for _, record in iter_jsonl_records(raw):
        source = record.get(args.field, "")
        if not isinstance(source, str):
            raise ValueError(f"Field {args.field!r} must be a string")
        record[args.output_field] = extract_formula_blocks(source)
        json.dump(record, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", help="Read from a text or JSONL file. Default: stdin.")
    parser.add_argument("--jsonl", action="store_true", help="Treat input as JSONL and augment each record.")
    parser.add_argument(
        "--field",
        help="Source field to read when using --jsonl, for example reference_solution or mask_text.",
    )
    parser.add_argument(
        "--output-field",
        default="formula_blocks",
        help="Destination field name in JSONL mode. Default: formula_blocks.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON in plain-text mode.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.jsonl:
            return process_jsonl_mode(args)
        return process_text_mode(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
