#!/usr/bin/env python3
"""Apply one exact [MASK] replacement to proof text or JSONL records."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


MASK_TOKEN = "[MASK]"


@dataclass(frozen=True)
class ResolvedSpan:
    start: int
    end: int
    text: str


def read_input(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return sys.stdin.read()


def find_all_occurrences(source: str, target: str) -> list[int]:
    positions: list[int] = []
    cursor = 0
    while True:
        index = source.find(target, cursor)
        if index == -1:
            return positions
        positions.append(index)
        cursor = index + 1


def validate_source(source: str, mask_token: str) -> None:
    if mask_token in source:
        raise ValueError(f"Source already contains {mask_token!r}")


def resolve_from_offsets(
    source: str,
    start: int,
    end: int,
    expected_text: str | None = None,
) -> ResolvedSpan:
    if start < 0 or end < 0:
        raise ValueError("start and end must be non-negative")
    if start >= end:
        raise ValueError("start must be smaller than end")
    if end > len(source):
        raise ValueError("end exceeds source length")
    text = source[start:end]
    if not text:
        raise ValueError("Selected span is empty")
    if expected_text is not None and text != expected_text:
        raise ValueError("Selected span does not match expected text")
    return ResolvedSpan(start=start, end=end, text=text)


def resolve_from_text(source: str, target: str, match_index: int | None) -> ResolvedSpan:
    if not target:
        raise ValueError("mask content must be non-empty")
    positions = find_all_occurrences(source, target)
    if not positions:
        raise ValueError("mask content does not occur in source")
    if match_index is None:
        if len(positions) != 1:
            raise ValueError(
                f"mask content is ambiguous: found {len(positions)} matches; provide match_index"
            )
        start = positions[0]
    else:
        if match_index < 0 or match_index >= len(positions):
            raise ValueError(
                f"match_index {match_index} out of range for {len(positions)} matches"
            )
        start = positions[match_index]
    end = start + len(target)
    return ResolvedSpan(start=start, end=end, text=target)


def apply_mask(source: str, span: ResolvedSpan, mask_token: str) -> dict[str, Any]:
    validate_source(source, mask_token)
    rebuilt = source[: span.start] + span.text + source[span.end :]
    if rebuilt != source:
        raise ValueError("Internal consistency check failed before replacement")
    mask_text = source[: span.start] + mask_token + source[span.end :]
    if mask_text.count(mask_token) != 1:
        raise ValueError("Result must contain exactly one mask token")
    return {
        "start": span.start,
        "end": span.end,
        "mask_content": span.text,
        "mask_text": mask_text,
    }


def coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, not a boolean")
    if isinstance(value, int):
        return value
    raise ValueError(f"{field_name} must be an integer")


def coerce_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def resolve_record_candidate(candidate: Any) -> tuple[int | None, int | None, str | None, int | None]:
    if not isinstance(candidate, dict):
        raise ValueError("candidate field must be a JSON object")
    start = candidate.get("start")
    end = candidate.get("end")
    expected_text = candidate.get("text", candidate.get("mask_content"))
    match_index = candidate.get("match_index")

    start_value = None if start is None else coerce_int(start, "candidate.start")
    end_value = None if end is None else coerce_int(end, "candidate.end")
    text_value = None if expected_text is None else coerce_str(expected_text, "candidate.text")
    index_value = None if match_index is None else coerce_int(match_index, "candidate.match_index")
    return start_value, end_value, text_value, index_value


def resolve_span_for_record(record: dict[str, Any], source: str, args: argparse.Namespace) -> ResolvedSpan:
    if args.candidate_field:
        candidate = record.get(args.candidate_field)
        if candidate is None:
            raise ValueError(f"Missing candidate field {args.candidate_field!r}")
        start, end, expected_text, match_index = resolve_record_candidate(candidate)
        if start is not None or end is not None:
            if start is None or end is None:
                raise ValueError("candidate.start and candidate.end must appear together")
            return resolve_from_offsets(source, start, end, expected_text)
        if expected_text is None:
            raise ValueError("candidate must provide either start/end or text")
        return resolve_from_text(source, expected_text, match_index)

    if args.start_field or args.end_field:
        if not args.start_field or not args.end_field:
            raise ValueError("--start-field and --end-field must be provided together")
        if args.start_field not in record or args.end_field not in record:
            raise ValueError("Missing start or end field in record")
        start = coerce_int(record[args.start_field], args.start_field)
        end = coerce_int(record[args.end_field], args.end_field)
        expected_text = None
        if args.expected_field:
            if args.expected_field not in record:
                raise ValueError(f"Missing expected field {args.expected_field!r}")
            expected_text = coerce_str(record[args.expected_field], args.expected_field)
        return resolve_from_offsets(source, start, end, expected_text)

    if args.mask_content_field:
        if args.mask_content_field not in record:
            raise ValueError(f"Missing mask content field {args.mask_content_field!r}")
        target = coerce_str(record[args.mask_content_field], args.mask_content_field)
        match_index = None
        if args.match_index_field:
            if args.match_index_field not in record:
                raise ValueError(f"Missing match index field {args.match_index_field!r}")
            match_index = coerce_int(record[args.match_index_field], args.match_index_field)
        return resolve_from_text(source, target, match_index)

    raise ValueError("No JSONL selector provided")


def ensure_output_fields(record: dict[str, Any], args: argparse.Namespace) -> None:
    for field_name in (args.mask_text_field, args.mask_content_output_field):
        if not args.overwrite and field_name in record:
            raise ValueError(
                f"Output field {field_name!r} already exists; rerun with --overwrite to replace it"
            )


def iter_jsonl_records(raw: str) -> Iterator[tuple[int, dict[str, Any]]]:
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object at line {line_number}")
        yield line_number, record


def process_plain_text(args: argparse.Namespace) -> int:
    source = read_input(args.input_file)
    if args.start is not None or args.end is not None:
        if args.start is None or args.end is None:
            raise ValueError("--start and --end must be provided together")
        span = resolve_from_offsets(source, args.start, args.end, args.expected_text)
    elif args.mask_content is not None:
        span = resolve_from_text(source, args.mask_content, args.match_index)
    else:
        raise ValueError("Provide either --start/--end or --mask-content in plain-text mode")

    output = apply_mask(source, span, args.mask_token)
    json.dump(output, sys.stdout, ensure_ascii=False, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    return 0


def process_jsonl(args: argparse.Namespace) -> int:
    raw = read_input(args.input_file)
    for line_number, record in iter_jsonl_records(raw):
        if args.source_field not in record:
            raise ValueError(
                f"Missing source field {args.source_field!r} in JSONL record at line {line_number}"
            )
        source = coerce_str(record[args.source_field], args.source_field)
        ensure_output_fields(record, args)
        span = resolve_span_for_record(record, source, args)
        result = apply_mask(source, span, args.mask_token)
        record[args.mask_text_field] = result["mask_text"]
        record[args.mask_content_output_field] = result["mask_content"]
        json.dump(record, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", help="Read from a text file or JSONL file. Default: stdin.")
    parser.add_argument("--jsonl", action="store_true", help="Treat input as JSONL and emit JSONL.")
    parser.add_argument("--source-field", default="reference_solution", help="JSONL source field.")
    parser.add_argument("--candidate-field", help="JSONL field containing a selected span object.")
    parser.add_argument("--start", type=int, help="Plain-text mode start offset.")
    parser.add_argument("--end", type=int, help="Plain-text mode end offset.")
    parser.add_argument("--expected-text", help="Optional exact text expected at start:end.")
    parser.add_argument("--mask-content", help="Plain-text mode exact literal span to replace.")
    parser.add_argument("--match-index", type=int, help="Zero-based match index for --mask-content.")
    parser.add_argument("--start-field", help="JSONL field storing the start offset.")
    parser.add_argument("--end-field", help="JSONL field storing the end offset.")
    parser.add_argument("--expected-field", help="JSONL field storing expected exact span text.")
    parser.add_argument("--mask-content-field", help="JSONL field storing exact literal span text.")
    parser.add_argument(
        "--match-index-field",
        help="JSONL field storing the zero-based match index when text matching is ambiguous.",
    )
    parser.add_argument("--mask-text-field", default="mask_text", help="JSONL output field for masked proof.")
    parser.add_argument(
        "--mask-content-output-field",
        default="mask_content",
        help="JSONL output field for the replaced span.",
    )
    parser.add_argument("--mask-token", default=MASK_TOKEN, help="Mask token to insert.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output fields.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print plain-text JSON output.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.jsonl:
            return process_jsonl(args)
        return process_plain_text(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
