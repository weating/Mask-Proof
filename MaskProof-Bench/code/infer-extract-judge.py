#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import re
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import tqdm
try:
    from openai import AsyncOpenAI
    from openai import (
        BadRequestError,
        APIStatusError,
        APITimeoutError,
        RateLimitError,
        APIConnectionError,
    )
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ==============================================================================
# Production-grade logging configuration
# ==============================================================================

import logging

try:
    from loguru import logger as loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
        except Exception:
            self.handleError(record)


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname:8s}{self.RESET}"
        return super().format(record)


def setup_logger(log_level: str = "INFO", log_file: str = None):
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = LOG_DIR / f"pipeline_{timestamp}.log"

    if LOGURU_AVAILABLE:
        loguru_logger.remove()
        loguru_logger.add(
            lambda msg: tqdm.tqdm.write(msg, end=""),
            level=log_level.upper(),
            colorize=True,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )
        loguru_logger.add(
            str(log_file),
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="100 MB",
            encoding="utf-8",
        )
        return loguru_logger
    else:
        logger = logging.getLogger("pipeline")
        logger.setLevel(getattr(logging, log_level.upper()))
        logger.handlers.clear()

        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
        return logger


logger = setup_logger("INFO")

# ==============================================================================
# Upgraded prompts (kept as-is)
# ==============================================================================

INFERENCE_PROMPT = """
# Mathematical Proof Derivation Restoration Task
## Task Description
You are an expert mathematician. Your task is to restore the content masked by `[MASK]` tokens in a given mathematical proof. 
To do this effectively, you must utilize both the **Problem Statement** (which provides definitions, constraints, and the goal) and the **Masked Proof** (the partial derivation).
The masks usually represent **intermediate derivation steps**, logical transitions, or specific mathematical terms required to bridge the preceding and succeeding expressions.

## Crucial Constraints
1. **No Shortcuts**: If a mask appears in a chain (e.g., `A = [MASK] = C`), the restored content must be the **logical bridge** `B`. Do not simply repeat `A` or `C`.
2. **Granularity Match**: The complexity of the restored step must match the surrounding proof. If the context shows detailed algebraic manipulation, the mask must not just result in a final number; it must show the operation.
3. **Consistency with Problem**: Ensure all restored steps adhere to the constraints defined in the **Problem Statement** (e.g., if the problem states $n$ is an integer, do not introduce continuous variable assumptions unless justified).
4. **Structural Integrity**: Ensure the restored LaTeX code fits syntactically into the surrounding text without breaking the compilation or the logical flow.
5. **Forward Reference Consistency**: Critically analyze how the text *following* the mask refers to the masked content.
    - If the subsequent text says "Substituting this into Eq (2)...", your restoration must match the form required by Eq (2).
    - If the subsequent text says "which implies $x > 0$", your restoration must mathematically lead to that implication.

## Input Data
**Problem Statement:**
{problem}
**Masked Proof Segment:**
{proof_text}

## Analysis Steps
1. **Goal Alignment**: Read the **Problem Statement** first to understand the objective, variable definitions, and initial constraints.
2. **Contextual Analysis**: 
   - **Look Back**: What variables and definitions are available immediately before the mask?
   - **Look Ahead (Crucial)**: How is the result of this mask **used** in the immediate next sentence? Does the next step perform an operation (like squaring, integrating) on your result?
3. **Derivation Logic**: 
   - Identify the transformation $f$ such that \\text{{Pre-Mask}} \\xrightarrow{{f}} \\text{{Post-Mask}}.
4. **Symbol Consistency**: Verify that any variables introduced in the mask match the notation used in both the Problem Statement and the Proof.

## Output Format
For the mask found in the text, provide the followin Strictly!:

**Reasoning & Step-by-Step Derivation:**
* **Context from Problem**: [Reference constraints]
* **Pre-Mask State**: [State before mask]
* **Downstream Dependency**: [Analyze how the text AFTER the mask relies on this specific result]
* **Operation**: [Explain the math operation]
**Verification:**
Verify that substituting the restoration back into the original text results in a mathematically valid derivation that flows naturally into the subsequent text.
**[MASK] Restoration:**
$$<Your restored LaTeX code here>$$
"""

EXTRACTION_PROMPT_TEMPLATE = r"""
You are an expert data extraction tool. Your task is to extract the restored mathematical formula from the provided text.

**Context:**
The input text contains the restoration result for a **single mask**.
The section header will contain the keywords "MASK" and "Restoration", but it might **not** contain a number (ID).

**Instructions:**
1. **Locate the Target Section:**
   Find the **bolded** line that functions as a title and contains both "MASK" and "Restoration".
   - It usually looks like: `**[MASK] Restoration**`, `**[MASK] Restoration:**`, or `**<MASK> Restoration**`.
   - Do NOT require a number to be present in the title.

2. **Extract Mask ID (Default Logic):**
   - If the title contains a specific number (e.g., `_1`, ` 1`), extract it.
   - **Important:** If NO number is found in the title, assign `0` as the `mask_id` (indicating the single unique mask).

3. **Extract Formula:**
   - Extract the complete LaTeX formula immediately following the target section title.
   - Capture the content regardless of the delimiters (e.g., `$$...$$`, `\[...\]`, or just raw latex).

4. **Output Format:**
   Return the result in strict JSON format.

**Input Text:**
---
{raw_model_response}
---

**Output JSON Schema:**
{{
  "mask_id": integer,
  "formula": "string"
}}
"""
SYSTEM_PROMPT_EVAL = r"""
You are an expert mathematician evaluating a 'fill-in-the-mask' task within a formal mathematical proof.
Your goal is to determine if the Generated Answer (GA) is **Structurally Isomorphic** to the Ground Truth (GT).

### Core Philosophy: "The State Machine Principle"
The Ground Truth (GT) defines the **Target State** for the current step.
The Generated Answer (GA) is Correct ONLY if it aligns with the GT's state in Value, Stage, and Scope.

### Pre-Evaluation Parsing Rule (Chain of Equalities)
**IF** the Generated Answer (GA) contains a chain of equalities (e.g., $$ A = B = C $$) **AND** the Ground Truth (GT) is a single expression:
- You must isolate the **FINAL result** (the expression after the last equal sign) of the GA.
- Compare ONLY this final expression against the GT.
- *Example:* If GT is "2x" and GA is "x + x = 2x", evaluate "2x" vs "2x". This is CORRECT.

### Evaluation Standard: The 3-Layer Filter
To be marked **Correct**, the (parsed) answer must pass ALL three layers:

#### Layer 1: Granularity & State Fidelity (The "Step" Check)
- **Rule:** If GT is in State $N$ (e.g., Setup, Expansion, or Result), GA must be in State $N$.
- **Operator Conservation:** Major operators (\sum, \int, \lim) present in the GT **must be preserved** in the GA unless trivial re-ordering.

#### Layer 2: Mathematical Equivalence (The Baseline)
- **Check:** Does the GA represent the exact same mathematical value, set, or proposition as the GT?

#### Layer 3: Allowable Deformations (Horizontal Movements)
**✅ ALLOWED (Isomorphic):** Factoring, Notation Styles ($e^x$ vs $\exp(x)$), Commutativity, Variable Renaming.
**❌ FORBIDDEN (Structural Alterations):** Term Merging (GT: $A+B$, GA: $C$), Definition Expansion.

### Supplemental Rule: Long GT Compression & Target Extraction (Minimal Add-on)
- The Ground Truth (GT) may contain a long chain of equalities whose purpose is to derive a single final expression.
- You must treat the GT as defining a single **Target Expression** corresponding to its final logical state.
- The Generated Answer (GA) is NOT required to reproduce intermediate transformations (e.g., integration by parts, boundary term cancellation, divergence theorem applications) if these steps are already resolved in the GT.

### Target Extraction Rule (GT-side)
- If the GT contains multiple equalities or derivation steps, you must conceptually extract the **final resolved expression** that represents the target state.
- Comparison with GA should be performed against this extracted target, not against the full derivation length.

### Length Mismatch Clarification
- A shorter GA can still be correct if it directly matches the GT’s final target state.
- Differences in derivation length, verbosity, or omission of already-eliminated terms MUST NOT be treated as incorrectness.

### Final Decision Constraint
- If GA directly expresses the same final mathematical object that GT reaches at the end of its derivation, then GA is CORRECT, provided Layers 1–3 are satisfied.

### Output Format
Return ONLY a JSON object:
{
    "is_correct": boolean,
    "explanation": "1. Parsing (Did you strip equality chain?). 2. Math Equivalence. 3. Logical State. 4. Deformations."
}
"""

USER_PROMPT_EVAL_TEMPLATE = """QUESTION (Context): {question}

GROUND TRUTH (Target State): {ground_truth}

GENERATED ANSWER: {generated_answer}

Evaluate based on "Structural Isomorphism".
Return ONLY a JSON object:
{{
    "is_correct": boolean,
    "explanation": "Analyze: 1. Math Equivalence. 2. Logical State. 3. Deformations."
}}
"""

# ==============================================================================
# Configuration management
# ==============================================================================

@dataclass
class PhaseConfig:
    temperature: float
    top_p: float
    max_tokens: int
    reasoning_effort: Optional[str] = None

# ==============================================================================
# IO helpers
# ==============================================================================

def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            data.append(obj)
                    except Exception:
                        # Preserve the original behavior: skip invalid lines.
                        pass
    return data


def save_jsonl_atomic(savepath: str, data: List[Dict]):
    tmp_path = savepath + ".tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)
        with open(tmp_path, 'w', encoding='utf-8') as fw:
            for d in data:
                fw.write(json.dumps(d, ensure_ascii=False) + '\n')
        shutil.move(tmp_path, savepath)
    except Exception as e:
        logger.error(f"Save failed: {e}")


def normalize_base_url(base_url: str) -> str:
    if not base_url:
        return ""
    url = base_url.strip().rstrip("/")
    if not url:
        return ""
    for suffix in ("/chat/completions", "/v1/chat/completions"):
        if url.endswith(suffix):
            url = url[:-len(suffix)]
            url = url.rstrip("/")
    return url


def normalize_reasoning_effort(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    lowered = _as_str(value).strip().lower()
    if lowered in ("", "none", "null", "off", "false", "0"):
        return None
    return value

# ==============================================================================
# Strict schema sanitization (added: string-level input fallback + schema convergence)
# ==============================================================================

def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return ""


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    parts.append(_as_str(part.get("text", "")))
                elif "content" in part:
                    parts.append(_as_str(part.get("content", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts).strip()
    return _as_str(content)


def _normalize_masks(mask_text: str) -> str:
    # Normalize legacy placeholders to a single [MASK] token.
    if "<mask>" in mask_text:
        mask_text = mask_text.replace("<mask>", "[MASK]")
    mask_text = re.sub(r"\[MASK\]" + r"_\d+", "[MASK]", mask_text)
    return mask_text


def sanitize_item(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Force a raw sample into the pipeline's internal schema:
    - Ensure all key fields exist, even when they are empty strings or empty lists.
    - Prevent downstream crashes caused by `None` values or missing fields.
    """
    item = dict(row) if isinstance(row, dict) else {}
    item["original_index"] = idx

    # problem
    if "problem" not in item or item.get("problem") is None:
        item["problem"] = _as_str(item.get("user_content", ""))

    # input / question
    mask_text = _as_str(item.get("mask_text", ""))
    if not mask_text:
        mask_text = _as_str(item.get("input", ""))
    mask_text = _normalize_masks(mask_text)
    item["input"] = mask_text
    item["question"] = mask_text

    # answers (ground truth)
    if "answers" not in item or item.get("answers") is None:
        content = _as_str(item.get("mask_content", ""))

        # Handle the case where assistant_content itself is a JSON string.
        if not content and "assistant_content" in item and item["assistant_content"] is not None:
            try:
                ac = json.loads(_as_str(item["assistant_content"]))
                if isinstance(ac, dict):
                    content = _as_str(ac.get("mask_content", ""))
            except Exception:
                pass

        if content:
            item["answers"] = [{"content": content, "source": "mask_content"}]
        else:
            # Mark it as explicitly empty so later illegal checks stay simple.
            item["answers"] = []

    # Ensure type safety.
    if not isinstance(item.get("answers"), list):
        item["answers"] = []

    if item.get("problem") is None:
        item["problem"] = ""
    if item.get("input") is None:
        item["input"] = ""
    if item.get("question") is None:
        item["question"] = ""

    return item


def prepare_dataset(raw_data: List[Dict]) -> List[Dict]:
    prepared = []
    for idx, row in enumerate(raw_data):
        try:
            item = sanitize_item(row, idx)
            # Preserve the original requirement that both input and problem
            # must exist before enqueueing, but do it more safely.
            if _as_str(item.get("input", "")).strip() and _as_str(item.get("problem", "")).strip():
                prepared.append(item)
        except Exception as e:
            logger.warning(f"Sanitize failed at idx={idx}: {e}")
    return prepared

# ==============================================================================
# Parsing / Extraction helpers
# ==============================================================================

def extract_regex(text: str) -> Optional[Dict]:
    if not text:
        return None
    pattern = r'\*\*(?:\[|<)?MASK(?:\]|>)?(?:\s*[_\s]*\{?([0-9]+|[A-Za-z]+)\}?)?\s+Restoration\s*(?:\*\*:|:\*\*)\s*(?:\$\$(.*?)\$\$|\\\[(.*?)\\\])'

    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        m = matches[0]
        formula = (m[1] if m[1] else m[2]).strip()
        return {'mask_id': str(m[0]).strip(), 'formula': formula}
    return None


def parse_loose_json_object(content: str) -> Optional[Dict]:
    """
    Loose JSON parsing for Phase 2 (mask_id/formula) and other JSON-only scenarios.
    """
    if not content:
        return None
    content = content.strip()
    try:
        obj = json.loads(content)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # code fence
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # first {...}
    match = re.search(r'(\{.*\})', content, re.DOTALL)
    if match:
        snippet = match.group(1)
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    return None


def parse_judge_json(content: str) -> Optional[Dict]:
    """
    Judge-specific parsing: try to recover {"is_correct": ..., "explanation": ...} from text.
    """
    if not content:
        return None
    content = content.strip()
    try:
        obj = json.loads(content)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # heuristic: small json object containing is_correct
    match = re.search(r'(\{[^{}]*"is_correct"[^{}]*\})', content, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # extremely short boolean answers
    if 'true' in content.lower() and len(content) < 20:
        return {"is_correct": True, "explanation": "Parsed from boolean keyword"}
    if 'false' in content.lower() and len(content) < 20:
        return {"is_correct": False, "explanation": "Parsed from boolean keyword"}

    return None

# ==============================================================================
# Validators (by phase)
# ==============================================================================

def validate_inference_output(content: str) -> bool:
    """Phase 1: content must be non-empty and have a reasonable length."""
    if not content:
        return False
    if len(content.strip()) < 10:
        return False
    return True


def validate_phase2_extraction_json(content: str) -> bool:
    """Phase 2: must parse into a dict and include mask_id/formula keys."""
    obj = parse_loose_json_object(content)
    if obj is None:
        return False
    # Null values are allowed, but the schema keys must exist.
    return ("mask_id" in obj) and ("formula" in obj)


def validate_phase3_judge_json(content: str) -> bool:
    """Phase 3: must parse into a dict and include is_correct."""
    obj = parse_judge_json(content)
    if obj is None:
        return False
    return "is_correct" in obj


def validate_judge_fields(parsed: Optional[Dict]) -> Tuple[bool, str]:
    """
    Judge field validation:
    - is_correct must exist and be a bool.
    - explanation must exist and be non-None, though an empty string is allowed.
    """
    if parsed is None:
        return False, "parse_failed"

    if "is_correct" not in parsed:
        return False, "missing_is_correct"
    if not isinstance(parsed.get("is_correct"), bool):
        return False, "invalid_is_correct_type"

    if "explanation" not in parsed:
        return False, "missing_explanation"
    if parsed.get("explanation") is None:
        return False, "null_explanation"

    return True, "ok"


def is_illegal_for_judge(item: Dict) -> Tuple[bool, str]:
    gt_list = item.get("answers", [])
    if not isinstance(gt_list, list) or len(gt_list) == 0:
        return True, "missing_ground_truth"
    gt = _as_str(gt_list[0].get("content", "")) if isinstance(gt_list[0], dict) else ""
    if not gt.strip():
        return True, "empty_ground_truth"
    return False, "ok"

# ==============================================================================
# Pipeline Stats
# ==============================================================================

@dataclass
class PipelineStats:
    inference_done: int = 0
    inference_success: int = 0
    extraction_done: int = 0
    extraction_success: int = 0
    evaluation_done: int = 0
    evaluation_correct: int = 0

    # API totals
    total_api_calls: int = 0
    total_api_success: int = 0
    total_api_failed: int = 0

    # Failure taxonomy (added)
    p1_api_failed: int = 0
    p2_api_failed: int = 0
    p3_api_failed: int = 0
    p2_parse_failed: int = 0
    p3_parse_failed: int = 0
    p3_missing_fields: int = 0
    illegal_samples: int = 0

# ==============================================================================
# Checkpoint system (enhanced: completion depends on failure type)
# ==============================================================================

class PipelineCheckpoint:
    def __init__(self, work_dir: str, prefix: str, rerun_rate_limit: bool = False):
        self.ckpt_dir = os.path.join(work_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.prefix = prefix
        self.rerun_rate_limit = rerun_rate_limit
        self.phase1_path = os.path.join(self.ckpt_dir, f"{prefix}_phase1_inference.jsonl")
        self.phase2_path = os.path.join(self.ckpt_dir, f"{prefix}_phase2_extraction.jsonl")
        self.phase3_path = os.path.join(self.ckpt_dir, f"{prefix}_phase3_evaluation.jsonl")

        self._phase1_buffer: List[Dict] = []
        self._phase2_buffer: List[Dict] = []
        self._phase3_buffer: List[Dict] = []
        self._buffer_size = 10
        self._lock = asyncio.Lock()

    def _is_valid_phase1(self, item: Dict) -> bool:
        """
        Phase 1 completion criteria:
        - model_responses must exist and contain at least one item.
        - At least one attempt must succeed and pass validate_inference_output.
        """
        resps = item.get('model_responses', [])
        if not isinstance(resps, list) or not resps:
            return False
        if self.rerun_rate_limit and self._has_rate_limit_errors(resps):
            return False
        for r in resps:
            if r.get("success") and validate_inference_output(_as_str(r.get("response", ""))):
                return True
        return False

    def _has_rate_limit_errors(self, responses: List[Dict]) -> bool:
        for r in responses:
            error_type = _as_str(r.get("error_type", ""))
            http_status = r.get("http_status", None)
            if error_type in ("rate_limit", "http_error_overload"):
                return True
            if http_status in (429, 503):
                return True
        return False

    def _is_valid_phase2(self, item: Dict) -> bool:
        """
        Phase 2 completion criteria:
        - extract_attempts must exist and contain at least one item.
        - If error_type is in {"p2_llm_api_failed","p2_llm_parse_failed"},
          the item is incomplete and should be rerun.
        - Pure regex_fail / no_match outcomes are treated as complete because
          additional API retries would not improve them.
        """
        attempts = item.get("extract_attempts", None)
        if not isinstance(attempts, list) or len(attempts) == 0:
            return False
        for a in attempts:
            et = a.get("error_type")
            if et in ("p2_llm_api_failed", "p2_llm_parse_failed"):
                return False
        return True

    def _is_valid_phase3(self, item: Dict) -> bool:
        """
        Phase 3 completion criteria:
        - If illegal_reason exists, treat the item as complete without evaluation.
        - eval_results must exist and contain at least one item.
        - If error_type is in {"judge_api_failed","judge_parse_failed","judge_missing_fields"},
          the item is incomplete and should be rerun.
        """
        if item.get("illegal_reason"):
            return True
        results = item.get("eval_results", None)
        if not isinstance(results, list) or len(results) == 0:
            return False
        for r in results:
            et = r.get("error_type")
            if et in ("judge_api_failed", "judge_parse_failed", "judge_missing_fields"):
                return False
        return True

    def load_state(self) -> Tuple[Dict[int, Dict], Dict[int, Dict], Dict[int, Dict], Dict[int, Dict]]:
        phase1_done = {}
        phase1_seen = {}
        phase2_done = {}
        phase3_done = {}

        logger.info("⚡ Loading checkpoints with Cascading Dependency Check...")

        # Phase1
        p1_raw = load_jsonl(self.phase1_path)
        invalid_p1 = 0
        for item in p1_raw:
            idx = item.get('original_index')
            if idx is not None:
                phase1_seen[int(idx)] = item
                if self._is_valid_phase1(item):
                    phase1_done[int(idx)] = item
                else:
                    invalid_p1 += 1

        # Phase2 (depends on Phase1)
        p2_raw = load_jsonl(self.phase2_path)
        invalid_p2 = 0
        orphan_p2 = 0
        for item in p2_raw:
            idx = item.get('original_index')
            if idx is not None:
                idx = int(idx)
                is_valid_struct = self._is_valid_phase2(item)
                has_parent = idx in phase1_done
                if is_valid_struct and has_parent:
                    phase2_done[idx] = item
                else:
                    if not is_valid_struct:
                        invalid_p2 += 1
                    if not has_parent:
                        orphan_p2 += 1

        # Phase3 (depends on Phase2)
        p3_raw = load_jsonl(self.phase3_path)
        invalid_p3 = 0
        orphan_p3 = 0
        for item in p3_raw:
            idx = item.get('original_index')
            if idx is not None:
                idx = int(idx)
                is_valid_struct = self._is_valid_phase3(item)
                has_parent = idx in phase2_done or item.get("illegal_reason")  # Illegal items may have no parent.
                if is_valid_struct and has_parent:
                    phase3_done[idx] = item
                else:
                    if not is_valid_struct:
                        invalid_p3 += 1
                    if not has_parent:
                        orphan_p3 += 1

        logger.info("Checkpoint Loaded Summary:")
        logger.info(f"  Phase 1: {len(phase1_done)} valid | {invalid_p1} invalid (re-infer)")
        logger.info(f"  Phase 2: {len(phase2_done)} valid | {invalid_p2} invalid, {orphan_p2} orphaned (re-extract)")
        logger.info(f"  Phase 3: {len(phase3_done)} valid | {invalid_p3} invalid, {orphan_p3} orphaned (re-eval)")

        return phase1_done, phase2_done, phase3_done, phase1_seen

    async def save_phase1(self, item: Dict):
        if not self._is_valid_phase1(item):
            return
        async with self._lock:
            self._phase1_buffer.append(item)
            if len(self._phase1_buffer) >= self._buffer_size:
                self._flush_buffer(self.phase1_path, self._phase1_buffer)
                self._phase1_buffer = []

    async def save_phase2(self, item: Dict):
        if not self._is_valid_phase2(item):
            return
        async with self._lock:
            self._phase2_buffer.append(item)
            if len(self._phase2_buffer) >= self._buffer_size:
                self._flush_buffer(self.phase2_path, self._phase2_buffer)
                self._phase2_buffer = []

    async def save_phase3(self, item: Dict):
        if not self._is_valid_phase3(item):
            return
        async with self._lock:
            self._phase3_buffer.append(item)
            if len(self._phase3_buffer) >= self._buffer_size:
                self._flush_buffer(self.phase3_path, self._phase3_buffer)
                self._phase3_buffer = []

    def _flush_buffer(self, filepath: str, buffer: List[Dict]):
        if buffer:
            try:
                with open(filepath, 'a', encoding='utf-8') as f:
                    for item in buffer:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            except Exception as e:
                logger.error(f"Error flushing checkpoint to {filepath}: {e}")

    async def flush_all(self):
        async with self._lock:
            self._flush_buffer(self.phase1_path, self._phase1_buffer)
            self._flush_buffer(self.phase2_path, self._phase2_buffer)
            self._flush_buffer(self.phase3_path, self._phase3_buffer)
            self._phase1_buffer = []
            self._phase2_buffer = []
            self._phase3_buffer = []

    def cleanup(self):
        for path in [self.phase1_path, self.phase2_path, self.phase3_path]:
            if os.path.exists(path):
                os.remove(path)
        logger.info("Checkpoints cleaned up")

# ==============================================================================
# Pipeline Controller
# ==============================================================================

class PipelineController:
    def __init__(
        self,
        base_url: str,
        inf_base_url: Optional[str],
        ext_base_url: Optional[str],
        eval_base_url: Optional[str],
        model_name: str,
        inf_model_name: Optional[str],
        ext_model_name: Optional[str],
        eval_model_name: Optional[str],
        api_provider: str,
        api_key: Optional[str],
        inf_api_key: Optional[str],
        ext_api_key: Optional[str],
        eval_api_key: Optional[str],
        rerun_rate_limit: bool,
        global_concurrency: int,
        inf_config: PhaseConfig,
        ext_config: PhaseConfig,
        eval_config: PhaseConfig,
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip('/')
        self.inf_base_url = (inf_base_url or self.base_url).rstrip('/')
        self.ext_base_url = (ext_base_url or self.base_url).rstrip('/')
        self.eval_base_url = (eval_base_url or self.base_url).rstrip('/')
        self.model_name = model_name
        self.inf_model_name = inf_model_name or model_name
        self.ext_model_name = ext_model_name or model_name
        self.eval_model_name = eval_model_name or model_name
        self.api_provider = api_provider
        self.api_key = api_key
        self.inf_api_key = inf_api_key or api_key
        self.ext_api_key = ext_api_key or api_key
        self.eval_api_key = eval_api_key or api_key
        self.rerun_rate_limit = rerun_rate_limit
        self._headers = self._build_headers(self.api_key)
        self.inf_config = inf_config
        self.ext_config = ext_config
        self.eval_config = eval_config
        self.timeout = timeout
        self.max_retries = max_retries
        self.global_semaphore = asyncio.Semaphore(global_concurrency)

        self.inference_to_extraction: asyncio.Queue = asyncio.Queue()
        self.extraction_to_evaluation: asyncio.Queue = asyncio.Queue()

        self.stats = PipelineStats()
        self._stats_lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None
        self._openai_clients: Dict[Tuple[str, str], Any] = {}

        logger.info(
            "PipelineController initialized | Provider: %s | Base URL: %s | Max Retries: %s | Models: inf=%s ext=%s eval=%s",
            self.api_provider,
            self.base_url,
            max_retries,
            self.inf_model_name,
            self.ext_model_name,
            self.eval_model_name,
        )
        if self.inf_base_url != self.base_url or self.ext_base_url != self.base_url or self.eval_base_url != self.base_url:
            logger.info(
                "Phase Base URLs | inf=%s ext=%s eval=%s",
                self.inf_base_url,
                self.ext_base_url,
                self.eval_base_url,
            )

    def _build_headers(self, api_key: Optional[str]) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_provider == "openai":
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["Authorization"] = f"Bearer {api_key or 'EMPTY'}"
        return headers

    async def _get_openai_client(self, base_url: str, api_key: Optional[str]):
        if self.api_provider != "openai":
            return None
        if not OPENAI_AVAILABLE:
            logger.error("openai package not installed; please install `openai` to use --api-provider=openai.")
            return None
        if not api_key:
            logger.error("Missing API key for OpenAI client (base_url=%s).", base_url)
            return None
        client_key = (base_url, api_key or "")
        if client_key not in self._openai_clients:
            self._openai_clients[client_key] = AsyncOpenAI(api_key=api_key, base_url=base_url)
        return self._openai_clients[client_key]

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(limit=200, limit_per_host=200)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=self._headers
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        if self._openai_clients:
            for client in self._openai_clients.values():
                try:
                    await client.aclose()
                except Exception:
                    pass

    async def _api_call(
        self,
        messages: List[Dict],
        config: PhaseConfig,
        validator_fn: Optional[Callable[[str], bool]] = None,
        phase_tag: str = "unknown",  # "p1" / "p2" / "p3"
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.api_provider == "openai":
            return await self._api_call_openai(messages, config, validator_fn, phase_tag, model_name, base_url, api_key)
        """
        Unified API call entry point:
        - Retries for network/HTTP/timeout failures.
        - Retries for content validator failures with distinct handling.
        - Returns structured error_type/http_status metadata for stats and checkpoint recovery.
        """
        async with self.global_semaphore:
            async with self._stats_lock:
                self.stats.total_api_calls += 1

            session = await self._get_session()
            request_base_url = (base_url or self.base_url).rstrip("/")
            url = f"{request_base_url}/chat/completions"
            effective_api_key = api_key if api_key is not None else self.api_key
            request_headers = None
            if effective_api_key != self.api_key:
                request_headers = self._build_headers(effective_api_key)

            payload = {
                "model": model_name or self.model_name,
                "messages": messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "stream": False,
            }
            if config.reasoning_effort:
                payload["reasoning_effort"] = config.reasoning_effort

            last_error_type = None
            last_http_status = None

            for attempt in range(self.max_retries):
                try:
                    async with session.post(url, json=payload, headers=request_headers) as response:
                        last_http_status = response.status

                        if response.status == 200:
                            data = await response.json()
                            choice = data.get("choices", [{}])[0]
                            message = choice.get("message", {})
                            content = _content_to_text(message.get("content", ""))

                            if validator_fn and not validator_fn(content):
                                last_error_type = "validation_failed"
                                await asyncio.sleep(1.5 ** attempt)
                                continue

                            async with self._stats_lock:
                                self.stats.total_api_success += 1

                            return {
                                "success": True,
                                "content": content,
                                "reasoning": _as_str(message.get("reasoning_content", message.get("reasoning", ""))),
                                "error_type": None,
                                "http_status": response.status,
                            }

                        # HTTP error
                        is_overload = response.status in [429, 503]
                        last_error_type = "http_error_overload" if is_overload else "http_error"
                        backoff = 5 * (2 ** attempt) if is_overload else (1.5 ** attempt)
                        await asyncio.sleep(min(backoff, 60))

                except asyncio.TimeoutError:
                    last_error_type = "timeout"
                    await asyncio.sleep(2 * (attempt + 1))
                except aiohttp.ClientError as e:
                    last_error_type = f"client_error:{type(e).__name__}"
                    await asyncio.sleep(1.5 ** attempt)
                except Exception as e:
                    last_error_type = f"exception:{type(e).__name__}"
                    await asyncio.sleep(1)

            # failed after retries
            async with self._stats_lock:
                self.stats.total_api_failed += 1
                if phase_tag == "p1":
                    self.stats.p1_api_failed += 1
                elif phase_tag == "p2":
                    self.stats.p2_api_failed += 1
                elif phase_tag == "p3":
                    self.stats.p3_api_failed += 1

            return {
                "success": False,
                "content": "",
                "reasoning": "",
                "error_type": last_error_type or "unknown",
                "http_status": last_http_status,
            }

    async def _api_call_openai(
        self,
        messages: List[Dict],
        config: PhaseConfig,
        validator_fn: Optional[Callable[[str], bool]],
        phase_tag: str,
        model_name: Optional[str],
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> Dict[str, Any]:
        async with self.global_semaphore:
            async with self._stats_lock:
                self.stats.total_api_calls += 1

            request_base_url = (base_url or self.base_url).rstrip("/")
            effective_api_key = api_key if api_key is not None else self.api_key
            client = await self._get_openai_client(request_base_url, effective_api_key)
            if client is None:
                return {
                    "success": False,
                    "content": "",
                    "reasoning": "",
                    "error_type": "missing_openai_client",
                    "http_status": None,
                }

            target_model = model_name or self.model_name
            last_error_type = None
            last_http_status = None

            for attempt in range(self.max_retries):
                try:
                    response = await client.chat.completions.create(
                        model=target_model,
                        messages=messages,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        stream=False,
                    )

                    choice = response.choices[0]
                    message = choice.message
                    content = _content_to_text(message.content)

                    if validator_fn and not validator_fn(content):
                        last_error_type = "validation_failed"
                        await asyncio.sleep(1.5 ** attempt)
                        continue

                    async with self._stats_lock:
                        self.stats.total_api_success += 1

                    return {
                        "success": True,
                        "content": content,
                        "reasoning": _as_str(getattr(message, "reasoning_content", getattr(message, "reasoning", ""))),
                        "error_type": None,
                        "http_status": last_http_status,
                    }

                except RateLimitError as e:
                    last_error_type = "rate_limit"
                    last_http_status = getattr(e, "status_code", None) or getattr(e, "status", None)
                    await asyncio.sleep(min(5 * (2 ** attempt), 60))
                except APITimeoutError:
                    last_error_type = "timeout"
                    last_http_status = None
                    await asyncio.sleep(2 * (attempt + 1))
                except APIStatusError as e:
                    last_http_status = getattr(e, "status_code", None) or getattr(e, "status", None)
                    is_overload = last_http_status in [429, 503]
                    last_error_type = "http_error_overload" if is_overload else f"http_error_{last_http_status or 'unknown'}"
                    backoff = 5 * (2 ** attempt) if is_overload else (1.5 ** attempt)
                    await asyncio.sleep(min(backoff, 60))
                except APIConnectionError:
                    last_error_type = "connection_error"
                    last_http_status = None
                    await asyncio.sleep(1.5 ** attempt)
                except BadRequestError as e:
                    last_error_type = "bad_request"
                    last_http_status = getattr(e, "status_code", None) or getattr(e, "status", None)
                    # A 400 usually means the parameters are unsupported, so no retry is needed.
                    break
                except Exception as e:
                    last_error_type = f"exception:{type(e).__name__}"
                    last_http_status = None
                    await asyncio.sleep(1)

            async with self._stats_lock:
                self.stats.total_api_failed += 1
                if phase_tag == "p1":
                    self.stats.p1_api_failed += 1
                elif phase_tag == "p2":
                    self.stats.p2_api_failed += 1
                elif phase_tag == "p3":
                    self.stats.p3_api_failed += 1

            return {
                "success": False,
                "content": "",
                "reasoning": "",
                "error_type": last_error_type or "unknown",
                "http_status": last_http_status,
            }

    def _is_rate_limit_response(self, response: Dict[str, Any]) -> bool:
        error_type = _as_str(response.get("error_type", ""))
        http_status = response.get("http_status", None)
        if error_type in ("rate_limit", "http_error_overload"):
            return True
        return http_status in (429, 503)

    async def inference_worker(self, item: Dict, n_responses: int, input_key: str) -> Dict:
        responses = []
        existing = item.get("model_responses", [])
        if self.rerun_rate_limit and isinstance(existing, list) and existing:
            for r in existing:
                if self._is_rate_limit_response(r):
                    continue
                responses.append(r)

        remaining = max(0, n_responses - len(responses))
        for _ in range(remaining):
            query = INFERENCE_PROMPT.format(
                problem=_as_str(item.get("problem", "N/A")),
                proof_text=_as_str(item.get(input_key, item.get("mask_text", "")))
            )
            result = await self._api_call(
                messages=[{"role": "user", "content": query}],
                config=self.inf_config,
                validator_fn=validate_inference_output,
                phase_tag="p1",
                model_name=self.inf_model_name,
                base_url=self.inf_base_url,
                api_key=self.inf_api_key,
            )
            responses.append({
                "response": result.get("content", ""),
                "reasoning": result.get("reasoning", ""),
                "success": result.get("success", False),
                "error_type": result.get("error_type"),
                "http_status": result.get("http_status"),
            })

        item['model_responses'] = responses

        async with self._stats_lock:
            self.stats.inference_done += 1
            if any(r['success'] for r in responses):
                self.stats.inference_success += 1

        return item

    async def extraction_worker(self, item: Dict) -> Dict:
        """
        Phase 2:
        - Keep the original regex-first behavior with LLM fallback.
        - Record error_type / method / raw snippets for each attempt to support
          debugging and checkpoint decisions.
        """
        extract_attempts = []
        extracts_compact = []

        for resp in item.get('model_responses', []):
            text = _as_str(resp.get('response', ""))
            method = "regex"
            parsed = extract_regex(text)

            if parsed:
                extract_attempts.append({
                    "success": True,
                    "method": method,
                    "mask_id": parsed.get("mask_id", ""),
                    "formula": parsed.get("formula", ""),
                    "error_type": None,
                })
                extracts_compact.append({"mask_id": parsed.get("mask_id", ""), "formula": parsed.get("formula", "")})
                continue

            # regex failed
            method = "llm_fallback"
            if not text:
                extract_attempts.append({
                    "success": False,
                    "method": "regex",
                    "mask_id": "",
                    "formula": "",
                    "error_type": "p2_empty_input_text",
                })
                extracts_compact.append(None)
                continue

            prompt = EXTRACTION_PROMPT_TEMPLATE.format(raw_model_response=text[:15000])
            llm_result = await self._api_call(
                messages=[{"role": "user", "content": prompt}],
                config=self.ext_config,
                validator_fn=validate_phase2_extraction_json,
                phase_tag="p2",
                model_name=self.ext_model_name,
                base_url=self.ext_base_url,
                api_key=self.ext_api_key,
            )

            if not llm_result.get("success"):
                extract_attempts.append({
                    "success": False,
                    "method": method,
                    "mask_id": "",
                    "formula": "",
                    "error_type": "p2_llm_api_failed",
                    "api_error_type": llm_result.get("error_type"),
                    "http_status": llm_result.get("http_status"),
                })
                extracts_compact.append(None)
                continue

            obj = parse_loose_json_object(llm_result.get("content", ""))
            if not isinstance(obj, dict):
                async with self._stats_lock:
                    self.stats.p2_parse_failed += 1
                extract_attempts.append({
                    "success": False,
                    "method": method,
                    "mask_id": "",
                    "formula": "",
                    "error_type": "p2_llm_parse_failed",
                    "raw_output": llm_result.get("content", "")[:500],
                })
                extracts_compact.append(None)
                continue

            mask_id = obj.get("mask_id", None)
            formula = obj.get("formula", None)

            # Null is allowed, but record it as "no extraction".
            if mask_id is None or formula is None:
                extract_attempts.append({
                    "success": True,  # The API succeeded and returned valid JSON, but no extraction was found.
                    "method": method,
                    "mask_id": "" if mask_id is None else _as_str(mask_id),
                    "formula": "" if formula is None else _as_str(formula),
                    "error_type": "p2_no_extraction",
                })
                extracts_compact.append(None)
                continue

            parsed = {"mask_id": _as_str(mask_id).strip(), "formula": _as_str(formula).strip()}
            ok = bool(parsed["mask_id"]) and bool(parsed["formula"])
            extract_attempts.append({
                "success": ok,
                "method": method,
                "mask_id": parsed["mask_id"],
                "formula": parsed["formula"],
                "error_type": None if ok else "p2_partial_fields",
            })
            extracts_compact.append(parsed if ok else None)

        item["extract_attempts"] = extract_attempts
        item['extract_answers'] = extracts_compact  # Preserve the existing field for downstream compatibility.

        async with self._stats_lock:
            self.stats.extraction_done += 1
            if any((e is not None and _as_str(e.get("formula", "")).strip()) for e in extracts_compact if isinstance(e, dict)):
                self.stats.extraction_success += 1

        return item

    async def evaluation_worker(self, item: Dict) -> Dict:
        """
        Phase 3: Judge
        - First identify illegal inputs (for example missing GT), then filter and count them.
        - For each extraction attempt, distinguish API failures, parse failures,
          missing fields, and explicit model decisions.
        - eval_results stores error_type so checkpoints can decide whether reruns are needed.
        """
        illegal, reason = is_illegal_for_judge(item)
        if illegal:
            item["illegal_reason"] = reason
            item["eval_results"] = []
            item["correct_attempts"] = 0
            async with self._stats_lock:
                self.stats.evaluation_done += 1
                self.stats.illegal_samples += 1
            return item

        gt_list = item.get('answers', [])
        extracts = item.get('extract_answers', [])
        question = _as_str(item.get('question', item.get('problem', '')))

        gt_content = _as_str(gt_list[0].get('content', '')) if isinstance(gt_list[0], dict) else ""
        correct_count = 0
        eval_results = []

        for idx, extract in enumerate(extracts):
            if not extract or not isinstance(extract, dict) or not _as_str(extract.get('formula', '')).strip():
                eval_results.append({
                    'attempt_idx': idx,
                    'is_correct': False,
                    'error_type': 'extraction_failed',
                    'explanation': None
                })
                continue

            prompt = USER_PROMPT_EVAL_TEMPLATE.format(
                question=question,
                ground_truth=gt_content,
                generated_answer=_as_str(extract.get('formula', ''))
            )

            result = await self._api_call(
                messages=[{"role": "system", "content": SYSTEM_PROMPT_EVAL}, {"role": "user", "content": prompt}],
                config=self.eval_config,
                validator_fn=validate_phase3_judge_json,
                phase_tag="p3",
                model_name=self.eval_model_name,
                base_url=self.eval_base_url,
                api_key=self.eval_api_key,
            )

            # 1) API failure
            if not result.get("success"):
                eval_results.append({
                    'attempt_idx': idx,
                    'is_correct': False,
                    'error_type': 'judge_api_failed',
                    'api_error_type': result.get("error_type"),
                    'http_status': result.get("http_status"),
                    'explanation': None
                })
                continue

            # 2) Parse failure (the validator passed, but parsing can still fail in edge cases)
            parsed = parse_judge_json(result.get("content", ""))
            if not isinstance(parsed, dict):
                async with self._stats_lock:
                    self.stats.p3_parse_failed += 1
                eval_results.append({
                    'attempt_idx': idx,
                    'is_correct': False,
                    'error_type': 'judge_parse_failed',
                    'raw_output': result.get("content", "")[:500],
                    'explanation': None
                })
                continue

            # 3) Missing or empty fields (this is a structural I/O issue and should trigger a rerun)
            ok_fields, miss_reason = validate_judge_fields(parsed)
            if not ok_fields:
                async with self._stats_lock:
                    self.stats.p3_missing_fields += 1
                eval_results.append({
                    'attempt_idx': idx,
                    'is_correct': False,
                    'error_type': 'judge_missing_fields',
                    'missing_reason': miss_reason,
                    'raw_output': result.get("content", "")[:500],
                    'explanation': None
                })
                continue

            # 4) Explicit model decision
            is_correct = bool(parsed.get('is_correct', False))
            explanation = _as_str(parsed.get('explanation', ""))  # Empty strings are allowed because that is valid model behavior.
            if is_correct:
                correct_count += 1

            eval_results.append({
                'attempt_idx': idx,
                'is_correct': is_correct,
                'error_type': 'model_decision',
                'explanation': explanation
            })

        item['eval_results'] = eval_results
        item['correct_attempts'] = correct_count

        async with self._stats_lock:
            self.stats.evaluation_done += 1
            if correct_count > 0:
                self.stats.evaluation_correct += 1

        return item

    async def run_pipeline(
        self,
        data_list,
        n_responses,
        input_key,
        work_dir,
        output_prefix,
        num_inference_workers,
        num_extraction_workers,
        num_evaluation_workers,
        checkpoint: Optional[PipelineCheckpoint],
    ):
        total = len(data_list)
        results_map = {}
        results_lock = asyncio.Lock()

        pending_inference, pending_extraction, pending_evaluation = [], [], []
        if checkpoint:
            phase1_done, phase2_done, phase3_done, phase1_seen = checkpoint.load_state()
            for item in data_list:
                idx = int(item.get('original_index'))
                if idx in phase3_done:
                    results_map[idx] = phase3_done[idx]
                elif idx in phase2_done:
                    pending_evaluation.append(phase2_done[idx])
                elif idx in phase1_done:
                    pending_extraction.append(phase1_done[idx])
                else:
                    if idx in phase1_seen:
                        pending_inference.append(phase1_seen[idx])
                    else:
                        pending_inference.append(item)
            logger.info(
                f"Resume: {len(results_map)} done, "
                f"{len(pending_inference)} infer, {len(pending_extraction)} extract, {len(pending_evaluation)} eval"
            )
        else:
            pending_inference = data_list.copy()

        input_queue = asyncio.Queue()
        for item in pending_inference:
            await input_queue.put(item)
        for item in pending_extraction:
            await self.inference_to_extraction.put(item)
        for item in pending_evaluation:
            await self.extraction_to_evaluation.put(item)

        done_inf = total - len(pending_inference)
        done_ext = total - len(pending_inference) - len(pending_extraction)
        done_eval = len(results_map)

        pbar_inf = tqdm.tqdm(total=total, initial=done_inf, desc="🚀 Inference", position=0)
        pbar_ext = tqdm.tqdm(total=total, initial=done_ext, desc="🔍 Extraction", position=1)
        pbar_eval = tqdm.tqdm(total=total, initial=done_eval, desc="⚖️ Evaluation", position=2)

        inf_count, ext_count = [done_inf], [done_ext]

        async def inference_task():
            while True:
                try:
                    item = input_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                result = await self.inference_worker(item, n_responses, input_key)
                if checkpoint:
                    await checkpoint.save_phase1(result)
                await self.inference_to_extraction.put(result)
                pbar_inf.update(1)
                inf_count[0] += 1

        async def extraction_task():
            while True:
                if self.inference_to_extraction.empty() and inf_count[0] >= total:
                    break
                try:
                    item = await asyncio.wait_for(self.inference_to_extraction.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if inf_count[0] >= total and self.inference_to_extraction.empty():
                        break
                    continue
                result = await self.extraction_worker(item)
                if checkpoint:
                    await checkpoint.save_phase2(result)
                await self.extraction_to_evaluation.put(result)
                pbar_ext.update(1)
                ext_count[0] += 1

        async def evaluation_task():
            while True:
                if self.extraction_to_evaluation.empty() and ext_count[0] >= total:
                    break
                try:
                    item = await asyncio.wait_for(self.extraction_to_evaluation.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if ext_count[0] >= total and self.extraction_to_evaluation.empty():
                        break
                    continue
                result = await self.evaluation_worker(item)
                if checkpoint:
                    await checkpoint.save_phase3(result)
                async with results_lock:
                    results_map[int(result.get('original_index'))] = result
                pbar_eval.update(1)

        tasks = []
        tasks.extend([asyncio.create_task(inference_task()) for _ in range(num_inference_workers)])
        tasks.extend([asyncio.create_task(extraction_task()) for _ in range(num_extraction_workers)])
        tasks.extend([asyncio.create_task(evaluation_task()) for _ in range(num_evaluation_workers)])

        await asyncio.gather(*tasks)
        if checkpoint:
            await checkpoint.flush_all()

        pbar_inf.close()
        pbar_ext.close()
        pbar_eval.close()

        final_data = [results_map[k] for k in sorted(results_map.keys())]
        save_jsonl_atomic(os.path.join(work_dir, f"{output_prefix}_evaluation.jsonl"), final_data)

        # Emit a summary without affecting existing outputs.
        async with self._stats_lock:
            logger.info("========== Pipeline Stats ==========")
            logger.info(f"P1 done={self.stats.inference_done} success={self.stats.inference_success} p1_api_failed={self.stats.p1_api_failed}")
            logger.info(f"P2 done={self.stats.extraction_done} success={self.stats.extraction_success} p2_api_failed={self.stats.p2_api_failed} p2_parse_failed={self.stats.p2_parse_failed}")
            logger.info(f"P3 done={self.stats.evaluation_done} correct_samples={self.stats.evaluation_correct} p3_api_failed={self.stats.p3_api_failed} p3_parse_failed={self.stats.p3_parse_failed} p3_missing_fields={self.stats.p3_missing_fields} illegal={self.stats.illegal_samples}")
            logger.info(f"API calls total={self.stats.total_api_calls} success={self.stats.total_api_success} failed={self.stats.total_api_failed}")
            logger.info("===================================")

        return final_data

# ==============================================================================
# Phase 4 filtering (kept as-is)
# ==============================================================================

def run_filtering(data_list, max_correct, work_dir, output_prefix):
    print(f"\n🌪️ [Phase 4] Filtering (Keep if correct_attempts <= {max_correct})")
    correct_dist = defaultdict(int)
    illegal_dist = defaultdict(int)

    for item in data_list:
        if item.get("illegal_reason"):
            illegal_dist[item["illegal_reason"]] += 1
        correct_dist[item.get("correct_attempts", 0)] += 1

    if illegal_dist:
        logger.info("Illegal sample distribution:")
        for k, v in sorted(illegal_dist.items(), key=lambda x: (-x[1], x[0])):
            logger.info(f"  illegal_reason={k}: {v}")

    for k in sorted(correct_dist.keys()):
        marker = "✓" if k <= max_correct else "✗"
        logger.info(f"  {marker} correct_attempts={k}: {correct_dist[k]} samples")

    filtered = [item for item in data_list if item.get("correct_attempts", 0) <= max_correct and not item.get("illegal_reason")]
    save_jsonl_atomic(os.path.join(work_dir, f"{output_prefix}_filtered_hard.jsonl"), filtered)

# ==============================================================================
# Main
# ==============================================================================

async def main_async(args):
    global logger
    logger = setup_logger(args.log_level)
    os.makedirs(args.work_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Coarse Filter Pipeline - Flagship++ (Auto-Retry & Sanitization & Failure Taxonomy)")
    logger.info("=" * 60)

    raw_data = load_jsonl(args.input_file)
    if not raw_data:
        logger.error("Input file is empty or invalid JSONL.")
        return

    data = prepare_dataset(raw_data)
    
    if not data:
        logger.error("No valid samples after sanitization (need non-empty problem & input).")
        return

    checkpoint = None
    if args.resume or not args.no_checkpoint:
        checkpoint = PipelineCheckpoint(
            args.work_dir,
            args.output_prefix,
            rerun_rate_limit=args.rerun_rate_limit,
        )

    api_provider = args.api_provider
    default_base_url = args.base_url
    if not default_base_url:
        if api_provider == "openai":
            default_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        else:
            default_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    base_url = normalize_base_url(default_base_url)
    if "/v1" not in base_url:
        logger.warning("Base URL does not include /v1; ensure %s is correct for /chat/completions", base_url)

    inf_base_url = normalize_base_url(args.inf_base_url or base_url)
    ext_base_url = normalize_base_url(args.ext_base_url or base_url)
    eval_base_url = normalize_base_url(args.eval_base_url or base_url)

    api_key = args.api_key
    if api_provider == "openai":
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "")
    else:
        if not api_key:
            api_key = os.getenv("VLLM_API_KEY", "EMPTY")

    inf_api_key = args.inf_api_key or api_key
    ext_api_key = args.ext_api_key or api_key
    eval_api_key = args.eval_api_key or api_key
    if api_provider == "openai":
        missing_keys = []
        if not inf_api_key:
            missing_keys.append("inf")
        if not ext_api_key:
            missing_keys.append("ext")
        if not eval_api_key:
            missing_keys.append("eval")
        if missing_keys:
            logger.error(
                "Missing API key(s) for phases: %s. Provide --api-key/OPENAI_API_KEY or per-phase keys.",
                ",".join(missing_keys),
            )
            return

    inf_reasoning_effort = normalize_reasoning_effort(args.inf_reasoning_effort)
    eval_reasoning_effort = normalize_reasoning_effort(args.eval_reasoning_effort)

    inf_config = PhaseConfig(
        temperature=args.inf_temp, top_p=args.inf_top_p, max_tokens=args.inf_max_tokens,
        reasoning_effort=inf_reasoning_effort
    )
    ext_config = PhaseConfig(
        temperature=args.ext_temp, top_p=args.ext_top_p, max_tokens=args.ext_max_tokens,
        reasoning_effort=None
    )
    eval_config = PhaseConfig(
        temperature=args.eval_temp, top_p=args.eval_top_p, max_tokens=args.eval_max_tokens,
        reasoning_effort=eval_reasoning_effort
    )

    controller = PipelineController(
        base_url=base_url,
        inf_base_url=inf_base_url,
        ext_base_url=ext_base_url,
        eval_base_url=eval_base_url,
        model_name=args.model_path,
        inf_model_name=args.inf_model_path or args.model_path,
        ext_model_name=args.ext_model_path or args.model_path,
        eval_model_name=args.eval_model_path or args.model_path,
        api_provider=api_provider,
        api_key=api_key,
        inf_api_key=inf_api_key,
        ext_api_key=ext_api_key,
        eval_api_key=eval_api_key,
        rerun_rate_limit=args.rerun_rate_limit,
        global_concurrency=args.global_concurrency,
        inf_config=inf_config,
        ext_config=ext_config,
        eval_config=eval_config,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    try:
        results = await controller.run_pipeline(
            data_list=data,
            n_responses=args.n_responses,
            input_key=args.input_key,
            work_dir=args.work_dir,
            output_prefix=args.output_prefix,
            num_inference_workers=args.inference_workers,
            num_extraction_workers=args.extraction_workers,
            num_evaluation_workers=args.evaluation_workers,
            checkpoint=checkpoint,
        )
        run_filtering(results, args.max_correct, args.work_dir, args.output_prefix)

        if checkpoint:
            logger.info(f"Checkpoints preserved for debugging at: {checkpoint.ckpt_dir}")

    finally:
        await controller.close()


def main():
    parser = argparse.ArgumentParser(description="Coarse Filter Pipeline")
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--work-dir', type=str, required=True)
    parser.add_argument('--output-prefix', type=str, default='pipeline_run')
    parser.add_argument('--api-provider', type=str, default='vllm', choices=['vllm', 'openai'])
    parser.add_argument('--api-key', type=str, default=None)
    parser.add_argument('--base-url', type=str, default=None)
    parser.add_argument('--inf-api-key', type=str, default=None)
    parser.add_argument('--ext-api-key', type=str, default=None)
    parser.add_argument('--eval-api-key', type=str, default=None)
    parser.add_argument('--inf-base-url', type=str, default=None)
    parser.add_argument('--ext-base-url', type=str, default=None)
    parser.add_argument('--eval-base-url', type=str, default=None)
    # -- Model Settings --
    # model-path sets the shared default model, while inf/ext/eval-model-path can override it per phase.
    parser.add_argument('--model-path', type=str, default="openai/gpt-oss-20b")
    parser.add_argument('--inf-model-path', type=str, default=None)
    parser.add_argument('--ext-model-path', type=str, default=None)
    parser.add_argument('--eval-model-path', type=str, default=None)
    
    parser.add_argument('--global-concurrency', type=int, default=128)
    parser.add_argument('--inference-workers', type=int, default=4)
    parser.add_argument('--extraction-workers', type=int, default=8)
    parser.add_argument('--evaluation-workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=60000.0)
    parser.add_argument('--max-retries', type=int, default=3, help="Max retries per API call (default: 3)")
    parser.add_argument('--n-responses', type=int, default=4)
    parser.add_argument('--max-correct', type=int, default=1)
    parser.add_argument('--input-key', type=str, default="input")

    # Phase 1
    parser.add_argument('--inf-temp', type=float, default=1.0)
    parser.add_argument('--inf-top-p', type=float, default=1.0)
    parser.add_argument('--inf-max-tokens', type=int, default=40000)
    parser.add_argument('--inf-reasoning-effort', type=str, default="high")

    # Phase 2
    parser.add_argument('--ext-temp', type=float, default=0.0)
    parser.add_argument('--ext-top-p', type=float, default=1.0)
    parser.add_argument('--ext-max-tokens', type=int, default=10000)

    # Phase 3
    parser.add_argument('--eval-temp', type=float, default=1.0)
    parser.add_argument('--eval-top-p', type=float, default=1.0)
    parser.add_argument('--eval-max-tokens', type=int, default=80000)
    parser.add_argument('--eval-reasoning-effort', type=str, default="high")

    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--keep-checkpoints', action='store_true')
    parser.add_argument('--rerun-rate-limit', action='store_true')
    parser.add_argument('--log-level', type=str, default='INFO')

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
