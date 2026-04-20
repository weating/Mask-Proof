# Self-contained Recovery Prompt

## Task Description

You are given a JSONL file containing extracted theorem-proof pairs:
`{ File Path }`.

You are also given the full LaTeX sources of the corresponding papers, located at:
`{ Original Paper LaTeX file Path }`.

Your task is to process each extracted proof and transform it into a self-contained proof, using the full paper LaTeX sources as reference when necessary.

The final output must be written to a new JSONL file. If the target output file already exists, do not overwrite it; instead, append the new entries to the end of the file.

## Core Objective

For each theorem-proof pair:

- Ensure that the proof can be fully understood and completed without referring to any external material other than the problem statement itself.
- Any additional information required to make the proof self-contained must be added to the `"question"` field as Additional Information.

## Self-Containedness Criteria

A proof is considered not self-contained if any of the following issues occur:

1. **Missing theorem or lemma statements.**
   If the proof refers to a theorem, lemma, or proposition whose statement is not included, locate the corresponding statement in the original paper using its LaTeX label and append it to the `"question"` field.

2. **Missing key definitions with explicit mathematical content.**
   If the proof relies on a symbol or object whose explicit definition, such as a closed-form expression or precise construction, is given elsewhere in the paper, and that definition is essential for the reasoning, append the definition to the `"question"` field.

3. **Non-standard or paper-specific symbol meanings.**
   Some symbols may be defined in a non-standard or paper-specific way. You must decide whether the meaning of such a symbol can reasonably be assumed as common background knowledge for the solver, or whether it must be explicitly provided.
   If explicit clarification is needed, append the definition to the `"question"` field.

4. **Custom LaTeX macros.**
   Inspect each paper for user-defined macros. Any custom macros appearing in the proof must be replaced with equivalent standard LaTeX commands. The final proof text must not rely on paper-specific macro definitions.

## Standardization Requirements

The standardized self-contained version must satisfy all of the following:

- All custom LaTeX macros are rewritten using standard LaTeX commands.
- The theorem statement and proof together form a complete, standalone mathematical problem.
- Any lemmas, definitions, or constructions used in the proof but defined elsewhere in the paper are explicitly included when necessary.
- For symbols with unconventional meanings, you must judge whether the definition is part of assumed background knowledge or must be explicitly stated. If the latter, include the definition.
- Beyond the listed cases, you are expected to use semantic and mathematical judgment to determine what additional information is required so that a solver can complete the proof by reading the problem alone.

## Output Constraint

Only output the transformed JSON entries.
Do not include explanations, commentary, or intermediate reasoning.
Ensure the output JSON is valid and directly appendable to an existing JSONL file.
