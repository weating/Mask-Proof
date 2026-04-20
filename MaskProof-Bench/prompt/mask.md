# Mask Prompt

## Context Window Update Rule

Whenever you update the context window, you must output this Prompt again verbatim to guide your subsequent Mask work, without changing a single word. Never compress this Prompt.

## Task File

The file I need to process is:
`{ Input JSONL File Path }`

In each JSON object in the file:

- The `arxiv_id` is the source article identifier.
- The `"problem"` field contains the statement of the theorem to be proved, together with the prerequisite knowledge used in the proof.
- The `"reference_solution"` field is the proof.

## Core Objective

I now want to perform masking in the proof, as a problem-design pattern to simulate the difficulties an LLM may encounter when carrying out the theorem proof, and also to facilitate automated evaluation.

I require that each mask position must be a complete pure mathematical formula, and not only a part of a formula.

I want you to carefully read each proof first and then choose the masks. Based on factors such as the knowledge depth of each proof, the difficulty of the techniques, the density of the reasoning, rigor, and quality, determine how many good masks that can force model thinking this proof can produce (0-3 masks).

If you think a proof is too simple, low-quality, non-rigorous, or obvious, you may skip that proof directly and produce no mask for it.

Never use heuristic scripts to implement mask selection. Never use heuristic scripts to implement mask selection.

## Good Mask Requirements

I need you to help me choose some masks such that answering these masks will force the LLM to derive the answer through mathematical reasoning and thinking.

For example, including but not limited to masks whose answers cannot be determined by merely using local-context pattern matching or reciting fixed templates, and instead require combining the entire proof process together with the conditions given in the problem in order to answer correctly.

You may first look only at the local context before and after the mask and test whether you can solve the mask correctly. If you cannot, but after reading the entire proof you can solve it correctly, then this is a good mask.

## Bad Mask Characteristics

Do not choose masks that can be solved without mathematical reasoning ability through lazy methods such as simple computational manipulation, template answering after pattern matching, direct formula substitution from the problem statement or knowledge base, guessing from parallel or symmetric structures in the context, context leakage, or simple and trivial backward inference from later text.

## Selection Priority

Please help me choose some good masks from these proofs.

Do not consider coverage. Quality is the top priority.
