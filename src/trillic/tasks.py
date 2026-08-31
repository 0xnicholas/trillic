"""Downstream task framing per load_type (issue #7: 三类负载分派接线).

The golden set's three load types map to three downstream tasks:

- rag — RAG question answering: answer the question using only the
  provided context;
- system_prompt — constraint adherence: adopt the system prompt and reply
  to the user message honoring every rule (format / refusal / tone);
- dialogue — conversation memory: continue as the assistant and answer the
  final user message using facts established in the history.

`task_prompt` frames ONE payload (the original prompt or its compressed
form) with the load type's instruction. The framing is identical for both
conditions of a pair — only the payload differs — so the measured quality
delta isolates what compression did to the payload.
"""

from trillic.golden import LOAD_TYPES

TASK_INSTRUCTIONS: dict[str, str] = {
    "rag": (
        "Answer the question in the material below using only what it "
        "states. If the material does not contain the answer, say exactly "
        "that. Keep the answer short and factual."
    ),
    "system_prompt": (
        "Adopt the role defined by the system prompt below and reply to "
        "its user message as that assistant would, honoring every rule it "
        "states (format, refusals, and tone included). Reply only with the "
        "assistant's answer."
    ),
    "dialogue": (
        "You are the assistant in the conversation below. Using the "
        "conversation history, answer the final user message. Facts the "
        "history establishes must come from the history, not from "
        "guesswork."
    ),
}


def task_prompt(load_type: str, payload: str) -> str:
    """Instruction for `load_type` followed by the payload under test."""
    try:
        instruction = TASK_INSTRUCTIONS[load_type]
    except KeyError:
        raise ValueError(
            f"unknown load_type {load_type!r} — task dispatch covers "
            f"{list(LOAD_TYPES)}"
        ) from None
    return f"{instruction}\n\n{payload}"
