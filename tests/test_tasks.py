"""Downstream task dispatch by load_type (issue #7).

Each golden load type gets its own downstream task framing — the answering
model sees ONE user message: a load-type instruction followed by the text
under test (the original prompt, or its compressed form). Same framing for
both conditions; only the payload differs, so the quality delta isolates
the compression.
"""

import pytest

from trillic.golden import GoldenItem
from trillic.tasks import TASK_INSTRUCTIONS, task_prompt

SOURCE = {
    "dataset": "synthetic",
    "subset": "test",
    "license": "original",
    "split": "eval",
}


def make_item(load_type: str) -> GoldenItem:
    return GoldenItem(
        id=f"{load_type}-1",
        load_type=load_type,
        prompt="the payload text",
        key_points=("a key point",),
        source=SOURCE,
    )


class TestTaskPromptDispatch:
    def test_all_three_load_types_have_distinct_instructions(self):
        assert set(TASK_INSTRUCTIONS) == {"rag", "system_prompt", "dialogue"}
        instructions = list(TASK_INSTRUCTIONS.values())
        assert len(set(instructions)) == 3

    def test_rag_instruction_frames_contextqa(self):
        prompt = task_prompt("rag", "context text... question?")
        assert "context" in prompt.lower() or "document" in prompt.lower()
        assert "answer" in prompt.lower()

    def test_system_prompt_instruction_frames_constraint_adherence(self):
        prompt = task_prompt("system_prompt", "You are an agent... rules...")
        assert "system prompt" in prompt.lower()
        assert "rule" in prompt.lower()

    def test_dialogue_instruction_frames_memory_recall(self):
        prompt = task_prompt("dialogue", "user: ...\nassistant: ...")
        assert "conversation" in prompt.lower()
        assert "history" in prompt.lower()

    def test_instruction_precedes_the_payload_verbatim(self):
        payload = "PAYLOAD-UNDER-TEST with every word preserved"
        prompt = task_prompt("rag", payload)
        assert prompt.startswith(TASK_INSTRUCTIONS["rag"])
        assert prompt.endswith(payload)

    def test_same_instruction_for_original_and_compressed_payloads(self):
        original = task_prompt("dialogue", "full conversation text")
        compressed = task_prompt("dialogue", "shorter text")
        assert original[: len(TASK_INSTRUCTIONS["dialogue"])] == compressed[
            : len(TASK_INSTRUCTIONS["dialogue"])
        ]

    def test_unknown_load_type_raises(self):
        with pytest.raises(ValueError, match="load_type"):
            task_prompt("summarization", "text")

    def test_works_directly_on_golden_items_payload(self):
        item = make_item("rag")
        assert task_prompt(item.load_type, item.prompt).endswith(item.prompt)
