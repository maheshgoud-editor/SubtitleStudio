"""Focused stdlib tests for caption grouping helpers in app.py.

The helpers are extracted with AST so this test does not need to import the
full FastAPI/Whisper application or download any speech models.
"""
import ast
import difflib
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
TARGETS = {
    "_align_corrected_words",
    "_rebalance_short_caption_tail",
    "split_caption_rows",
}
HELPERS = [
    node for node in TREE.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name in TARGETS
]
NAMESPACE = {"difflib": difflib, "re": re}
exec(compile(ast.Module(body=HELPERS, type_ignores=[]), str(ROOT / "app.py"), "exec"), NAMESPACE)
split_caption_rows = NAMESPACE["split_caption_rows"]


class CaptionGroupingTests(unittest.TestCase):
    def test_default_groups_are_five_words_and_balances_short_tail(self):
        rows = split_caption_rows([
            {"start": 0, "end": 7, "text": "one two three four five six seven", "words": []}
        ])
        counts = [len(row["text"].split()) for row in rows]
        self.assertEqual(counts, [4, 3])
        self.assertTrue(all(count <= 5 for count in counts))

    def test_user_selected_four_word_limit_is_respected(self):
        rows = split_caption_rows([
            {"start": 0, "end": 9, "text": "one two three four five six seven eight nine", "words": []}
        ], max_words=4)
        self.assertTrue(all(len(row["text"].split()) <= 4 for row in rows))
        self.assertEqual([len(row["text"].split()) for row in rows], [3, 3, 3])
        self.assertEqual(
            " ".join(row["text"] for row in rows),
            "one two three four five six seven eight nine",
        )

    def test_does_not_rebalance_across_sentence_punctuation(self):
        rows = split_caption_rows([
            {"start": 0, "end": 7, "text": "one two three four five. six seven", "words": []}
        ], max_words=5)
        self.assertEqual([len(row["text"].split()) for row in rows], [5, 2])

    def test_translated_text_is_used_with_source_word_timestamps(self):
        source_words = [
            {"start": 0.0, "end": 0.5, "text": "hello"},
            {"start": 0.5, "end": 1.0, "text": "world"},
        ]
        rows = split_caption_rows([
            {"start": 0.0, "end": 1.0, "text": "नमस्ते दुनिया", "words": source_words}
        ])
        self.assertEqual(" ".join(row["text"] for row in rows), "नमस्ते दुनिया")
        self.assertEqual(
            [word["text"] for row in rows for word in row["words"]],
            ["नमस्ते", "दुनिया"],
        )

    def test_timed_words_are_grouped_without_losing_text(self):
        words = [
            {"start": float(i), "end": float(i + 1), "text": f"word{i + 1}"}
            for i in range(7)
        ]
        rows = split_caption_rows([
            {"start": 0.0, "end": 7.0, "text": " ".join(w["text"] for w in words), "words": words}
        ])
        counts = [len(row["words"]) for row in rows]
        self.assertEqual(counts, [4, 3])
        self.assertEqual(" ".join(row["text"] for row in rows), " ".join(w["text"] for w in words))


if __name__ == "__main__":
    unittest.main()
