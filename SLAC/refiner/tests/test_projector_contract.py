"""Offline, model-free regression tests for chunk-budget and text contracts."""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slac_refiner.decoding.projector import (
    ProjectorBudgetError,
    ProjectorConfig,
    boundary_vector_to_spans,
    project_boundary_vector,
    rebuild_chunks_from_boundary_vector,
    token_len_proxy,
)


def config(**overrides):
    values = dict(max_chunk_atoms=64, min_chunk_atoms=0, max_chunk_chars=10000,
                  min_chunk_chars=0, max_chunk_tokens=384, min_chunk_tokens=0)
    values.update(overrides)
    return ProjectorConfig(**values)


class ProjectorContractTests(unittest.TestCase):
    def test_soft_min_cannot_reintroduce_390_token_chunk(self):
        atoms = [" ".join(["x"] * count) for count in (380, 10, 380)]
        result = project_boundary_vector(atoms, [1, 1], config(min_chunk_tokens=48), strict=True)
        self.assertEqual([token_len_proxy(unit["text"]) for unit in result["projected_units"]], [380, 10, 380])
        self.assertTrue(result["hard_max_satisfied"])
        self.assertEqual([item["start_atom"] for item in result["short_spans"]], [1])

    def test_merge_checks_all_maxima_and_full_text_characters(self):
        # Two two-character atoms become five characters after a newline join.
        result = project_boundary_vector(["ab", "cd"], [1],
                                         config(max_chunk_chars=4, min_chunk_chars=3))
        self.assertEqual(result["projected_b"], [1])
        self.assertTrue(result["hard_max_satisfied"])
        # Token/character budgets permit this merge, but the atom maximum does not.
        result = project_boundary_vector(["a", "b"], [1],
                                         config(max_chunk_atoms=1, min_chunk_tokens=2))
        self.assertEqual(result["projected_b"], [1])

    def test_unfeasible_preferred_side_does_not_prevent_other_side(self):
        atoms = [" ".join(["x"] * count) for count in (380, 10, 30)]
        result = project_boundary_vector(atoms, [1, 1], config(min_chunk_tokens=48), [0.0, 1.0])
        self.assertEqual(result["spans_after_merge"], [(0, 1), (1, 3)])

    def test_unmergeable_short_span_does_not_block_later_merges(self):
        atoms = ["x", " ".join(["x"] * 384), "a", "b"]
        result = project_boundary_vector(atoms, [1, 1, 1], config(min_chunk_tokens=2))
        self.assertEqual(result["spans_after_merge"], [(0, 1), (1, 2), (2, 4)])

    def test_indivisible_atom_is_reported_and_strict_mode_raises(self):
        result = project_boundary_vector(["x" * 5], [], config(max_chunk_chars=4))
        self.assertFalse(result["hard_max_satisfied"])
        self.assertEqual(result["overlong_spans"][0]["violated_dimensions"], ["chars"])
        self.assertEqual(result["overlong_spans"][0]["reason"], "indivisible_atom")
        with self.assertRaises(ProjectorBudgetError) as caught:
            project_boundary_vector(["x" * 5], [], config(max_chunk_chars=4), strict=True)
        self.assertEqual(caught.exception.overlong_spans, result["overlong_spans"])

    def test_nonadditive_tokenizer_measures_whole_returned_text(self):
        seen = []

        def counter(text):
            seen.append(text)
            # Simulates boundary-sensitive encoding: combined count exceeds sum.
            return 7 if "\n" in text else 2

        result = project_boundary_vector(["a", "b"], [0],
                                         config(max_chunk_tokens=5, min_chunk_tokens=3),
                                         token_counter=counter, strict=True)
        self.assertEqual(result["projected_b"], [1])
        self.assertIn("a\nb", seen)
        self.assertEqual(result["token_count_mode"], "custom_whole_span")
        # Special-token overhead makes combined count smaller than the sum.
        result = project_boundary_vector(["a", "b"], [1],
                                         config(max_chunk_tokens=3, min_chunk_tokens=3),
                                         token_counter=lambda text: len(text.split()) + 1)
        self.assertEqual(result["projected_b"], [0])

    def test_lossless_source_mode_retains_separators_and_outer_whitespace(self):
        source = "  alpha\r\n\r\nbeta\t gamma  "
        atoms = ["alpha", "beta", "gamma"]
        offsets = [(source.index(atom), source.index(atom) + len(atom)) for atom in atoms]
        result = rebuild_chunks_from_boundary_vector(
            atoms, [0, 0], config(max_chunk_atoms=1), source_text=source,
            atom_char_spans=offsets, token_counter=len, strict=True,
        )
        units = result["projected_units"]
        self.assertEqual("".join(unit["text"] for unit in units), source)
        self.assertEqual(result["text_mode"], "source_spans")
        for unit in units:
            self.assertEqual(unit["text"], source[unit["start_char"]:unit["end_char"]])

    def test_source_separators_count_toward_hard_limits(self):
        source = "a     b"
        result = project_boundary_vector(["a", "b"], [1], config(max_chunk_chars=4),
                                         source_text=source, atom_char_spans=[(0, 1), (6, 7)])
        self.assertFalse(result["hard_max_satisfied"])
        self.assertEqual(result["overlong_spans"][0]["stats"]["chars"], 6)

    def test_empty_single_and_legacy_text_mode(self):
        empty = project_boundary_vector([], [], strict=True)
        self.assertEqual(empty["projected_units"], [])
        self.assertTrue(empty["hard_max_satisfied"])
        single = project_boundary_vector(["  alpha  "], [], strict=True)
        self.assertEqual(single["projected_units"][0]["text"], "alpha")
        self.assertEqual(single["text_mode"], "legacy_newline_strip")
        self.assertEqual(single["token_count_mode"], "regex_proxy")

    def test_split_and_merge_are_not_limited_to_32_rounds(self):
        atoms = ["a"] * 100
        result = project_boundary_vector(atoms, [0] * 99, config(max_chunk_atoms=1),
                                         list(reversed(range(99))), strict=True)
        self.assertEqual(len(result["projected_units"]), 100)
        result = project_boundary_vector(atoms, [1] * 99,
                                         config(max_chunk_atoms=100, min_chunk_atoms=100))
        self.assertEqual(len(result["projected_units"]), 1)

    def test_invalid_boundaries_config_counter_and_source_are_rejected(self):
        for boundaries in ([0, 0], [], [2]):
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError):
                project_boundary_vector(["a", "b"], boundaries)
        with self.assertRaises(ValueError):
            project_boundary_vector(["a"], [], config(max_chunk_tokens=0))
        with self.assertRaises(ValueError):
            project_boundary_vector(["a"], [], token_counter=lambda text: 1.2)
        with self.assertRaises(ValueError):
            project_boundary_vector(["a", "b"], [0], gap_scores=[])
        with self.assertRaises(ValueError):
            project_boundary_vector(["a"], [], source_text="a")
        with self.assertRaises(ValueError):
            project_boundary_vector(["a"], [], source_text="b", atom_char_spans=[(0, 1)])
        with self.assertRaises(ValueError):
            project_boundary_vector([], [], source_text="a", atom_char_spans=[])

    def test_random_partitions_preserve_order_and_simultaneous_maxima(self):
        rng = random.Random(20260926)
        cfg = config(max_chunk_atoms=5, min_chunk_atoms=2, max_chunk_chars=35,
                     min_chunk_chars=12, max_chunk_tokens=8, min_chunk_tokens=4)
        for _ in range(200):
            atoms = [" ".join(["x"] * rng.randint(1, 5)) for _ in range(rng.randint(1, 35))]
            result = project_boundary_vector(atoms, [rng.randrange(2) for _ in atoms[1:]], cfg, strict=True)
            spans = result["spans_after_merge"]
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], len(atoms))
            self.assertEqual(boundary_vector_to_spans(len(atoms), result["projected_b"]), spans)
            for left, right in zip(spans, spans[1:]):
                self.assertEqual(left[1], right[0])
            for unit in result["projected_units"]:
                self.assertLessEqual(unit["end_atom"] - unit["start_atom"], cfg.max_chunk_atoms)
                self.assertLessEqual(len(unit["text"]), cfg.max_chunk_chars)
                self.assertLessEqual(token_len_proxy(unit["text"]), cfg.max_chunk_tokens)


if __name__ == "__main__":
    unittest.main()
