"""Offline regressions for the legal decoder language; no model or corpus load."""
from itertools import product
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slac_refiner.decoding.dp_edit_decode import batch_decode, decode_one
from slac_refiner.label_contract import derive_canonical_labels, replay_labels
from slac_refiner.models.heads import RefinerHeads
from scripts import infer_bestofn


def _oracle_edit_score(b0, logits, K, lambda_del=1.0, lambda_shift=0.25):
    """Exhaustive independent oracle over all classes, retaining DEL history."""
    initial = [g for g, value in enumerate(b0) if value]
    log_probs = logits.log_softmax(-1)
    best = -float("inf")
    for actions in product(range(2 * K + 2), repeat=len(initial)):
        last, score = -1, 0.0
        for row, (g, action) in enumerate(zip(initial, actions)):
            score += float(log_probs[row, action])
            if action == 0:
                score -= lambda_del
                continue
            shift = action - 1 - K
            pos = g + shift
            if not last < pos < len(b0):
                break
            last = pos
            score -= lambda_shift * abs(shift)
        else:
            best = max(best, score)
    return best


def _trace_score(labels, logits, K, lambda_del=1.0, lambda_shift=0.25):
    log_probs = logits.log_softmax(-1)
    score = 0.0
    for row, item in enumerate(labels):
        label = item["y"]
        shift = 0 if label in ("KEEP", "DEL") else int(label.split(":")[1])
        cls = 0 if label == "DEL" else 1 + K + shift
        score += float(log_probs[row, cls]) - (lambda_del if label == "DEL" else lambda_shift * abs(shift))
    return score


def test_dp_matches_exhaustive_objective():
    rng = random.Random(712)
    generator = torch.Generator().manual_seed(82)
    for _ in range(60):
        G, K = rng.randint(1, 6), rng.randint(0, 2)
        initial = sorted(rng.sample(range(G), rng.randint(0, min(G, 4))))
        b0 = [int(g in initial) for g in range(G)]
        logits = torch.randn((len(initial), 2 * K + 2), generator=generator) * 3
        result = decode_one(b0, initial, logits, torch.full((G,), -40.0), K=K)
        assert _trace_score(result["pred_edit_labels"], logits, K) == pytest.approx(
            _oracle_edit_score(b0, logits, K), abs=2e-6
        )
        assert replay_labels(b0, {"edit": result["pred_edit_labels"], "insert": result["pred_insert_labels"]}, K=K) == result["pred_b"]


def test_deletion_cannot_reset_monotonic_state():
    # The previous decoder took SHIFT:+2, DEL, SHIFT:-2 and hid the crossing
    # by sorting/deduplicating the final set. Its trace could not legally replay.
    logits = torch.full((3, 6), -30.0)
    logits[0, 5] = 30  # 0 -> 2
    logits[1, 0] = 30  # delete gap 1
    logits[2, 1] = 30  # 2 -> 0 (illegal after 2)
    result = decode_one([1, 1, 1], [0, 1, 2], logits, torch.full((3,), -40.0), K=2)
    assert _trace_score(result["pred_edit_labels"], logits, 2) == pytest.approx(_oracle_edit_score([1, 1, 1], logits, 2))
    assert replay_labels([1, 1, 1], {"edit": result["pred_edit_labels"], "insert": result["pred_insert_labels"]}, K=2) == result["pred_b"]


def test_adjacent_inserts_survive_and_duplicate_is_suppressed():
    result = decode_one([0, 1, 0], [1], torch.tensor([[-30.0, 30.0]]), torch.full((3,), 30.0), K=0)
    assert result["pred_b"] == [1, 1, 1]
    assert result["pred_insert_labels"] == [1, 0, 1]


@pytest.mark.parametrize("G", [0, 1, 4])
def test_no_initial_boundaries_and_padding_rows(G):
    result = decode_one([0] * G, [-1], torch.empty((0, 4)), torch.full((G,), 30.0), K=1)
    assert result["pred_b"] == [1] * G
    assert result["pred_edit_labels"] == []


def test_true_gap_lengths_trim_insert_and_edit_candidates():
    # Padding logits actively demand boundaries; those gaps must never be used.
    b0 = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    logits = torch.full((3, 1, 6), -40.0)
    logits[:, 0, 5] = 40.0  # SHIFT:+2, invalid for second document
    logits[:, 0, 3] = 30.0  # KEEP is its best legal candidate
    result = batch_decode(b0, torch.tensor([[0], [0], [-1]]), logits,
                          torch.full((3, 4), 40.0), K=2, num_gaps=[4, 1, 0])
    assert result.pred_b == [[1, 1, 1, 1], [1], []]
    assert result.pred_gaps == [[0, 1, 2, 3], [0], []]
    assert result.pred_edit_labels[1] == [{"g": 0, "y": "KEEP"}]
    assert [len(row) for row in result.pred_insert_labels] == [4, 1, 0]


def test_oracle_logits_replay_canonical_mixed_length_targets():
    cases = [([1, 0, 1, 0, 0], [0, 1, 0, 1, 1]), ([0, 0], [1, 1]), ([], [])]
    K, width = 2, 5
    seeds = torch.zeros((3, width), dtype=torch.long)
    positions = torch.full((3, 2), -1, dtype=torch.long)
    edits = torch.full((3, 2, 2 * K + 2), -30.0)
    inserts = torch.full((3, width), 30.0)  # padding would be false positives
    for row, (b0, target) in enumerate(cases):
        labels = derive_canonical_labels(b0, target, K)
        seeds[row, :len(b0)] = torch.tensor(b0)
        for col, item in enumerate(labels["edit"]):
            positions[row, col] = item["g"]
            edits[row, col, infer_bestofn.label_to_class_idx(item["y"], K)] = 30
        inserts[row, :len(b0)] = torch.tensor([30.0 if v else -30.0 for v in labels["insert"]])
    output = batch_decode(seeds, positions, edits, inserts, K=K, num_gaps=[len(x[0]) for x in cases])
    assert output.pred_b == [target for _, target in cases]


def test_padded_boundary_rows_preserve_logit_alignment():
    logits = torch.full((3, 2), -30.0)
    logits[0, 1], logits[2, 1] = 30, 30
    logits[1, 0] = 30
    result = decode_one([1, 0, 1], [0, -1, 2], logits, torch.full((3,), -30.0), K=0)
    assert result["pred_b"] == [1, 0, 1]


def test_head_masks_real_lengths_and_matches_unpadded_forward():
    torch.manual_seed(8)
    heads = RefinerHeads(4, K=2, dropout=0).eval()
    h = torch.randn(3, 6, 4)
    h[1, 3:] = 10000
    mask = torch.tensor([[1] * 6, [1, 1, 1, 0, 0, 0], [0] * 6], dtype=torch.bool)
    g0 = torch.tensor([[0, 4], [1, -1], [-1, -1]])
    output = heads(h, g0, mask)
    single = heads(h[1:2, :3], g0[1:2])
    torch.testing.assert_close(output.insert_logits[1, :2], single.insert_logits[0])
    torch.testing.assert_close(output.edit_choice_logits[1], single.edit_choice_logits[0])
    assert torch.equal(output.insert_logits.sigmoid()[1, 2:], torch.zeros(3))
    assert torch.equal(output.insert_logits.sigmoid()[2], torch.zeros(5))
    # g0=1 has only SHIFT:-1 and KEEP inside its real two-gap document.
    probabilities = output.edit_choice_logits.softmax(-1)[1, 0]
    assert probabilities[1] == probabilities[4] == probabilities[5] == 0


@pytest.mark.parametrize("T", [0, 1])
@pytest.mark.parametrize("N", [0, 1])
def test_heads_support_no_real_gaps(T, N):
    heads = RefinerHeads(4, K=2, dropout=0)
    output = heads(torch.zeros(2, T, 4), torch.full((2, N), -1), torch.ones(2, T, dtype=torch.bool))
    assert output.insert_logits.shape == (2, 0)
    assert output.edit_choice_logits.shape == (2, N, 6)
    assert torch.isfinite(output.edit_choice_logits).all()


def test_sampler_uses_same_legal_solver_as_greedy(monkeypatch):
    monkeypatch.setattr(infer_bestofn, "gumbel_noise", lambda rng: 0.0)
    logits = torch.tensor([[-30., -30., -30., -30., -30., 30.],
                           [30., -30., -30., -30., -30., -30.],
                           [-30., 30., -30., -30., -30., -30.]])
    greedy = decode_one([1, 1, 1], [0, 1, 2], logits, torch.full((3,), -40.0), K=2)
    positions, trace = infer_bestofn.sample_monotonic_edit_decode([1, 1, 1], [0, 1, 2], logits, 2, 1.0, 6, 1.0, 0.25)
    assert positions == greedy["pred_gaps"]
    assert [t["to"] for t in trace if t["to"] is not None] == positions


def test_sampler_trace_remains_legal_across_random_perturbations():
    logits = torch.randn(3, 6, generator=torch.Generator().manual_seed(1))
    for seed in range(30):
        positions, trace = infer_bestofn.sample_monotonic_edit_decode([1, 1, 1], [0, 1, 2], logits, 2, 1.0, seed, 1.0, 0.25)
        assert positions == sorted(set(positions))
        assert len(trace) == 3
        assert all(0 <= p < 3 for p in positions)


def test_sample_insert_has_same_adjacency_contract():
    labels, gaps = infer_bestofn.sample_insert_decode(torch.full((3,), 40.0), [1], 1.0, 0.5, 0, 0)
    assert labels == [1, 0, 1]
    assert [g["gap"] for g in gaps] == [0, 2]


def test_final_supervision_does_not_reuse_last_pass_trace():
    original, final = [1, 0, 0, 0, 0], [0, 0, 0, 0, 1]
    raw_passes = [{"edit_trace": [{"g0": 2, "action": "SHIFT", "shift": 2, "to": 4, "prob": 0.9}]}]
    result = infer_bestofn.build_canonical_prediction(original, final, K=2, raw_passes=raw_passes)
    assert replay_labels(original, result["canonical_labels"], K=2) == final
    assert result["edit_trace"] == [{"g0": 0, "action": "DEL", "shift": None, "to": None, "prob": None}]
    assert result["insert_gaps"] == [{"gap": 4, "prob": None}]
    assert result["raw_passes"] == raw_passes


@pytest.mark.parametrize("mode", ["greedy", "sample"])
def test_multiple_pass_export_uses_real_token_budget_and_final_labels(monkeypatch, mode):
    encoded = []

    class Tokenizer:
        def encode(self, text, *, add_special_tokens, truncation):
            assert add_special_tokens and not truncation
            encoded.append(text)
            return [0] + text.split() + [1]

    def fake_forward(model, atoms, b_dense, use_autocast):
        positions = [g for g, value in enumerate(b_dense) if value]
        logits = torch.full((len(positions), 6), -40.0)
        logits[:, 0] = 40  # raw model deletes all boundaries on every pass
        return {"g0_positions": positions, "edit_choice_logits": logits,
                "insert_logits": torch.full((len(b_dense),), -40.0)}

    monkeypatch.setattr(infer_bestofn, "forward_one_doc", fake_forward)
    monkeypatch.setattr(infer_bestofn, "DEFAULT_PROJECTOR_CFG", infer_bestofn.ProjectorConfig(
        max_chunk_atoms=2, min_chunk_atoms=1, max_chunk_chars=100,
        min_chunk_chars=1, max_chunk_tokens=4, min_chunk_tokens=1,
    ))
    model = SimpleNamespace(atom_encoder=SimpleNamespace(tokenizer=Tokenizer()))
    args = SimpleNamespace(k_shift=2, refine_passes=2, insert_min_sep=0,
                           lambda_del=1.0, lambda_ins=1.0, lambda_shift=0.25,
                           disable_autocast=True, ckpt="mock")
    document = {"doc_id": "synthetic", "atoms": ["a", "b", "c", "d"],
                "b0": [0, 1, 0], "chunk0_units": []}
    if mode == "greedy":
        result = infer_bestofn.run_greedy_candidate(model, document, args)
    else:
        result = infer_bestofn.run_sample_candidate(model, document, args, 1.0, 0.5, 3)
    prediction = result["prediction"]
    final = infer_bestofn.sparse_to_dense(prediction["b_pred_sparse"], 4)
    assert replay_labels(document["b0"], prediction["canonical_labels"], K=2) == final
    assert len(prediction["raw_passes"]) == 2
    assert all(p["b_raw"] == [0, 0, 0] for p in prediction["raw_passes"])
    assert prediction["raw_passes"][-1]["b_projected"] == final
    assert result["chunk_stats"]["max_chunk_tokens"] <= 4
    assert result["chunk_stats"]["num_hard_violations"] == 0
    assert result["chunk_stats"]["token_count_mode"] == "custom_whole_span"
    assert any("\n" in text for text in encoded)
    assert result["teacher_stats"]["scope"] == "last_raw_pass"
