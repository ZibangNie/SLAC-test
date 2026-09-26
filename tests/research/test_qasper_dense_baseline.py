"""CPU-only checks for BGE dense pooling, audited token IDs and ranking."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
from run_qasper_dense_baseline import audit_token_lengths, dense_cls, dense_ranking, encode_token_ids


class TinyTokenizer:
    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is True and truncation is False
        return [10, *range(len(text.split())), 11]

    def pad(self, features, *, padding, return_tensors):
        assert padding is True and return_tensors == "pt"
        longest = max(map(len, features["input_ids"]))
        return {
            "input_ids": torch.tensor([ids + [0] * (longest - len(ids)) for ids in features["input_ids"]]),
            "attention_mask": torch.tensor([[1] * len(ids) + [0] * (longest - len(ids)) for ids in features["input_ids"]]),
        }


class TinyModel:
    def __init__(self):
        self.seen, self.training = [], True

    def eval(self):
        self.training = False
        return self

    def __call__(self, input_ids, attention_mask):
        assert not self.training and not torch.is_grad_enabled()
        self.seen.extend([ids[mask.bool()].tolist() for ids, mask in zip(input_ids, attention_mask)])
        value = input_ids.float()
        return SimpleNamespace(last_hidden_state=torch.stack((value, torch.ones_like(value)), -1))


def test_pooling_is_cls_and_l2_not_token_mean():
    states = torch.tensor([[[3., 4.], [100., -100.], [500., 500.]]], dtype=torch.float16)
    result = dense_cls(states)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([[.6, .8]]))
    assert torch.allclose(result.norm(dim=-1), torch.ones(1))


@pytest.mark.parametrize("states", [torch.zeros(1, 2, 3), torch.full((1, 2, 3), float("nan")), torch.full((1, 2, 3), float("inf"))])
def test_pooling_rejects_invalid_vectors(states):
    with pytest.raises(ValueError):
        dense_cls(states)


def test_length_audit_counts_specials_and_never_drops_overlong_text():
    ids, report = audit_token_lengths(TinyTokenizer(), ["a b", "a b c"], max_length=4)
    assert list(map(len, ids)) == [4, 5]
    assert report["count"] == 2 and report["total_tokens"] == 9
    assert report["above_max_length"] == 1 and report["truncation"] is False


def test_sorted_microbatches_preserve_exact_input_ids_and_output_order():
    ids = [[3, 10, 11, 12, 13], [2, 11], [1, 4, 11]]
    model = TinyModel()
    actual = encode_token_ids(model, TinyTokenizer(), ids, batch_size=2,
                              device="cpu", deadline=float("inf"), max_length=5)
    expected = torch.tensor([[3., 1.], [2., 1.], [1., 1.]])
    expected = expected / expected.norm(dim=-1, keepdim=True)
    assert torch.allclose(actual, expected)
    assert model.seen == [ids[1], ids[2], ids[0]]


def test_overflow_stops_before_any_model_call():
    model = TinyModel()
    with pytest.raises(ValueError, match="truncation is forbidden"):
        encode_token_ids(model, TinyTokenizer(), [[3, 4, 5]], batch_size=1,
                         device="cpu", deadline=float("inf"), max_length=2)
    assert model.seen == []


def test_deadline_stops_before_any_model_call():
    model = TinyModel()
    with pytest.raises(TimeoutError):
        encode_token_ids(model, TinyTokenizer(), [[3, 4]], batch_size=1,
                         device="cpu", deadline=0, max_length=2)
    assert model.seen == []


def test_dense_ranking_uses_similarity_then_original_source_order():
    units = [SimpleNamespace(order=7), SimpleNamespace(order=2), SimpleNamespace(order=3)]
    actual = dense_ranking(torch.tensor([1., 0.]), torch.tensor([[.5, .3], [.5, .7], [-1., 0.]]), units)
    assert actual == [1, 0, 2]


def test_shape_mismatch_or_nan_ranking_is_rejected():
    with pytest.raises(ValueError):
        dense_ranking(torch.ones(3), torch.ones(2, 3), [SimpleNamespace(order=0)])
    with pytest.raises(ValueError):
        dense_ranking(torch.tensor([float("nan")]), torch.ones(1, 1), [SimpleNamespace(order=0)])
