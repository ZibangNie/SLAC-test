"""Small offline regressions for weighting, padding, and frozen features."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slac_refiner.models.losses import RefinerLoss
from slac_refiner.models.atom_encoder import AtomEncoder
from slac_refiner.models.refiner import BoundaryRefinerModel
from slac_refiner.models.doc_encoder import DocEncoder
from slac_refiner.models.heads import RefinerHeads


def test_confidence_changes_batch_one_loss_and_gradient():
    criterion = RefinerLoss(insert_pos_weight=1)
    logits = torch.tensor([[0.7, -0.2]], requires_grad=True)
    args = (logits, torch.tensor([[1., 0.]]), torch.ones(1, 2, dtype=torch.bool))
    full = criterion.compute_insert_loss(*args, sample_weight=torch.tensor([1.]))
    small = criterion.compute_insert_loss(*args, sample_weight=torch.tensor([0.2]))
    assert torch.allclose(small, full * .2)
    g1 = torch.autograd.grad(full, logits, retain_graph=True)[0]
    g2 = torch.autograd.grad(small, logits)[0]
    assert torch.allclose(g2, g1 * .2)


def test_legacy_normalized_weighting_is_explicit():
    criterion = RefinerLoss(sample_weight_reduction="normalized")
    assert criterion._weighted_mean(torch.tensor([3.]), torch.tensor([.2])).item() == pytest.approx(3.)


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf")])
def test_reject_invalid_weights(weight):
    with pytest.raises(ValueError):
        RefinerLoss()._weighted_mean(torch.tensor([1.]), torch.tensor([weight]))


def test_masked_logits_do_not_make_loss_nan_or_change_gradient():
    criterion = RefinerLoss(insert_pos_weight=1)
    logits = torch.tensor([[.2, float("-inf")]], requires_grad=True)
    loss = criterion.compute_insert_loss(logits, torch.tensor([[1., 0.]]), torch.tensor([[True, False]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1] == 0
    edits = torch.full((1, 1, 14), float("-inf"), requires_grad=True)
    eloss = criterion.compute_edit_loss(edits, torch.tensor([[-100]]))
    eloss.backward()
    assert eloss == 0 and torch.isfinite(edits.grad).all()


def test_cost_regularizer_ignores_padding_and_empty_gaps():
    criterion = RefinerLoss(beta_cost=1)
    logits = torch.tensor([[0., 100.]])
    edit = torch.full((1, 1, 14), float("-inf"))
    result = criterion.compute_cost_regularizer(logits, edit, torch.tensor([[False]]), insert_mask=torch.tensor([[True, False]]))
    assert result.item() == pytest.approx(.5)
    empty = criterion.compute_cost_regularizer(torch.empty(1, 0), torch.empty(1, 0, 14), torch.empty(1, 0, dtype=torch.bool))
    assert empty.item() == 0


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        lengths = [len(t) + 2 for t in texts]
        if kwargs.get("return_length"):
            return {"length": lengths}
        width = kwargs["max_length"]
        return {"input_ids": torch.zeros(len(texts), width, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), width, dtype=torch.long)}


def bare_encoder(policy="error"):
    encoder = AtomEncoder.__new__(AtomEncoder)
    nn.Module.__init__(encoder)
    encoder.backbone = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.5))
    encoder.freeze = True
    encoder.tokenizer = FakeTokenizer()
    encoder.max_length = 6
    encoder.overflow_policy = policy
    encoder.truncated_atoms = 0
    return encoder


def test_parent_train_keeps_frozen_backbone_in_eval():
    encoder = bare_encoder()
    parent = nn.Sequential(encoder)
    parent.train()
    assert parent.training and encoder.training
    assert not encoder.backbone.training
    assert not encoder.backbone[1].training


def test_atom_overflow_includes_special_tokens_and_is_explicit():
    encoder = bare_encoder()
    encoder.tokenize(["1234"])
    with pytest.raises(ValueError, match="exceed encoder"):
        encoder.tokenize(["12345"])
    legacy = bare_encoder("truncate")
    legacy.tokenize(["12345"])
    assert legacy.truncated_atoms == 1


def cached_model():
    model = BoundaryRefinerModel.__new__(BoundaryRefinerModel)
    nn.Module.__init__(model)
    model.atom_encoder = bare_encoder()
    model.max_doc_atoms = 8
    model.doc_encoder = DocEncoder(4, hidden_size=8, num_layers=1, num_heads=2, dropout=0, window_size=1)
    model.heads = RefinerHeads(8, K=1, dropout=0)
    return model


def test_half_cached_features_support_float_training_and_are_detached():
    model = cached_model()
    features = torch.randn(1, 3, 4, dtype=torch.float16, requires_grad=True)
    output = model({"atom_embeddings": features, "atom_mask": torch.ones(1, 3, dtype=torch.bool),
                    "g0_positions": torch.tensor([[0]]), "num_atoms": torch.tensor([3]), "num_gaps": torch.tensor([2])})
    output.insert_logits.sum().backward()
    assert output.atom_embeddings.dtype == torch.float32
    assert features.grad is None
    assert torch.isfinite(model.doc_encoder.in_proj.weight.grad).all()


@pytest.mark.parametrize("mask,extra,match", [
    ([True, False, True], {}, "right padding"),
    ([True, True, True], {"num_atoms": torch.tensor([2])}, "num_atoms"),
    ([True, True, True], {"num_gaps": torch.tensor([1])}, "num_gaps"),
    ([True, True, True], {"atoms_text": [["a", "b"]]}, "source atom counts"),
])
def test_cache_rejects_misaligned_atom_metadata(mask, extra, match):
    with pytest.raises(ValueError, match=match):
        cached_model()({"atom_embeddings": torch.randn(1, 3, 4), "atom_mask": torch.tensor([mask]),
                        "g0_positions": torch.tensor([[0]]), **extra})
