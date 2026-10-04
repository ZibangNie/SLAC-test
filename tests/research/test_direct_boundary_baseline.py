"""CPU synthetic contracts for the cached b0-conditioned direct baseline.

These tests exercise small tensors and gradients only. They do not load an atom
encoder, checkpoint, tokenizer, dataset, or run an optimization step.
"""
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "SLAC" / "refiner"))
from slac_refiner.models.direct_boundary import (  # noqa: E402
    CachedSeedBoundaryClassifier,
    DirectBoundaryHead,
)


def make_head(hidden_size=4, K=1, width=7):
    head = DirectBoundaryHead(hidden_size, K, width, dropout=0.0).double().eval()
    # Explicit nonzero weights make conditioning/gradient checks deterministic.
    with torch.no_grad():
        head.classifier[0].weight.fill_(0.02)
        head.classifier[0].bias.fill_(0.10)
        head.classifier[3].weight.fill_(0.03)
        head.classifier[3].bias.fill_(0.02)
    return head


def mixed_inputs():
    h = torch.arange(1, 41, dtype=torch.float64).reshape(2, 5, 4) / 20
    mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    b0 = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 0]], dtype=torch.int64)
    return h, b0, mask


def masked_loss(output):
    logits = output.logits[output.gap_mask]
    labels = (torch.arange(logits.numel(), device=logits.device) % 2).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, labels)


def test_feature_layout_uses_target_then_ordered_seed_slots_and_presence():
    head = make_head(hidden_size=2)
    h = torch.tensor([[[1., 2.], [3., 5.], [7., 11.], [13., 17.]]], dtype=torch.float64)
    b0 = torch.tensor([[1, 0, 1]])
    mask = torch.ones((1, 4), dtype=torch.bool)
    features, gap_mask = head.build_features(h, b0, mask)
    # Independently written 4H gap representations; no production helper oracle.
    a = [1, 2, 3, 5, 2, 3, 3, 10]
    b = [3, 5, 7, 11, 4, 6, 21, 55]
    c = [7, 11, 13, 17, 6, 6, 91, 187]
    z = [0] * 8
    expected = torch.tensor([[a + z + a + z + [0, 1, 0],
                              b + a + z + c + [1, 0, 1],
                              c + z + c + z + [0, 1, 0]]], dtype=h.dtype)
    assert head.input_dim == 35
    assert gap_mask.dtype == torch.bool and gap_mask.tolist() == [[True] * 3]
    torch.testing.assert_close(features, expected, rtol=0, atol=0)


def test_b0_changes_real_output_with_identical_h_and_mask():
    head = make_head()
    h, _, mask = mixed_inputs()
    before = torch.zeros((2, 4), dtype=torch.int64)
    after = before.clone()
    after[0, 1] = 1
    first = head(h, before, mask)
    second = head(h, after, mask)
    assert second.logits[0, 1] > first.logits[0, 1]
    torch.testing.assert_close(first.logits[1], second.logits[1], rtol=0, atol=0)
    torch.testing.assert_close(head(h, after.bool(), mask).logits, second.logits)


@pytest.mark.parametrize("padding_value", [1e100, float("nan"), float("inf"), -float("inf")])
def test_padding_is_scrubbed_before_products_and_gradients(padding_value):
    head = make_head()
    h, b0, mask = mixed_inputs()

    def observe(value):
        head.zero_grad(set_to_none=True)
        x = h.clone()
        x[~mask] = value
        x.requires_grad_()
        output = head(x, b0, mask)
        features, gap_mask = head.build_features(x, b0, mask)
        assert torch.isfinite(features).all()
        assert torch.count_nonzero(features[~gap_mask]) == 0
        assert torch.isfinite(output.logits[output.gap_mask]).all()
        masked_loss(output).backward()
        assert torch.isfinite(x.grad).all()
        assert torch.count_nonzero(x.grad[~mask]) == 0
        grads = {name: parameter.grad.detach().clone() for name, parameter in head.named_parameters()}
        assert all(torch.isfinite(value).all() for value in grads.values())
        return output.logits[output.gap_mask].detach(), x.grad[mask].clone(), grads

    expected, actual = observe(0.), observe(padding_value)
    torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
    torch.testing.assert_close(expected[1], actual[1], rtol=0, atol=0)
    for name in expected[2]:
        torch.testing.assert_close(expected[2][name], actual[2][name], rtol=0, atol=0)


def test_mixed_lengths_single_document_and_permuted_batch_agree():
    head = make_head()
    h, b0, mask = mixed_inputs()
    h = torch.cat([h, torch.ones((1, 5, 4), dtype=h.dtype)])
    b0 = torch.cat([b0, torch.zeros((1, 4), dtype=b0.dtype)])
    mask = torch.cat([mask, torch.tensor([[True, False, False, False, False]])])
    batch = head(h, b0, mask)
    assert batch.gap_mask.sum(1).tolist() == [4, 2, 0]
    for row, length in enumerate([5, 3, 1]):
        single = head(h[row:row + 1, :length], b0[row:row + 1, :length - 1],
                      mask[row:row + 1, :length])
        torch.testing.assert_close(batch.logits[row, :length - 1], single.logits[0])
    order = torch.tensor([2, 0, 1])
    permuted = head(h[order], b0[order], mask[order])
    torch.testing.assert_close(permuted.logits, batch.logits[order])
    assert torch.equal(permuted.gap_mask, batch.gap_mask[order])


def test_one_atom_returns_empty_gap_and_features():
    head = make_head()
    h = torch.ones((2, 1, 4), dtype=torch.float64)
    b0 = torch.empty((2, 0), dtype=torch.int64)
    mask = torch.ones((2, 1), dtype=torch.bool)
    features, gap_mask = head.build_features(h, b0, mask)
    output = head(h, b0, mask)
    assert features.shape == (2, 0, head.input_dim)
    assert output.logits.shape == gap_mask.shape == output.gap_mask.shape == (2, 0)


def test_all_head_parameters_receive_finite_nonzero_gradient_signal():
    head = make_head()
    h, b0, mask = mixed_inputs()
    h.requires_grad_()
    masked_loss(head(h, b0, mask)).backward()
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    assert torch.isfinite(h.grad).all()
    assert torch.count_nonzero(h.grad[mask]) > 0
    assert torch.count_nonzero(h.grad[~mask]) == 0


@pytest.mark.parametrize("case", [
    "h_rank", "h_width", "h_integer", "b0_shape", "b0_float", "b0_nonbinary",
    "mask_shape", "mask_integer", "mask_hole", "b0_padding", "real_nan",
    "real_inf", "empty_batch", "empty_atoms", "all_padding_row",
])
def test_invalid_inputs_fail_closed(case):
    head = make_head()
    h, b0, mask = mixed_inputs()
    if case == "h_rank": h = h[0]
    elif case == "h_width": h = h[:, :, :3]
    elif case == "h_integer": h = h.long()
    elif case == "b0_shape": b0 = b0[:, :-1]
    elif case == "b0_float": b0 = b0.double()
    elif case == "b0_nonbinary": b0[0, 0] = 2
    elif case == "mask_shape": mask = mask[:, :-1]
    elif case == "mask_integer": mask = mask.long()
    elif case == "mask_hole": mask[0, 1] = False
    elif case == "b0_padding": b0[1, -1] = 1
    elif case == "real_nan": h[0, 0, 0] = float("nan")
    elif case == "real_inf": h[0, 0, 0] = float("inf")
    elif case == "empty_batch": h, b0, mask = h[:0], b0[:0], mask[:0]
    elif case == "empty_atoms": h, b0, mask = h[:, :0], b0[:, :0], mask[:, :0]
    elif case == "all_padding_row": mask[1] = False; b0[1] = 0
    with pytest.raises(ValueError):
        head(h, b0, mask)
    with pytest.raises(ValueError):
        head.build_features(h, b0, mask)


def test_local_information_range_with_fixed_contextual_h():
    head = make_head(hidden_size=2, K=1)
    h = torch.arange(1, 17, dtype=torch.float64).reshape(1, 8, 2) / 10
    b0 = torch.ones((1, 7), dtype=torch.int64)
    mask = torch.ones((1, 8), dtype=torch.bool)
    original = head(h, b0, mask).logits[0, 3]
    # Gap3 with K1 may read gap2..4, hence only atoms2..5 from fixed h.
    outside = h.clone()
    outside[0, 0] += 30
    far_b0 = b0.clone()
    far_b0[0, 0] = 0
    torch.testing.assert_close(head(outside, far_b0, mask).logits[0, 3], original, rtol=0, atol=0)
    inside = h.clone()
    inside[0, 2] += 1
    assert not torch.equal(head(inside, b0, mask).logits[0, 3], original)
    near_b0 = b0.clone()
    near_b0[0, 2] = 0
    assert not torch.equal(head(h, near_b0, mask).logits[0, 3], original)


def test_declared_128_6_37_head_parameter_count():
    head = DirectBoundaryHead(128, 6, 37, dropout=0.0)
    assert head.input_dim == 4 * 128 * (1 + 13) + 13 == 7181
    assert sum(p.numel() for p in head.parameters()) == 265772


def make_cached_classifier():
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = CachedSeedBoundaryClassifier(
            atom_dim=3, hidden_size=4, doc_layers=1, doc_heads=2, window_size=1,
            K=1, mlp_hidden_size=7, dropout=0.0, max_doc_atoms=6,
        ).double().eval()
    return model


def test_cached_embeddings_detach_and_wrapper_matches_explicit_components():
    model = make_cached_classifier()
    x = (torch.arange(1, 31, dtype=torch.float64).reshape(2, 5, 3) / 10).requires_grad_()
    _, b0, mask = mixed_inputs()
    output = model(x, b0, mask)
    clean = x.detach().masked_fill(~mask.unsqueeze(-1), 0)
    direct = model.head(model.doc(clean, mask).h, b0, mask)
    torch.testing.assert_close(output.logits, direct.logits)
    assert torch.equal(output.gap_mask, direct.gap_mask)
    masked_loss(output).backward()
    assert x.grad is None
    for name, parameter in model.head.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.doc.parameters())


def test_cached_wrapper_padding_and_single_row_agree():
    model = make_cached_classifier()
    x = torch.arange(1, 31, dtype=torch.float64).reshape(2, 5, 3) / 10
    _, b0, mask = mixed_inputs()
    expected = model(x, b0, mask)
    poisoned = x.clone()
    poisoned[1, 3] = float("nan")
    poisoned[1, 4] = float("inf")
    actual = model(poisoned, b0, mask)
    torch.testing.assert_close(actual.logits[actual.gap_mask], expected.logits[expected.gap_mask])
    for row, length in enumerate([5, 3]):
        single = model(x[row:row + 1, :length], b0[row:row + 1, :length - 1],
                       mask[row:row + 1, :length])
        torch.testing.assert_close(single.logits[0], expected.logits[row, :length - 1])


def test_cached_wrapper_rejects_real_nonfinite_and_document_cap():
    model = make_cached_classifier()
    x = torch.ones((1, 3, 3), dtype=torch.float64)
    x[0, 1, 0] = float("nan")
    with pytest.raises(ValueError):
        model(x, torch.zeros((1, 2), dtype=torch.int64), torch.ones((1, 3), dtype=torch.bool))
    with pytest.raises(ValueError):
        model(torch.ones((1, 7, 3), dtype=torch.float64),
              torch.zeros((1, 6), dtype=torch.int64), torch.ones((1, 7), dtype=torch.bool))
