"""Exercise the training entrypoint's real epoch/evaluation/checkpoint path offline."""
import math
from types import SimpleNamespace

import pytest
import torch

from scripts import train_loop
from slac_refiner.decoding.projector import ProjectorConfig


def test_train_evaluate_and_checkpoint_roundtrip(tiny_refiner, refiner_batch, tmp_path, monkeypatch):
    model = tiny_refiner
    assert model.heads.K == 2  # Evaluation must derive this instead of assuming K=6.
    args = SimpleNamespace(insert_pos_weight=2, alpha_insert=3, alpha_edit=1,
                           beta_cost=0.05, sample_weight_reduction="absolute", k_shift=2)
    criterion = train_loop.build_criterion(args)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    before = model.heads.insert_head[-1].weight.detach().clone()
    stats = train_loop.train_one_epoch(model, [refiner_batch], criterion, optimizer, epoch=3)
    assert math.isfinite(stats["loss"]) and stats["loss"] > 0
    assert not torch.equal(before, model.heads.insert_head[-1].weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert not model.atom_encoder.backbone.training

    # Choose a known raw output after exercising a real optimizer step. The real
    # model still runs; the resulting classifier always DELs and never INSERTs.
    with torch.no_grad():
        for parameter in model.heads.parameters():
            parameter.zero_()
        model.heads.insert_head[-1].bias.fill_(-30)
        model.heads.delete_head[-1].bias.fill_(30)

    # Poison ONLY padded gold positions. Correct per-document cropping must make
    # these nonexistent gaps irrelevant to all boundary/INSERT metrics.
    batch = dict(refiner_batch)
    batch["b_gold"] = refiner_batch["b_gold"].clone()
    batch["insert_labels"] = refiner_batch["insert_labels"].clone()
    batch["b_gold"][~batch["b_gold_mask"]] = 1
    batch["insert_labels"][~batch["insert_mask"]] = 1
    projections = []
    real_projector = train_loop.rebuild_chunks_from_boundary_vector

    def checked_projector(*args, **kwargs):
        assert kwargs["strict"] is True
        assert callable(kwargs["token_counter"])
        result = real_projector(*args, **kwargs)
        assert result["hard_max_satisfied"]
        assert result["token_count_mode"] == "custom_whole_span"
        projections.append(result)
        return result

    monkeypatch.setattr(train_loop, "rebuild_chunks_from_boundary_vector", checked_projector)
    cfg = ProjectorConfig(max_chunk_atoms=1, min_chunk_atoms=1, max_chunk_chars=100,
                          min_chunk_chars=0, max_chunk_tokens=8, min_chunk_tokens=0)
    dev = train_loop.evaluate(model, [batch], projector_cfg=cfg)
    assert len(projections) == 3
    assert [p["projected_b"] for p in projections] == [[1, 1, 1, 1], [1], []]
    assert dev["projection_changed_documents"] == 2
    assert dev["boundary_raw"]["f1"] == 0
    assert dev["boundary_raw"]["fn"] == pytest.approx(4 / 3)
    assert dev["boundary"]["f1"] == pytest.approx((6 / 7 + 1) / 3)
    assert dev["insert_acc"]["total"] == pytest.approx(5 / 3)
    assert dev["insert_acc"]["acc"] == pytest.approx(0.25)
    assert dev["length"]["num_chunks"] == pytest.approx(8 / 3)
    assert dev["length"]["max_atoms_per_chunk"] == 1
    assert all(math.isfinite(value) for group in dev.values() if isinstance(group, dict) for value in group.values())

    # Run the same save/load entrypoints as main, including optimizer metadata.
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    train_loop.maybe_save(model, optimizer, str(tmp_path), epoch=3, dev_metrics=dev, args=args)
    checkpoint = tmp_path / "epoch_3.pt"
    assert checkpoint.is_file()
    with torch.no_grad():
        model.heads.delete_head[-1].bias.zero_()
    real_load = torch.load
    load_options = []

    def checked_load(*args, **kwargs):
        load_options.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    assert train_loop.load_init_ckpt(model, str(checkpoint)) == 3
    assert load_options[0]["weights_only"] is True
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert train_loop.load_init_ckpt(model, None) == 0
