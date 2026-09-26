import torch
from slac_refiner.models.losses import RefinerLoss


def test_refiner_loss_backpropagates_with_padding_and_frozen_backbone(tiny_refiner, refiner_batch):
    model = tiny_refiner.train()
    outputs = model(refiner_batch)
    result = RefinerLoss(insert_pos_weight=2, beta_cost=0.05)(outputs, refiner_batch)
    assert all(torch.isfinite(x) for x in (result.loss, result.loss_insert, result.loss_edit, result.loss_cost_reg))
    result.loss.backward()
    assert model.heads.insert_head[0].weight.grad.abs().sum() > 0
    assert model.doc_encoder.in_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.atom_encoder.backbone.parameters())
    assert not model.atom_encoder.backbone.training
