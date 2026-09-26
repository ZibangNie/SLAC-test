import torch


def test_refiner_forward_respects_mixed_document_lengths(tiny_refiner, refiner_batch):
    model = tiny_refiner.eval()
    output = model(refiner_batch)
    assert output.atom_embeddings.shape == (3, 5, 4)
    assert output.doc_hidden.shape == (3, 5, 8)
    assert output.insert_logits.shape == (3, 4)
    assert output.edit_choice_logits.shape == (3, 2, 6)
    assert output.atom_mask.sum(1).tolist() == [5, 2, 1]
    assert torch.isfinite(output.doc_hidden).all()
    assert output.insert_logits[1, 1:].sigmoid().sum() == 0
    assert output.insert_logits[2].sigmoid().sum() == 0
    cached = model({"atom_embeddings": output.atom_embeddings.detach(),
                    "atom_mask": output.atom_mask, "g0_positions": refiner_batch["g0_positions"]})
    torch.testing.assert_close(cached.insert_logits, output.insert_logits)
    torch.testing.assert_close(cached.edit_choice_logits, output.edit_choice_logits)
