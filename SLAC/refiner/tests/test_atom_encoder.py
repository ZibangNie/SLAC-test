import torch
from slac_refiner.models.atom_encoder import AtomEncoder


def test_atom_encoder_microbatch_pooling_and_frozen_state(offline_pretrained):
    encoder = AtomEncoder(model_name="offline-test-fixture", max_length=12,
                          freeze=True, device="cpu", encode_batch_size=2)
    atoms = ["甲", "英文", "abc", "d", "最后"]
    output = encoder.encode(atoms)
    assert output.input_ids.shape == (5, 12)
    assert output.atom_embeddings.shape == (5, 4)
    expected = []
    for atom in atoms:
        ids = torch.tensor(encoder.tokenizer.encode(atom))
        expected.append(encoder.backbone.embedding(ids).mean(0))
    torch.testing.assert_close(output.atom_embeddings, torch.stack(expected))
    assert not any(p.requires_grad for p in encoder.backbone.parameters())
    encoder.train()
    assert not encoder.backbone.training
    normalized = encoder.encode(atoms, normalize=True)
    torch.testing.assert_close(normalized.atom_embeddings.norm(dim=1), torch.ones(5))
