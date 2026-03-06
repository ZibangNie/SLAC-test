import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slac_refiner.models.atom_encoder import AtomEncoder


def main():
    atoms = [
        "第一段。",
        "第二句。",
        "Third unit.",
        "Another sentence.",
        "最后一段。",
    ]

    encoder = AtomEncoder(
        model_name="BAAI/bge-m3",
        max_length=64,
        freeze=True,
    )

    out = encoder.encode(atoms, normalize=False)

    print("device =", encoder.device)
    print("hidden_size =", out.hidden_size)
    print("input_ids shape =", tuple(out.input_ids.shape))
    print("attention_mask shape =", tuple(out.attention_mask.shape))
    print("atom_embeddings shape =", tuple(out.atom_embeddings.shape))
    print("dtype =", out.atom_embeddings.dtype)

    assert out.atom_embeddings.ndim == 2
    assert out.atom_embeddings.shape[0] == len(atoms)
    assert out.atom_embeddings.shape[1] == out.hidden_size

    # freeze check
    trainable = [p.requires_grad for p in encoder.backbone.parameters()]
    assert all(x is False for x in trainable), "Backbone should be frozen"

    # numeric sanity
    assert torch.isfinite(out.atom_embeddings).all()

    print("AtomEncoder test passed.")


if __name__ == "__main__":
    main()