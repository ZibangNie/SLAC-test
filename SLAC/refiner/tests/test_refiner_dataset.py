import json
import pytest
import torch
from torch.utils.data import DataLoader
from slac_refiner.datasets.refiner_dataset import RefinerDenoiseDataset
from slac_refiner.datasets.collate import refiner_collate_fn


def test_dataset_parses_legal_edit_classes_and_nested_weights(fixture_jsonl):
    dataset = RefinerDenoiseDataset(str(fixture_jsonl), sample_weight_field="meta.confidence", expected_k=2)
    assert len(dataset) == 3
    first = dataset[0]
    assert first["atoms_text"] == ["甲", "乙", "丙", "丁", "戊"]
    assert first["g0_positions"].tolist() == [0, 2]
    assert first["edit_choice"].tolist() == [4, 3]  # K=2: SHIFT:+1, KEEP
    assert first["insert_labels"].tolist() == [0., 0., 0., 1.]
    assert first["sample_weight"].item() == pytest.approx(0.4)
    assert dataset[1]["sample_weight"].item() == 1
    assert dataset[2]["num_gaps"] == 0


def test_dataloader_preserves_true_lengths_and_padding_masks(fixture_jsonl):
    dataset = RefinerDenoiseDataset(str(fixture_jsonl), expected_k=2)
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=refiner_collate_fn)))
    assert batch["num_atoms"].tolist() == [5, 2, 1]
    assert batch["num_gaps"].tolist() == [4, 1, 0]
    assert batch["b0"].shape == (3, 4)
    assert batch["insert_mask"].tolist() == [[True] * 4, [True, False, False, False], [False] * 4]
    assert batch["g0_positions"].tolist() == [[0, 2], [-1, -1], [-1, -1]]
    assert batch["edit_choice"].tolist() == [[4, 3], [-100, -100], [-100, -100]]
    assert torch.equal(batch["g0_mask"], batch["edit_choice_mask"])
    assert torch.equal(batch["b0_mask"], batch["insert_mask"])


def test_dataset_rejects_labels_that_do_not_replay(tmp_path, legal_records):
    record = legal_records[0]
    record["b_gold"] = [1, 0, 0, 0]
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="do not reproduce"):
        RefinerDenoiseDataset(str(path), expected_k=2)[0]


def test_dataset_requires_explicit_radius_matching(tmp_path, legal_records):
    record = legal_records[0]
    record["meta"]["K"] = 1
    path = tmp_path / "radius_one.jsonl"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        RefinerDenoiseDataset(str(path))[0]
    first = RefinerDenoiseDataset(str(path), expected_k=1)[0]
    assert first["edit_choice"].tolist() == [3, 2]
