import pytest
import torch
from slac_refiner.decoding.dp_edit_decode import batch_decode
from slac_refiner.eval.metrics import boundary_prf, edit_action_accuracy, insert_accuracy


def test_oracle_decode_reports_expected_metrics_from_portable_dataset(refiner_batch, legal_records):
    edits = torch.full((3, 2, 6), -30.0)
    for row in range(3):
        for col, label in enumerate(refiner_batch["edit_choice"][row]):
            if label >= 0:
                edits[row, col, label] = 30
    insert = refiner_batch["insert_labels"] * 60 - 30
    result = batch_decode(refiner_batch["b0"], refiner_batch["g0_positions"], edits, insert,
                          K=2, num_gaps=refiner_batch["num_gaps"])
    assert result.pred_b == [row["b_gold"] for row in legal_records]
    assert boundary_prf(result.pred_b[0], legal_records[0]["b_gold"])["f1"] == 1
    assert edit_action_accuracy(result.pred_edit_labels[0], legal_records[0]["labels"]["edit"])["acc"] == 1
    assert insert_accuracy(result.pred_insert_labels[0], legal_records[0]["labels"]["insert"])["acc"] == 1
    imperfect = boundary_prf([1, 1, 0, 0], [0, 1, 1, 1])
    assert imperfect["tp"] == 1 and imperfect["fp"] == 1 and imperfect["fn"] == 2
    assert imperfect["f1"] == pytest.approx(0.4)
