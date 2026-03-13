import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slac_refiner.datasets.refiner_dataset import RefinerDenoiseDataset
from slac_refiner.datasets.collate import refiner_collate_fn
from slac_refiner.models.refiner import BoundaryRefinerModel
from slac_refiner.decoding.dp_edit_decode import batch_decode
from slac_refiner.decoding.projector import rebuild_chunks_from_boundary_vector
from slac_refiner.eval.metrics import (
    boundary_prf,
    edit_action_accuracy,
    insert_accuracy,
    chunk_length_stats_from_spans,
)
from slac_refiner.decoding.projector import rebuild_chunks_from_boundary_vector, ProjectorConfig


LOCAL_BGE_M3_DIR = r"/root/autodl-tmp/models/bge-m3/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"


def main():
    data_path = r"D:\code\Github\SLAC-test\SLAC\refiner\data\interim\atoms_b0\refiner_train_demo.jsonl"

    ds = RefinerDenoiseDataset(data_path)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=refiner_collate_fn)
    batch = next(iter(loader))

    model = BoundaryRefinerModel(
        atom_model_name=LOCAL_BGE_M3_DIR,
        atom_max_length=64,
        atom_freeze=True,
        doc_hidden_size=768,
        doc_layers=2,
        doc_heads=12,
        doc_dropout=0.1,
        window_size=16,
    )

    model.eval()
    out = model(batch)

    dec = batch_decode(
        b0=batch["b0"],
        g0_positions=batch["g0_positions"],
        edit_choice_logits=out.edit_choice_logits,
        insert_logits=out.insert_logits,
        K=6,
        insert_threshold=0.5,
        min_sep=1,
        lambda_del=1.0,
        lambda_ins=1.0,
        lambda_shift=0.25,
    )

    pred_b = dec.pred_b[0]
    pred_edit = dec.pred_edit_labels[0]
    pred_insert = dec.pred_insert_labels[0]

    gold_b = batch["b_gold"][0].tolist()
    gold_insert = batch["insert_labels"][0].tolist()

    # 从原始样本里取 gold edit labels
    raw_sample = ds.samples[0]
    gold_edit = raw_sample["labels"]["edit"]

    print("pred_b =", pred_b)
    print("gold_b =", gold_b)
    print("pred_edit =", pred_edit)
    print("gold_edit =", gold_edit)
    print("pred_insert =", pred_insert)
    print("gold_insert =", gold_insert)

    atoms_text = batch["atoms_text"][0]
    projector_cfg = ProjectorConfig(
        max_chunk_atoms=64,
        min_chunk_atoms=2,
        max_chunk_chars=1600,
        min_chunk_chars=20,
        max_chunk_tokens=384,
        min_chunk_tokens=48,
    )

    # 用当前 insert logits 作为 gap_scores 的近似来源
    gap_scores = out.insert_logits[0].detach().cpu().tolist()

    rebuilt = rebuild_chunks_from_boundary_vector(
        atoms_text=atoms_text,
        b=pred_b,
        cfg=projector_cfg,
        gap_scores=gap_scores,
    )

    # print("rebuilt spans =", rebuilt["spans"])
    # print("rebuilt units =")
    # for u in rebuilt["units"]:
    #     print(u)

    print("spans before =", rebuilt["spans_before"])
    print("spans after split =", rebuilt["spans_after_split"])
    print("spans after merge =", rebuilt["spans_after_merge"])
    print("projected_b =", rebuilt["projected_b"])
    print("projected units =")
    for u in rebuilt["projected_units"]:
        print(u)

    m_boundary = boundary_prf(pred_b, gold_b)
    m_edit = edit_action_accuracy(pred_edit, gold_edit)
    m_insert = insert_accuracy(pred_insert, gold_insert)
    m_len = chunk_length_stats_from_spans(rebuilt["spans_after_merge"])

    print("boundary metrics =", m_boundary)
    print("edit metrics =", m_edit)
    print("insert metrics =", m_insert)
    print("length stats =", m_len)

    assert len(pred_b) == len(gold_b)
    assert "f1" in m_boundary
    assert "acc" in m_edit
    assert "acc" in m_insert
    assert "num_chunks" in m_len

    print("Decode + metrics test passed.")


if __name__ == "__main__":
    main()