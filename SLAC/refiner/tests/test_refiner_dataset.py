import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from slac_refiner.datasets.refiner_dataset import RefinerDenoiseDataset
from slac_refiner.datasets.collate import refiner_collate_fn


data_path = r"D:\code\Github\SLAC-test\SLAC\refiner\data\interim\atoms_b0\refiner_train_demo.jsonl"

ds = RefinerDenoiseDataset(data_path)
print("dataset size =", len(ds))

x = ds[0]
print("sample_id =", x["sample_id"])
print("atoms_text =", x["atoms_text"])
print("b0 =", x["b0"].tolist())
print("g0_positions =", x["g0_positions"].tolist())
print("edit_cls =", x["edit_cls"].tolist())
print("edit_offset =", x["edit_offset"].tolist())
print("insert_labels =", x["insert_labels"].tolist())
print("b_gold =", x["b_gold"].tolist())

loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=refiner_collate_fn)
batch = next(iter(loader))

print("batch keys =", batch.keys())
print("batch b0 shape =", tuple(batch["b0"].shape))
print("batch insert_labels shape =", tuple(batch["insert_labels"].shape))
print("batch edit_cls shape =", tuple(batch["edit_cls"].shape))
print("batch g0_positions =", batch["g0_positions"])