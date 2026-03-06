import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slac_refiner.atomize.normalize import normalize_text
from slac_refiner.atomize.splitter import split_text_to_atoms
from slac_refiner.atomize.mapping import build_atoms_b0_from_units


def run_case(name, units):
    normalized_units = [
        {"unit_id": u["unit_id"], "text": normalize_text(u["text"])}
        for u in units
    ]
    result = build_atoms_b0_from_units(
        normalized_units=normalized_units,
        splitter_fn=split_text_to_atoms,
    )

    print("=" * 60)
    print(name)
    print("atoms:")
    for i, a in enumerate(result["atoms"]):
        print(f"  {i}: {a!r}")

    print("unit2atom_span:")
    for s in result["unit2atom_span"]:
        print(f"  unit_id={s.unit_id}, span=[{s.start_atom},{s.end_atom})")

    print("b0:", result["b0"])
    print("meta:", result["meta"])
    return result


# Case 1: 正常非空 unit
case1 = [
    {"unit_id": 0, "text": "第一段。第二句。"},
    {"unit_id": 1, "text": "Third unit. Another sentence."},
]

# Case 2: 中间空 unit
case2 = [
    {"unit_id": 0, "text": "第一段。第二句。"},
    {"unit_id": 1, "text": "   "},
    {"unit_id": 2, "text": "Third unit. Another sentence."},
]

# Case 3: 全空 unit
case3 = [
    {"unit_id": 0, "text": "   "},
    {"unit_id": 1, "text": "\n\n"},
]

r1 = run_case("CASE 1: normal", case1)
r2 = run_case("CASE 2: empty middle", case2)
r3 = run_case("CASE 3: all empty", case3)

# -------- 基本断言 --------

# case1
assert len(r1["atoms"]) == 4
assert [(x.start_atom, x.end_atom) for x in r1["unit2atom_span"]] == [(0, 2), (2, 4)]
assert r1["b0"] == [0, 1, 0]

# case2
assert len(r2["atoms"]) == 4
assert [(x.start_atom, x.end_atom) for x in r2["unit2atom_span"]] == [(0, 2), (2, 2), (2, 4)]
assert r2["b0"] == [0, 0, 0]
assert r2["meta"]["projection_fix"] == 2

# case3
assert r3["atoms"] == []
assert r3["b0"] == []
assert r3["meta"]["all_units_empty"] is True
assert [(x.start_atom, x.end_atom) for x in r3["unit2atom_span"]] == [(0, 0), (0, 0)]

print("\nAll manual mapping tests passed.")