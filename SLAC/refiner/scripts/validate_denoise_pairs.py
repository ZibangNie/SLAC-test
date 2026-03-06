from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate refiner denoise training samples.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL path.")
    parser.add_argument("--max_errors", type=int, default=20)
    return parser.parse_args()


def vector_to_gaps(b: Sequence[int]) -> List[int]:
    return [i for i, x in enumerate(b) if int(x) == 1]


def gaps_to_vector(gaps: Sequence[int], num_gaps: int) -> List[int]:
    out = [0] * num_gaps
    for g in gaps:
        if not (0 <= g < num_gaps):
            raise ValueError(f"Gap out of range: g={g}, num_gaps={num_gaps}")
        out[g] = 1
    return out


def parse_shift(label: str) -> int:
    if not label.startswith("SHIFT:"):
        raise ValueError(f"Invalid SHIFT label: {label}")
    return int(label.split(":", 1)[1])


def apply_labels_to_b0(
    b0: Sequence[int],
    edit_labels: Sequence[Dict],
    insert_labels: Sequence[int],
) -> List[int]:
    num_gaps = len(b0)
    g0 = vector_to_gaps(b0)

    edit_map: Dict[int, str] = {}
    for item in edit_labels:
        g = int(item["g"])
        y = str(item["y"])
        if g in edit_map:
            raise ValueError(f"Duplicate edit label for initial boundary g={g}")
        edit_map[g] = y

    if set(edit_map.keys()) != set(g0):
        raise ValueError(
            f"Edit labels must be defined exactly on G0. "
            f"G0={g0}, edit_keys={sorted(edit_map.keys())}"
        )

    final_gaps: List[int] = []

    for g in g0:
        y = edit_map[g]
        if y == "KEEP":
            final_gaps.append(g)
        elif y == "DEL":
            continue
        elif y.startswith("SHIFT:"):
            k = parse_shift(y)
            t = g + k
            if not (0 <= t < num_gaps):
                raise ValueError(f"SHIFT target out of range: g={g}, k={k}, num_gaps={num_gaps}")
            final_gaps.append(t)
        else:
            raise ValueError(f"Unknown edit label: {y}")

    if len(insert_labels) != num_gaps:
        raise ValueError(
            f"insert length mismatch: len(insert)={len(insert_labels)}, num_gaps={num_gaps}"
        )

    for g, x in enumerate(insert_labels):
        if int(x) == 1:
            final_gaps.append(g)

    final_gaps = sorted(final_gaps)

    # strict monotonic + no duplicates
    for i in range(1, len(final_gaps)):
        if final_gaps[i] <= final_gaps[i - 1]:
            raise ValueError(f"Non-strict final gaps after apply: {final_gaps}")

    return gaps_to_vector(final_gaps, num_gaps)


def validate_one(sample: Dict) -> None:
    atoms = sample["atoms"]
    if len(atoms) == 0:
        raise ValueError("atoms must be non-empty in training samples")

    num_gaps = len(atoms) - 1
    b0 = sample["b0"]
    b_gold = sample["b_gold"]
    labels = sample["labels"]

    if len(b0) != num_gaps:
        raise ValueError(f"len(b0)={len(b0)} != len(atoms)-1={num_gaps}")

    if len(b_gold) != num_gaps:
        raise ValueError(f"len(b_gold)={len(b_gold)} != len(atoms)-1={num_gaps}")

    if len(labels["insert"]) != num_gaps:
        raise ValueError(f"len(insert)={len(labels['insert'])} != len(atoms)-1={num_gaps}")

    # edit.g must come from G0
    g0 = set(vector_to_gaps(b0))
    edit_g = {int(x["g"]) for x in labels["edit"]}
    if edit_g != g0:
        raise ValueError(f"edit.g must equal G0. edit_g={sorted(edit_g)}, G0={sorted(g0)}")

    # chunk0/unit2atom_span consistency
    spans = sample["unit2atom_span"]
    if len(spans) != len(sample["chunk0_units"]):
        raise ValueError("unit2atom_span length must equal chunk0_units length")

    cursor = 0
    for i, sp in enumerate(spans):
        s = int(sp["s"])
        e = int(sp["e"])
        if s != cursor:
            raise ValueError(f"span coverage error at idx={i}: expected s={cursor}, got s={s}")
        if e < s:
            raise ValueError(f"span order error at idx={i}: s={s}, e={e}")
        cursor = e
    if cursor != len(atoms):
        raise ValueError(f"spans do not cover atoms: last_end={cursor}, atoms={len(atoms)}")

    # apply(edit, insert, b0) == b_gold
    rebuilt = apply_labels_to_b0(
        b0=b0,
        edit_labels=labels["edit"],
        insert_labels=labels["insert"],
    )
    if rebuilt != b_gold:
        raise ValueError(f"apply(labels, b0) != b_gold. rebuilt={rebuilt}, gold={b_gold}")


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Failed parsing line {line_no}: {e}") from e


def main() -> None:
    args = parse_args()
    path = Path(args.input)

    total = 0
    bad = 0

    for i, sample in enumerate(iter_jsonl(path), start=1):
        total += 1
        try:
            validate_one(sample)
        except Exception as e:
            bad += 1
            print(f"[BAD] line={i}, sample_id={sample.get('sample_id')}, err={e}")
            if bad >= args.max_errors:
                print(f"Stop early: reached max_errors={args.max_errors}")
                break

    good = total - bad
    print(f"Validation done. total={total}, good={good}, bad={bad}")


if __name__ == "__main__":
    main()