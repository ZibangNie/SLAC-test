from slac_refiner.atomize.normalize import normalize_text
from slac_refiner.atomize.splitter import split_text_to_atoms

samples = [
    "第一段。这是第一段的第二句。这里有 3.14，不该在小数点处分开。中文里 A.B. 不一定常见。",
    "Second unit. Another sentence. Dr. Smith arrived at 3.14 p.m. He said hello.",
    "第一行\n第二行\n第三行\n第四行",
    "This is a very long line, with many clauses, and maybe OCR style fragments, and more text: keep splitting if necessary.",
    "短。 很短。 A. B. C.",
]

for i, s in enumerate(samples, 1):
    print("=" * 40)
    print(f"sample {i}")
    s = normalize_text(s)
    atoms = split_text_to_atoms(s)
    for j, a in enumerate(atoms):
        print(j, repr(a))