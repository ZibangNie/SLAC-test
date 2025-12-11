from datasets import load_dataset
import json
from pathlib import Path


def load_multieurlex_english(
    split: str = "train",
    n_docs: int = 50,
    min_chars: int | None = None,
    max_chars: int | None = None,
):
    ds = load_dataset(
        "coastalcph/multi_eurlex",
        "en",
        split=split,
        trust_remote_code=True,
    )

    def length_filter(example):
        l = len(example["text"])
        if min_chars is not None and l < min_chars:
            return False
        if max_chars is not None and l > max_chars:
            return False
        return True

    ds_filtered = ds.filter(length_filter)

    if n_docs is not None and n_docs < len(ds_filtered):
        ds_filtered = ds_filtered.shuffle(seed=42).select(range(n_docs))

    docs = []
    for idx, ex in enumerate(ds_filtered):
        docs.append(
            {
                "celex_id": ex["celex_id"],
                "text": ex["text"],
                "labels": ex["labels"],
            }
        )
    return docs



def export_each_as_json(docs, output_dir: str):
    """
    将每条文档单独保存成一个 JSON 文件：
    {output_dir}/{celex_id}.json
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, doc in enumerate(docs):
        celex_id = doc.get("celex_id") or f"no_celex_{idx}"
        # 避免特殊字符
        safe_celex = "".join(c if c.isalnum() else "_" for c in str(celex_id))

        out_path = out_dir / f"{safe_celex}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 目标：十几二十页的长文档
    docs = load_multieurlex_english(
        split="train",
        n_docs=30,          # 先取 30 部看看
        min_chars=35000,    # ≈ 10 页起步
        max_chars=120000,   # 上限相当于 30–40 页级别
    )
    print(f"Loaded {len(docs)} long EN laws from MultiEURLEX")

    export_each_as_json(
        docs,
        r"D:\code\Github\SLAC-test\data\A_structure\laws\en_long",
    )
    print("Saved long-law JSON files to ...\\laws\\en_long")
