import json
import random
import shutil
from pathlib import Path

# 固定随机种子，保证每次划分一致
RANDOM_SEED = 42

# 旧数据根目录
DATA_ROOT = Path(r"D:\code\Github\SLAC-test\data")

# 新数据集根目录
TARGET_ROOT = Path(r"D:\code\Github\SLAC-test\structure_dataset")

# index 目录
INDEX_DIR = TARGET_ROOT / "index"


def doc_id_from_filename(path: Path) -> str:
    """
    根据文件名生成 doc_id：
    - xxx.tree.json -> 去掉 .tree.json
    - xxx.json      -> 去掉 .json
    - 其它情况      -> 用 stem
    """
    name = path.name
    if name.endswith(".tree.json"):
        return name[:-len(".tree.json")]
    elif name.endswith(".json"):
        return name[:-len(".json")]
    else:
        return path.stem


def split_and_move(
    files,
    target_base: Path,
    source_type: str,
    domain: str,
    lang: str,
    index_all: list,
    index_A: list,
    index_B: list,
    subdomain: str = None,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    use_lang_subdir: bool = True,
):
    """
    将 files 划分为 train/dev/test，移动到目标目录，
    并写入 index 记录。

    use_lang_subdir:
      - True:  目标目录形如 target_base/lang/split
      - False: 目标目录形如 target_base/split  （用于 A/papers）
    """
    files = sorted(files)
    n = len(files)
    if n == 0:
        print(f"[WARN] No files found for {source_type}/{domain}/{lang}, subdomain={subdomain}")
        return

    random.shuffle(files)

    # 简单 8:1:1 划分
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    n_test = n - n_train - n_dev

    # 样本极少时，保证至少有一个 train
    if n_train == 0 and n > 0:
        n_train = 1
        n_dev = 0
        n_test = n - 1

    print(
        f"[INFO] {source_type}/{domain}/{lang}"
        + (f"/{subdomain}" if subdomain else "")
        + f": total={n}, train={n_train}, dev={n_dev}, test={n_test}"
    )

    def get_split(idx: int) -> str:
        if idx < n_train:
            return "train"
        elif idx < n_train + n_dev:
            return "dev"
        else:
            return "test"

    for idx, src in enumerate(files):
        split = get_split(idx)

        # 决定目标目录结构
        if use_lang_subdir:
            dest_dir = target_base / lang / split
        else:
            dest_dir = target_base / split

        dest_dir.mkdir(parents=True, exist_ok=True)

        # 论文带 subdomain 时，为避免不同子目录下重名，前缀子目录名
        if subdomain is not None and domain == "paper":
            dest_name = f"{subdomain}__{src.name}"
        else:
            dest_name = src.name

        dest_path = dest_dir / dest_name

        # 真正移动文件；如果只想复制，请改成 shutil.copy2
        shutil.move(str(src), str(dest_path))

        doc_id = doc_id_from_filename(src)
        record = {
            "doc_id": doc_id,
            "abs_path": str(dest_path),  # 绝对路径
            "rel_path": str(dest_path.relative_to(TARGET_ROOT)),  # 相对 structure_dataset 的路径
            "source_type": source_type,  # "A" or "B"
            "domain": domain,            # "law" / "standard" / "paper" / "short"
            "lang": lang,                # "en" / "zh" / ...
            "split": split,              # "train" / "dev" / "test"
        }
        if subdomain is not None:
            record["subdomain"] = subdomain  # 比如 arxiv_cs / arxiv_math / root 等

        index_all.append(record)
        if source_type == "A":
            index_A.append(record)
        elif source_type == "B":
            index_B.append(record)


def main():
    random.seed(RANDOM_SEED)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_all = []
    index_A = []
    index_B = []

    # =========================================================
    # A 类：法律（laws_structure/en|zh）
    # =========================================================
    for lang in ["en", "zh"]:
        src_dir = DATA_ROOT / "A_structure" / "laws_structure" / lang
        if not src_dir.exists():
            print(f"[WARN] Source dir not found: {src_dir}")
            continue
        files = list(src_dir.glob("*.json"))
        target_base = TARGET_ROOT / "A" / "laws"
        split_and_move(
            files=files,
            target_base=target_base,
            source_type="A",
            domain="law",
            lang=lang,
            index_all=index_all,
            index_A=index_A,
            index_B=index_B,
            use_lang_subdir=True,
        )

    # =========================================================
    # A 类：国家 / 行业标准（national_standards_structure/en|zh）
    # =========================================================
    for lang in ["en", "zh"]:
        src_dir = DATA_ROOT / "A_structure" / "national_standards_structure" / lang
        if not src_dir.exists():
            print(f"[WARN] Source dir not found: {src_dir}")
            continue
        files = list(src_dir.glob("*.json"))
        target_base = TARGET_ROOT / "A" / "standards"
        split_and_move(
            files=files,
            target_base=target_base,
            source_type="A",
            domain="standard",
            lang=lang,
            index_all=index_all,
            index_A=index_A,
            index_B=index_B,
            use_lang_subdir=True,
        )

    # =========================================================
    # A 类：论文（papers_structure 根目录 + 各 arxiv_* 子目录）
    # 目标目录：D:\code\Github\SLAC-test\structure_dataset\A\papers\train|dev|test
    # =========================================================
    papers_root = DATA_ROOT / "A_structure" / "papers_structure"
    if papers_root.exists():
        target_base = TARGET_ROOT / "A" / "papers"

        # 1）根目录直接放的论文 JSON
        root_files = list(papers_root.glob("*.json"))
        if root_files:
            split_and_move(
                files=root_files,
                target_base=target_base,
                source_type="A",
                domain="paper",
                lang="en",          # 这里统一标记为 en；如果后续有中文论文，可再细分处理
                index_all=index_all,
                index_A=index_A,
                index_B=index_B,
                subdomain="root",   # 标记来自根目录
                use_lang_subdir=False,  # A/papers 下无 en/zh 子目录
            )

        # 2）各 arxiv_* 子目录中的论文 JSON
        for subdir in papers_root.iterdir():
            if not subdir.is_dir():
                continue
            subdomain = subdir.name  # 例如 arxiv_cs, arxiv_math 等
            files = list(subdir.glob("*.json"))
            if not files:
                continue
            split_and_move(
                files=files,
                target_base=target_base,
                source_type="A",
                domain="paper",
                lang="en",          # 同上，先统一标为 en
                index_all=index_all,
                index_A=index_A,
                index_B=index_B,
                subdomain=subdomain,
                use_lang_subdir=False,  # A/papers 只有 train/dev/test 三级
            )
    else:
        print(f"[WARN] Papers root not found: {papers_root}")

    # =========================================================
    # B 类：中英文短文（半结构）
    # 目标目录：B\en|zh\train|dev|test
    # =========================================================

    # 英文短文
    src_dir_en = DATA_ROOT / "B_semi-structure" / "en-tech" / "json"
    if src_dir_en.exists():
        files_en = list(src_dir_en.glob("*.json"))
        target_base = TARGET_ROOT / "B"
        split_and_move(
            files=files_en,
            target_base=target_base,
            source_type="B",
            domain="short",
            lang="en",
            index_all=index_all,
            index_A=index_A,
            index_B=index_B,
            use_lang_subdir=True,
        )
    else:
        print(f"[WARN] Source dir not found: {src_dir_en}")

    # 中文短文
    src_dir_zh = DATA_ROOT / "B_semi-structure" / "zh-tech" / "json"
    if src_dir_zh.exists():
        files_zh = list(src_dir_zh.glob("*.json"))
        target_base = TARGET_ROOT / "B"
        split_and_move(
            files=files_zh,
            target_base=target_base,
            source_type="B",
            domain="short",
            lang="zh",
            index_all=index_all,
            index_A=index_A,
            index_B=index_B,
            use_lang_subdir=True,
        )
    else:
        print(f"[WARN] Source dir not found: {src_dir_zh}")

    # =========================================================
    # 写出 index 文件（jsonl，每行一条记录）
    # =========================================================
    all_index_path = INDEX_DIR / "all_index.jsonl"
    A_index_path = INDEX_DIR / "A_index.jsonl"
    B_index_path = INDEX_DIR / "B_index.jsonl"

    def write_index(path: Path, records: list):
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[INFO] Wrote {len(records)} records to {path}")

    write_index(all_index_path, index_all)
    write_index(A_index_path, index_A)
    write_index(B_index_path, index_B)

    print("[DONE] Dataset split & index generation finished.")


if __name__ == "__main__":
    main()
