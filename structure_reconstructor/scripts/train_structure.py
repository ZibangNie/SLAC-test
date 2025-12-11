# scripts/train_structure.py
import argparse
from pathlib import Path
import sys

# 把项目根目录（structure_reconstructor）加入 sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.struct_reconstructor import StructRecConfig
from src.training.trainer import train_one_run


def parse_args():
    parser = argparse.ArgumentParser(description="Train SLAC Structure Reconstructor")
    parser.add_argument(
        "--index_path",
        type=str,
        default=r"D:\code\Github\SLAC-test\structure_dataset\index\all_index.jsonl",
        help="结构数据集的 index jsonl 路径",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=r"D:\code\Github\SLAC-test\structure_reconstructor\runs\debug_run",
        help="本次实验的输出目录",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_unit_len", type=int, default=256)
    parser.add_argument(
        "--lm_name",
        type=str,
        default="bert-base-multilingual-cased",
        help="预训练 LM 名称/路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = StructRecConfig(
        lm_name=args.lm_name,
        # 如果你用别的 LM，记得把 lm_hidden_size 改成对应 hidden 维度
        lm_hidden_size=768,
        doc_hidden_size=512,
        num_doc_layers=4,
        num_heads=8,
        max_level=8,
    )

    Path(args.run_dir).mkdir(parents=True, exist_ok=True)

    train_one_run(
        index_path=args.index_path,
        run_dir=args.run_dir,
        config=cfg,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_unit_len=args.max_unit_len,
    )


if __name__ == "__main__":
    main()
