from __future__ import annotations

from pathlib import Path

# 你本地代码里当前写死的旧路径
OLD_PATH_RAW = r"D:\code\Github\SLAC-test\SLAC\refiner\slac_refiner\models\bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"

# 服务器上的新路径
NEW_PATH = "/root/autodl-tmp/models/bge-m3/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

# 扫描范围：按你的项目结构改
TARGET_DIR = Path("SLAC/refiner")

# 要处理的文件类型；当前先只改 .py，最稳
FILE_PATTERNS = ["*.py"]


def main() -> None:
    if not TARGET_DIR.exists():
        raise FileNotFoundError(f"Target dir not found: {TARGET_DIR.resolve()}")

    old_path_escaped = OLD_PATH_RAW.replace("\\", "\\\\")
    changed_files = []

    for pattern in FILE_PATTERNS:
        for path in TARGET_DIR.rglob(pattern):
            text = path.read_text(encoding="utf-8")

            new_text = text
            # 替换原始写法，例如 r"D:\code\..."
            new_text = new_text.replace(OLD_PATH_RAW, NEW_PATH)
            # 替换转义写法，例如 "D:\\code\\..."
            new_text = new_text.replace(old_path_escaped, NEW_PATH)

            if new_text != text:
                path.write_text(new_text, encoding="utf-8")
                changed_files.append(path)

    print(f"Changed {len(changed_files)} file(s).")
    for p in changed_files:
        print(p)


if __name__ == "__main__":
    main()