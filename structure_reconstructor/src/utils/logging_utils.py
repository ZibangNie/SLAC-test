# src/utils/logging_utils.py
import logging
from pathlib import Path


def init_logger(log_file: str | Path) -> logging.Logger:
    """
    同时把日志打印到控制台和 log_file。
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 避免重复输出

    # 如果重复调用 init_logger，先清掉旧 handler
    if logger.handlers:
        logger.handlers.clear()

    # 控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # 文件
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    return logger
