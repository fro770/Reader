"""Настройка логирования приложения.

Два уровня вывода:
  • консоль — только основные этапы работы (INFO, краткий формат);
  • файл ``output/reader.log`` — подробно (DEBUG), включая логи чанков,
    ударений и внутренних модулей сегментации.
"""
from __future__ import annotations

import logging
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: Path | None = None) -> logging.Logger:
    """Единая точка настройки логов и подавления шумных предупреждений torch/TTS."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # --- Консоль: только основные этапы -----------------------------------
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt="%(levelname)-7s | %(message)s"))

    class _StageFilter(logging.Filter):
        """В консоль: сообщения reader.* полностью, сторонние — только WARNING+."""

        def filter(self, record: logging.LogRecord) -> bool:
            if record.name == "reader" or record.name.startswith("reader."):
                return True
            return record.levelno >= logging.WARNING

    console.addFilter(_StageFilter())
    root.addHandler(console)

    # --- Файл: подробно, с временными метками ------------------------------
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                log_dir / "reader.log",
                maxBytes=2_000_000, backupCount=3, encoding="utf-8",
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            root.addHandler(fh)
        except OSError:
            logging.getLogger("reader").warning("Не удалось создать файл лога")

    # Шумные сторонние библиотеки — не ниже WARNING
    for noisy in ("TTS", "torch", "numba", "matplotlib", "urllib3",
                  "razdel", "pysbd", "espeakngloader"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*")
    warnings.filterwarnings("ignore", message=r".*isin_mps_friendly.*")
    warnings.filterwarnings("ignore", message=r".*smallest subnormal.*")
    return logging.getLogger("reader")
