"""Настройка логирования приложения."""
from __future__ import annotations

import logging
import warnings


def setup_logging() -> logging.Logger:
    """Единая точка настройки логов и подавления шумных предупреждений torch/TTS."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*")
    warnings.filterwarnings("ignore", message=r".*isin_mps_friendly.*")
    warnings.filterwarnings("ignore", message=r".*smallest subnormal.*")
    return logging.getLogger("reader")
