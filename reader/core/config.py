"""Конфигурация приложения: пути, параметры модели и постобработки."""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- CPU-потоки
# ВАЖНО: переменные окружения выставляем ДО импорта torch, иначе MKL их игнорирует.
CPU_COUNT = os.cpu_count() or 4
for _env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env, str(CPU_COUNT))

# ---------------------------------------------------------------- Параметры XTTS
LANGUAGE = "ru"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

XTTS_KWARGS = dict(
    temperature=0.75,
    length_penalty=1.0,
    repetition_penalty=5.0,
    top_k=50,
    top_p=0.85,
    enable_text_splitting=False,
)

# ---------------------------------------------------------------- Пути
BASE_DIR = Path(__file__).resolve().parents[2]
VOICES_DIR = BASE_DIR / "voices"
CACHE_DIR = VOICES_DIR / ".cache"
OUTPUT_DIR = BASE_DIR / "output"

for _d in (VOICES_DIR, CACHE_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Нарезка текста
DEFAULT_CHAR_LIMIT = 182      # жёсткий лимит символов XTTS на один вызов
CHUNK_SAFETY_MARGIN = 10      # запас до лимита
MIN_CHUNK_LEN = 40
INSERT_PAUSE_MS = 150         # пауза между чанками, мс

# ---------------------------------------------------------------- Постобработка
HP_FREQ = 70.0                # частота среза high-pass, Гц
LP_FREQ = 7500.0              # частота среза low-pass, Гц
FADE_MS = 15                  # длина фейда начала/конца, мс
GATE_THRESH_DB = -50.0        # порог noise-gate, дБ

# ---------------------------------------------------------------- Прочее
MP3_BITRATE = 192
FP16_VRAM_THRESHOLD_GB = 3.0  # при VRAM ниже — используем FP16
