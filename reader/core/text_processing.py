"""Разбиение текста на предложения и чанки, понятные XTTS.

Приоритет сегментации: razdel → pysbd → regex-фолбэк.
razdel лучше всего понимает русские сокращения («т.д.», «г.», «ул.»).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("reader.text")

try:
    from razdel import sentenize as _razdel_sentenize  # type: ignore
    HAS_RAZDEL = True
except Exception:
    _razdel_sentenize = None  # type: ignore
    HAS_RAZDEL = False

try:
    import pysbd as _pysbd  # type: ignore
    _PY_SBD_SEG = _pysbd.Segmenter(language="ru", clean=False)
    HAS_PYSBD = True
except Exception:
    _PY_SBD_SEG = None
    HAS_PYSBD = False


def describe_splitter() -> str:
    """Короткое описание активного сегментатора — для статуса/логов."""
    if HAS_RAZDEL:
        return "razdel"
    if HAS_PYSBD:
        return "pysbd (razdel не установлен)"
    return "regex (razdel/pysbd не установлены)"


def sentenize(text: str) -> list[str]:
    """Разбивает текст на предложения, сохраняя пунктуацию."""
    if not text:
        return []

    if HAS_RAZDEL and _razdel_sentenize is not None:
        try:
            return [s.text.strip() for s in _razdel_sentenize(text) if s.text.strip()]
        except Exception:
            logger.exception("razdel.sentenize упал, откат на pysbd/regex")

    if HAS_PYSBD and _PY_SBD_SEG is not None:
        try:
            return [s.strip() for s in _PY_SBD_SEG.segment(text) if s.strip()]
        except Exception:
            logger.exception("pysbd упал, откат на regex")

    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _split_long_sentence(sentence: str, max_len: int) -> list[str]:
    """Режет слишком длинное предложение по границам придаточных,
    а если не помогло — по пробелам."""
    parts = re.split(r"(?<=[,;:—])\s+", sentence)
    chunks: list[str] = []
    cur = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_len:
            cur = cur + " " + p
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)

    result: list[str] = []
    for c in chunks:
        if len(c) <= max_len:
            result.append(c)
            continue
        buf = ""
        for w in c.split():
            if len(w) > max_len:
                if buf:
                    result.append(buf)
                    buf = ""
                for k in range(0, len(w), max_len):
                    result.append(w[k:k + max_len])
                continue
            if not buf:
                buf = w
            elif len(buf) + 1 + len(w) <= max_len:
                buf = buf + " " + w
            else:
                result.append(buf)
                buf = w
        if buf:
            result.append(buf)
    return result


def split_into_chunks(text: str, max_len: int) -> list[str]:
    """Разбивает текст на чанки, группируя короткие предложения.

    Алгоритм:
      1. Нормализуем пробелы.
      2. Разбиваем на предложения через razdel/pysbd/regex.
      3. Сливаем последовательные предложения, пока очередное влезает в max_len.
      4. Слишком длинные предложения режем отдельно.
    """
    if not text:
        return []

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    sentences = sentenize(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for s in sentences:
        if len(s) > max_len:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_sentence(s, max_len))
            continue

        if not current:
            current = s
        elif len(current) + 1 + len(s) <= max_len:
            current = current + " " + s
        else:
            chunks.append(current)
            current = s

    if current:
        chunks.append(current)

    return [c for c in chunks if c]
