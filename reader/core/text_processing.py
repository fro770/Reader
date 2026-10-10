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


# ------------------------------------------------------------------ Защита от повторов
_STRESS = "\u0301"


def _norm_key(chunk: str) -> str:
    """Ключ сравнения чанков: без ударений, регистра и пунктуации."""
    t = chunk.replace(_STRESS, "").lower()
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def _is_duplicate(prev: str, cur: str) -> bool:
    """Один и тот же фрагмент? (полное совпадение или сильное перекрытие хвоста)."""
    a, b = _norm_key(prev), _norm_key(cur)
    if not a or not b:
        return False
    if a == b:
        return True
    # хвост prev начинается с начала cur (или наоборот) — типичный артефакт
    # XTTS, когда модель «доедает» последние слова фразы в следующем чанке
    tail_a = a[-25:]
    head_b = b[:25]
    if len(tail_a) >= 8 and len(head_b) >= 8 and tail_a == head_b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return len(shorter) >= 10 and shorter in longer


def split_into_chunks(text: str, max_len: int) -> list[str]:
    """Разбивает текст на чанки, группируя короткие предложения.

    Алгоритм:
      1. Нормализуем пробелы.
      2. Разбиваем на предложения через razdel/pysbd/regex.
      3. Сливаем последовательные предложения, пока очередное влезает в max_len.
      4. Слишком длинные предложения режем отдельно.
      5. Убираем случайные повторы соседних чанков (защита от галлюцинаций XTTS).
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

    chunks = [c for c in chunks if c]

    deduped: list[str] = []
    for c in chunks:
        if deduped and _is_duplicate(deduped[-1], c):
            logger.warning("Повторяющийся чанк пропущен: %r", c[:60])
            continue
        deduped.append(c)
    if len(deduped) != len(chunks):
        logger.info("Дедупликация чанков: %d → %d", len(chunks), len(deduped))
    return deduped


def trim_overlap(chunk_text: str, prev_text: str) -> tuple[int, float]:
    """Оценивает, сколько слов следующего чанка модель могла «доесть» за предыдущим.

    Возвращает (кол-во повторяющихся слов на стыке, примерную длительность в
    секундах при среднем темпе русской речи ~2.7 слова/сек). Используется
    воркером для подрезки начала текущего аудио, чтобы на слух не было
    двойного прочтения фразы.
    """
    a_words = _norm_key(prev_text).split()
    b_words = _norm_key(chunk_text).split()
    if not a_words or not b_words:
        return 0, 0.0

    max_k = min(len(a_words), len(b_words), 8)
    for k in range(max_k, 2, -1):
        if a_words[-k:] == b_words[:k]:
            return k, k / 2.7
    return 0, 0.0
