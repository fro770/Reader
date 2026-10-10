"""Голоса: папки с .wav-записями, кэш латентов, добавление новых голосов.

Один голос = одна папка в voices/. Внутри — один или несколько .wav (6–15 с).
Латенты, посчитанные по всем записям, кэшируются в voices/.cache.
"""
from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path

import numpy as np

from reader.core import config

logger = logging.getLogger("reader.voices")


def list_voice_dirs() -> list[Path]:
    """Папки в voices/, содержащие хотя бы один .wav."""
    result: list[Path] = []
    for d in sorted(config.VOICES_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if any(d.glob("*.wav")):
            result.append(d)
    return result


def resolve_voice_dir(name: str) -> Path | None:
    """Возвращает папку голоса по имени, если она существует и содержит .wav."""
    name = (name or "").strip()
    if not name:
        return None
    d = config.VOICES_DIR / name
    if d.is_dir() and any(d.glob("*.wav")):
        return d
    return None


def voice_wavs(voice_dir: Path) -> list[Path]:
    return sorted(voice_dir.glob("*.wav"))


def folder_signature(voice_dir: Path) -> str:
    """Хэш по всем *.wav внутри папки: имя|size|mtime."""
    parts: list[str] = []
    for f in voice_wavs(voice_dir):
        try:
            st = f.stat()
            parts.append(f"{f.name}|{st.st_size}|{int(st.st_mtime)}")
        except OSError:
            continue
    if not parts:
        return "empty"
    return "||".join(parts)


def cache_file_for(voice_dir: Path) -> Path:
    sig = folder_signature(voice_dir)
    h = hashlib.md5(sig.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^\w\-]+", "_", voice_dir.name, flags=re.UNICODE).strip("_") or "voice"
    return config.CACHE_DIR / f"{safe}__{h}.pt"


def sanitize_name(name: str) -> str:
    """Безопасное имя папки из пользовательского ввода."""
    return re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE).strip("_")


def add_voice(name: str, files: list[str]) -> tuple[str, int]:
    """Создаёт папку голоса и копирует/конвертирует выбранные записи в .wav.

    Возвращает (безопасное_имя, количество_добавленных_файлов).
    """
    safe = sanitize_name(name)
    if not safe:
        raise ValueError("Недопустимое имя папки.")

    dst_dir = config.VOICES_DIR / safe
    dst_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    for src in files:
        try:
            src_p = Path(src)
            if src_p.suffix.lower() == ".wav":
                shutil.copyfile(src_p, dst_dir / src_p.name)
            else:
                import soundfile as sf  # type: ignore
                data, sr = sf.read(src, always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                out_name = src_p.with_suffix(".wav").name
                sf.write(dst_dir / out_name, data.astype(np.float32), sr)
            added += 1
        except Exception:
            logger.exception("Не удалось добавить %s", src)

    logger.info("Голос «%s»: добавлено %d файл(ов) → %s", safe, added, dst_dir)
    return safe, added


def invalidate_cache(voice_dir: Path, memory_cache: dict) -> None:
    """Удаляет дисковый кэш латентов голоса и соответствующие записи памяти."""
    old_cache = cache_file_for(voice_dir)
    if old_cache.exists():
        try:
            old_cache.unlink()
            logger.info("Старый кэш удалён: %s", old_cache.name)
        except Exception:
            logger.exception("Не удалось удалить старый кэш")

    for k in list(memory_cache):
        if k[0] == str(voice_dir):
            del memory_cache[k]
