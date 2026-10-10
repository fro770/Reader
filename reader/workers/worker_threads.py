"""QThread-воркеры: длительные операции вынесены из GUI-потока.

Каждый воркер общается с UI только через сигналы и (для синтеза)
через предиктор остановки `stop_check`.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from reader.core import config
from reader.core.audio_processing import normalize_peak, postprocess, write_mp3, write_wav
from reader.core.model_manager import ModelManager
from reader.core.stress import prepare_for_tts, stress_text
from reader.core.text_processing import split_into_chunks, trim_overlap

logger = logging.getLogger("reader.workers")


class ModelLoader(QThread):
    """Загрузка XTTS-v2 при старте приложения."""
    status_changed = pyqtSignal(str)
    loaded = pyqtSignal(int, str)   # sample_rate, device
    failed = pyqtSignal(str)

    def __init__(self, manager: ModelManager, want_gpu: bool) -> None:
        super().__init__()
        self._manager = manager
        self._want_gpu = want_gpu

    def run(self) -> None:
        try:
            self._manager.load(self._want_gpu, status_cb=self.status_changed.emit)
            self.loaded.emit(self._manager.sample_rate, self._manager.device)
        except Exception as e:
            logger.exception("Ошибка загрузки модели")
            self.failed.emit(str(e))


class DeviceSwitcher(QThread):
    """Перенос модели между CPU и GPU."""
    status_changed = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, manager: ModelManager, target: str) -> None:
        super().__init__()
        self._manager = manager
        self._target = target

    def run(self) -> None:
        try:
            if self._manager.switch_device(self._target,
                                           status_cb=self.status_changed.emit):
                self.done.emit(self._manager.device)
            else:
                self.failed.emit("CUDA OOM или ошибка — остаёмся на CPU")
        except Exception as e:
            logger.exception("Ошибка переключения устройства")
            self.failed.emit(str(e))


class TrainWorker(QThread):
    """Считает латенты по всем *.wav в папке голоса и сохраняет в кэш."""
    status_changed = pyqtSignal(str)
    finished_ok = pyqtSignal(str, str)
    failed = pyqtSignal(str)

    def __init__(self, manager: ModelManager, voice_dir: Path) -> None:
        super().__init__()
        self._manager = manager
        self._voice_dir = voice_dir

    def run(self) -> None:
        try:
            dt = self._manager.train_voice(self._voice_dir,
                                           status_cb=self.status_changed.emit)
            wavs = list(self._voice_dir.glob("*.wav"))
            self.finished_ok.emit(
                self._voice_dir.name,
                f"Голос «{self._voice_dir.name}» обучен на {len(wavs)} записях "
                f"({dt:.1f} с)",
            )
        except Exception as e:
            logger.exception("Ошибка обучения голоса")
            self.failed.emit(str(e))


class SynthWorker(QThread):
    """Синтез текста в аудиофайл (MP3/WAV) чанками с прогрессом."""
    status_changed = pyqtSignal(str)
    progress_changed = pyqtSignal(float)
    elapsed_changed = pyqtSignal(str)
    finished_ok = pyqtSignal(str, float, float)  # имя файла, длительность, время
    failed = pyqtSignal(str)

    def __init__(
        self,
        manager: ModelManager,
        text: str,
        voice_dir: Path,
        cleanup: bool,
        out_path: Path,
        stop_check: Callable[[], bool],
        auto_stress: bool = True,
    ) -> None:
        super().__init__()
        self._manager = manager
        self._text = text
        self._voice_dir = voice_dir
        self._cleanup = cleanup
        self._out_path = out_path
        self._stop_check = stop_check
        self._auto_stress = auto_stress

    @staticmethod
    def _fade_head(a: np.ndarray, sr: int, ms: int = 15) -> np.ndarray:
        """Короткий фейд-ин в начале фрагмента — чтобы подрезка не щёлкала."""
        n = min(int(sr * ms / 1000.0), len(a))
        if n > 1:
            a = a.copy()
            a[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
        return a

    def run(self) -> None:
        t_start = time.time()
        sr = self._manager.sample_rate
        try:
            gpt_cond, spk_emb = self._manager.get_latents(
                self._voice_dir, status_cb=self.status_changed.emit)

            # --- Подготовка текста: контекстные ударения → нормализация для XTTS
            text = self._text
            if self._auto_stress:
                self.status_changed.emit("✍ Расстановка ударений…")
                text = stress_text(text)
            text = prepare_for_tts(text)

            chunks = split_into_chunks(text, max_len=self._manager.chunk_max)
            total = max(1, len(chunks))
            if not chunks:
                self.failed.emit("Не удалось разбить текст на чанки")
                return

            logger.info("Разбито на %d фрагмент(ов) по ≤ %d симв.",
                        total, self._manager.chunk_max)

            parts: list[np.ndarray] = []
            prev_text = ""
            for i, chunk in enumerate(chunks, 1):
                if self._stop_check():
                    logger.info("Прервано пользователем на %d/%d", i, total)
                    break

                t_chunk = time.time()
                self.status_changed.emit(f"🎧 Синтез… {i}/{total}")
                logger.debug("[%d/%d] (%d симв.) %s", i, total, len(chunk), chunk)

                audio = self._manager.synthesize(chunk, gpt_cond, spk_emb)

                # Защита от повторов: если XTTS «доела» начало чанка из конца
                # предыдущего — подрезаем дублирующееся начало текущего аудио.
                if prev_text and parts:
                    n_words, est_sec = trim_overlap(chunk, prev_text)
                    if n_words >= 3:
                        cut = min(int(est_sec * sr), max(0, len(audio) - sr // 4))
                        if cut > 0:
                            logger.warning(
                                "[%d/%d] стык: повтор %d слов (~%.1f с) — подрезано",
                                i, total, n_words, est_sec)
                            audio = self._fade_head(audio[cut:], sr)

                parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
                prev_text = chunk

                dt_chunk = time.time() - t_chunk
                self.progress_changed.emit((i / total) * 100.0)

                elapsed_total = time.time() - t_start
                avg = elapsed_total / i
                remaining = avg * (total - i)
                eta_min = int(remaining // 60)
                eta_sec = int(remaining % 60)
                eta_str = f"{eta_min}м {eta_sec:02d}с" if eta_min else f"{eta_sec}с"
                logger.debug("[%d/%d] готово за %.1f с (осталось ~%s)",
                             i, total, dt_chunk, eta_str)
                self.elapsed_changed.emit(
                    f"⏱ {int(elapsed_total // 60)}м {int(elapsed_total % 60):02d}с, "
                    f"ETA ~{eta_str}"
                )

            if not parts:
                self.failed.emit("Ничего не сгенерировано")
                return

            # Склейка с паузами между чанками
            if len(parts) == 1:
                audio = parts[0]
            else:
                pause = np.zeros(int(sr * config.INSERT_PAUSE_MS / 1000.0),
                                 dtype=np.float32)
                pieces: list[np.ndarray] = []
                for i, p in enumerate(parts):
                    pieces.append(p)
                    if i < len(parts) - 1:
                        pieces.append(pause)
                audio = np.concatenate(pieces)

            audio = audio.astype(np.float32)

            # Постобработка один раз на всём аудио
            audio = postprocess(audio, sr) if self._cleanup else normalize_peak(audio)

            self.status_changed.emit("💾 Сохранение файла…")
            ext = self._out_path.suffix.lower()
            if ext == ".mp3":
                write_mp3(str(self._out_path), audio, sr)
            else:
                write_wav(str(self._out_path), audio, sr)

            dur = len(audio) / sr
            elapsed = time.time() - t_start
            rtf = dur / elapsed if elapsed > 0 else 0.0
            logger.info(
                "Сохранено: %s (%.1f с аудио за %.1f с синтеза, ×%.2f realtime, %s)",
                self._out_path, dur, elapsed, rtf, self._manager.device.upper(),
            )
            self.finished_ok.emit(self._out_path.name, dur, elapsed)
        except Exception as e:
            logger.exception("Ошибка синтеза")
            self.failed.emit(str(e))
