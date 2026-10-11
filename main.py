"""
Читалка — русский TTS с клонированием голоса на Coqui XTTS-v2 (PyQt6).

Архитектура проекта:
    reader/
        core/
            config.py            — пути, параметры модели, потоки CPU (импортировать первым)
            logging_setup.py     — настройка логов и подавление шумных warnings
            model_manager.py     — ModelManager: загрузка XTTS, CPU/GPU, прогрев, латенты,
                                   патчи совместимости (torchaudio→soundfile, safe globals)
            text_processing.py   — сегментация текста (razdel → pysbd → regex), чанки
            audio_processing.py  — постобработка звука и запись WAV/MP3
            voices.py            — папки голосов, кэш латентов, добавление голосов
        workers/
            worker_threads.py    — QThread-воркеры: загрузка модели, переключение
                                   устройства, обучение голоса, синтез в файл
        ui/
            theme.py             — тёмная тема: палитра, QSS, анимации плавного появления
            widgets.py           — переиспользуемые блоки: карточки-секции, шапка
            main_window.py       — главное окно: секции «Голос», «Текст», «Результат»

Голоса — папки в voices/: одна папка = один голос, внутри .wav-записи по 6–15 с.
Файлы озвучки сохраняются в output/.

Зависимости: PyQt6, torch, torchaudio, soundfile, sounddevice, numpy, scipy,
razdel (опционально), lameenc (для MP3).
"""
from __future__ import annotations

import sys

# config выставляет OMP/MKL-переменные ДО импорта torch — это важно для
# производительности на CPU; все остальные модули приложения импортируются после.
from reader.core import config  # noqa: F401  (побочный эффект: переменные окружения)

import torch

from PyQt6.QtWidgets import QApplication

from reader.core.logging_setup import setup_logging
from reader.ui.main_window import MainWindow
from reader.ui.theme import build_qss


def _configure_torch_threads() -> None:
    try:
        torch.set_num_threads(config.CPU_COUNT)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        torch.set_flush_denormal(True)
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def main() -> None:
    setup_logging()
    _configure_torch_threads()

    if sys.platform == "win32":
        try:
            from ctypes import windll  # type: ignore
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("XTTS Reader")
    app.setStyleSheet(build_qss())

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
