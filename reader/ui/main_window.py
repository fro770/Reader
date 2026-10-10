"""Главное окно приложения: секции, переходы и связывание UI с воркерами."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QUrl, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from reader.core import config, voices

# ВАЖНО: config импортируется первым среди модулей приложения (см. main.py) —
# он выставляет OMP/MKL-переменные до импорта torch; повторный импорт здесь
# безопасен и гарантирует порядок при прямом импорте этого модуля.
from reader.core.model_manager import ModelManager, default_use_gpu
from reader.core.text_processing import describe_splitter
from reader.ui import theme, widgets
from reader.workers.worker_threads import (
    DeviceSwitcher,
    ModelLoader,
    SynthWorker,
    TrainWorker,
)

logger = logging.getLogger("reader.main")


def open_local_path(path: Path) -> None:
    try:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
    except Exception:
        logger.exception("Не удалось открыть папку: %s", path)


class MainWindow(QMainWindow):
    """Окно «Читалка»: загрузка модели, голоса, текст, синтез."""

    sig_status = pyqtSignal(str)
    sig_progress = pyqtSignal(float)
    sig_elapsed = pyqtSignal(str)
    sig_enable = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Читалка — XTTS-v2")
        self.resize(1080, 860)
        self.setMinimumSize(860, 620)

        self.manager = ModelManager()
        self.is_working: bool = False
        self.stop_flag: bool = False

        self._loader_thread: Optional[ModelLoader] = None
        self._switch_thread: Optional[DeviceSwitcher] = None
        self._worker_thread: Optional[QWidget] = None

        self.sig_status.connect(self._on_status)
        self.sig_progress.connect(self._on_progress)
        self.sig_elapsed.connect(self._on_elapsed)
        self.sig_enable.connect(self._on_enable)

        self._build_ui()
        self._refresh_voice_list()
        self._load_model_async()
        self._log_startup_info()

    # ================================================================ UI
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 14)
        root.setSpacing(16)

        # ---- Шапка -------------------------------------------------
        header = widgets.make_header(
            "Читалка",
            "Русский TTS с клонированием голоса на Coqui XTTS-v2",
        )
        root.addWidget(header)

        # ---- Прокручиваемая область секций --------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        pages_host = QWidget()
        pages_host.setStyleSheet("background: transparent;")
        pages_lay = QVBoxLayout(pages_host)
        pages_lay.setContentsMargins(0, 0, 0, 0)
        pages_lay.setSpacing(16)
        scroll.setWidget(pages_host)
        root.addWidget(scroll, stretch=1)

        pages_lay.addWidget(self._build_voice_section())
        pages_lay.addWidget(self._build_text_section())
        pages_lay.addWidget(self._build_output_section())
        pages_lay.addStretch(1)

        # ---- Прогресс + статус --------------------------------------
        prog_card, prog_body = widgets.make_section("Прогресс")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        prog_body.addWidget(self.progress)

        info_row = QHBoxLayout()
        info_row.setSpacing(10)
        self.lbl_status = QLabel("⏳ Загрузка модели…")
        self.lbl_status.setObjectName("StatusWorking")
        self.lbl_elapsed = QLabel("")
        self.lbl_elapsed.setObjectName("Muted")
        info_row.addWidget(self.lbl_status)
        info_row.addStretch(1)
        info_row.addWidget(self.lbl_elapsed)
        prog_body.addLayout(info_row)
        root.addWidget(prog_card)

        hint = widgets.muted(
            "Лицензия XTTS-v2 (CPML) допускает только некоммерческое использование."
        )
        root.addWidget(hint)

        self._collect_action_buttons()

        # Плавное появление секций при старте
        for i, w in enumerate((prog_card,)):
            theme.fade_in(w, duration=350 + i * 100)

    # ------------------------------------------------------- секция «Голос»
    def _build_voice_section(self) -> QWidget:
        card, body = widgets.make_section("Голос")

        row = QHBoxLayout()
        row.setSpacing(10)

        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(240)
        row.addWidget(QLabel("Текущий:"))
        row.addWidget(self.voice_combo)

        self.btn_add_voice = QPushButton("➕  Добавить голос")
        self.btn_open_voices = QPushButton("📁  Папка голосов")
        self.btn_train = QPushButton("🎓  Обучить голос")
        row.addWidget(self.btn_add_voice)
        row.addWidget(self.btn_open_voices)
        row.addWidget(self.btn_train)
        row.addStretch(1)
        body.addLayout(row)

        row2 = QHBoxLayout()
        row2.setSpacing(16)
        self.chk_cleanup = QCheckBox("✨ Улучшение звука (фильтры, noise-gate)")
        self.chk_cleanup.setChecked(True)

        self.chk_gpu = QCheckBox("🚀 GPU (CUDA)")
        self.chk_gpu.setChecked(default_use_gpu(self.manager.cuda))
        self.chk_gpu.setEnabled(self.manager.cuda["available"])

        gpu_text = (
            f"{self.manager.cuda['name']} · {self.manager.cuda['vram_gb']:.0f} ГБ VRAM"
            if self.manager.cuda["available"] else "CUDA не обнаружена — работаем на CPU"
        )
        row2.addWidget(self.chk_cleanup)
        row2.addWidget(self.chk_gpu)
        row2.addWidget(widgets.muted(gpu_text))
        row2.addStretch(1)
        body.addLayout(row2)

        self.btn_add_voice.clicked.connect(self.add_voice)
        self.btn_open_voices.clicked.connect(lambda: open_local_path(config.VOICES_DIR))
        self.btn_train.clicked.connect(self.train_voice)
        self.chk_gpu.toggled.connect(self._on_gpu_toggle)

        theme.fade_in(card)
        return card

    # ------------------------------------------------------- секция «Текст»
    def _build_text_section(self) -> QWidget:
        card, body = widgets.make_section("Текст")

        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)
        self.btn_open = QPushButton("📄  Открыть .txt")
        toolbar.addWidget(self.btn_open)
        toolbar.addStretch(1)
        self.lbl_charcount = widgets.muted("0 символов")
        toolbar.addWidget(self.lbl_charcount)
        body.addLayout(toolbar)

        self.text_area = QPlainTextEdit()
        self.text_area.setFont(QFont(theme.FONT_FAMILY, 12))
        self.text_area.setPlaceholderText(
            "Вставьте или наберите текст для озвучивания…\n\n"
            "Текст автоматически режется на фразы по границам предложений "
            "(razdel), короткие предложения склеиваются, между фрагментами "
            "вставляются паузы 150 мс."
        )
        self.text_area.setMinimumHeight(260)
        body.addWidget(self.text_area, stretch=1)

        self.btn_open.clicked.connect(self.open_file)
        self.text_area.textChanged.connect(self._update_charcount)

        theme.fade_in(card, duration=550)
        return card

    # ------------------------------------------------------- секция «Результат»
    def _build_output_section(self) -> QWidget:
        card, body = widgets.make_section("Результат")

        row = QHBoxLayout()
        row.setSpacing(10)

        row.addWidget(QLabel("Формат:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["MP3", "WAV"])
        self.format_combo.setCurrentText("MP3")
        self.format_combo.setFixedWidth(90)
        row.addWidget(self.format_combo)

        self.btn_synth = QPushButton("🎧  Озвучить в файл")
        self.btn_synth.setObjectName("PrimaryButton")
        self.btn_stop = QPushButton("⏹  Стоп")
        self.btn_stop.setObjectName("DangerButton")
        self.btn_stop.setEnabled(False)
        self.btn_open_out = QPushButton("🗂  Открыть output")

        row.addWidget(self.btn_synth)
        row.addWidget(self.btn_stop)
        row.addWidget(self.btn_open_out)
        row.addStretch(1)
        body.addLayout(row)

        self.btn_synth.clicked.connect(self.synthesize_to_file)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_open_out.clicked.connect(lambda: open_local_path(config.OUTPUT_DIR))

        theme.fade_in(card, duration=650)
        return card

    # ------------------------------------------------------- кнопки действий
    def _collect_action_buttons(self) -> None:
        self._action_buttons = [
            self.btn_open, self.btn_synth, self.btn_train,
            self.btn_open_out, self.btn_add_voice, self.btn_open_voices,
        ]

    # ============================================================ голоса
    def _refresh_voice_list(self) -> None:
        current = self.voice_combo.currentText()
        self.voice_combo.clear()
        for d in voices.list_voice_dirs():
            self.voice_combo.addItem(d.name)
        if current and self.voice_combo.findText(current) >= 0:
            self.voice_combo.setCurrentText(current)
        logger.debug("Список голосов: %s",
                     [d.name for d in voices.list_voice_dirs()] or "— пусто —")

    def _current_voice_dir(self) -> Optional[Path]:
        return voices.resolve_voice_dir(self.voice_combo.currentText())

    def add_voice(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите .wav-записи одного голоса (можно несколько)",
            str(config.BASE_DIR),
            "Аудио (*.wav *.mp3 *.flac *.ogg);;Все файлы (*.*)",
        )
        if not files:
            return

        name, ok = QInputDialog.getText(
            self, "Имя голоса", "Название папки голоса:"
        )
        if not ok or not name.strip():
            return

        try:
            safe, added = voices.add_voice(name, files)
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", str(e))
            return

        self._refresh_voice_list()
        idx = self.voice_combo.findText(safe)
        if idx >= 0:
            self.voice_combo.setCurrentIndex(idx)
        self.sig_status.emit(f"➕ Голос «{safe}»: {added} файл(ов)")

    # ============================================================ модель
    def _load_model_async(self) -> None:
        want_gpu = self.chk_gpu.isChecked()
        self._loader_thread = ModelLoader(self.manager, want_gpu)
        self._loader_thread.status_changed.connect(self.sig_status.emit)
        self._loader_thread.loaded.connect(self._on_model_loaded)
        self._loader_thread.failed.connect(self._on_model_failed)
        self._loader_thread.start()

    def _on_model_loaded(self, sr: int, device: str) -> None:
        self.chk_gpu.blockSignals(True)
        self.chk_gpu.setChecked(device == "cuda")
        self.chk_gpu.blockSignals(False)
        self.sig_status.emit(f"✅ XTTS-v2 готов ({device.upper()}, {sr} Гц)")
        self.sig_enable.emit(True)

    def _on_model_failed(self, err: str) -> None:
        self.sig_status.emit(f"❌ Ошибка загрузки модели: {err}")
        QMessageBox.critical(self, "Ошибка загрузки модели", err)

    # ============================================================ GPU toggle
    def _on_gpu_toggle(self, checked: bool) -> None:
        if self.is_working:
            logger.warning("Переключение устройства невозможно: идёт работа")
            self.chk_gpu.blockSignals(True)
            self.chk_gpu.setChecked(self.manager.device == "cuda")
            self.chk_gpu.blockSignals(False)
            return
        target = "cuda" if (checked and self.manager.cuda["available"]) else "cpu"
        if target == self.manager.device or not self.manager.ready:
            return

        self.is_working = True
        self.sig_enable.emit(False)
        self.sig_status.emit(f"🔄 Переключение на {target.upper()}…")

        self._switch_thread = DeviceSwitcher(self.manager, target)
        self._switch_thread.status_changed.connect(self.sig_status.emit)
        self._switch_thread.done.connect(self._on_device_switched)
        self._switch_thread.failed.connect(self._on_device_switch_failed)
        self._switch_thread.finished.connect(self._on_worker_done)
        self._switch_thread.start()

    def _on_device_switched(self, device: str) -> None:
        self.chk_gpu.blockSignals(True)
        self.chk_gpu.setChecked(device == "cuda")
        self.chk_gpu.blockSignals(False)
        self.sig_status.emit(f"✅ Готово ({device.upper()})")

    def _on_device_switch_failed(self, err: str) -> None:
        logger.warning("Переключение устройства: %s", err)
        self.sig_status.emit(f"⚠ {err}")
        self.chk_gpu.blockSignals(True)
        self.chk_gpu.setChecked(False)
        self.chk_gpu.blockSignals(False)

    # ============================================================ обучение
    def train_voice(self) -> None:
        if not self.manager.ready:
            self.sig_status.emit("⚠ Модель ещё загружается")
            return
        if self.is_working:
            return
        voice_dir = self._current_voice_dir()
        if voice_dir is None:
            self.sig_status.emit("⚠ Выберите голос")
            return

        voices.invalidate_cache(voice_dir, self.manager.latents_cache)

        self.is_working = True
        self.stop_flag = False
        self.sig_enable.emit(False)
        self.sig_progress.emit(0.0)
        self.sig_status.emit("🎓 Обучение голоса…")

        self._worker_thread = TrainWorker(self.manager, voice_dir)
        self._worker_thread.status_changed.connect(self.sig_status.emit)
        self._worker_thread.finished_ok.connect(self._on_train_done)
        self._worker_thread.failed.connect(self._on_worker_error)
        self._worker_thread.finished.connect(self._on_worker_done)
        self._worker_thread.start()

    def _on_train_done(self, voice_name: str, msg: str) -> None:
        self.sig_progress.emit(100.0)
        self.sig_status.emit(f"✅ {msg}")
        QMessageBox.information(self, "Обучение завершено", msg)

    # ============================================================ синтез
    def synthesize_to_file(self) -> None:
        if not self.manager.ready:
            self.sig_status.emit("⚠ Модель ещё загружается")
            return
        if self.is_working:
            return

        voice_dir = self._current_voice_dir()
        if voice_dir is None:
            self.sig_status.emit("⚠ Выберите голос")
            return

        text = self.text_area.toPlainText().strip()
        if not text:
            logger.info("Пустой текст")
            self.sig_status.emit("⚠ Текст пуст — вставьте текст или откройте .txt")
            return

        fmt = self.format_combo.currentText().upper()
        ext = ".mp3" if fmt == "MP3" else ".wav"
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = config.OUTPUT_DIR / f"output_{ts}{ext}"

        cleanup = self.chk_cleanup.isChecked()

        self.is_working = True
        self.stop_flag = False
        self.sig_enable.emit(False)
        self.sig_progress.emit(0.0)
        self.sig_status.emit("🎧 Синтез…")
        self.sig_elapsed.emit("")

        logger.info(
            "Синтез: %d симв., голос=%s, формат=%s, очистка=%s, "
            "устройство=%s → %s (чанк ≤ %d)",
            len(text), voice_dir.name, fmt, cleanup,
            self.manager.device.upper(), out_path.name, self.manager.chunk_max,
        )

        self._worker_thread = SynthWorker(
            manager=self.manager,
            text=text,
            voice_dir=voice_dir,
            cleanup=cleanup,
            out_path=out_path,
            stop_check=lambda: self.stop_flag,
        )
        self._worker_thread.status_changed.connect(self.sig_status.emit)
        self._worker_thread.progress_changed.connect(self.sig_progress.emit)
        self._worker_thread.elapsed_changed.connect(self.sig_elapsed.emit)
        self._worker_thread.finished_ok.connect(self._on_synth_done)
        self._worker_thread.failed.connect(self._on_worker_error)
        self._worker_thread.finished.connect(self._on_worker_done)
        self._worker_thread.start()

    def _on_synth_done(self, out_name: str, dur: float, elapsed: float) -> None:
        rtf = dur / elapsed if elapsed > 0 else 0.0
        self.sig_progress.emit(100.0)
        self.sig_status.emit(f"✅ Готово: {out_name} ({dur:.0f} с)")
        self.sig_elapsed.emit(f"⏱ {elapsed:.0f} с, ×{rtf:.1f} realtime")

    # ============================================================ worker lifecycle
    def _on_worker_done(self) -> None:
        self.is_working = False
        self.sig_enable.emit(True)

    def _on_worker_error(self, err: str) -> None:
        logger.error("Worker error: %s", err)
        self.sig_status.emit(f"❌ {err}")
        if self.manager.ready and self.manager.device == "cuda" \
                and "out of memory" in err.lower():
            self.manager.move_to_cpu()
            self.chk_gpu.blockSignals(True)
            self.chk_gpu.setChecked(False)
            self.chk_gpu.blockSignals(False)
            self.sig_status.emit("⚠ OOM: откатились на CPU, попробуйте ещё раз")

    # ============================================================ Stop
    def stop(self) -> None:
        if not self.is_working:
            return
        logger.info("Запрошена остановка")
        self.stop_flag = True
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self.sig_status.emit("⏹ Остановка…")

    # ============================================================ файл
    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть текст", str(config.BASE_DIR),
            "Текстовые файлы (*.txt);;Все файлы (*.*)",
        )
        if not path:
            return
        content = ""
        for enc in ("utf-8", "cp1251", "koi8-r"):
            try:
                with open(path, "r", encoding=enc) as f:
                    content = f.read()
                logger.info("Открыт файл: %s (%s, %d симв.)", path, enc, len(content))
                break
            except UnicodeDecodeError:
                continue
        if not content:
            logger.error("Не удалось определить кодировку: %s", path)
            QMessageBox.warning(self, "Ошибка", "Не удалось прочитать файл.")
            return
        self.text_area.setPlainText(content)
        self.sig_status.emit(f"📄 {Path(path).name}")

    def _update_charcount(self) -> None:
        n = len(self.text_area.toPlainText().strip())
        self.lbl_charcount.setText(f"{n:,} символов".replace(",", " "))

    # ============================================================ слоты
    def _on_status(self, msg: str) -> None:
        self.lbl_status.setText(msg)
        if msg.startswith("✅"):
            self.lbl_status.setObjectName("StatusReady")
        elif msg.startswith(("❌", "⚠")):
            self.lbl_status.setObjectName("StatusError")
        else:
            self.lbl_status.setObjectName("StatusWorking")
        # пере-применение стиля после смены objectName
        style = self.style()
        style.unpolish(self.lbl_status)
        style.polish(self.lbl_status)

    def _on_progress(self, value: float) -> None:
        v = int(max(0.0, min(100.0, float(value))))
        self.progress.setValue(v)

    def _on_elapsed(self, text: str) -> None:
        self.lbl_elapsed.setText(text)

    def _on_enable(self, enabled: bool) -> None:
        for b in self._action_buttons:
            b.setEnabled(enabled)
        self.btn_stop.setEnabled(not enabled and self.is_working)
        self.chk_gpu.setEnabled(enabled and self.manager.cuda["available"])

    # ============================================================ логи при старте
    def _log_startup_info(self) -> None:
        logger.info("Приложение запущено.")
        logger.info("Логических ядер CPU: %d", config.CPU_COUNT)
        if self.manager.cuda["available"]:
            logger.info("GPU: %s, VRAM: %.1f ГБ (порог FP16: %.1f ГБ)",
                        self.manager.cuda["name"], self.manager.cuda["vram_gb"],
                        config.FP16_VRAM_THRESHOLD_GB)
        else:
            logger.info("GPU: недоступна, работаем на CPU")
        logger.info("Токенизация: %s", describe_splitter())
        logger.info("Голоса (папки): %s", config.VOICES_DIR)
        logger.info("Кэш латентов:   %s", config.CACHE_DIR)
        logger.info("Вывод:          %s", config.OUTPUT_DIR)

    # ============================================================ close
    def closeEvent(self, a0: Optional[QCloseEvent]) -> None:  # noqa: N802
        self.stop_flag = True
        for th in (self._worker_thread, self._switch_thread, self._loader_thread):
            if th is not None and getattr(th, "isRunning", lambda: False)():
                th.wait(2000)
        if a0 is not None:
            a0.accept()
