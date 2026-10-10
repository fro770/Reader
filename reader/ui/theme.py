"""Тёмная тема приложения: палитра, QSS-стили и анимации переходов."""
from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, pyqtProperty
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QWidget

# ---------------------------------------------------------------- Палитра
ACCENT = "#7c5cff"          # основной акцент (фиолетовый)
ACCENT_HOVER = "#9070ff"
ACCENT_DOWN = "#6a4de0"
BG = "#12141c"              # фон окна
SURFACE = "#1a1e2a"         # фон карточек/секций
SURFACE_ALT = "#232838"     # поля ввода
TEXT = "#e8eaf2"
TEXT_MUTED = "#8a90a6"
BORDER = "#2b3144"
SUCCESS = "#3ecf8e"
DANGER = "#ff5c6c"

FONT_FAMILY = "Segoe UI"


def build_qss() -> str:
    """Глобальная таблица стилей приложения."""
    return f"""
    QMainWindow, QWidget {{
        background-color: {BG};
        color: {TEXT};
        font-family: "{FONT_FAMILY}", "Inter", "Noto Sans", sans-serif;
        font-size: 14px;
    }}

    /* ---------- Секции-карточки ---------- */
    QFrame#SectionCard {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
    }}
    QLabel#SectionTitle {{
        color: {TEXT_MUTED};
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 2px;
        padding: 0px;
    }}

    /* ---------- Заголовок ---------- */
    QLabel#AppTitle {{
        font-size: 22px;
        font-weight: 700;
        color: {TEXT};
        background: transparent;
    }}
    QLabel#AppSubtitle {{
        font-size: 13px;
        color: {TEXT_MUTED};
        background: transparent;
    }}

    /* ---------- Кнопки ---------- */
    QPushButton {{
        background-color: {SURFACE_ALT};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 9px 16px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: #2c3346;
        border-color: {ACCENT};
    }}
    QPushButton:pressed {{
        background-color: {ACCENT_DOWN};
    }}
    QPushButton:disabled {{
        color: {TEXT_MUTED};
        background-color: {SURFACE};
        border-color: {BORDER};
    }}
    QPushButton#PrimaryButton {{
        background-color: {ACCENT};
        border: none;
        color: white;
        font-weight: 700;
        padding: 10px 22px;
    }}
    QPushButton#PrimaryButton:hover {{
        background-color: {ACCENT_HOVER};
    }}
    QPushButton#PrimaryButton:pressed {{
        background-color: {ACCENT_DOWN};
    }}
    QPushButton#PrimaryButton:disabled {{
        background-color: {SURFACE_ALT};
        color: {TEXT_MUTED};
    }}
    QPushButton#DangerButton {{
        background-color: transparent;
        border: 1px solid {DANGER};
        color: {DANGER};
    }}
    QPushButton#DangerButton:hover {{
        background-color: {DANGER};
        color: white;
    }}
    QPushButton#DangerButton:disabled {{
        border-color: {BORDER};
        color: {TEXT_MUTED};
        background: transparent;
    }}

    /* ---------- Поля и списки ---------- */
    QPlainTextEdit {{
        background-color: {SURFACE_ALT};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 10px;
        font-size: 15px;
        color: {TEXT};
    }}
    QPlainTextEdit:focus {{
        border: 1px solid {ACCENT};
    }}
    QComboBox {{
        background-color: {SURFACE_ALT};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 7px 12px;
        color: {TEXT};
    }}
    QComboBox:hover {{ border-color: {ACCENT}; }}
    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox QAbstractItemView {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        selection-background-color: {ACCENT};
        selection-color: white;
        outline: none;
    }}

    /* ---------- Чекбоксы ---------- */
    QCheckBox {{
        color: {TEXT};
        spacing: 8px;
        font-weight: 500;
    }}
    QCheckBox::indicator {{
        width: 18px; height: 18px;
        border: 2px solid {BORDER};
        border-radius: 6px;
        background-color: {SURFACE_ALT};
    }}
    QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
    QCheckBox::indicator:checked {{
        background-color: {ACCENT};
        border-color: {ACCENT};
    }}
    QCheckBox:disabled {{ color: {TEXT_MUTED}; }}

    /* ---------- Прогресс ---------- */
    QProgressBar {{
        background-color: {SURFACE_ALT};
        border: none;
        border-radius: 8px;
        height: 10px;
        text-align: center;
        color: transparent;
    }}
    QProgressBar::chunk {{
        border-radius: 8px;
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {ACCENT}, stop:1 {SUCCESS});
    }}

    /* ---------- Статус-бар ---------- */
    QLabel#StatusReady   {{ color: {SUCCESS}; font-weight: 600; }}
    QLabel#StatusWorking {{ color: {ACCENT_HOVER}; font-weight: 600; }}
    QLabel#StatusError   {{ color: {DANGER}; font-weight: 600; }}
    QLabel#StatusInfo    {{ color: {TEXT_MUTED}; }}
    QLabel#Muted {{ color: {TEXT_MUTED}; }}
    """


class FadeInWidget(QWidget):
    """Виджет с плавным появлением (анимация opacity при показе)."""

    def __init__(self, parent: QWidget | None = None,
                 duration: int = 450) -> None:
        super().__init__(parent)
        self._effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._effect)
        self._duration = duration
        self._opacity = 0.0

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        anim = QPropertyAnimation(self._effect, b"opacity", self)
        anim.setDuration(self._duration)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start()
        self._anim = anim  # держим ссылку до завершения

    def set_opacity(self, v: float) -> None:
        self._effect.setOpacity(v)

    def opacity(self) -> float:
        return self._effect.opacity()

    opacity_prop = pyqtProperty(float, fget=opacity, fset=set_opacity)


def fade_in(widget: QWidget, duration: int = 450) -> None:
    """Плавно проявить уже существующий виджет (одноразово)."""
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim.finished.connect(lambda: widget.setGraphicsEffect(None))
    anim.start()
    widget._fade_anim = anim  # noqa: SLF001 — удерживаем ссылку
