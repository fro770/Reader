"""Общие строительные блоки интерфейса: карточки-секции, заголовки."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from reader.ui import theme


def make_section(title: str) -> tuple[QFrame, QVBoxLayout]:
    """Карточка-секция с заголовком. Возвращает (frame, layout_for_content)."""
    frame = QFrame()
    frame.setObjectName("SectionCard")
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(18, 14, 18, 16)
    outer.setSpacing(10)

    label = QLabel(title.upper())
    label.setObjectName("SectionTitle")
    outer.addWidget(label)

    body = QVBoxLayout()
    body.setSpacing(8)
    outer.addLayout(body)
    return frame, body


def make_header(title: str, subtitle: str) -> QFrame:
    """Верхняя шапка приложения."""
    frame = QFrame()
    lay = QHBoxLayout(frame)
    lay.setContentsMargins(4, 0, 4, 0)
    lay.setSpacing(12)

    icon = QLabel("🎙")
    icon.setStyleSheet("font-size: 30px; background: transparent;")
    lay.addWidget(icon)

    col = QVBoxLayout()
    col.setSpacing(1)
    t = QLabel(title)
    t.setObjectName("AppTitle")
    s = QLabel(subtitle)
    s.setObjectName("AppSubtitle")
    col.addWidget(t)
    col.addWidget(s)
    lay.addLayout(col)
    lay.addStretch(1)
    return frame


def muted(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("Muted")
    return lbl


def hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f"background-color: {theme.BORDER}; border: none;")
    line.setFixedHeight(1)
    return line


class ClickableRow(QWidget):
    """Строка 'label: widget …' — выравнивание подписей в секциях."""

    def __init__(self, label_text: str, widget: QWidget,
                 stretch_after: bool = True) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lbl = QLabel(label_text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(lbl)
        lay.addWidget(widget)
        if stretch_after:
            lay.addStretch(1)
