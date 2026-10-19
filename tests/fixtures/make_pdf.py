"""Генерация маленьких PDF для тестов: каждая строка текста рисуется отдельной строкой."""

import os
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONT_NAME = "TestCyrillic"
FONT_CANDIDATES = (
    os.environ.get("SAMPLE_FONT", ""),
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _register_font() -> str:
    if FONT_NAME in pdfmetrics.getRegisteredFontNames():
        return FONT_NAME
    for candidate in FONT_CANDIDATES:
        if candidate and Path(candidate).exists():
            pdfmetrics.registerFont(TTFont(FONT_NAME, candidate))
            return FONT_NAME
    raise RuntimeError("Не найден TTF-шрифт с кириллицей, задайте SAMPLE_FONT")


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Создаёт PDF: одна строка списка pages = одна страница; пустая строка = пустая страница."""
    font = _register_font()
    pdf = canvas.Canvas(str(path), pagesize=A4)
    _, height = A4
    for page_text in pages:
        y = height - 60
        pdf.setFont(font, 11)
        for line in page_text.splitlines():
            pdf.drawString(50, y, line)
            y -= 16
        pdf.showPage()
    pdf.save()
    return path
