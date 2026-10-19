"""Поиск разделов документа по нумерованным заголовкам уровня 1 и 2.

Заголовок — отдельная строка вида «4. ОТВЕТСТВЕННОСТЬ СТОРОН» или «4.2 Штрафные санкции»:
номер из одного-двух уровней, затем короткое название с заглавной буквы без точки в конце.
Пункты «4.2.1. Поставщик обязан…», суммы и предложения заголовками не считаются.
"""

import re
from dataclasses import dataclass

INTRO_SECTION_TITLE = "Вводная часть"

HEADING_RE = re.compile(
    r"^\d{1,2}(?:\.\d{1,2})?\.?[ \t]+[А-ЯЁA-Z][^\n]{0,98}$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Section:
    title: str
    start: int
    end: int


def _heading_starts(text: str) -> list[tuple[int, str]]:
    return [
        (match.start(), match.group().strip())
        for match in HEADING_RE.finditer(text)
        if not match.group().rstrip().endswith(".")
    ]


def find_sections(text: str) -> list[Section]:
    """Разделы, покрывающие весь текст без пропусков и пересечений."""
    headings = _heading_starts(text)
    if not headings:
        return [Section(INTRO_SECTION_TITLE, 0, len(text))]

    starts = [(0, INTRO_SECTION_TITLE)] if text[: headings[0][0]].strip() else []
    starts += headings
    starts[0] = (0, starts[0][1])  # пробелы до первого заголовка относятся к нему

    ends = [start for start, _ in starts[1:]] + [len(text)]
    return [Section(title, start, end) for (start, title), end in zip(starts, ends, strict=True)]
