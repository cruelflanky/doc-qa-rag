"""Чтение PDF в один сплошной текст с картой начала страниц."""

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.errors import DocumentError

# Не "\n\n": сплиттер режет по нему в первую очередь, и факт на стыке страниц
# разорвался бы на два фрагмента.
PAGE_SEPARATOR = "\n"
PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True)
class ParsedDocument:
    text: str
    page_offsets: tuple[int, ...]  # page_offsets[i] — начало страницы i+1 в text
    page_count: int


def _open_pdf(path: Path) -> PdfReader:
    with path.open("rb") as file:
        if file.read(len(PDF_MAGIC)) != PDF_MAGIC:
            raise DocumentError("Файл не является корректным PDF")
    try:
        return PdfReader(path)
    except PdfReadError as exc:
        raise DocumentError("Файл не является корректным PDF") from exc


def load_pdf(path: Path, max_pages: int) -> ParsedDocument:
    reader = _open_pdf(path)
    page_count = len(reader.pages)
    if page_count > max_pages:
        raise DocumentError(f"Документ содержит {page_count} страниц, максимум {max_pages}")

    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    if not any(pages):
        raise DocumentError("В документе нет текстового слоя (возможно, это скан)")

    offsets: list[int] = []
    position = 0
    for page in pages:
        offsets.append(position)
        position += len(page) + len(PAGE_SEPARATOR)
    return ParsedDocument(
        text=PAGE_SEPARATOR.join(pages),
        page_offsets=tuple(offsets),
        page_count=len(pages),
    )


def page_for_offset(page_offsets: tuple[int, ...], offset: int) -> int:
    """Номер страницы (с 1), на которую попадает символ с данным смещением."""
    return max(1, bisect_right(page_offsets, offset))
