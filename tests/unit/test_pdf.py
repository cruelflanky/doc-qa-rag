import pytest

from app.errors import DocumentError
from app.ingest.pdf import load_pdf, page_for_offset
from tests.fixtures.make_pdf import make_pdf


def test_load_pdf_builds_page_offsets(tmp_path):
    path = make_pdf(tmp_path / "doc.pdf", ["Первая страница", "Вторая страница", "Третья"])

    doc = load_pdf(path, max_pages=10)

    assert doc.page_count == 3
    assert len(doc.page_offsets) == 3
    assert doc.page_offsets[0] == 0
    assert doc.text[doc.page_offsets[1]:].startswith("Вторая страница")
    assert doc.text[doc.page_offsets[2]:].startswith("Третья")


def test_load_pdf_rejects_too_many_pages(tmp_path):
    path = make_pdf(tmp_path / "doc.pdf", ["a", "b", "c"])

    with pytest.raises(DocumentError, match="максимум 2"):
        load_pdf(path, max_pages=2)


def test_load_pdf_rejects_pdf_without_text(tmp_path):
    path = make_pdf(tmp_path / "scan.pdf", ["", ""])

    with pytest.raises(DocumentError, match="нет текстового слоя"):
        load_pdf(path, max_pages=10)


def test_load_pdf_rejects_non_pdf(tmp_path):
    path = tmp_path / "fake.pdf"
    path.write_text("это не pdf")

    with pytest.raises(DocumentError, match="не является корректным PDF"):
        load_pdf(path, max_pages=10)


def test_page_for_offset_boundaries():
    offsets = (0, 100, 250)

    assert page_for_offset(offsets, 0) == 1
    assert page_for_offset(offsets, 99) == 1
    assert page_for_offset(offsets, 100) == 2
    assert page_for_offset(offsets, 249) == 2
    assert page_for_offset(offsets, 10_000) == 3
