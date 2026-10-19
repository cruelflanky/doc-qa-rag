from app.ingest.chunker import chunk_document
from app.ingest.pdf import ParsedDocument
from app.ingest.sections import find_sections

SENTENCE = "Поставщик обязан передать товар Покупателю в согласованный срок. "


def build_doc() -> ParsedDocument:
    page1 = "1. ПРЕДМЕТ ДОГОВОРА\n" + SENTENCE * 8 + "\n2. ЦЕНА ДОГОВОРА\n" + SENTENCE * 3
    page2 = SENTENCE * 3 + "\n2.1 Порядок оплаты\n" + SENTENCE * 6
    text = page1 + "\n\n" + page2
    return ParsedDocument(text=text, page_offsets=(0, len(page1) + 2), page_count=2)


def test_chunk_offsets_match_text():
    doc = build_doc()

    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=30)

    assert chunks
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text


def test_chunk_never_crosses_section():
    doc = build_doc()
    sections = find_sections(doc.text)

    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=30)

    for chunk in chunks:
        owner = next(s for s in sections if s.start <= chunk.char_start < s.end)
        assert chunk.char_end <= owner.end
        assert chunk.section == owner.title


def test_chunk_spanning_page_break_has_page_range():
    doc = build_doc()

    chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=100)

    section_2 = [c for c in chunks if c.section == "2. ЦЕНА ДОГОВОРА"]
    assert len(section_2) == 1
    assert (section_2[0].page_from, section_2[0].page_to) == (1, 2)


def test_chunk_indexes_are_sequential():
    chunks = chunk_document(build_doc(), chunk_size=200, chunk_overlap=30)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
