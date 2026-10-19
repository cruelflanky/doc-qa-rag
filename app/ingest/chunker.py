"""Разбиение документа на фрагменты: сначала по разделам, затем рекурсивно внутри раздела."""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingest.pdf import ParsedDocument, page_for_offset
from app.ingest.sections import find_sections


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    page_from: int
    page_to: int
    section: str


def chunk_document(doc: ParsedDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks: list[Chunk] = []
    for section in find_sections(doc.text):
        section_text = doc.text[section.start : section.end]
        for piece in splitter.create_documents([section_text]):
            start = section.start + piece.metadata["start_index"]
            end = start + len(piece.page_content)
            chunks.append(
                Chunk(
                    chunk_index=len(chunks),
                    text=piece.page_content,
                    char_start=start,
                    char_end=end,
                    page_from=page_for_offset(doc.page_offsets, start),
                    page_to=page_for_offset(doc.page_offsets, end - 1),
                    section=section.title,
                )
            )
    return chunks
