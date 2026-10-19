from app.rag.answer import NOT_FOUND_MESSAGE, build_messages, finalize
from app.schemas import LLMAnswer
from app.store.models import Candidate


def candidate(i: int, page: int = 1) -> Candidate:
    return Candidate(
        point_id=f"p{i}",
        chunk_index=i,
        text=f"Текст фрагмента {i}",
        page_from=page,
        page_to=page + 1,
        section=f"{i}. РАЗДЕЛ",
        fused_score=0.5,
        fused_rank=i + 1,
    )


FRAGMENTS = [candidate(10, page=3), candidate(20, page=7), candidate(30, page=9)]


def test_finalize_builds_citations_from_fragments_not_llm():
    llm = LLMAnswer(found=True, answer="Ответ", used_fragments=[2])

    response = finalize(llm, FRAGMENTS)

    assert response.found is True
    assert response.answer == "Ответ"
    [citation] = response.citations
    assert citation.text == "Текст фрагмента 20"
    assert (citation.page, citation.page_to) == (7, 8)
    assert citation.section == "20. РАЗДЕЛ"
    assert citation.chunk_index == 20


def test_finalize_drops_out_of_range_and_duplicate_ids():
    llm = LLMAnswer(found=True, answer="Ответ", used_fragments=[2, 9, 2, 0, 1])

    response = finalize(llm, FRAGMENTS)

    assert [c.chunk_index for c in response.citations] == [20, 10]


def test_finalize_found_without_valid_ids_becomes_not_found():
    llm = LLMAnswer(found=True, answer="Придуманный ответ", used_fragments=[7])

    response = finalize(llm, FRAGMENTS)

    assert response.found is False
    assert response.citations == []
    assert response.answer == NOT_FOUND_MESSAGE


def test_finalize_not_found_ignores_ids():
    llm = LLMAnswer(found=False, answer="Не нашёл", used_fragments=[1, 2])

    response = finalize(llm, FRAGMENTS)

    assert response.found is False
    assert response.citations == []
    assert response.answer == NOT_FOUND_MESSAGE


def test_build_messages_numbers_fragments_with_page_and_section():
    messages = build_messages("Какой штраф?", FRAGMENTS)

    user_text = messages[-1].content
    assert "[1] (стр. 3–4, раздел «10. РАЗДЕЛ»)\nТекст фрагмента 10" in user_text
    assert "[3] (стр. 9–10, раздел «30. РАЗДЕЛ»)" in user_text
    assert "Какой штраф?" in user_text
    assert "только" in messages[0].content.lower()
