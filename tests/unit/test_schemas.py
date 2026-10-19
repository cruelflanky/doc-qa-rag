import pytest
from pydantic import ValidationError

from app.schemas import AnswerResponse, AskRequest, Citation

TZ_EXAMPLE = {
    "answer": "Штраф за просрочку поставки составляет 0,1% от стоимости товара "
    "за каждый день просрочки.",
    "found": True,
    "citations": [
        {
            "text": "За нарушение сроков поставки Поставщик уплачивает Покупателю неустойку "
            "в размере 0,1% ...",
            "page": 12,
            "page_to": 12,
            "section": "4.2 Штрафные санкции",
            "chunk_index": 41,
        }
    ],
}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def test_answer_response_matches_tz_example_shape():
    response = AnswerResponse.model_validate(TZ_EXAMPLE)

    dumped = response.model_dump(exclude_none=True)
    assert set(dumped) == {"answer", "found", "citations"}
    assert dumped["citations"][0]["page"] == 12
    assert dumped["citations"][0]["section"] == "4.2 Штрафные санкции"


def test_answer_response_found_false_requires_no_citations():
    citation = Citation(text="t", page=1, page_to=1, section="s", chunk_index=0)

    with pytest.raises(ValidationError):
        AnswerResponse(answer="нет", found=False, citations=[citation])


def test_answer_response_found_true_requires_citations():
    with pytest.raises(ValidationError):
        AnswerResponse(answer="да", found=True, citations=[])


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_ask_request_rejects_blank_and_too_long_question(question):
    with pytest.raises(ValidationError):
        AskRequest(question=question)


def test_ask_request_strips_question_and_defaults_debug_false():
    request = AskRequest(question="  Какой штраф?  ")

    assert request.question == "Какой штраф?"
    assert request.debug is False
