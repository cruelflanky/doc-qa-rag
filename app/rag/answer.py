"""Промпт для LLM и детерминированная постобработка её ответа.

Модель получает пронумерованные фрагменты и возвращает только номера тех, на которые опирается.
Текст, страница и раздел цитат берутся из найденных фрагментов, а не из ответа модели,
поэтому выдумать цитату модель не может.
"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.schemas import AnswerResponse, Citation, LLMAnswer
from app.store.models import Candidate

NOT_FOUND_MESSAGE = "В документе нет сведений, необходимых для ответа на этот вопрос."

SYSTEM_PROMPT = """Ты отвечаешь на вопросы по документу, используя только приведённые фрагменты.

Правила:
1. Используй только сведения из фрагментов. Общие знания и догадки запрещены.
2. Каждое утверждение ответа должно опираться на фрагменты; укажи их номера в used_fragments.
3. Если во фрагментах нет сведений для ответа, верни found=false, пустой answer
   и пустой used_fragments. Не пытайся ответить частично или «примерно».
4. Точные обозначения (номера пунктов, суммы, даты, проценты, наименования) переноси дословно.
5. Отвечай на русском, кратко и по существу."""


def _format_fragment(number: int, fragment: Candidate) -> str:
    header = (
        f"[{number}] (стр. {fragment.page_from}–{fragment.page_to}, раздел «{fragment.section}»)"
    )
    return f"{header}\n{fragment.text}"


def build_messages(question: str, fragments: list[Candidate]) -> list[BaseMessage]:
    numbered = "\n\n".join(_format_fragment(i, f) for i, f in enumerate(fragments, start=1))
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Фрагменты документа:\n\n{numbered}\n\nВопрос: {question}"),
    ]


def not_found_response() -> AnswerResponse:
    return AnswerResponse(answer=NOT_FOUND_MESSAGE, found=False, citations=[])


def _valid_numbers(numbers: list[int], fragment_count: int) -> list[int]:
    """Номера в диапазоне 1..fragment_count без повторов, в исходном порядке."""
    return list(dict.fromkeys(n for n in numbers if 1 <= n <= fragment_count))


def _citation(fragment: Candidate) -> Citation:
    return Citation(
        text=fragment.text,
        page=fragment.page_from,
        page_to=fragment.page_to,
        section=fragment.section,
        chunk_index=fragment.chunk_index,
    )


def finalize(llm_answer: LLMAnswer, fragments: list[Candidate]) -> AnswerResponse:
    if not llm_answer.found:
        return not_found_response()
    numbers = _valid_numbers(llm_answer.used_fragments, len(fragments))
    if not numbers or not llm_answer.answer.strip():
        return not_found_response()
    return AnswerResponse(
        answer=llm_answer.answer.strip(),
        found=True,
        citations=[_citation(fragments[n - 1]) for n in numbers],
    )
