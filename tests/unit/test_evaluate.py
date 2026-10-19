"""Логика зачёта ответа в scripts/evaluate.py."""

from scripts.evaluate import Outcome

ITEM = {
    "id": "exact_buyer",
    "kind": "exact",
    "expect_found": True,
    "expect_pages": [72],
    "expect_section": "13.2 Реквизиты Покупателя",
    "expect_contains": "Северный Терминал",
    "also_found_in": [{"pages": [1], "section": "Вводная часть"}],
}


def outcome(pages, sections, item=ITEM, answer="ООО «Северный Терминал»"):
    return Outcome(
        item=item,
        found=True,
        answer=answer,
        pages=pages,
        sections=sections,
        reranked_pages=[p for p, _ in pages],
        latency_ms=1,
    )


def test_answer_cited_from_main_location_is_correct():
    assert outcome([(72, 72)], ["13.2 Реквизиты Покупателя"]).correct


def test_answer_cited_from_alternative_location_is_correct():
    result = outcome([(1, 1)], ["Вводная часть"])

    assert result.page_hit and result.section_hit and result.retrieval_hit and result.correct


def test_alternative_page_with_wrong_section_is_not_correct():
    assert not outcome([(1, 1)], ["9.3 Подсудность"]).correct


def test_item_without_alternatives_keeps_strict_location():
    item = {k: v for k, v in ITEM.items() if k != "also_found_in"}

    assert not outcome([(1, 1)], ["Вводная часть"], item=item).correct
