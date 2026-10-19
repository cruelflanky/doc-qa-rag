"""Проверки сгенерированных образцов (scripts/generate_samples.py) и эталонного набора."""

import json
import re
from pathlib import Path

import pytest

from app.ingest.pdf import load_pdf
from app.ingest.sections import INTRO_SECTION_TITLE, find_sections

SAMPLES = Path(__file__).resolve().parents[2] / "samples"
GOLDEN = json.loads((SAMPLES / "golden.json").read_text()) if SAMPLES.exists() else []
ABSENT_TERMS = ["страхов", "директор", "мирово"]

pytestmark = pytest.mark.skipif(not GOLDEN, reason="образцы не сгенерированы")


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


@pytest.fixture(scope="module")
def supply():
    return load_pdf(SAMPLES / "contract_supply.pdf", max_pages=100)


def page_texts(doc):
    bounds = [*doc.page_offsets, len(doc.text)]
    return [doc.text[bounds[i] : bounds[i + 1]] for i in range(doc.page_count)]


def test_supply_contract_page_count_in_range(supply):
    assert 60 <= supply.page_count <= 100


def test_golden_covers_all_kinds():
    kinds = {item["kind"] for item in GOLDEN}

    assert kinds == {"exact", "similar_numbers", "paraphrase", "cross_page", "absent", "isolation"}
    assert sum(item["kind"] == "absent" for item in GOLDEN) >= 3


def test_golden_facts_located_on_expected_pages(supply):
    pages = [normalized(p) for p in page_texts(supply)]
    for item in GOLDEN:
        if item["doc"] != "contract_supply.pdf" or not item["expect_found"]:
            continue
        assert item["expect_pages"], item["id"]
        value = normalized(item["expect_contains"])
        found_on = [n for n in item["expect_pages"] if value in pages[n - 1]]
        joined = " ".join(pages[n - 1] for n in item["expect_pages"])
        assert found_on or value in joined, item["id"]


def test_cross_page_fact_spans_two_pages():
    [item] = [i for i in GOLDEN if i["kind"] == "cross_page"]

    assert len(item["expect_pages"]) == 2
    assert item["expect_pages"][1] == item["expect_pages"][0] + 1


def test_absent_questions_terms_not_in_document(supply):
    text = normalized(supply.text)

    for term in ABSENT_TERMS:
        assert term not in text


def test_sections_detected_in_sample(supply):
    titles = [s.title for s in find_sections(supply.text)]

    assert "4.2 Штрафные санкции" in titles
    assert "7.3 Неустойка за просрочку оплаты" in titles
    assert len(titles) >= 36
    level = re.compile(r"^\d{1,2}(\.\d{1,2})?\.? ")
    assert all(t == INTRO_SECTION_TITLE or level.match(t) for t in titles)


def test_expected_sections_exist_in_document(supply):
    titles = {s.title for s in find_sections(supply.text)}

    for item in GOLDEN:
        if item["doc"] == "contract_supply.pdf" and item["expect_section"]:
            assert item["expect_section"] in titles, item["id"]


def test_short_contract_has_different_penalty():
    short = load_pdf(SAMPLES / "contract_short.pdf", max_pages=100)

    assert 3 <= short.page_count <= 5
    assert "0,3%" in short.text
    assert "0,1%" not in short.text


def test_alternative_locations_contain_the_value(supply):
    pages = [normalized(p) for p in page_texts(supply)]
    titles = {s.title for s in find_sections(supply.text)}
    buyer = next(i for i in GOLDEN if i["id"] == "exact_buyer")

    assert buyer["also_found_in"], "преамбула тоже называет Покупателя"
    for location in buyer["also_found_in"]:
        assert location["section"] in titles
        assert any(normalized(buyer["expect_contains"]) in pages[n - 1] for n in location["pages"])
