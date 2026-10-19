"""Генерация синтетических договоров и эталонного набора вопросов.

Запуск: uv run python scripts/generate_samples.py
Результат: samples/contract_supply.pdf (~70 стр.), samples/contract_short.pdf, samples/golden.json.

Генерация детерминирована (random.Random(42)). Страницы ответов в golden.json не задаются руками,
а вычисляются поиском текста факта в извлечённом из PDF тексте.
"""

import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable, Paragraph, SimpleDocTemplate, Spacer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sample_content import (  # noqa: E402
    FACTS,
    FILLER_TEMPLATES,
    PREAMBLE,
    SHORT_CONTRACT,
    SUPPLY_SECTIONS,
    Fact,
)

OUT = Path(__file__).resolve().parents[1] / "samples"
FONT = "SampleFont"
FONT_CANDIDATES = (
    os.environ.get("SAMPLE_FONT", ""),
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
CLAUSES_PER_SUBSECTION = (16, 20)
CROSS_PAGE_RESERVE_PT = 32  # сколько места оставить на странице перед фактом «на стыке»


class FillToNearPageEnd(Flowable):
    """Занимает место так, чтобы на странице осталось reserve пунктов: следующий абзац порвётся."""

    def __init__(self, reserve: float) -> None:
        super().__init__()
        self.reserve = reserve

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        return avail_width, max(0.0, avail_height - self.reserve)

    def draw(self) -> None:
        pass


@dataclass(frozen=True)
class Styles:
    title: ParagraphStyle
    heading1: ParagraphStyle
    heading2: ParagraphStyle
    body: ParagraphStyle


def register_font() -> None:
    for candidate in FONT_CANDIDATES:
        if candidate and Path(candidate).exists():
            pdfmetrics.registerFont(TTFont(FONT, candidate))
            return
    raise SystemExit("Не найден TTF-шрифт с кириллицей, задайте SAMPLE_FONT")


def make_styles() -> Styles:
    body = ParagraphStyle("body", fontName=FONT, fontSize=10.5, leading=14, spaceAfter=4)
    return Styles(
        title=ParagraphStyle("title", parent=body, fontSize=14, leading=18, alignment=1),
        heading1=ParagraphStyle("h1", parent=body, fontSize=12, leading=16, spaceBefore=10),
        heading2=ParagraphStyle("h2", parent=body, fontSize=11, leading=15, spaceBefore=6),
        body=body,
    )


def filler_clause(rng: random.Random) -> str:
    """Пункт из двух разных шаблонных предложений."""
    first, second = rng.sample(FILLER_TEMPLATES, 2)
    return " ".join(
        t.format(days=rng.randint(3, 30), copies=rng.randint(2, 4)) for t in (first, second)
    )


def subsection_story(
    number: str, clauses: list[str], facts: list[Fact], styles: Styles
) -> list[Flowable]:
    story: list[Flowable] = []
    for index, clause in enumerate(clauses, start=1):
        fact = next((f for f in facts if f.clause == f"{number}.{index}"), None)
        if fact and fact.cross_page:
            story.append(FillToNearPageEnd(CROSS_PAGE_RESERVE_PT))
        text = fact.text if fact else clause
        story.append(Paragraph(f"{number}.{index}. {text}", styles.body))
    return story


def contract_story(
    title: str, preamble: str, sections, facts: list[Fact], rng: random.Random, clauses_range
) -> list[Flowable]:
    styles = make_styles()
    story: list[Flowable] = [Paragraph(title, styles.title), Spacer(1, 0.4 * cm)]
    story.append(Paragraph(preamble, styles.body))
    for number, name, subsections in sections:
        story.append(Paragraph(f"{number}. {name}", styles.heading1))
        for sub_number, sub_name in enumerate(subsections, start=1):
            full = f"{number}.{sub_number}"
            story.append(Paragraph(f"{full} {sub_name}", styles.heading2))
            count = rng.randint(*clauses_range)
            clauses = [filler_clause(rng) for _ in range(count)]
            story.extend(subsection_story(full, clauses, facts, styles))
    return story


def build_pdf(path: Path, story: list[Flowable]) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=path.stem,
    )
    doc.build(story)


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def locate_pages(path: Path, needle: str) -> list[int]:
    """Страницы, на которых лежит текст needle (он может переходить через границу страниц)."""
    pages = [normalized(page.extract_text() or "") for page in PdfReader(path).pages]
    target = normalized(needle)
    joined, starts = "", []
    for page in pages:
        starts.append(len(joined))
        joined += page + " "
    position = joined.find(target)
    if position < 0:
        raise SystemExit(f"Факт не найден в PDF: {needle[:60]}")
    end = position + len(target) - 1
    return [
        n
        for n, start in enumerate(starts, start=1)
        if start <= end and (n == len(starts) or starts[n] > position)
    ]


def golden_item(fact: Fact, doc_path: Path) -> dict:
    needle = fact.text if fact.cross_page else fact.locate  # «на стыке» ищем целиком
    pages = locate_pages(doc_path, needle) if fact.expect_found else []
    return {
        "id": fact.key,
        "doc": doc_path.name,
        "kind": fact.kind,
        "question": fact.question,
        "expect_found": fact.expect_found,
        "expect_pages": pages,
        "expect_section": fact.section,
        "expect_contains": fact.expect_contains,
    }


def main() -> None:
    register_font()
    OUT.mkdir(exist_ok=True)
    supply_path, short_path = OUT / "contract_supply.pdf", OUT / "contract_short.pdf"
    supply_facts = [f for f in FACTS if f.doc == supply_path.name]
    short_facts = [f for f in FACTS if f.doc == short_path.name]

    build_pdf(
        supply_path,
        contract_story(
            "ДОГОВОР ПОСТАВКИ № 47/2026",
            PREAMBLE,
            SUPPLY_SECTIONS,
            supply_facts,
            random.Random(42),
            CLAUSES_PER_SUBSECTION,
        ),
    )
    build_pdf(
        short_path,
        contract_story(
            "ДОГОВОР ПОСТАВКИ № 12/2026",
            SHORT_CONTRACT["preamble"],
            SHORT_CONTRACT["sections"],
            short_facts,
            random.Random(7),
            (3, 4),
        ),
    )

    golden = [golden_item(f, OUT / f.doc) for f in FACTS]
    (OUT / "golden.json").write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n")
    pages = len(PdfReader(supply_path).pages)
    print(f"contract_supply.pdf: {pages} стр.; golden.json: {len(golden)} вопросов")


if __name__ == "__main__":
    main()
