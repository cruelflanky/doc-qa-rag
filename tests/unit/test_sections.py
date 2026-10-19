from app.ingest.sections import INTRO_SECTION_TITLE, find_sections

TEXT = (
    "ДОГОВОР ПОСТАВКИ № 15/24\n"
    "г. Москва\n"
    "4. ОТВЕТСТВЕННОСТЬ СТОРОН\n"
    "4.1 Общие положения\n"
    "4.1.1. Стороны несут ответственность в соответствии с законом.\n"
    "4.2 Штрафные санкции\n"
    "4.2.1. Поставщик обязан уплатить неустойку 0,1% за каждый день просрочки.\n"
    "Общая сумма составляет\n"
    "12 480 000,00 рублей.\n"
    "1. Стороны договорились, что спор решается в суде.\n"
)


def titles(text):
    return [s.title for s in find_sections(text)]


def test_find_sections_levels_1_and_2():
    found = titles(TEXT)

    assert "4. ОТВЕТСТВЕННОСТЬ СТОРОН" in found
    assert "4.1 Общие положения" in found
    assert "4.2 Штрафные санкции" in found
    assert not any(t.startswith("4.2.1") for t in found)
    assert not any(t.startswith("4.1.1") for t in found)


def test_find_sections_ignores_amounts_and_sentences():
    found = titles(TEXT)

    assert not any("480" in t for t in found)
    assert not any(t.startswith("1. Стороны") for t in found)


def test_find_sections_intro_and_full_coverage():
    sections = find_sections(TEXT)

    assert sections[0].title == INTRO_SECTION_TITLE
    assert sections[0].start == 0
    assert sections[-1].end == len(TEXT)
    for left, right in zip(sections, sections[1:], strict=False):
        assert left.end == right.start


def test_find_sections_without_headings_is_single_intro():
    sections = find_sections("Просто текст без заголовков.")

    assert len(sections) == 1
    assert sections[0].title == INTRO_SECTION_TITLE


def test_heading_at_start_has_no_empty_intro():
    sections = find_sections("1. ПРЕДМЕТ ДОГОВОРА\nТекст раздела.")

    assert sections[0].title == "1. ПРЕДМЕТ ДОГОВОРА"
    assert sections[0].start == 0
