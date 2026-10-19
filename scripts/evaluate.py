"""Оценка качества по эталонному набору samples/golden.json на живом сервисе.

Запуск (стек поднят, документы загрузятся сами при необходимости):
    uv run python scripts/evaluate.py --base-url http://localhost:8000 --out docs/eval-report.md

Код выхода 1, если цели из спеки не достигнуты.
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
POLL_INTERVAL_S = 2
INDEX_TIMEOUT_S = 900
TARGETS = {"absent_correct": 1.0, "exact_hit": 1.0, "page_hit": 0.9, "isolation_ok": 1.0}


@dataclass(frozen=True)
class Outcome:
    item: dict
    found: bool
    answer: str
    pages: list[tuple[int, int]]
    sections: list[str]
    reranked_pages: list[int]
    latency_ms: int

    def _norm(self, text: str) -> str:
        return " ".join(text.replace(" ", " ").split()).lower()

    @property
    def locations(self) -> list[tuple[set[int], str | None]]:
        """Основное место факта и альтернативные (факт честно указан в нескольких местах)."""
        main = (set(self.item["expect_pages"]), self.item["expect_section"])
        extra = [(set(a["pages"]), a["section"]) for a in self.item.get("also_found_in", [])]
        return [main, *extra]

    def _cited(self, pages: set[int], section: str | None) -> bool:
        return any(
            pages & set(range(a, b + 1)) and (not section or section == cited)
            for (a, b), cited in zip(self.pages, self.sections, strict=True)
        )

    @property
    def page_hit(self) -> bool:
        return any(
            pages & set(range(a, b + 1)) for pages, _ in self.locations for a, b in self.pages
        )

    @property
    def section_hit(self) -> bool:
        return any(not section or section in self.sections for _, section in self.locations)

    @property
    def location_hit(self) -> bool:
        """Есть цитата, у которой и страница, и раздел совпадают с одним из мест факта."""
        return any(self._cited(pages, section) for pages, section in self.locations)

    @property
    def contains_ok(self) -> bool:
        return self._norm(self.item["expect_contains"]) in self._norm(self.answer)

    @property
    def retrieval_hit(self) -> bool:
        return any(pages & set(self.reranked_pages) for pages, _ in self.locations)

    @property
    def correct(self) -> bool:
        if not self.item["expect_found"]:
            return not self.found and not self.pages
        return self.found and self.location_hit and self.contains_ok


def ensure_documents(client: httpx.Client, names: set[str]) -> dict[str, str]:
    """doc filename → doc_id; загружает недостающие документы и ждёт индексации."""
    existing = {d["title"]: d["doc_id"] for d in client.get("/documents").json()}
    doc_ids = {}
    for name in sorted(names):
        title = Path(name).stem
        if title in existing:
            doc_ids[name] = existing[title]
            continue
        with (ROOT / "samples" / name).open("rb") as file:
            upload = client.post("/documents", files={"file": (name, file)}, data={"title": title})
        upload.raise_for_status()
        doc_ids[name] = wait_for_index(client, upload.json())
    return doc_ids


def wait_for_index(client: httpx.Client, upload: dict) -> str:
    deadline = time.monotonic() + INDEX_TIMEOUT_S
    while time.monotonic() < deadline:
        status = client.get(upload["status_url"]).json()
        if status["state"] == "SUCCESS":
            return upload["doc_id"]
        if status["state"] == "FAILURE":
            raise SystemExit(f"Индексация упала: {status.get('error')}")
        time.sleep(POLL_INTERVAL_S)
    raise SystemExit("Индексация не завершилась за отведённое время")


def ask(client: httpx.Client, doc_id: str, item: dict) -> Outcome:
    started = time.perf_counter()
    response = client.post(
        f"/documents/{doc_id}/ask", json={"question": item["question"], "debug": True}
    )
    response.raise_for_status()
    body = response.json()
    return Outcome(
        item=item,
        found=body["found"],
        answer=body["answer"],
        pages=[(c["page"], c["page_to"]) for c in body["citations"]],
        sections=[c["section"] for c in body["citations"]],
        reranked_pages=[
            p for r in body["debug"]["reranked"] for p in range(r["page"], r["page_to"] + 1)
        ],
        latency_ms=round((time.perf_counter() - started) * 1000),
    )


def share(outcomes: list[Outcome], predicate) -> float:
    return sum(map(predicate, outcomes)) / len(outcomes) if outcomes else 1.0


def compute_metrics(outcomes: list[Outcome]) -> dict[str, float]:
    of_kind = lambda *kinds: [o for o in outcomes if o.item["kind"] in kinds]  # noqa: E731
    answerable = [o for o in outcomes if o.item["expect_found"]]
    latencies = sorted(o.latency_ms for o in outcomes)
    return {
        "absent_correct": share(of_kind("absent"), lambda o: o.correct),
        "exact_hit": share(of_kind("exact", "similar_numbers"), lambda o: o.correct),
        "page_hit": share(answerable, lambda o: o.page_hit),
        "isolation_ok": share(of_kind("isolation"), lambda o: o.correct),
        "retrieval_hit": share(answerable, lambda o: o.retrieval_hit),
        "overall_correct": share(outcomes, lambda o: o.correct),
        "latency_avg_ms": statistics.mean(latencies),
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
    }


def mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def render_report(outcomes: list[Outcome], metrics: dict[str, float], label: str) -> str:
    lines = [
        f"# Отчёт об оценке качества\n\nКонфигурация: {label}\n",
        "## Метрики\n",
        "| Метрика | Значение | Цель |",
        "|---|---|---|",
    ]
    for name, value in metrics.items():
        target = TARGETS.get(name)
        shown = f"{value:.0f} мс" if name.startswith("latency") else f"{value:.0%}"
        goal = f"≥ {target:.0%} {mark(value >= target)}" if target is not None else "—"
        lines.append(f"| {name} | {shown} | {goal} |")
    lines += [
        "\n## По вопросам\n",
        "| id | вид | вопрос | ожид. стр. | цит. стр. | раздел | значение | итог |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        item = o.item
        cited = ", ".join(f"{a}" if a == b else f"{a}–{b}" for a, b in o.pages) or "—"
        expected = ", ".join(map(str, item["expect_pages"])) or "—"
        lines.append(
            f"| {item['id']} | {item['kind']} | {item['question']} | {expected} | {cited} | "
            f"{mark(o.section_hit)} | {mark(o.contains_ok) if item['expect_found'] else '—'} | "
            f"{mark(o.correct)} |"
        )
    lines += ["\n## Ответы\n"]
    lines += [f"- **{o.item['id']}**: {o.answer}" for o in outcomes]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--golden", default=str(ROOT / "samples" / "golden.json"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "eval-report.md"))
    parser.add_argument("--label", default="значения по умолчанию из .env.example")
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text())
    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        doc_ids = ensure_documents(client, {item["doc"] for item in golden})
        outcomes = [ask(client, doc_ids[item["doc"]], item) for item in golden]

    metrics = compute_metrics(outcomes)
    Path(args.out).write_text(render_report(outcomes, metrics, args.label))
    for name, value in metrics.items():
        print(f"{name:16} {value:.2f}")
    failed = [n for n, t in TARGETS.items() if metrics[n] < t]
    print("Цели достигнуты" if not failed else f"Не достигнуты: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
