"""Conservative, document-linked sentence/paragraph sampling for paired views."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher


PARAGRAPH_GAP = re.compile(r"\n\s*\n+")
SENTENCE_END = re.compile(r"[.!?…]+(?:[\"”’»)]*)\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9\"“‘«(])")
ABBREVIATIONS = frozenset({"dr.", "dra.", "sr.", "sra.", "prof.", "profa.",
                            "art.", "n.º", "etc.", "ex.", "p.", "vol."})
WORD = re.compile(r"\w+", re.UNICODE)
SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Segment:
    text: str
    start: int
    end: int
    index: int
    total: int

    @property
    def midpoint(self) -> float:
        return ((self.start + self.end) / 2) / max(self.total, 1)


def segments(text: str, kind: str) -> list[Segment]:
    """Only explicit blank lines define paragraphs; sentence rules are conservative."""

    if kind not in {"paragraph", "sentence"}:
        raise ValueError(f"Unknown segment kind: {kind}")
    bounds: list[tuple[int, int]] = []
    if kind == "paragraph":
        cursor = 0
        for gap in PARAGRAPH_GAP.finditer(text):
            bounds.append((cursor, gap.start()))
            cursor = gap.end()
        bounds.append((cursor, len(text)))
    else:
        cursor = 0
        for boundary in SENTENCE_END.finditer(text):
            prefix = text[cursor:boundary.start() + 1].strip().lower()
            if any(prefix.endswith(abbreviation) for abbreviation in ABBREVIATIONS):
                continue
            bounds.append((cursor, boundary.end()))
            cursor = boundary.end()
        bounds.append((cursor, len(text)))
    output = []
    for start, end in bounds:
        stripped = text[start:end].strip()
        if stripped:
            left = start + len(text[start:end]) - len(text[start:end].lstrip())
            output.append(Segment(stripped, left, left + len(stripped), len(output), len(text)))
    return output


def select_middle(
    text: str, kind: str, record_id: str, *, min_words: int, max_words: int,
    middle_start: float, middle_end: float, max_fraction_of_document: float,
    seed: int,
) -> Segment | None:
    """Choose one eligible central unit, never fall back to front/back matter."""

    if not 0 <= middle_start < middle_end <= 1 or not 0 < max_fraction_of_document <= 1:
        raise ValueError("Middle window must be within [0, 1]")
    candidates = [
        unit for unit in segments(text, kind)
        if middle_start <= unit.midpoint <= middle_end
        and (unit.end - unit.start) / max(len(text), 1) <= max_fraction_of_document
        and min_words <= len(WORD.findall(unit.text)) <= max_words
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda unit: (
        abs(unit.midpoint - 0.5),
        hashlib.sha256(f"{seed}|{record_id}|{kind}|{unit.index}".encode()).hexdigest(),
    ))


def _canonical(value: str) -> str:
    return SPACE.sub(" ", value).strip().casefold()


def align_segment(
    original: Segment, new_text: str, kind: str, *, minimum_similarity: float,
    minimum_margin: float,
) -> tuple[Segment | None, str, float]:
    """Match only a unique, near-identical unit; changed boundaries are attrition."""

    options = segments(new_text, kind)
    if not options:
        return None, "missing", 0.0
    source = _canonical(original.text)
    exact = [unit for unit in options if _canonical(unit.text) == source]
    if len(exact) == 1:
        return exact[0], "exact", 1.0
    if len(exact) > 1:
        return None, "ambiguous", 1.0
    source_words = set(WORD.findall(source))
    ranked: list[tuple[float, Segment]] = []
    for unit in options:
        target = _canonical(unit.text)
        target_words = set(WORD.findall(target))
        overlap = len(source_words & target_words) / max(len(source_words | target_words), 1)
        if overlap < minimum_similarity - 0.15:
            continue
        ratio = SequenceMatcher(None, source, target, autojunk=False).ratio()
        if ratio >= minimum_similarity - minimum_margin:
            ranked.append((ratio, unit))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked or ranked[0][0] < minimum_similarity:
        return None, "missing_or_changed_boundary", ranked[0][0] if ranked else 0.0
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < minimum_margin:
        return None, "ambiguous", ranked[0][0]
    return ranked[0][1], "matched_changed", ranked[0][0]
