"""Query-free PT-BR adaptations of six RAID perturbation families.

Rules, doses and protected regions are fixed before observing detector scores.
Offsets always refer to the clean source. An attack never consumes a class label.
The conservative PT-BR spelling dictionary is versioned here, not learned from
the evaluation corpus. These sparse adaptations are not exact RAID replications.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import random
import re
from typing import Any

from horus_benchmark.audit.middle_segments import segments


ATTACKS = ("insert_paragraphs", "upper_lower", "whitespace", "misspelling",
           "zero_width_space", "homoglyph")
SPARSE_ATTACKS = tuple(name for name in ATTACKS if name != "insert_paragraphs")
WORD = re.compile(r"\b[^\W\d_]+\b", re.UNICODE)
PROTECTED = re.compile(
    r"```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]*`|"
    r"<(?:pre|code)\b[^>]*>[\s\S]*?</(?:pre|code)>|<[^>\n]+>|"
    r"https?://[^\s<>]+|www\.[^\s<>]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"\b\d[\w.,:/%+-]*", re.IGNORECASE)
SPELLING_VERSION = "ptbr-conservative-v1"
MISSPELLINGS = {
    "porque": "porqe", "também": "tambem", "então": "entao",
    "não": "nao", "você": "voce", "vocês": "voces", "além": "alem",
    "através": "atraves", "possível": "possivel", "necessário": "necessario",
    "importância": "importancia", "experiência": "experiencia",
    "análise": "analise", "conclusão": "conclusao", "informação": "informacao",
    "informações": "informacoes", "situação": "situacao", "relação": "relacao",
    "relações": "relacoes", "diferença": "diferençã", "processo": "proceso",
    "desenvolvimento": "desenvolvimeto", "portanto": "portato",
    "resultado": "resutado", "resultados": "resutados",
    "principal": "pricipal", "exemplo": "exempo", "problema": "probema",
    "importante": "importate", "conhecimento": "conhecimeto",
    "sociedade": "sociedae", "pessoas": "pesoas",
}
# A single fixed visually confusable replacement per Latin character. No digit
# substitutions: numerical content is protected and not part of this battery.
HOMOGLYPHS = {
    "a": "а", "A": "А", "B": "В", "e": "е", "E": "Е", "c": "с",
    "C": "С", "p": "р", "P": "Р", "K": "К", "O": "О", "o": "о",
    "M": "М", "H": "Н", "T": "Т", "X": "Х", "x": "х", "y": "у",
    "I": "І", "i": "і", "N": "Ν", "Z": "Ζ",
}


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    before: str
    after: str


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seed(parent_id: str, name: str, intensity: float, seed: int) -> int:
    return int(sha256_text(f"{seed}|{parent_id}|{name}|{intensity:.8g}")[:16], 16)


def _mask(text: str) -> list[bool]:
    result = [False] * len(text)
    for span in PROTECTED.finditer(text):
        result[span.start():span.end()] = [True] * (span.end() - span.start())
    return result


def _available(mask: list[bool], start: int, end: int) -> bool:
    return not any(mask[max(0, start):min(len(mask), max(end, start + 1))])


def replay(text: str, edits: list[Edit]) -> str:
    ordered = sorted(edits, key=lambda item: (item.start, item.end))
    cursor = 0
    chunks: list[str] = []
    for edit in ordered:
        if (not 0 <= edit.start <= edit.end <= len(text) or
                edit.start < cursor or text[edit.start:edit.end] != edit.before):
            raise ValueError("Invalid, overlapping, or mismatched source edit")
        chunks.extend((text[cursor:edit.start], edit.after))
        cursor = edit.end
    chunks.append(text[cursor:])
    return "".join(chunks)


def _candidates(text: str, attack: str, protected: list[bool]) -> list[Edit]:
    output: list[Edit] = []
    if attack in {"upper_lower", "misspelling"}:
        for word in WORD.finditer(text):
            start, end = word.span()
            if not _available(protected, start, end):
                continue
            value = word.group()
            if attack == "upper_lower":
                replacement = value[0].swapcase()
                if len(replacement) == 1 and replacement != value[0]:
                    output.append(Edit(start, start + 1, value[0], replacement))
            elif value.islower() and value in MISSPELLINGS:
                output.append(Edit(start, end, value, MISSPELLINGS[value]))
    elif attack == "whitespace":
        for span in re.finditer(r"(?<=\S) (?=\S)", text):
            if _available(protected, span.start() - 1, span.end() + 1):
                output.append(Edit(span.start(), span.end(), " ", "  "))
    elif attack == "homoglyph":
        output = [Edit(i, i + 1, char, HOMOGLYPHS[char])
                  for i, char in enumerate(text) if char in HOMOGLYPHS and
                  _available(protected, i, i + 1)]
    elif attack == "zero_width_space":
        # Only within alphabetic words, not at a combining mark or number.
        output = [Edit(i, i, "", "\u200b") for i in range(1, len(text))
                  if text[i - 1].isalpha() and text[i].isalpha() and
                  _available(protected, i - 1, i + 1)]
    elif attack == "insert_paragraphs":
        units = segments(text, "sentence")
        for previous, following in zip(units, units[1:]):
            start, end = previous.end, following.start
            middle = ((start + end) / 2) / max(len(text), 1)
            gap = text[start:end]
            if (0.35 <= middle <= 0.65 and gap and gap.isspace() and
                    "\n" not in gap and _available(protected, start, end)):
                output.append(Edit(start, end, gap, "\n\n"))
    else:
        raise ValueError(f"Unknown attack: {attack}")
    return output


def perturb(text: str, parent_id: str, attack: str, intensity: float,
            *, seed: int = 20261125) -> dict[str, Any]:
    """Apply one attack independently to clean text with auditable no-ops.

    For sparse attacks, floor(dose * eligible_positions) positions are sampled
    without replacement. No minimum-one edit is forced on short documents.
    Paragraph insertion uses a single central boundary and intensity=1.0 means
    one boundary, not 100% of the text. It has a separate `dose_unit` field.
    """
    if not isinstance(text, str) or not isinstance(parent_id, str) or not parent_id:
        raise ValueError("Text and nonempty parent ID strings are required")
    if attack not in ATTACKS:
        raise ValueError(f"Unknown attack: {attack}")
    if attack == "insert_paragraphs":
        if float(intensity) != 1.0:
            raise ValueError("Paragraph insertion has exactly one central edit")
    elif not 0 < float(intensity) <= 1:
        raise ValueError("Sparse dose must lie in (0, 1]")
    attack_seed = _seed(parent_id, attack, float(intensity), seed)
    candidates = _candidates(text, attack, _mask(text)) if text.strip() else []
    if attack == "insert_paragraphs" and candidates:
        chosen = [min(candidates, key=lambda item: (
            abs(((item.start + item.end) / 2) / len(text) - .5),
            sha256_text(f"{attack_seed}|{item.start}")))]
    else:
        count = math.floor(len(candidates) * float(intensity))
        chosen = random.Random(attack_seed).sample(candidates, count)
    chosen.sort(key=lambda item: (item.start, item.end))
    changed = replay(text, chosen)
    if attack in {"whitespace", "insert_paragraphs"} and changed.split() != text.split():
        raise AssertionError("Presentation attack altered the whitespace token sequence")
    if not text.strip():
        status, reason = "invalid", "empty_source"
    elif not candidates:
        status, reason = "no_op", "no_eligible_position"
    elif not chosen:
        status, reason = "no_op", "dose_rounds_to_zero"
    else:
        status, reason = "changed", ""
    parent_hash, content_hash = sha256_text(text), sha256_text(changed)
    variant_id = sha256_text(
        f"{parent_id}|{parent_hash}|{attack}|{intensity:.8g}|{seed}|{content_hash}")
    return {
        "record_id": parent_id, "parent_record_id": parent_id,
        "variant_id": variant_id, "content": changed,
        "content_hash": content_hash, "parent_content_hash": parent_hash,
        "attack": attack, "intensity": float(intensity), "attack_seed": str(attack_seed),
        "status": status, "reason": reason, "eligible_positions": len(candidates),
        "num_edits": len(chosen), "effective_fraction": len(chosen) / len(candidates)
        if candidates else 0.0,
        "dose_unit": "one_central_boundary" if attack == "insert_paragraphs"
        else "fraction_of_eligible_positions",
        "edits_json": json.dumps([asdict(item) for item in chosen],
                                  ensure_ascii=False, separators=(",", ":")),
        "spelling_dictionary_version": SPELLING_VERSION if attack == "misspelling" else "",
    }
