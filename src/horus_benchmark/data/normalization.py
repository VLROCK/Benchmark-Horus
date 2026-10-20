"""Conservative, class-blind transformations for the *full* Horus corpus.

This module does not select examples, infer authorship, or use split metadata.
Ambiguous phenomena are flagged for review rather than rewritten.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass


EMAIL = re.compile(r"(?<![\w@])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w@])", re.I)
URL = re.compile(r"(?<![\w@])(?:https?://|www\.)[^\s<>\]\[()]+", re.I)
HTML_ENTITY = re.compile(r"&(?:#[0-9]+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]+);")
BR_TAG = re.compile(r"<br\s*/?>", re.I)
PARA_TAG = re.compile(r"</?(?:p|div)\b[^>]*>", re.I)
HTML_TAG = re.compile(r"<\s*/?\s*[A-Za-z][^>]*>")
CODE_BLOCK = re.compile(r"(?m)^\s*(?:```|~~~)|<\s*(?:pre|code)\b", re.I)
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RUN = re.compile(r"[ \t\u00a0]{2,}")
TRAILING_SPACE = re.compile(r"(?m)[ \t]+$")
EXCESS_BLANK = re.compile(r"\n{4,}")
MOJIBAKE = re.compile(r"(?:Ã.|Â.|Ă.|â€|�)")
ABSTRACT = re.compile(r"(?im)^\s*(?:abstract|résumé|resumen|keywords|palabras clave)\s*:?\s*")
ABSTRACT_START = re.compile(r"(?im)^[ \t]*(?:abstract|résumé|resumen)[ \t]*:?[^\n]*")
PORTUGUESE_RESUMO = re.compile(r"(?im)^[ \t]*resumos?[ \t]*:?[^\n]*")
PORTUGUESE_INTRO = re.compile(r"(?im)^[ \t]*(?:1[.\s-]+)?introdução\b[^\n]*")
REFERENCES = re.compile(r"(?im)^\s*(?:referências(?: bibliográficas)?|bibliografia|references)\s*$")
PLACEHOLDER = re.compile(r"(?i)\[(?:insira|inserir|assinatura|seu nome|nome aqui)[^\]]{0,80}\]")
PREAMBLE = re.compile(r"(?is)^\s*(?:aqui (?:está|esta|segue)|segue (?:abaixo|o texto)|claro[,!])\b.{0,100}[:\n]")
FOOTER = re.compile(
    r"(?im)^\s*(?:copyright\s*(?:©|\(c\))?.*todos os direitos reservados\.?|"
    r"atualizado em:\s*[^\n]{4,90}|one reply to\s+[^\n]{1,120})\s*$"
)
MARKDOWN = re.compile(r"(?m)^\s{0,3}(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)|\*\*|```|~~~")
LITERAL_ESCAPE = re.compile(r"\\[nrt]")


@dataclass(frozen=True)
class RuleConfig:
    stage: str = "n1"
    repair_mojibake: bool = True
    replace_contacts: bool = True
    convert_editorial_html: bool = True
    remove_strict_terminal_footers: bool = True
    remove_bounded_parallel_abstracts: bool = True
    remove_strict_terminal_references: bool = True
    guard_min_original_words: int = 50
    guard_min_lost_words: int = 20
    guard_min_retained_fraction: float = 0.90

    def __post_init__(self) -> None:
        if self.stage not in {"n1", "n2"}:
            raise ValueError("stage must be n1 or n2")
        if not 0 <= self.guard_min_retained_fraction <= 1:
            raise ValueError("guard_min_retained_fraction must be in [0, 1]")
        if min(self.guard_min_original_words, self.guard_min_lost_words) < 0:
            raise ValueError("word-count guard values must be nonnegative")


@dataclass(frozen=True)
class NormalizedText:
    text: str
    operations: dict[str, int]
    flags: tuple[str, ...]
    status: str
    original_sha256: str
    normalized_sha256: str
    original_chars: int
    normalized_chars: int
    original_words: int
    normalized_words: int


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _replace(
    pattern: re.Pattern[str], text: str, replacement: str | Callable[[re.Match[str]], str],
    operations: Counter[str], name: str,
) -> str:
    result, count = pattern.subn(replacement, text)
    if count:
        operations[name] += count
    return result


def _repair_mojibake(text: str) -> str:
    """Accept a whole-text reverse decoding only when suspicion strictly falls.

    Whole-text encoding fails on mixed scripts/emoji; those cases stay intact
    and are flagged. No lossy replacement is ever used.
    """

    before = len(MOJIBAKE.findall(text))
    if before == 0:
        return text
    candidates: list[str] = []
    for codec in ("cp1252", "cp1250", "latin-1"):
        try:
            candidate = text.encode(codec).decode("utf-8")
        except (UnicodeError, LookupError):
            continue
        if "\ufffd" not in candidate and len(MOJIBAKE.findall(candidate)) < before:
            candidates.append(candidate)
    if not candidates:
        return text
    best_score = min(len(MOJIBAKE.findall(value)) for value in candidates)
    best = {value for value in candidates if len(MOJIBAKE.findall(value)) == best_score}
    # Multiple equally plausible decodings are ambiguous, so leave intact.
    return next(iter(best)) if len(best) == 1 else text


def _mask_url(match: re.Match[str]) -> str:
    raw = match.group()
    core = raw.rstrip(".,;:!?")
    return "URL" + raw[len(core):]


def _remove_parallel_abstract(text: str, operations: Counter[str]) -> str:
    """Remove only a clearly bounded non-Portuguese front-matter section."""

    start = ABSTRACT_START.search(text)
    if start is None or start.start() > len(text) // 4:
        return text
    if PORTUGUESE_RESUMO.search(text[:start.start()]) is None:
        return text
    introduction = PORTUGUESE_INTRO.search(text, start.end())
    if introduction is None or introduction.start() > len(text) * 0.6:
        return text
    if len(text[start.start():introduction.start()].split()) < 25:
        return text
    operations["PARALLEL_ABSTRACT_REMOVE"] += 1
    return (text[:start.start()].rstrip() + "\n\n" + text[introduction.start():]).strip()


def _remove_terminal_references(text: str, operations: Counter[str]) -> str:
    """Remove a terminal bibliography only when its shape is unmistakable."""

    matches = list(REFERENCES.finditer(text))
    if not matches:
        return text
    heading = matches[-1]
    prefix, suffix = text[:heading.start()], text[heading.end():]
    if heading.start() < len(text) * 0.6 or len(prefix.split()) < 200:
        return text
    if len(suffix.split()) < 30 or len(re.findall(r"\b(?:19|20)\d{2}\b", suffix)) < 2:
        return text
    operations["TERMINAL_REFERENCES_REMOVE"] += 1
    return prefix.rstrip()


def normalize_text(content: str, config: RuleConfig) -> NormalizedText:
    """Return a derived text and diagnostic flags; never mutate ``content``."""

    if not isinstance(content, str):
        raise TypeError("content must be a string")
    original = content
    flags: set[str] = set()
    operations: Counter[str] = Counter()

    if ABSTRACT.search(original):
        flags.add("LANG_SECTION")
    if REFERENCES.search(original):
        flags.add("REFERENCES")
    if PLACEHOLDER.search(original):
        flags.add("PLACEHOLDER")
    if PREAMBLE.search(original):
        flags.add("PREAMBLE")
    if MARKDOWN.search(original):
        flags.add("MARKUP")
    if LITERAL_ESCAPE.search(original):
        flags.add("LITERAL_ESCAPE")
    if CODE_BLOCK.search(original):
        flags.add("CODE_EXAMPLE")
    if MOJIBAKE.search(original):
        flags.add("ENCODING_CANDIDATE")

    text = unicodedata.normalize("NFC", original)
    if text != original:
        operations["UNICODE_NFC"] += 1
    repaired = _repair_mojibake(text) if config.repair_mojibake else text
    if repaired != text:
        text = repaired
        operations["MOJIBAKE_REPAIR"] += 1
    if MOJIBAKE.search(text):
        flags.add("ENCODING_REMAINS")

    converted = text.replace("\r\n", "\n").replace("\r", "\n")
    if converted != text:
        text = converted
        operations["NEWLINES"] += 1
    text = _replace(CONTROL, text, "", operations, "CONTROL_REMOVE")

    # Code and literal markup are legitimate content in some genres. In those
    # records, ambiguous HTML/entity conversion is intentionally skipped.
    code_like = bool(CODE_BLOCK.search(text))
    if config.convert_editorial_html and not code_like:
        text = _replace(BR_TAG, text, "\n", operations, "HTML_BR")
        text = _replace(PARA_TAG, text, "\n", operations, "HTML_PARAGRAPH")
        converted = HTML_ENTITY.sub(lambda match: html.unescape(match.group()), text)
        if converted != text:
            operations["HTML_ENTITIES"] += 1
            text = converted
    elif HTML_TAG.search(text) or HTML_ENTITY.search(text):
        flags.add("HTML_REVIEW")
    if HTML_TAG.search(text):
        flags.add("HTML_REMAINS")

    if config.replace_contacts and not code_like:
        text = _replace(EMAIL, text, "EMAIL", operations, "EMAIL_MASK")
        text = _replace(URL, text, _mask_url, operations, "URL_MASK")
    elif EMAIL.search(text) or URL.search(text):
        flags.add("CONTACT_REVIEW")

    text = _replace(TRAILING_SPACE, text, "", operations, "TRAILING_SPACE")
    text = _replace(SPACE_RUN, text, " ", operations, "SPACE_RUN")
    text = _replace(EXCESS_BLANK, text, "\n\n\n", operations, "BLANK_LINES")
    trimmed = text.strip()
    if trimmed != text:
        operations["EDGE_SPACE"] += 1
        text = trimmed

    if config.stage == "n2" and config.remove_strict_terminal_footers:
        lines = text.splitlines()
        removed = 0
        while lines and FOOTER.fullmatch(lines[-1].strip()):
            lines.pop()
            removed += 1
        if removed:
            text = "\n".join(lines).rstrip()
            operations["TERMINAL_FOOTER"] += removed
    elif FOOTER.search(text):
        flags.add("EDITORIAL_FOOTER")
    if config.stage == "n2" and config.remove_bounded_parallel_abstracts:
        text = _remove_parallel_abstract(text, operations)
    if config.stage == "n2" and config.remove_strict_terminal_references:
        text = _remove_terminal_references(text, operations)

    # Remaining structural issues are detection-only until segment boundaries
    # have been validated. No model rewrites the content.
    original_words = len(original.split())
    normalized_words = len(text.split())
    loss = original_words - normalized_words
    if (
        original_words >= config.guard_min_original_words
        and loss >= config.guard_min_lost_words
        and normalized_words / original_words < config.guard_min_retained_fraction
    ):
        flags.add("SUBSTANTIAL_LOSS")
    if not text:
        flags.add("EMPTY_AFTER")
    review_flags = {
        "LANG_SECTION", "REFERENCES", "PLACEHOLDER", "PREAMBLE",
        "ENCODING_REMAINS", "HTML_REVIEW", "HTML_REMAINS", "CONTACT_REVIEW",
        "SUBSTANTIAL_LOSS", "EMPTY_AFTER", "LITERAL_ESCAPE",
    }
    status = "review" if flags & review_flags else "ok"
    return NormalizedText(
        text=text, operations=dict(sorted(operations.items())),
        flags=tuple(sorted(flags)), status=status,
        original_sha256=_sha(original), normalized_sha256=_sha(text),
        original_chars=len(original), normalized_chars=len(text),
        original_words=original_words, normalized_words=normalized_words,
    )
