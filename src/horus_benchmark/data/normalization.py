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
BR_TAG = re.compile(r"<br[ \t]*/?>", re.I)
PARA_TAG = re.compile(r"</?(?:p|div)\b[^<>\n]{0,500}>", re.I)
HTML_TAG = re.compile(
    r"</?(?:a|article|b|blockquote|br|code|div|em|h[1-6]|hr|i|img|li|ol|p|pre|"
    r"section|span|strong|table|tbody|td|th|thead|tr|u|ul)\b[^<>\n]{0,500}>", re.I
)
CODE_BLOCK = re.compile(r"(?m)^\s*(?:```|~~~)|<\s*(?:pre|code)\b", re.I)
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RUN = re.compile(r"[ \t\u00a0]{2,}")
TRAILING_SPACE = re.compile(r"(?m)[ \t]+$")
EXCESS_BLANK = re.compile(r"\n{4,}")
MOJIBAKE = re.compile(r"(?:[ÃÂĂ][^\x00-\x7f]|â€|�)")
MIXED_C3_MOJIBAKE = re.compile(r"Ă([\u0080-\u00bf])")
ABSTRACT_START = re.compile(r"(?im)^[ \t]*abstract(?=[ \t]*:|[ \t]*$)[ \t]*:?[ \t]*[^\n]*")
PORTUGUESE_RESUMO = re.compile(r"(?im)^[ \t]*resumos?[ \t]*:?[^\n]*")
PORTUGUESE_INTRO = re.compile(r"(?im)^[ \t]*(?:1[.\s-]+)?introdução\b[^\n]*")
ABSTRACT_NEXT_SECTION = re.compile(
    r"(?i)^(?:(?:\d+[.\s-]+)?introdução|resumo|resumen|résumé|sumário)\b"
)
ENGLISH_KEYWORDS = re.compile(r"(?i)^(?:key[ -]?words?|index terms?)\s*:")
REFERENCES = re.compile(r"(?im)^[ \t]*(?:referências(?: bibliográficas)?|bibliografia|references)[ \t]*$")
PLACEHOLDER = re.compile(r"(?i)\[(?:insira|inserir|assinatura|seu nome|nome aqui)[^\]]{0,80}\]")
PREAMBLE = re.compile(r"(?is)^\s*(?:aqui (?:está|esta|segue)|segue (?:abaixo|o texto)|claro[,!])\b.{0,100}[:\n]")
FOOTER = re.compile(
    r"(?im)^\s*(?:copyright\s*(?:©|\(c\))?.*todos os direitos reservados\.?|"
    r"atualizado em:\s*[^\n]{4,90}|one reply to\s+[^\n]{1,120})\s*$"
)
MARKDOWN = re.compile(r"(?m)^\s{0,3}(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)|\*\*|```|~~~")
FENCE_LINE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})([^\n]*)$")
MD_LINK = re.compile(r"\[([^\]\n]{1,500})\]\(([^)\n]{1,1000})\)")
MD_INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
MD_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(?=\S)")
# Numbered sections are common in legal and academic prose, so their numbers
# are preserved even when they resemble Markdown list markers.
MD_BULLET = re.compile(r"^[ \t]*[-*+][ \t]+(?=\S)")
MD_STRONG = re.compile(r"(?<!\w)(?:\*\*([^*\n]+)\*\*|__([^_\n]+)__)(?!\w)")
MD_EMPHASIS = re.compile(r"(?<!\w)(?:\*([^*\n]+)\*|_([^_\n]+)_)(?!\w)")
LITERAL_ESCAPE = re.compile(r"\\[nrt]")
ENGLISH_MARKERS = frozenset(
    "the of and to in for with this that from are was were have has as we our "
    "these those which by on an is be study research results methods findings".split()
)
PORTUGUESE_MARKERS = frozenset(
    "de do da dos das que e em para uma um o a os as com por no na nos nas "
    "este esta estudo pesquisa resultados métodos".split()
)
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
TEXT_REVIEW_FLAGS = frozenset({
    "ENGLISH_ABSTRACT_CANDIDATE", "ENCODING_REMAINS", "HTML_DENSE_REMAINS",
    "SUBSTANTIAL_LOSS", "EMPTY_AFTER", "N2_GUARD_RETAINED",
    "N3_GUARD_RETAINED", "N3_UNCLOSED_FENCE",
})


@dataclass(frozen=True)
class RuleConfig:
    stage: str = "n1"
    repair_mojibake: bool = True
    replace_contacts: bool = True
    convert_editorial_html: bool = True
    remove_strict_terminal_footers: bool = True
    remove_bounded_parallel_abstracts: bool = True
    remove_strict_terminal_references: bool = False
    guard_min_original_words: int = 50
    guard_min_lost_words: int = 20
    guard_min_retained_fraction: float = 0.90
    min_english_abstract_words: int = 40
    min_html_residual_tags_for_review: int = 3
    neutralize_markdown: bool = False
    remove_explicit_placeholders: bool = False

    def __post_init__(self) -> None:
        if self.stage not in {"n1", "n2", "n3"}:
            raise ValueError("stage must be n1, n2, or n3")
        if self.stage != "n3" and (self.neutralize_markdown or self.remove_explicit_placeholders):
            raise ValueError("Presentation cleanup requires stage=n3")
        if not 0 <= self.guard_min_retained_fraction <= 1:
            raise ValueError("guard_min_retained_fraction must be in [0, 1]")
        if min(self.guard_min_original_words, self.guard_min_lost_words) < 0:
            raise ValueError("word-count guard values must be nonnegative")
        if self.min_english_abstract_words < 25 or self.min_html_residual_tags_for_review < 2:
            raise ValueError("Abstract and HTML review thresholds are too permissive")


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


@dataclass(frozen=True)
class N2Edit:
    """A proposed, class-blind span in the N1-normalized text."""

    kind: str
    start: int
    end: int
    replacement: str = ""
    count: int = 1


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
    """Reverse unambiguous byte pairs, then try whole-text decoding.

    Some UTF-8 C3 byte pairs were decoded through two different single-byte
    codecs, yielding e.g. ``Ă§`` for ``ç``. Reversing that exact byte pair
    also works when the rest of the text contains emoji or other scripts.
    Other repairs still require a unique, lower-suspicion whole-text result.
    No lossy replacement is ever used.
    """

    text = MIXED_C3_MOJIBAKE.sub(
        lambda match: bytes((0xC3, ord(match.group(1)))).decode("utf-8"), text
    )
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


def _english_evidence(segment: str, minimum_words: int) -> bool:
    """Conservative lexical evidence, not a general-purpose language detector."""

    words = [word.casefold() for word in WORD.findall(segment)]
    if len(words) < minimum_words:
        return False
    english = [word for word in words if word in ENGLISH_MARKERS]
    portuguese = sum(word in PORTUGUESE_MARKERS for word in words)
    return len(english) >= 10 and len(set(english)) >= 4 and len(english) > 2 * portuguese


def _english_abstract(text: str, minimum_words: int) -> tuple[int, int] | None:
    """Locate a front/terminal English abstract with an explicit heading."""

    for heading in ABSTRACT_START.finditer(text):
        if not (heading.start() <= len(text) // 4 or heading.start() >= len(text) * 0.7):
            continue
        introduction = PORTUGUESE_INTRO.search(text, heading.end())
        end = introduction.start() if introduction else min(len(text), heading.end() + 2000)
        if _english_evidence(text[heading.start():end], minimum_words):
            return heading.start(), end
    return None


def _parallel_abstract_span(text: str, minimum_words: int) -> tuple[int, int] | None:
    """Cut only a one-paragraph English abstract with a visible next section.

    Wrapped/multisection abstracts remain flagged, not shortened speculatively.
    In particular, a distant Introduction is never used as the cut boundary.
    """

    for heading in ABSTRACT_START.finditer(text):
        if heading.start() > len(text) // 4:
            continue
        previous_resumo = PORTUGUESE_RESUMO.search(text[:heading.start()])
        next_resumo = PORTUGUESE_RESUMO.search(text, heading.end())
        next_intro = PORTUGUESE_INTRO.search(text, heading.end())
        if not next_intro or next_intro.start() > len(text) * 0.6:
            continue
        if not previous_resumo and not (next_resumo and next_resumo.start() < next_intro.start()):
            continue
        # A single prose line is intentionally required. If the next line is
        # not a recognized keyword line or section, the boundary is ambiguous.
        lines = text[heading.start():next_intro.end()].splitlines(keepends=True)
        if not lines:
            continue
        first = re.sub(r"(?i)^\s*abstract\s*:?\s*", "", lines[0]).strip()
        body_index = 0 if _english_evidence(first, minimum_words) else 1
        if body_index >= len(lines) or not _english_evidence(lines[body_index], minimum_words):
            continue
        index = body_index + 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        if (index < len(lines) and ENGLISH_KEYWORDS.match(lines[index].strip())
                and len(lines[index].split()) <= 25):
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
        elif index < len(lines):
            keywords = lines[index].strip()
            if keywords.count(";") >= 2 and len(keywords.split()) <= 25:
                index += 1
                while index < len(lines) and not lines[index].strip():
                    index += 1
        if index < len(lines) and ABSTRACT_NEXT_SECTION.match(lines[index].strip()):
            end = heading.start() + sum(len(line) for line in lines[:index])
            return heading.start(), end
    return None


def _terminal_references_start(text: str) -> int | None:
    """Find a high-confidence terminal bibliography, without changing text."""

    matches = list(REFERENCES.finditer(text))
    if not matches:
        return None
    heading = matches[-1]
    # Some scraped pages contain an empty "Referências" heading immediately
    # before "Bibliografia". Keep neither orphaned in the derived text.
    for previous in reversed(matches[:-1]):
        if text[previous.end():heading.start()].strip():
            break
        heading = previous
    prefix, suffix = text[:heading.start()], text[heading.end():]
    if heading.start() < len(text) * 0.6 or len(prefix.split()) < 200:
        return None
    dated_lines = [line for line in suffix.splitlines()
                   if re.search(r"\b(?:19|20)\d{2}\b", line)]
    if len(suffix.split()) < 30 or len(dated_lines) < 3:
        return None
    return heading.start()


def propose_n2_edits(text: str, config: RuleConfig) -> tuple[N2Edit, ...]:
    """Locate auditable N2 cuts without using label, source, or split metadata."""

    if config.stage not in {"n2", "n3"}:
        raise ValueError("N2 proposals require a stage=n2 or n3 configuration")
    candidates: list[N2Edit] = []
    if config.remove_bounded_parallel_abstracts:
        span = _parallel_abstract_span(text, config.min_english_abstract_words)
        if span is not None:
            candidates.append(N2Edit("PARALLEL_ABSTRACT_REMOVE", *span, "\n\n"))
    if config.remove_strict_terminal_references:
        start = _terminal_references_start(text)
        if start is not None:
            candidates.append(N2Edit("TERMINAL_REFERENCES_REMOVE", start, len(text)))
    if config.remove_strict_terminal_footers:
        lines = text.splitlines(keepends=True)
        start, removed = len(text), 0
        for line in reversed(lines):
            if not FOOTER.fullmatch(line.strip()):
                break
            start -= len(line)
            removed += 1
        if removed:
            candidates.append(N2Edit("TERMINAL_FOOTER", start, len(text), count=removed))
    accepted: list[N2Edit] = []
    for edit in sorted(candidates, key=lambda item: (item.start, -item.end)):
        if not accepted or edit.start >= accepted[-1].end:
            accepted.append(edit)
    return tuple(accepted)


def apply_n2_edits(text: str, edits: tuple[N2Edit, ...]) -> str:
    """Apply validated, non-overlapping spans relative to the N1 text."""

    result = text
    for edit in reversed(edits):
        if not 0 <= edit.start < edit.end <= len(text):
            raise ValueError("Invalid N2 span")
        result = result[:edit.start] + edit.replacement + result[edit.end:]
    return result.strip()


def n2_loss_guard(before: str, after: str, config: RuleConfig) -> bool:
    """Veto a substantial N2-only loss; N1's existing changes are unaffected."""

    original_words, retained_words = len(before.split()), len(after.split())
    return (
        original_words >= config.guard_min_original_words
        and original_words - retained_words >= config.guard_min_lost_words
        and retained_words / original_words < config.guard_min_retained_fraction
    )


def _n3_presentation_cleanup(text: str, config: RuleConfig) -> tuple[str, Counter[str], set[str]]:
    """Remove only explicit presentation syntax; keep prose and code content.

    Fenced code bodies are copied verbatim. An unclosed fence is retained and
    flagged rather than guessing where prose resumes. Inline code content is
    protected while prose on the same line is cleaned.
    """

    operations: Counter[str] = Counter()
    flags: set[str] = set()
    # HTML/CSS examples may contain Markdown and placeholder-looking literals.
    # Retain the whole record when their boundaries cannot be safely isolated.
    if re.search(r"<\s*(?:pre|code)\b", text, re.I):
        flags.add("N3_HTML_CODE_RETAINED")
        return text, operations, flags
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        bare = line.rstrip("\n")
        opening = FENCE_LINE.fullmatch(bare) if config.neutralize_markdown else None
        if opening:
            marker = opening.group(1)
            closing = next((end for end in range(index + 1, len(lines))
                            if (candidate := FENCE_LINE.fullmatch(lines[end].rstrip("\n")))
                            and candidate.group(1)[0] == marker[0]
                            and len(candidate.group(1)) >= len(marker)
                            and not candidate.group(2).strip()), None)
            if closing is None:
                flags.add("N3_UNCLOSED_FENCE")
                output.extend(lines[index:])
                break
            output.extend(lines[index + 1:closing])
            operations["MARKDOWN_FENCE"] += 2
            index = closing + 1
            continue

        newline = "\n" if line.endswith("\n") else ""
        prose = bare
        protected: list[str] = []
        if config.neutralize_markdown:
            def protect_code(match: re.Match[str]) -> str:
                protected.append(match.group(1))
                operations["MARKDOWN_INLINE_CODE"] += 1
                return f"\x00HORUS_CODE_{len(protected) - 1}\x00"

            prose = MD_INLINE_CODE.sub(protect_code, prose)
            prose = _replace(MD_LINK, prose, lambda match: match.group(1), operations,
                             "MARKDOWN_LINK")
            prose = _replace(MD_STRONG, prose, lambda match: match.group(1) or match.group(2),
                             operations, "MARKDOWN_EMPHASIS")
            prose = _replace(MD_EMPHASIS, prose, lambda match: match.group(1) or match.group(2),
                             operations, "MARKDOWN_EMPHASIS")
            prose = _replace(MD_HEADING, prose, "", operations, "MARKDOWN_HEADING")
            prose = _replace(MD_BULLET, prose, "", operations, "MARKDOWN_BULLET")
        if config.remove_explicit_placeholders:
            prose = _replace(PLACEHOLDER, prose, "", operations, "PLACEHOLDER_REMOVE")
        prose = _replace(SPACE_RUN, prose, " ", operations, "N3_SPACE_RUN")
        for code_index, code in enumerate(protected):
            prose = prose.replace(f"\x00HORUS_CODE_{code_index}\x00", code)
        output.append(prose + newline)
        index += 1
    result = "".join(output)
    # Keep the established whitespace policy without changing paragraph breaks.
    result = TRAILING_SPACE.sub("", result).strip()
    return result, operations, flags


def _n3_loss_guard(before: str, after: str, config: RuleConfig) -> bool:
    """Count lexical words, so removal of list glyphs does not mimic content loss."""

    if before and not after:
        return True
    original_words = len(WORD.findall(before))
    retained_words = len(WORD.findall(after))
    return (
        original_words >= config.guard_min_original_words
        and original_words - retained_words >= config.guard_min_lost_words
        and retained_words / original_words < config.guard_min_retained_fraction
    )


def normalize_text(content: str, config: RuleConfig) -> NormalizedText:
    """Return a derived text and diagnostic flags; never mutate ``content``."""

    if not isinstance(content, str):
        raise TypeError("content must be a string")
    original = content
    flags: set[str] = set()
    operations: Counter[str] = Counter()

    if ABSTRACT_START.search(original):
        flags.add("LANG_SECTION")
    if _english_abstract(original, config.min_english_abstract_words):
        flags.add("ENGLISH_ABSTRACT_CANDIDATE")
    if REFERENCES.search(original):
        flags.add("REFERENCES")
    if _terminal_references_start(original) is not None:
        flags.add("TERMINAL_REFERENCES_CANDIDATE")
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
    residual_tags = list(HTML_TAG.finditer(text))
    if residual_tags:
        flags.add("HTML_REMAINS")
        if not code_like and len(residual_tags) >= config.min_html_residual_tags_for_review:
            flags.add("HTML_DENSE_REMAINS")

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

    if config.stage in {"n2", "n3"}:
        edits = propose_n2_edits(text, config)
        if edits:
            proposed = apply_n2_edits(text, edits)
            if n2_loss_guard(text, proposed, config):
                flags.add("N2_GUARD_RETAINED")
            else:
                text = proposed
                for edit in edits:
                    operations[edit.kind] += edit.count
    elif FOOTER.search(text):
        flags.add("EDITORIAL_FOOTER")

    if config.stage == "n3":
        presented, presentation_operations, presentation_flags = _n3_presentation_cleanup(text, config)
        flags.update(presentation_flags)
        if _n3_loss_guard(text, presented, config):
            flags.add("N3_GUARD_RETAINED")
        else:
            if presented != text:
                flags.add("N3_CHANGED")
            if any(name.startswith("MARKDOWN_") for name in presentation_operations):
                flags.add("N3_MARKDOWN_CHANGED")
            if presentation_operations["PLACEHOLDER_REMOVE"]:
                flags.add("N3_PLACEHOLDER_CHANGED")
            text = presented
            operations.update(presentation_operations)

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
    # Broad markers remain available for prevalence analyses, but do not
    # automatically put authentic text in the high-precision review queue.
    status = "review" if flags & TEXT_REVIEW_FLAGS else "ok"
    return NormalizedText(
        text=text, operations=dict(sorted(operations.items())),
        flags=tuple(sorted(flags)), status=status,
        original_sha256=_sha(original), normalized_sha256=_sha(text),
        original_chars=len(original), normalized_chars=len(text),
        original_words=original_words, normalized_words=normalized_words,
    )
