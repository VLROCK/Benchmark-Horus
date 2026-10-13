"""Observable text features and independent hashes for shortcut auditing.

These measurements are diagnostic. They do not edit the input or define the
future benchmark's inclusion rules.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from functools import lru_cache


WORD_RE = re.compile(r"\w+", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
BULLET_RE = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)")
HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
FENCE_RE = re.compile(r"(?m)^\s*```|^\s*~~~")
HTML_TAG_RE = re.compile(r"<\s*/?\s*[A-Za-z][^>]*>")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
PARAGRAPH_RE = re.compile(r"\n\s*\n+")


@lru_cache(maxsize=4096)
def _is_punctuation(character: str) -> bool:
    return unicodedata.category(character).startswith("P")


@lru_cache(maxsize=4096)
def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def sha256_utf8(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def conservative_comparison_text(value: str) -> str:
    """Normalize only for independent duplicate comparison, never for output.

    Unicode NFKC, casefolding and whitespace collapse can create false
    positives for semantically distinct texts, so matches require review.
    """

    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def extract_text_features(value: str) -> dict[str, int | float | bool | str]:
    """Measure length, punctuation and presentation on one unchanged text."""

    characters = Counter(value)
    words = value.split()
    word_count = len(words)
    lexical_tokens = WORD_RE.findall(value)
    punctuation_count = sum(n for char, n in characters.items() if _is_punctuation(char))
    letter_count = sum(n for char, n in characters.items() if _is_letter(char))
    normalized_lines = value.replace("\r\n", "\n").replace("\r", "\n")
    nonempty_lines = sum(bool(line.strip()) for line in normalized_lines.split("\n"))
    paragraph_count = (
        len(PARAGRAPH_RE.split(normalized_lines.strip())) if value.strip() else 0
    )
    divisor = max(word_count, 1)

    return {
        "content_sha256_audit": sha256_utf8(value),
        "comparison_sha256_audit": sha256_utf8(conservative_comparison_text(value)),
        "char_count_audit": len(value),
        "word_count_audit": word_count,
        "lexical_token_count": len(lexical_tokens),
        "letters_per_word": letter_count / divisor,
        "chars_per_word": len(value) / divisor,
        "punctuation_per_100_words": 100 * punctuation_count / divisor,
        "comma_per_100_words": 100 * (characters[","] + characters["，"]) / divisor,
        "period_per_100_words": 100 * (characters["."] + characters["…"]) / divisor,
        "semicolon_per_100_words": 100 * characters[";"] / divisor,
        "colon_per_100_words": 100 * characters[":"] / divisor,
        "question_per_100_words": 100 * characters["?"] / divisor,
        "exclamation_per_100_words": 100 * characters["!"] / divisor,
        "quote_per_100_words": 100 * sum(
            characters[char] for char in ('"', "'", "“", "”", "‘", "’", "«", "»")
        ) / divisor,
        "parenthesis_per_100_words": 100 * sum(
            characters[char] for char in ("(", ")", "[", "]", "{", "}")
        ) / divisor,
        "digit_per_100_words": 100 * sum(n for char, n in characters.items() if char.isdigit()) / divisor,
        "newline_per_100_words": 100 * normalized_lines.count("\n") / divisor,
        "paragraph_count": paragraph_count,
        "nonempty_line_count": nonempty_lines,
        "markdown_bullet": bool(BULLET_RE.search(normalized_lines)),
        "markdown_heading": bool(HEADING_RE.search(normalized_lines)),
        "markdown_code_fence": bool(FENCE_RE.search(normalized_lines)),
        "html_tag": bool(HTML_TAG_RE.search(value)),
        "url_in_content": bool(URL_RE.search(value)),
        "replacement_character": "\ufffd" in value,
        "tab_character": "\t" in value,
    }


NUMERIC_FEATURES = (
    "char_count_audit",
    "word_count_audit",
    "lexical_token_count",
    "letters_per_word",
    "chars_per_word",
    "punctuation_per_100_words",
    "comma_per_100_words",
    "period_per_100_words",
    "semicolon_per_100_words",
    "colon_per_100_words",
    "question_per_100_words",
    "exclamation_per_100_words",
    "quote_per_100_words",
    "parenthesis_per_100_words",
    "digit_per_100_words",
    "newline_per_100_words",
    "paragraph_count",
    "nonempty_line_count",
)

BOOLEAN_FEATURES = (
    "markdown_bullet",
    "markdown_heading",
    "markdown_code_fence",
    "html_tag",
    "url_in_content",
    "replacement_character",
    "tab_character",
)
