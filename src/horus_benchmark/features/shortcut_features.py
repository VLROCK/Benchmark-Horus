"""Text-only feature groups for the frozen-release shortcut probes.

All measurements come from the supplied string. Source, generator, theme and
generation metadata are deliberately excluded. The groups are hypotheses to
audit, not an attempt to identify causal authorship features.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

from horus_benchmark.audit.text_features import WORD_RE, extract_text_features


_UNORDERED_LIST = re.compile(r"(?m)^[ \t]*[-*+][ \t]+\S")
_NUMBERED_LIST = re.compile(r"(?m)^[ \t]*\d+[.)][ \t]+\S")
_EMAIL = re.compile(r"(?i)\b(?:EMAIL|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})\b")
_URL_TOKEN = re.compile(r"\bURL\b")
_INLINE_CODE = re.compile(r"(?<!`)`[^`\n]+`(?!`)")
_MARKDOWN_LINK = re.compile(r"\[[^\]\n]+\]\([^\)\n]+\)")


LENGTH_FEATURES = (
    "log_words", "log_chars", "log_lexical_tokens",
)
PRESENTATION_FEATURES = (
    "paragraph_count", "nonempty_line_count", "newline_per_100_words",
    "markdown_bullet", "unordered_list_marker", "numbered_list_marker",
    "markdown_heading", "markdown_code_fence", "markdown_inline_code",
    "markdown_link", "html_tag", "url_marker", "email_marker",
    "replacement_character", "tab_character",
)
PUNCTUATION_FEATURES = (
    "punctuation_per_100_words", "comma_per_100_words",
    "period_per_100_words", "semicolon_per_100_words",
    "colon_per_100_words", "question_per_100_words",
    "exclamation_per_100_words", "quote_per_100_words",
    "parenthesis_per_100_words", "digit_per_100_words",
    "uppercase_per_100_letters", "unique_lexical_fraction",
    "chars_per_word",
)
ALL_SURFACE_FEATURES = tuple(dict.fromkeys(
    (*LENGTH_FEATURES, *PRESENTATION_FEATURES, *PUNCTUATION_FEATURES)
))
SURFACE_GROUPS = {
    "surface_length": LENGTH_FEATURES,
    "surface_presentation": PRESENTATION_FEATURES,
    "surface_punctuation": PUNCTUATION_FEATURES,
    "surface_all": ALL_SURFACE_FEATURES,
}


def extract_shortcut_features(text: str) -> dict[str, float]:
    """Return finite, reproducible features for one unchanged text."""

    if not isinstance(text, str):
        raise TypeError("Each document must be a string")
    base = extract_text_features(text)
    lexical = [token.casefold() for token in WORD_RE.findall(text)]
    letters = [char for char in text if char.isalpha()]
    n_letters = max(len(letters), 1)
    result = {
        "log_words": math.log1p(float(base["word_count_audit"])),
        "log_chars": math.log1p(float(base["char_count_audit"])),
        "log_lexical_tokens": math.log1p(float(base["lexical_token_count"])),
        "chars_per_word": float(base["chars_per_word"]),
        "paragraph_count": float(base["paragraph_count"]),
        "nonempty_line_count": float(base["nonempty_line_count"]),
        "newline_per_100_words": float(base["newline_per_100_words"]),
        "markdown_bullet": float(base["markdown_bullet"]),
        "unordered_list_marker": float(bool(_UNORDERED_LIST.search(text))),
        "numbered_list_marker": float(bool(_NUMBERED_LIST.search(text))),
        "markdown_heading": float(base["markdown_heading"]),
        "markdown_code_fence": float(base["markdown_code_fence"]),
        "markdown_inline_code": float(bool(_INLINE_CODE.search(text))),
        "markdown_link": float(bool(_MARKDOWN_LINK.search(text))),
        "html_tag": float(base["html_tag"]),
        "url_marker": float(base["url_in_content"] or bool(_URL_TOKEN.search(text))),
        "email_marker": float(bool(_EMAIL.search(text))),
        "replacement_character": float(base["replacement_character"]),
        "tab_character": float(base["tab_character"]),
        "punctuation_per_100_words": float(base["punctuation_per_100_words"]),
        "comma_per_100_words": float(base["comma_per_100_words"]),
        "period_per_100_words": float(base["period_per_100_words"]),
        "semicolon_per_100_words": float(base["semicolon_per_100_words"]),
        "colon_per_100_words": float(base["colon_per_100_words"]),
        "question_per_100_words": float(base["question_per_100_words"]),
        "exclamation_per_100_words": float(base["exclamation_per_100_words"]),
        "quote_per_100_words": float(base["quote_per_100_words"]),
        "parenthesis_per_100_words": float(base["parenthesis_per_100_words"]),
        "digit_per_100_words": float(base["digit_per_100_words"]),
        "uppercase_per_100_letters": 100 * sum(char.isupper() for char in letters) / n_letters,
        "unique_lexical_fraction": len(set(lexical)) / max(len(lexical), 1),
    }
    if set(result) != set(ALL_SURFACE_FEATURES):
        raise AssertionError("Surface feature schema drift")
    return result


def surface_matrix(texts: Sequence[str], group: str) -> np.ndarray:
    """Extract a text-only feature matrix with stable column order."""

    if group not in SURFACE_GROUPS:
        raise ValueError(f"Unknown surface group: {group}")
    names = SURFACE_GROUPS[group]
    matrix = np.asarray(
        [[features[name] for name in names]
         for features in (extract_shortcut_features(text) for text in texts)],
        dtype=np.float32,
    )
    if matrix.size == 0:
        return np.empty((0, len(names)), dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite surface feature")
    return matrix
