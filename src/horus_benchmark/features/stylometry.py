"""Versioned PT-BR stylometric measurements, independent of corpus labels.

Syllables are estimated from the pt_BR Pyphen hyphenation dictionary. They are
an approximation, not a phonological annotation. The Portuguese Flesch formula
is used as a readability descriptor, never as an English school-grade score.
UPOS annotations must be supplied by the caller's pinned Portuguese tagger.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import math
import re
from typing import Sequence

import numpy as np


FEATURE_VERSION = "horus-ptbr-stylometry-v1"
FUNCTION_WORD_VERSION = "pt-closed-class-v1"
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
SENTENCE = re.compile(r"(?<=[.!?…])\s+")
UPOS = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
        "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X")
# Closed-class surface forms fixed before the experiment. No corpus counts or
# class/source metadata are involved in defining this list.
FUNCTION_WORDS = tuple(sorted(set("""
a as o os um uma uns umas de do da dos das em no na nos nas por pelo pela
pelos pelas para ao aos à às com sem sob sobre entre até após ante contra
desde durante perante trás e ou mas porém contudo todavia nem porque pois
que se quando enquanto embora como conforme caso ainda já também só apenas
não sim muito mais menos tão tanto eu tu ele ela nós vós eles elas me te lhe
lhes mim ti si comigo contigo consigo meu minha meus minhas teu tua teus
tuas seu sua seus suas nosso nossa nossos nossas vosso vossa este esta
estes estas esse essa esses essas isto isso aquele aquela aqueles aquelas
aquilo quem qual quais quanto quantos quanta quantas cujo cuja cujos cujas
algum alguma alguns algumas nenhum nenhuma todos todas todo toda cada
outro outra outros outras qualquer quaisquer ambos ambas aqui ali lá onde
era eram é são foi foram será serão ser estar está estão estava estavam
ter tem têm tinha tinham haver há pode podem deve devem
""".split())))


def words(text: str) -> list[str]:
    return [value.casefold() for value in WORD.findall(text)]


@lru_cache(maxsize=1)
def _hyphenator():
    try:
        import pyphen
    except ImportError as error:
        raise RuntimeError("Install the detector extra (pyphen) for PT readability") from error
    return pyphen.Pyphen(lang="pt_BR")


@lru_cache(maxsize=100_000)
def syllables(word: str) -> int:
    return max(1, len(_hyphenator().positions(word)) + 1)


def mattr(tokens: Sequence[str], window: int = 50) -> float:
    """Moving average type/token ratio; shorter texts use one available window."""
    if window < 1:
        raise ValueError("MATTR window must be positive")
    if not tokens:
        return 0.0
    width = min(window, len(tokens))
    counts = Counter(tokens[:width])
    values = [len(counts) / width]
    for index in range(width, len(tokens)):
        old = tokens[index - width]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        counts[tokens[index]] += 1
        values.append(len(counts) / width)
    return float(np.mean(values))


def _moments(values: Sequence[float]) -> list[float]:
    array = np.asarray(values if len(values) else [0.0], dtype=float)
    return [float(array.mean()), float(array.std()),
            *[float(value) for value in np.quantile(array, [.25, .5, .75, .9])]]


READABILITY_NAMES = (
    "log_words", "log_chars", "log_sentences", "mean_word_chars",
    "std_word_chars", "q25_word_chars", "median_word_chars", "q75_word_chars",
    "q90_word_chars", "mean_sentence_words", "std_sentence_words",
    "q25_sentence_words", "median_sentence_words", "q75_sentence_words",
    "q90_sentence_words", "syllables_per_word", "polysyllabic_fraction",
    "long_word_fraction", "words_per_sentence", "chars_per_sentence",
    "flesch_pt",
)


def readability(text: str) -> np.ndarray:
    lexical = words(text)
    n = max(len(lexical), 1)
    sentence_words = [len(words(part)) for part in SENTENCE.split(text.strip())
                      if words(part)]
    sentence_count = max(len(sentence_words), 1)
    syl = [syllables(word) for word in lexical]
    asw = sum(syl) / n
    asl = len(lexical) / sentence_count
    result = np.asarray([
        math.log1p(len(lexical)), math.log1p(len(text)),
        math.log1p(len(sentence_words)),
        *_moments([len(word) for word in lexical]),
        *_moments(sentence_words), asw,
        sum(value >= 3 for value in syl) / n,
        sum(len(value) >= 7 for value in lexical) / n,
        asl, len(text) / sentence_count,
        248.835 - 1.015 * asl - 84.6 * asw,
    ], dtype=float)
    if result.shape != (len(READABILITY_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Invalid Portuguese readability feature vector")
    return result


def pos_frequencies(tags: Sequence[str]) -> np.ndarray:
    if isinstance(tags, str) or not isinstance(tags, (list, tuple, np.ndarray)):
        raise ValueError("UPOS must be a sequence from the pinned tagger")
    counter = Counter(str(tag) for tag in tags)
    if set(counter) - set(UPOS):
        raise ValueError(f"Unknown UPOS tags: {sorted(set(counter) - set(UPOS))}")
    denominator = max(len(tags), 1)
    return np.asarray([counter[tag] / denominator for tag in UPOS], dtype=float)


PUNCTUATION = (",", ".", ";", ":", "?", "!", "…", "—", "-", '"',
               "'", "(", ")", "[", "]")
STYLE_EXTRA_NAMES = (
    "mattr50", "hapax_fraction", "repeat_word_fraction", "uppercase_fraction",
    "digit_fraction", "newline_per_word", "blank_line_per_word",
    *[f"punct_{ord(char)}_per_word" for char in PUNCTUATION],
    *[f"function_{word}" for word in FUNCTION_WORDS],
    *[f"upos_{tag}" for tag in UPOS],
)
STYLE_NAMES = (*READABILITY_NAMES, *STYLE_EXTRA_NAMES)
STRUCTURE_NAMES = {
    "log_words", "log_chars", "log_sentences", "newline_per_word",
    "blank_line_per_word", "words_per_sentence", "chars_per_sentence",
    "mean_sentence_words", "std_sentence_words", "q25_sentence_words",
    "median_sentence_words", "q75_sentence_words", "q90_sentence_words",
}


def stylometry(text: str, tags: Sequence[str], *, without_structure: bool = False
               ) -> np.ndarray:
    lexical = words(text)
    counts = Counter(lexical)
    n = max(len(lexical), 1)
    letters = [char for char in text if char.isalpha()]
    result = np.r_[readability(text), np.asarray([
        mattr(lexical), sum(value == 1 for value in counts.values()) / n,
        sum(max(value - 1, 0) for value in counts.values()) / n,
        sum(char.isupper() for char in letters) / max(len(letters), 1),
        sum(char.isdigit() for char in text) / max(len(text), 1),
        text.count("\n") / n, len(re.findall(r"\n\s*\n", text)) / n,
        *[text.count(char) / n for char in PUNCTUATION],
        *[counts[word] / n for word in FUNCTION_WORDS],
    ]), pos_frequencies(tags)]
    if result.shape != (len(STYLE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Invalid stylometry feature vector")
    if without_structure:
        result = result[[name not in STRUCTURE_NAMES for name in STYLE_NAMES]]
    return result
