"""Normalized / non-normalized classifier — skeleton for lab 1.

Run as a script to score yourself on the development set::

    python text_filter.py
"""
import csv
import re

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

DEV_SET_PATH = "data/dev_sentences.csv"

_REJECT_SUBPATTERNS = (
    r"\d",                                                              # digits
    r"[A-Za-z]",                                                         # latin letters
    r"[%°$€£₽№©§]",                                                      # designations
    r"[*/@+<>=_#\\^~|]",                                                 # tech symbols
    r"[()\[\]{}]",                                                       # brackets
    r"[!?]\.{1,}|[!?]{2,}",                                              # broken punctuation
    r"\b[А-ЯЁ]\.\s*[А-ЯЁ]?\.",                                           # initials
    r"\b(?i:г|ул|д|им|пр|т\.д|т\.п|т\.е|т\.к|руб|коп|прим|см|стр)\.",     # abbreviations
    r"\b[А-ЯЁ]{2,}\b",                                                   # acronyms
    r"https?://\S+|www\.\S+",                                            # urls
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",              # email addresses
    r"\+?\d[\d\s\-\(\)]{7,}\d",                                         # phone numbers
    r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b",                                    # dates (e.g. 18.10.2026)
    r"(?::|;|=)(?:-)?(?:\)|PR|Р|p|P|\(|D|S|O|o)",                       # text emoticons (e.g. :Р, :-D)
    r"\s[.,!?:;…]",                                                      # space BEFORE punctuation mark
    r"[.,!?:;…][А-ЯЁа-яё]",                                              # missing space AFTER punctuation mark
    r"(?<!\.)\.{2}(?!\.)",                                               # exactly two periods (..)
    r"\b(?i:хм+|ммм+|гмм+|ыы+|хехе|мда)\b",                              # interjections and filler words
    r"\b[А-ЯЁ]\.\s+[А-ЯЁ][а-яё]+",                                       # single initials (e.g. I. Petrov)
    r"\bг-(?:жи|н|жа|же)\b",                                             # hyphenated abbreviations (e.g. г-н)
    r"\b[А-ЯЁ]{2,}[а-яё]+\b",                                            # inflected acronyms (e.g. в Минюсте)
    r"\?!\.",                                                            # specific punctuation combo (?!.)
)

_REJECT_PATTERN = re.compile("|".join(_REJECT_SUBPATTERNS))

# Fallback alphabet check: catches anything not covered above (emoji, rare
# symbols). Cyrillic + stress mark (U+0301) + standard punctuation only.
_ALLOWED_PATTERN = re.compile(r"""[А-ЯЁа-яё\u0301.,!?:;"'«»…—\- ]+""")


class TextFilter:
    """Decides whether an utterance is usable as a training example.

    Example:
        >>> textfilter = TextFilter()
        >>> textfilter.filter("Я вышел из дома.")
        1
        >>> textfilter.filter("Александрову Г. П.")
        0
    """

    def filter(self, text: str) -> int:
        """Classify a single utterance.

        Args:
            text: Utterance text, already passed through :class:`TextNormalizer`.

        Returns:
            ``1`` if the text is normalized and the utterance can be used for
            training; 
            ``0`` if it contains something the speaker pronounced
            differently from how it is written, and the utterance should be dropped.
        """
        if not text or not isinstance(text, str):
            return 0

        text = text.strip()
        if not text:
            return 0

        if _REJECT_PATTERN.search(text):
            return 0

        if not _ALLOWED_PATTERN.fullmatch(text):
            return 0

        return 1


if __name__ == "__main__":
    text_filter = TextFilter()

    dev_set = pd.read_csv(
        DEV_SET_PATH, sep="|", encoding="utf-8", quoting=csv.QUOTE_NONE, header=0
    )
    dev_set["predicted"] = [text_filter.filter(t) for t in dev_set["text"]]

    precision = precision_score(dev_set["is_normalized"], dev_set["predicted"])
    recall = recall_score(dev_set["is_normalized"], dev_set["predicted"])
    f1 = f1_score(dev_set["is_normalized"], dev_set["predicted"])
    print(f"F1 Score is {f1:.4f}, Precision is {precision:.4f}, Recall is {recall:.4f}")