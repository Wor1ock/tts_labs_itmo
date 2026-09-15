"""Russian text normalizer — skeleton for lab 1.

Brings corpus text into a form usable for training a speech synthesizer.
"""
import re
import unicodedata

from utils.yoficator import CustomYoficator

_TRANSLATION_MAP = {
    # non-standard hyphens -> ASCII hyphen
    "\u2010": "-", "\u2011": "-", "\u00ad": "-", "\ufe63": "-",
    # non-standard dashes -> em dash
    "\u2012": "—", "\u2013": "—", "\u2015": "—", "\u2212": "—",
    # opening quote variants -> «
    "\u201e": "«", "\u201c": "«",
    # closing quote variants -> »
    "\u201d": "»", "\u00bb": "»",
    # single-quote variants -> '
    "\u2018": "'", "\u2019": "'", "\u0060": "'",
    # non-breaking / zero-width spaces -> regular space
    "\u00a0": " ", "\u202f": " ", "\u200b": " ", "\ufeff": " ",
    # markup/tech symbols -> space (not "", to avoid merging adjacent words)
    "*": " ", "/": " ", "<": " ", ">": " ", "_": " ", "#": " ",
    "@": " ", "\\": " ", "^": " ", "~": " ",
}
_TRANSLATE_TABLE = str.maketrans(_TRANSLATION_MAP)

_RE_BROKEN_EXCL = re.compile(r"!\.{1,}")
_RE_BROKEN_QUEST = re.compile(r"\?\.{1,}")
_RE_MULTI_DOTS = re.compile(r"\.{4,}")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?:;…])")
_RE_MULTI_SPACE = re.compile(r"[ \t]+")

_YOFICATOR = CustomYoficator()


class TextNormalizer:
    """Normalizes text in Russian.


        "!.."           -> "!"
        "«цитата»"      -> '"цитата"'
        "текст * мусор" -> "текст мусор"
        "де‑факто"      -> "де-факто"      # U+2011 -> ordinary hyphen

    **Word-changing edits.** The alignment for that utterance becomes invalid and the
    row must be dropped from the training set — but the logic itself is still needed
    for lab 5, where arbitrary user input arrives with no alignment at all::

        "в 1995 г."     -> "в тысяча девятьсот девяносто пятом году"
        "прим. автора"  -> "примечание автора"

    Example:
        >>> normalizer = TextNormalizer()
        >>> normalizer.normalize("Расстреливать надо таких писателей!.")
        'Расстреливать надо таких писателей!'
    """
    def normalize(self, text: str) -> str:
        """Normalize a single line.

        Args:
            text: Raw utterance text, exactly as stored in the corpus metadata.

        Returns:
            The normalized text. Returning the input unchanged is valid and common —
            most lines need nothing done to them.

        Note:
            Do not strip the combining acute accent ``U+0301``. It looks like part of
            the letter and is easily lost to "unicode cleanup", but it marks explicit
            stress and becomes labelled data for stress placement in lab 3.

            Normalize to NFC. Strings in NFC and NFD render identically in a terminal
            and compare unequal.
        """
        if not text or not isinstance(text, str):
            return ""

        text = unicodedata.normalize("NFC", text)
        text = text.translate(_TRANSLATE_TABLE)
        text = _RE_BROKEN_EXCL.sub("!", text)
        text = _RE_BROKEN_QUEST.sub("?", text)
        text = _RE_MULTI_DOTS.sub("…", text)

        # е->ё restoration
        _, text = _YOFICATOR.check_text(text)

        text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = _RE_MULTI_SPACE.sub(" ", text)

        return text.strip()