"""Feature extraction for the pause predictor — lab 2.

Extracts, per token of a sentence, the features `PausePredictorLinear` and
`PausePredictorCatboost` train on:

    punct_class     -- type of punctuation right after the token (period, comma, ...)
                        -- the single strongest feature (~37% CatBoost importance)
    word_len        -- number of Cyrillic letters in the token
    rel_pos         -- relative position in the sentence, 0.0 (first) .. 1.0 (last)
    pos_in_sentence -- absolute 0-based position
    sent_len        -- number of tokens in the sentence
    is_last_word    -- 1 for the sentence's last word (present as a feature; the
                        *label* for this row is still excluded from training/eval
                        per the lab's protocol -- that filtering happens in the
                        predictor modules, not here)
    pos_prev2, pos_prev1, pos_curr, pos_next1, pos_next2
                    -- POS tags of the current token and its +/-2 neighbours,
                       tagged with Natasha (context-aware, sequence tagging)
    tokens_since_punct
                    -- how many tokens ago the last punctuation mark occurred (0 if
                       the *previous* token was punctuated, counted from sentence
                       start otherwise) -- signal for how deep into an unbroken,
                       comma-less stretch of the sentence we are
    tokens_to_punct -- how many tokens until the *next* punctuation mark (0 if the
                       current token itself is punctuated; distance to sentence end
                       if none follows) -- same idea, looking forward

Two entry points:
    FeatureExtractor.extract_sentence(tokens)   -- inference time, one sentence
    FeatureExtractor.extract_dataframe(df)      -- training time, batch over the
                                                    RUSLAN_pause_metadata.csv layout

Caching: extracting POS features over the whole corpus is slow (Natasha runs once
per sentence). `extract_and_cache_dataframe` / `load_cached_features` at the bottom
of this module let you run extraction once (see `build_features_cache.py`) and have
`pause_predictor_linear.py` / `pause_predictor_catboost.py` load the result instead
of recomputing it on every run.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import pandas as pd

try:
    from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
    _NATASHA_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without natasha installed
    _NATASHA_AVAILABLE = False


# --- punctuation classification -------------------------------------------------
# Same categories used during EDA (report.md / eda.ipynb §5), kept consistent so
# the feature distributions line up with the analysis that motivated them.

_PUNCT_PATTERNS = [
    (re.compile(r'\.\.\.$|…$'), 'ellipsis'),
    (re.compile(r'[.]$'), 'period'),
    (re.compile(r'[,]$'), 'comma'),
    (re.compile(r'[;]$'), 'semicolon'),
    (re.compile(r'[:]$'), 'colon'),
    (re.compile(r'[!]$'), 'exclaim'),
    (re.compile(r'[?]$'), 'question'),
    (re.compile(r'[—-]$'), 'dash'),
    (re.compile(r'[)\]»"]$'), 'closing'),
]
NO_PUNCT = 'none'
OTHER_PUNCT = 'other'

_LETTERS_RE = re.compile(r'[^а-яё]')

UNK_POS = 'UNK'    # Natasha/tokenization mismatch fallback
PAD_POS = 'NONE'   # out-of-sentence-bounds context (start/end of sentence)
CONTEXT_WINDOW = 2  # words before/after, as requested


def punct_class(label_raw: str) -> str:
    """Classify the punctuation that follows a token, from its `label_raw` form."""
    if not isinstance(label_raw, str):
        return NO_PUNCT
    s = label_raw.strip()
    for pattern, name in _PUNCT_PATTERNS:
        if pattern.search(s):
            return name
    return NO_PUNCT if re.search(r'[а-яёa-z]$', s.lower()) else OTHER_PUNCT


def clean_letters(label: str) -> str:
    """Keep only Cyrillic letters (drops punctuation/digits/spaces)."""
    if not isinstance(label, str):
        return ''
    return _LETTERS_RE.sub('', label.lower())


def word_len(label: str) -> int:
    """Number of Cyrillic letters in a token."""
    return len(clean_letters(label))


def punct_distances(punct_classes: list[str]) -> tuple[list[int], list[int]]:
    """Distance (in tokens) to the nearest punctuation mark, backward and forward.

    `tokens_since_punct[i]` -- tokens since the last punctuated token *before* i
    (0 means the immediately preceding token was punctuated); counted from the
    start of the sentence if there is no earlier punctuation.

    `tokens_to_punct[i]` -- tokens until the next punctuated token *at or after* i
    (0 means the token itself is punctuated); counted to the end of the sentence
    if no punctuation follows.
    """
    n = len(punct_classes)
    since = [0] * n
    to = [0] * n

    last_punct_idx = -1  # virtual punctuation boundary before the sentence starts
    for i in range(n):
        since[i] = i - last_punct_idx - 1
        if punct_classes[i] != NO_PUNCT:
            last_punct_idx = i

    next_punct_idx = n  # virtual punctuation boundary at the sentence end
    for i in range(n - 1, -1, -1):
        if punct_classes[i] != NO_PUNCT:
            next_punct_idx = i
        to[i] = next_punct_idx - i

    return since, to


# --- POS tagging (Natasha) -------------------------------------------------------

class PosTagger:
    """Thin wrapper around Natasha's morphology pipeline.

    `NewsEmbedding` is a sizeable model (downloaded on first use, cached locally
    afterwards) -- built once per `PosTagger` instance and reused, never rebuilt
    per sentence.
    """

    def __init__(self):
        if not _NATASHA_AVAILABLE:
            raise ImportError(
                "natasha is required for POS features. Install with `pip install natasha`."
            )
        self._segmenter = Segmenter()
        self._morph_vocab = MorphVocab()
        self._emb = NewsEmbedding()
        self._morph_tagger = NewsMorphTagger(self._emb)

    def tag_sentence(self, tokens: list[str]) -> list[str]:
        """POS-tag a sentence given as plain word tokens (letters only, no punctuation).

        Reconstructs a sentence string, runs Natasha's segmentation + morphology
        tagger on it (this is what makes the tagging *context-aware* -- it's not a
        per-word lookup), then aligns the resulting tags back to the input tokens by
        order.

        If Natasha's own tokenization doesn't line up 1:1 with the input (rare --
        e.g. an unusual token gets split), the mismatch is padded/truncated with
        `UNK_POS` rather than raised, so a single odd sentence never crashes a batch
        run over the whole corpus.
        """
        if not tokens:
            return []

        text = ' '.join(t for t in tokens if t)
        doc = Doc(text)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)

        word_tags = [t.pos for t in doc.tokens if t.pos != 'PUNCT']

        n = len(tokens)
        if len(word_tags) == n:
            return word_tags
        if len(word_tags) > n:
            return word_tags[:n]
        return word_tags + [UNK_POS] * (n - len(word_tags))


@lru_cache(maxsize=1)
def get_pos_tagger() -> PosTagger:
    """Process-wide singleton -- avoids reloading the embedding model repeatedly."""
    return PosTagger()


def pos_context(pos_tags: list[str], idx: int, window: int = CONTEXT_WINDOW) -> dict:
    """+/- `window` POS-tag context around position `idx`, padded with PAD_POS.

    Returns keys `pos_prev2, pos_prev1, pos_curr, pos_next1, pos_next2` for window=2.
    """
    feats = {}
    for offset in range(-window, window + 1):
        j = idx + offset
        if offset == 0:
            key = 'pos_curr'
        else:
            key = f'pos_{"prev" if offset < 0 else "next"}{abs(offset)}'
        feats[key] = pos_tags[j] if 0 <= j < len(pos_tags) else PAD_POS
    return feats


CATEGORICAL_FEATURES = ['punct_class', 'pos_prev2', 'pos_prev1', 'pos_curr', 'pos_next1', 'pos_next2']
NUMERIC_FEATURES = [
    'word_len', 'rel_pos', 'pos_in_sentence', 'sent_len',
    'tokens_since_punct', 'tokens_to_punct',
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass
class FeatureExtractor:
    """Builds the per-token feature table shared by both predictor implementations.

    Set `use_pos=False` to skip Natasha entirely (e.g. for a quick sanity run
    without the POS-context features, or if natasha isn't installed) -- POS columns
    are then filled with `UNK_POS`.
    """

    use_pos: bool = True
    _tagger: Optional[PosTagger] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.use_pos:
            self._tagger = get_pos_tagger()

    def extract_sentence(self, tokens: list[str]) -> pd.DataFrame:
        """Extract features for one sentence, given as ordered `label_raw` tokens.

        `tokens` is exactly what `PausePredictor.predict` receives: raw tokens with
        trailing punctuation, e.g. ["Я", "вышел", "из", "дома,", "когда", "стемнело."]
        """
        n = len(tokens)
        clean_tokens = [clean_letters(t) for t in tokens]
        punct_classes = [punct_class(t) for t in tokens]
        since_punct, to_punct = punct_distances(punct_classes)

        pos_tags = self._tagger.tag_sentence(clean_tokens) if self.use_pos else [UNK_POS] * n

        rows = []
        for i, tok in enumerate(tokens):
            row = {
                'punct_class': punct_classes[i],
                'word_len': word_len(tok),
                'rel_pos': i / max(n - 1, 1),
                'pos_in_sentence': i,
                'sent_len': n,
                'is_last_word': int(i == n - 1),
                'tokens_since_punct': since_punct[i],
                'tokens_to_punct': to_punct[i],
            }
            row.update(pos_context(pos_tags, i))
            rows.append(row)

        return pd.DataFrame(rows)

    def extract_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch feature extraction over a `RUSLAN_pause_metadata.csv`-shaped frame.

        Expects `id`, `label_raw` columns, with rows grouped and ordered by `id` (as
        written by `prepare_training_data.py`). Pass the *whole* sentence for every
        `id` present -- including its last word -- so position/context features are
        computed correctly; excluding the last-word *row* from training/evaluation
        is the caller's job (predictor modules), not this function's.

        Row order of the input is preserved in the output.
        """
        df = df.reset_index(drop=True)
        feat_rows: list = [None] * len(df)

        for _, group in df.groupby('id', sort=False):
            tokens = group.label_raw.tolist()
            feats = self.extract_sentence(tokens)
            for local_i, orig_i in enumerate(group.index):
                feat_rows[orig_i] = feats.iloc[local_i]

        feat_df = pd.DataFrame(feat_rows).reset_index(drop=True)
        return pd.concat([df, feat_df], axis=1)


# --- feature caching --------------------------------------------------------------
# Extracting POS features over the whole corpus (~250k tokens, one Natasha call per
# sentence) takes a while. Run `build_features_cache.py` once to compute them and
# write a cache file; `pause_predictor_linear.py` / `pause_predictor_catboost.py`
# then load that cache instead of recomputing on every run.

FEATURE_CACHE_COLUMNS = ['id'] + ALL_FEATURES


def extract_and_cache_dataframe(
    extractor: FeatureExtractor, df: pd.DataFrame, cache_path: str
) -> pd.DataFrame:
    """Extract features for the whole `df` and write them to `cache_path`.

    Returns `df` with the feature columns attached (same as `extract_dataframe`),
    so this can be used directly instead of `extract_dataframe` when you also want
    to persist the result.
    """
    result = extractor.extract_dataframe(df)
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    result[FEATURE_CACHE_COLUMNS].to_csv(cache_path, sep='|', index=False, quoting=csv.QUOTE_NONE)
    return result


def load_cached_features(df: pd.DataFrame, cache_path: str) -> Optional[pd.DataFrame]:
    """Load a features cache written by `extract_and_cache_dataframe`, if present and
    still matching `df` row-for-row (same `id` sequence).

    Returns `None` (rather than raising) if the cache is missing, stale, or from an
    older feature set (missing a column added since, or carrying columns dropped
    since), so callers fall back to recomputation instead of crashing or silently
    using outdated features.
    """
    if not os.path.exists(cache_path):
        return None

    df = df.reset_index(drop=True)
    cached = pd.read_csv(cache_path, sep='|', quoting=csv.QUOTE_NONE)

    if set(ALL_FEATURES) != set(cached.columns) - {'id'}:
        print(f'[features] Cache at {cache_path} has a different feature set than the '
              f'current code -- ignoring it, will recompute. Re-run build_features_cache.py '
              f'to refresh it.')
        return None

    if len(cached) != len(df) or not (cached.id.values == df.id.values).all():
        print(f'[features] Cache at {cache_path} does not match the current data '
              f'(different length or row order) -- ignoring it, will recompute.')
        return None

    return pd.concat([df, cached[ALL_FEATURES].reset_index(drop=True)], axis=1)
