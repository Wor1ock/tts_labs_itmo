"""Feature extraction for the pause predictor — lab 2.

Extracts, per token of a sentence, the features `PausePredictorLinear` and
`PausePredictorCatboost` train on:

    punct_class     -- type of punctuation right after the token (period, comma, ...)
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
    is_curr_gerund, is_prev1_gerund
                    -- whether the current / previous token is a gerund
                       ("деепричастие", VerbForm=Conv) -- a strong pause trigger
                       that a coarse POS tag alone does not capture (a gerund is
                       tagged VERB, same as a finite verb)
    next_is_sconj, next_is_cconj
                    -- whether the *next* token is a subordinating / coordinating
                       conjunction (SCONJ / CCONJ) -- these usually open a new
                       clause, which is a common (often comma-less) pause site
    next_is_marker  -- whether the next token's word form is a curated discourse
                       marker / conjunction / relative pronoun (see
                       CONJUNCTION_MARKERS below) -- a lexical backstop for cases
                       the POS tagger mis-tags or where the marker isn't a SCONJ

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
from tqdm import tqdm

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


# --- lexical markers --------------------------------------------------------------
# Small, curated set of Russian subordinating/coordinating conjunctions and relative
# pronouns that commonly open a new clause -- a lexical backstop alongside the SCONJ
# / CCONJ POS signal, for cases the tagger mislabels or that carry the meaning
# without being tagged as a conjunction (e.g. relative "который").

CONJUNCTION_MARKERS = frozenset({
    'однако', 'хотя', 'чтобы', 'если', 'пока', 'словно', 'будто', 'ибо',
    'потому', 'поэтому', 'зато', 'причём', 'притом', 'также', 'причем',
    'который', 'которая', 'которое', 'которые', 'которого', 'которой',
    'которых', 'которую', 'которым', 'которыми', 'котором',
})


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

    def tag_sentence(self, tokens: list[str]) -> tuple[list[str], list[bool]]:
        """POS-tag a sentence given as plain word tokens (letters only, no punctuation).

        Reconstructs a sentence string, runs Natasha's segmentation + morphology
        tagger on it (this is what makes the tagging *context-aware* -- it's not a
        per-word lookup), then aligns the results back to the input tokens by order.

        Returns `(pos_tags, is_gerund)`: POS tags, and whether each token is a
        gerund ("деепричастие", VerbForm=Conv -- tagged VERB same as a finite verb,
        so this needs the morphological features, not just the coarse POS).

        If Natasha's own tokenization doesn't line up 1:1 with the input (rare --
        e.g. an unusual token gets split), the mismatch is padded/truncated rather
        than raised, so a single odd sentence never crashes a batch run.
        """
        if not tokens:
            return [], []

        text = ' '.join(t for t in tokens if t)
        doc = Doc(text)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)

        word_tokens = [t for t in doc.tokens if t.pos != 'PUNCT']
        word_tags = [t.pos for t in word_tokens]
        gerund_flags = [bool(t.feats) and t.feats.get('VerbForm') == 'Conv' for t in word_tokens]

        n = len(tokens)
        if len(word_tags) == n:
            return word_tags, gerund_flags
        if len(word_tags) > n:
            return word_tags[:n], gerund_flags[:n]
        pad = n - len(word_tags)
        return word_tags + [UNK_POS] * pad, gerund_flags + [False] * pad


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
    'is_curr_gerund', 'is_prev1_gerund', 'next_is_sconj', 'next_is_cconj', 'next_is_marker',
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass
class FeatureExtractor:
    """Builds the per-token feature table shared by both predictor implementations.

    Set `use_pos=False` to skip Natasha entirely (e.g. for a quick sanity run
    without the POS-context/gerund/conjunction features, or if natasha isn't
    installed) -- POS-derived columns are then filled with defaults (`UNK_POS`, 0).
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

        if self.use_pos:
            pos_tags, gerund_flags = self._tagger.tag_sentence(clean_tokens)
        else:
            pos_tags, gerund_flags = [UNK_POS] * n, [False] * n

        rows = []
        for i, tok in enumerate(tokens):
            next_pos = pos_tags[i + 1] if i + 1 < n else PAD_POS
            next_word = clean_tokens[i + 1] if i + 1 < n else ''
            row = {
                'punct_class': punct_class(tok),
                'word_len': word_len(tok),
                'rel_pos': i / max(n - 1, 1),
                'pos_in_sentence': i,
                'sent_len': n,
                'is_last_word': int(i == n - 1),
                'is_curr_gerund': int(gerund_flags[i]) if i < len(gerund_flags) else 0,
                'is_prev1_gerund': int(gerund_flags[i - 1]) if 0 <= i - 1 < len(gerund_flags) else 0,
                'next_is_sconj': int(next_pos == 'SCONJ'),
                'next_is_cconj': int(next_pos == 'CCONJ'),
                'next_is_marker': int(next_word in CONJUNCTION_MARKERS),
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

        for _, group in tqdm(df.groupby('id', sort=False), total=df['id'].nunique(), desc="Extracting features by sentence"):
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
    older feature set (missing a column added since), so callers fall back to
    recomputation instead of crashing.
    """
    if not os.path.exists(cache_path):
        return None

    df = df.reset_index(drop=True)
    cached = pd.read_csv(cache_path, sep='|', quoting=csv.QUOTE_NONE)

    if not set(ALL_FEATURES).issubset(cached.columns):
        print(f'[features] Cache at {cache_path} is missing newer feature columns -- '
              f'ignoring it, will recompute. Re-run build_features_cache.py to refresh it.')
        return None

    if len(cached) != len(df) or not (cached.id.values == df.id.values).all():
        print(f'[features] Cache at {cache_path} does not match the current data '
              f'(different length or row order) -- ignoring it, will recompute.')
        return None

    return pd.concat([df, cached[ALL_FEATURES].reset_index(drop=True)], axis=1)
