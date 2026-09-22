"""Pause predictor — linear / logistic regression baseline, lab 2.

Design, in order of what changed after the first pass got F1~0.68-0.69 with low
precision (see report for the full diagnosis -- train/test gap was small, so the
problem was bias/calibration, not overfitting):

  1. Hard rule override for "strong" punctuation (period, ellipsis, !, ?): these
     have P(pause) >= 0.9 in the data -- forced to `is_pause=1` regardless of what
     the classifier says, guaranteeing near-total recall on the unambiguous part of
     the corpus instead of leaving it to a probabilistic model.
  2. Decision threshold tuned to maximize F1 on a held-out slice of `train` (not
     `test`), instead of relying on `class_weight='balanced'`, which optimizes a
     symmetric objective rather than F1 and was pushing recall up at precision's
     expense.
  3. `pause_duration` regressed in log-space (`log1p` / `expm1`) -- durations are
     strongly right-skewed, and a plain linear fit on raw seconds is pulled around
     by the long tail.
  4. New lexical/syntactic features from `features.py` (gerund flags, next-token
     SCONJ/CCONJ, curated conjunction markers) feed straight into the same model.

Run as a script to fit on `train` and score on both folds::

    python pause_predictor_linear.py
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import (
    FeatureExtractor, CATEGORICAL_FEATURES, NUMERIC_FEATURES, ALL_FEATURES,
    load_cached_features,
)

PAUSE_PREDICTOR_DATA = 'data/RUSLAN_pause_metadata.csv'
FEATURES_CACHE_PATH = 'data/RUSLAN_pause_features.csv'
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Punctuation classes with P(pause) >= ~0.9 in the EDA (period/multi-dot/!/?);
# "closing" (~0.75) and "semicolon" (1 sample in the corpus) are deliberately left
# out -- not reliable enough to force without the model's judgement.
STRONG_PUNCT_CLASSES = frozenset({'period', 'ellipsis', 'exclaim', 'question'})


def _build_preprocessor() -> ColumnTransformer:
    """One-hot for categorical (punctuation, POS context), scaling for numeric --
    linear models need this; tree-based CatBoost (other module) does not.
    """
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
        ('num', StandardScaler(), NUMERIC_FEATURES),
    ])


def _select_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Decision threshold maximizing F1 on the given (val) set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = int(np.nanargmax(f1s[:-1]))  # last P/R pair has no matching threshold
    return float(thresholds[best_idx])


class PausePredictorLinear:
    """Predicts pause placement and duration with linear/logistic regression, a
    hard rule for strong punctuation, and a tuned decision threshold.

    Same public contract as `pause_predictor.PausePredictor` (`predict`,
    `predict_durations`), plus a `fit()` used to train it first.
    """

    def __init__(self, use_pos: bool = True, strong_punct_classes: frozenset = STRONG_PUNCT_CLASSES,
                 val_size: float = 0.15, random_state: int = 42):
        self.feature_extractor = FeatureExtractor(use_pos=use_pos)
        self.strong_punct_classes = strong_punct_classes
        self.val_size = val_size
        self.random_state = random_state

        self.clf = Pipeline([
            ('prep', _build_preprocessor()),
            ('model', LogisticRegression(max_iter=1000)),
        ])
        self.reg = Pipeline([
            ('prep', _build_preprocessor()),
            ('model', LinearRegression()),
        ])
        self.threshold = 0.5
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> 'PausePredictorLinear':
        """Fit both models.

        `train_df` must be restricted to `set == 'train'` rows, but should include
        every sentence's last word -- features need full-sentence context (position,
        POS neighbours). The last-word exclusion from training targets (per the
        lab's protocol -- its trailing pause is a recording-boundary artifact, not
        natural speech) is handled internally here.
        """
        train_df = train_df.reset_index(drop=True)
        if set(ALL_FEATURES).issubset(train_df.columns):
            # Features already attached (e.g. loaded from the cache) -- skip
            # extraction, this is the whole point of the cache.
            feats = train_df
        else:
            feats = self.feature_extractor.extract_dataframe(train_df)

        trainable = train_df.is_last_word.values == 0
        X_cls = feats.loc[trainable, FEATURE_COLUMNS]
        y_cls = train_df.is_pause_after.values[trainable]

        # Threshold picked on a held-out slice of train, then the final classifier
        # is refit on the full trainable set so no data is wasted for the model
        # itself -- only for threshold selection.
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_cls, y_cls, test_size=self.val_size, random_state=self.random_state, stratify=y_cls,
        )
        self.clf.fit(X_fit, y_fit)
        val_proba = self.clf.predict_proba(X_val)[:, 1]
        self.threshold = _select_threshold(y_val, val_proba)
        self.clf.fit(X_cls, y_cls)

        tp_mask = trainable & (train_df.is_pause_after.values == 1)
        X_reg = feats.loc[tp_mask, FEATURE_COLUMNS]
        y_reg = np.log1p(train_df.pause_duration.values[tp_mask])  # log-space target
        self.reg.fit(X_reg, y_reg)

        self._fitted = True
        return self

    def _classify(self, feats: pd.DataFrame) -> np.ndarray:
        """Probability-threshold classification + hard override for strong
        punctuation, shared by `predict` and `predict_batch`.
        """
        X = feats[FEATURE_COLUMNS]
        proba = self.clf.predict_proba(X)[:, 1]
        is_pause = (proba >= self.threshold).astype(int)
        strong_mask = feats.punct_class.isin(self.strong_punct_classes).values
        is_pause[strong_mask] = 1
        return is_pause

    def _regress(self, feats: pd.DataFrame, is_pause: np.ndarray) -> np.ndarray:
        n = len(feats)
        pause_duration = np.zeros(n, dtype=float)
        mask = is_pause.astype(bool)
        if mask.any():
            log_durations = self.reg.predict(feats.loc[mask, FEATURE_COLUMNS])
            durations = np.expm1(log_durations)
            durations = np.clip(durations, a_min=1e-3, a_max=None)
            pause_duration[mask] = durations
        return pause_duration

    def predict(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Same contract as `pause_predictor.PausePredictor.predict`.

        `tokens` is one full sentence's `label_raw` tokens, in order.
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before predict().')

        tokens = list(tokens)
        feats = self.feature_extractor.extract_sentence(tokens)
        is_pause = self._classify(feats)
        pause_duration = self._regress(feats, is_pause)
        return is_pause, pause_duration

    def predict_durations(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Same contract as `pause_predictor.PausePredictor.predict_durations`."""
        def expand_is_pause(token, is_pause):
            return [token, '<SIL>'] if bool(is_pause) else [token]

        def expand_durations(pause_duration):
            return [-1., pause_duration] if pause_duration > 0. else [-1.]

        is_pause, durations = self.predict(tokens)
        tokens_w_pauses = np.concatenate([expand_is_pause(a, b) for a, b in zip(tokens, is_pause)])
        durations_w_pauses = np.concatenate([expand_durations(d) for d in durations]).astype(np.float32)
        return tokens_w_pauses, durations_w_pauses

    def predict_batch(self, feats_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized prediction directly from a frame that already carries
        `ALL_FEATURES` columns (e.g. loaded from the features cache).

        Skips feature extraction (no Natasha calls) entirely -- unlike `predict()`,
        which is for real-time single-sentence inference (lab 5), this is for fast
        bulk evaluation over an already-featurized train/test set.
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before predict_batch().')

        is_pause = self._classify(feats_df)
        pause_duration = self._regress(feats_df, is_pause)
        return is_pause, pause_duration


def _load_data_with_features() -> pd.DataFrame:
    """Load the metadata, attach cached features if available (built by
    `build_features_cache.py`), or fall back to extracting them now.
    """
    df = pd.read_csv(PAUSE_PREDICTOR_DATA, sep='|', quoting=csv.QUOTE_NONE)

    cached = load_cached_features(df, FEATURES_CACHE_PATH)
    if cached is not None:
        print(f'[pause_predictor_linear] Loaded features from {FEATURES_CACHE_PATH}')
        return cached

    print(f'[pause_predictor_linear] No usable cache at {FEATURES_CACHE_PATH} -- '
          f'extracting features now (consider running build_features_cache.py first).')
    return FeatureExtractor(use_pos=True).extract_dataframe(df)


if __name__ == '__main__':
    from pause_predictor import calc_metrics

    df = _load_data_with_features()
    train_df = df[df.set == 'train'].reset_index(drop=True)
    test_df = df[df.set == 'test'].reset_index(drop=True)

    pp = PausePredictorLinear().fit(train_df)
    print(f'Tuned decision threshold: {pp.threshold:.3f}')

    for name, part in [('train', train_df), ('test', test_df)]:
        evaluable = part[part.is_last_word == 0].copy()
        is_pause_hat, pause_duration_hat = pp.predict_batch(evaluable)
        evaluable['is_pause_hat'] = is_pause_hat
        evaluable['pause_duration_hat'] = pause_duration_hat
        print(f'-- {name} --')
        calc_metrics(evaluable)
