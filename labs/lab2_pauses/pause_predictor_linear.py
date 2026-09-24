"""Pause predictor — linear / logistic regression baseline, lab 2.

Design notes (see the report for the full diagnosis of the first two iterations):

  1. Hard rule override for "strong" punctuation (period, ellipsis, !, ?): these
     have P(pause) >= 0.9 in the data -- forced to `is_pause=1` regardless of what
     the classifier says.
  2. Decision threshold tuned to maximize F1 on a held-out slice of `train` (not
     `test`), searched over a coarse grid (0.10 .. 0.90, step 0.05 -- 17 points)
     instead of the full precision-recall curve, and instead of relying on
     `class_weight='balanced'` (which optimizes a symmetric objective, not F1).
  3. `pause_duration` regressed directly in seconds -- an earlier log1p/expm1
     attempt made MAE *worse* (Jensen's inequality: exp(E[log X]) systematically
     underestimates E[X] for right-skewed X), so it was reverted.
  4. Feature set: `punct_class` (dominant feature), `word_len`, `rel_pos`,
     `pos_in_sentence`, `sent_len`, POS context +/-2, and `tokens_since_punct` /
     `tokens_to_punct` -- distance to the nearest punctuation mark, meant to help
     the weak, unpunctuated ("none") segment specifically.

Run as a script to fit on `train` and score on both folds::

    python pause_predictor_linear.py
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, precision_score, recall_score
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

# Coarse threshold grid: 0.10, 0.15, ..., 0.90 -- 17 points.
THRESHOLD_GRID = np.round(np.arange(0.10, 0.90 + 1e-9, 0.05), 2)


def _build_preprocessor() -> ColumnTransformer:
    """One-hot for categorical (punctuation, POS context), scaling for numeric --
    linear models need this; tree-based CatBoost (other module) does not.
    """
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
        ('num', StandardScaler(), NUMERIC_FEATURES),
    ])


def _select_threshold(y_true: np.ndarray, y_proba: np.ndarray, grid=THRESHOLD_GRID) -> float:
    """Threshold from `grid` maximizing F1 on the given (validation) set."""
    best_threshold, best_f1 = 0.5, -1.0
    for t in grid:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    return float(best_threshold)


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
        y_reg = train_df.pause_duration.values[tp_mask]  # raw seconds, no log transform
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
            durations = self.reg.predict(feats.loc[mask, FEATURE_COLUMNS])
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

        print(f'\n-- {name} --')
        calc_metrics(evaluable)

        if name == 'test':
            print("\n=== Метрики по классам пунктуации (Test) ===")
            for cls in evaluable.punct_class.unique():
                sub = evaluable[evaluable.punct_class == cls]
                if sub.is_pause_after.sum() == 0:
                    continue
                prc = precision_score(sub.is_pause_after, sub.is_pause_hat, zero_division=0)
                rec = recall_score(sub.is_pause_after, sub.is_pause_hat, zero_division=0)
                print(f"{cls:<10} | Всего токенов: {sub.shape[0]:<6} | P = {prc:.3f} | R = {rec:.3f}")
